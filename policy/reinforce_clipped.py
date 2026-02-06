from typing import Union, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class REINFORCEClipped(REINFORCE):
    """REINFORCE algorithm with gradient clipping.
    Inherits from REINFORCE and adds gradient clipping functionality.

    Args:
        clip_val: Maximum allowed value for gradient clipping
        clip_type: Type of gradient clipping ('norm' or 'value')
        lr_decay: Learning rate decay type ('none', 'cosine', or 'linear')
        min_lr: Minimum learning rate at end of training
        **kwargs: Arguments passed to parent REINFORCE class
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        clip_val: float = 10.0,
        clip_type: str = "norm",
        lr_decay: str = "none",
        min_lr: float = 1e-6,
        exp_gamma: float | None = None,
        **kwargs,
    ):
        super().__init__(env, policy, baseline, **kwargs)
        
        self.clip_val = clip_val
        self.clip_type = clip_type
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.exp_gamma = exp_gamma
        
        assert clip_type in ["norm", "value"], "clip_type must be either 'norm' or 'value'"
        assert lr_decay in ["none", "cosine", "linear", "exponential"], "lr_decay must be 'none', 'cosine', 'linear', or 'exponential'"
        
        # Disable automatic optimization
        self.automatic_optimization = False
        
        self.save_hyperparameters(logger=False)

    def _dataloader_single(self, dataset, batch_size, shuffle=False):
        """Override RL4CO DataLoader to enable persistent workers when num_workers > 0."""
        num_workers = int(getattr(self, "dataloader_num_workers", 0) or 0)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            persistent_workers=num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def training_step(self, batch: Any, batch_idx: int):
        """Override training_step to add gradient clipping"""
        # Get optimizer
        opt = self.optimizers()
        opt.zero_grad()
        
        # Compute loss using parent's shared_step
        result = self.shared_step(batch, batch_idx, phase="train")
        
        # Get loss from result (could be dict or tensor)
        loss = result['loss'] if isinstance(result, dict) else result
        
        # Manually compute gradients
        self.manual_backward(loss)
        
        if batch_idx % 100 == 0:
            # Get pre-clipping stats
            grad_norms = [param.grad.norm().item() for param in self.parameters() if param.grad is not None]
            pre_max, pre_mean = max(grad_norms), sum(grad_norms)/len(grad_norms)
        
        # Clip gradients
        if self.clip_type == "norm":
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_val)
        else:  # clip_type == "value"
            torch.nn.utils.clip_grad_value_(self.parameters(), self.clip_val)
            
        if batch_idx % 100 == 0:
            # Get post-clipping stats
            grad_norms = [param.grad.norm().item() for param in self.parameters() if param.grad is not None]
            post_max, post_mean = max(grad_norms), sum(grad_norms)/len(grad_norms)
            # print(f"Batch {batch_idx} - Grads (max/mean): pre-clip=({pre_max:.3f}/{pre_mean:.3f}), post-clip=({post_max:.3f}/{post_mean:.3f})")
            
        # Update weights
        opt.step()

        # Lightning may retain `training_step` returns; ensure we don't return tensors
        # that keep the autograd graph alive (which can cause GPU memory blowups).
        if isinstance(result, dict):
            clean = {}
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    clean[k] = v.detach()
                else:
                    clean[k] = v
            return clean
        if isinstance(result, torch.Tensor):
            return result.detach()
        return result
    
    def on_train_epoch_end(self):
        """Step the scheduler if one is being used"""
        if self.lr_decay != "none":
            sch = self.lr_schedulers()
            sch.step()
        
        # Log the current learning rate regardless
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler based on lr_decay setting"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer_kwargs["lr"])
        
        # If no decay is requested, return just the optimizer
        if self.lr_decay == "none":
            return optimizer
        
        # Otherwise, set up the appropriate scheduler
        if self.lr_decay == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs,
                eta_min=self.min_lr
            )
        elif self.lr_decay == "linear":
            lambda_fn = lambda epoch: max(1.0 - (1 - self.min_lr / self.hparams.optimizer_kwargs["lr"]) * 
                                         (epoch / self.trainer.max_epochs), self.min_lr / self.hparams.optimizer_kwargs["lr"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda_fn
            )
        elif self.lr_decay == "exponential":
            if self.exp_gamma is not None:
                gamma = float(self.exp_gamma)
                if 0.0 < gamma < 1.0:
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
                else:
                    log.warning(f"Invalid exp_gamma={gamma} for exponential decay; expected 0<gamma<1. Disabling decay.")
                    return optimizer
            else:
                base_lr = float(self.hparams.optimizer_kwargs["lr"])
                min_lr = float(self.min_lr)
                if min_lr <= 0 or base_lr <= 0 or min_lr >= base_lr:
                    log.warning(
                        f"Invalid min_lr/base_lr for exponential decay (base_lr={base_lr}, min_lr={min_lr}); disabling decay."
                    )
                    return optimizer
                gamma = (min_lr / base_lr) ** (1.0 / max(1, int(self.trainer.max_epochs)))
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
