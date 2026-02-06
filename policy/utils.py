from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import torch
from torchrl.data import Composite

from policy.reinforce_clipped import REINFORCEClipped


def patch_env_specs(env) -> None:
    def _patch(spec):
        if isinstance(spec, Composite):
            if not hasattr(spec, "data_cls"):
                spec.data_cls = None
            if not hasattr(spec, "step_mdp_static"):
                spec.step_mdp_static = False
            for child in spec.values():
                if child is not None:
                    _patch(child)

    for spec_name in ["input_spec", "output_spec", "observation_spec", "reward_spec"]:
        spec = getattr(env, spec_name, None)
        if spec is not None:
            _patch(spec)


def _patch_rl4co_env_setstate() -> None:
    """Coerce legacy RNG state into ByteTensor during checkpoint unpickling."""
    try:
        from rl4co.envs.routing.tsp.env import TSPEnv
    except Exception:
        return

    if getattr(TSPEnv, "_codex_rng_patch", False):
        return

    def _patched_setstate(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
            rng_state = state.get("rng")
        else:
            self.__dict__.update(state)
            rng_state = None

        self.rng = torch.manual_seed(0)
        if rng_state is None:
            return
        try:
            if not isinstance(rng_state, torch.ByteTensor):
                if torch.is_tensor(rng_state):
                    rng_state = rng_state.to(torch.uint8)
                elif isinstance(rng_state, (bytes, bytearray)):
                    rng_state = torch.tensor(list(rng_state), dtype=torch.uint8)
                else:
                    rng_state = torch.as_tensor(rng_state, dtype=torch.uint8)
            self.rng.set_state(rng_state)
        except Exception:
            # Legacy RNG state formats can be ignored for inference-only use.
            pass

    TSPEnv.__setstate__ = _patched_setstate
    TSPEnv._codex_rng_patch = True


def _resolve_policy_checkpoint(
    run_dir: Path,
    *,
    checkpoint_path: Optional[Path] = None,
    checkpoint_epoch: Optional[int] = None,
) -> Path:
    """Resolve a policy checkpoint under runs/<run>/checkpoints."""
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        return path

    if checkpoint_epoch is not None:
        epoch = int(checkpoint_epoch)
        if epoch <= 0:
            raise ValueError(f"checkpoint_epoch must be >= 1, got {epoch}")
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found: {path}")
        return path

    candidates = list(checkpoint_dir.glob("checkpoint_epoch_*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(candidates, key=lambda p: int(p.stem.split("checkpoint_epoch_")[1]))


def load_env_and_policy(
    run_dir: Path,
    device: torch.device,
    *,
    checkpoint_path: Optional[Path] = None,
    checkpoint_epoch: Optional[int] = None,
):
    _patch_rl4co_env_setstate()
    env_path = run_dir / "env.pkl"
    config_path = run_dir / "config.json"

    if not env_path.exists():
        raise FileNotFoundError(f"Environment pickle missing: {env_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config missing: {config_path}")

    with open(env_path, "rb") as fp:
        env = pickle.load(fp)

    with open(config_path, "r") as fp:
        config = json.load(fp)

    patch_env_specs(env)

    ckpt = _resolve_policy_checkpoint(run_dir, checkpoint_path=checkpoint_path, checkpoint_epoch=checkpoint_epoch)

    from policy.policy_hooked import EnhancedHookedPolicy

    policy = EnhancedHookedPolicy(
        env_name=env.name,
        embed_dim=config["embed_dim"],
        num_encoder_layers=config["n_encoder_layers"],
        num_heads=int(config.get("num_heads", 8)),
        temperature=config["temperature"],
        dropout=config.get("dropout", 0.0),
        attention_dropout=config.get("attention_dropout", 0.0),
    )

    model = REINFORCEClipped.load_from_checkpoint(
        ckpt,
        env=env,
        policy=policy,
        strict=False,
        map_location=device,
    )
    policy = model.policy.to(device)
    policy.eval()
    return {"env": env, "config": config}, policy
