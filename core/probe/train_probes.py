#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train what-if linear probes on extracted representations.")
    p.add_argument("--reps_path", type=str, required=True, help="Path to probe_reps.pt from extract_representations.py")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory for models/metrics (default: <reps_dir>/probe_artifacts)")
    p.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=["linear", "mlp", "deepset", "transformer"],
        help="Probe model family (default: linear).",
    )
    p.add_argument("--mlp_hidden_dim", type=int, default=256, help="Hidden dim for --model mlp.")
    p.add_argument("--mlp_layers", type=int, default=1, help="Number of hidden layers for --model mlp.")
    p.add_argument("--mlp_dropout", type=float, default=0.0, help="Dropout probability for --model mlp.")
    p.add_argument("--tfm_dim", type=int, default=256, help="Model dim for --model transformer.")
    p.add_argument("--tfm_layers", type=int, default=2, help="Number of Transformer encoder layers.")
    p.add_argument("--tfm_heads", type=int, default=8, help="Number of attention heads for Transformer.")
    p.add_argument("--tfm_ff_mult", type=int, default=4, help="FFN multiplier for Transformer (ff_dim = tfm_dim*mult).")
    p.add_argument("--tfm_dropout", type=float, default=0.0, help="Dropout probability for Transformer.")
    p.add_argument(
        "--target",
        type=str,
        default="length",
        choices=["length", "time", "both"],
        help="Which label(s) to predict (default: length).",
    )
    p.add_argument(
        "--objective",
        type=str,
        default="regression",
        choices=["regression", "best_node_ce", "soft_ce", "pairwise_rank"],
        help=(
            "Training objective. regression predicts target values; best_node_ce classifies the best node per instance; "
            "soft_ce uses a softmax over target values as a distributional target; "
            "pairwise_rank uses a sampled pairwise ranking loss within each instance."
        ),
    )
    p.add_argument("--soft_ce_tau", type=float, default=1.0, help="Temperature for --objective soft_ce (higher = softer).")
    p.add_argument(
        "--pairwise_pairs_per_instance",
        type=int,
        default=128,
        help="Pairs per instance for --objective pairwise_rank (default: 128).",
    )
    p.add_argument(
        "--pairwise_margin",
        type=float,
        default=0.0,
        help="Optional margin: ignore pairs with |delta_i-delta_j| < margin (default: 0).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="Device string, e.g. cuda, cpu.")
    p.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size (rows for regression; instances for best_node_ce).",
    )
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--l1_lambda", type=float, default=0.0, help="Optional L1 penalty on weights.")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--standardize_x", action="store_true", help="Standardize X using train-set mean/std.")
    p.add_argument(
        "--instance_standardize_x",
        action="store_true",
        help="Standardize X per instance using per-feature mean/std over nodes (helps remove pool-specific fingerprints).",
    )
    p.add_argument(
        "--node_standardize_x",
        action="store_true",
        help="Standardize X per node using mean/std over features (layernorm-like).",
    )
    p.add_argument(
        "--save_models",
        action="store_true",
        help="Also write model checkpoint(s) to out_dir (default: metrics-only).",
    )
    p.add_argument(
        "--use_extra_features",
        action="store_true",
        help="If reps_path contains X_extra, concatenate it to X before training (simple heuristic-feature ensemble).",
    )
    return p


def _as_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().flatten().to(torch.float64)
    y = y.detach().flatten().to(torch.float64)
    n = int(x.numel())
    if n < 2:
        return float("nan")

    x_order = torch.argsort(x)
    y_order = torch.argsort(y)

    x_rank = torch.empty_like(x_order, dtype=torch.float64)
    y_rank = torch.empty_like(y_order, dtype=torch.float64)
    x_rank[x_order] = torch.arange(n, dtype=torch.float64)
    y_rank[y_order] = torch.arange(n, dtype=torch.float64)

    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denom = x_rank.std(unbiased=False) * y_rank.std(unbiased=False)
    if float(denom) == 0.0:
        return float("nan")
    return float((x_rank * y_rank).mean().item() / denom.item())


@dataclass(frozen=True)
class SplitMasks:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


def make_instance_splits(instance_ids: torch.Tensor, train_frac: float, val_frac: float, seed: int) -> SplitMasks:
    instance_ids = instance_ids.detach().cpu().to(torch.int64)
    num_instances = int(instance_ids.max().item()) + 1 if instance_ids.numel() else 0
    if num_instances <= 0:
        raise ValueError("No instances found in reps file")

    if not (0.0 < train_frac < 1.0) or not (0.0 <= val_frac < 1.0):
        raise ValueError("Bad split fractions")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_instances, generator=g)
    n_train = int(round(train_frac * num_instances))
    n_val = int(round(val_frac * num_instances))
    n_train = max(1, min(num_instances - 2, n_train))
    n_val = max(1, min(num_instances - n_train - 1, n_val))
    train_ids = perm[:n_train]
    val_ids = perm[n_train : n_train + n_val]
    test_ids = perm[n_train + n_val :]

    def _mask_for_ids(ids: torch.Tensor) -> torch.Tensor:
        id_set = set(int(i) for i in ids.tolist())
        return torch.tensor([int(i) in id_set for i in instance_ids.tolist()], dtype=torch.bool)

    return SplitMasks(
        train=_mask_for_ids(train_ids),
        val=_mask_for_ids(val_ids),
        test=_mask_for_ids(test_ids),
    )


def _standardize_targets(y: torch.Tensor, train_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_train = y[train_mask]
    mean = y_train.mean(dim=0)
    std = y_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (y - mean) / std, mean, std


def _standardize_x(X: torch.Tensor, train_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X_train = X[train_mask]
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (X - mean) / std, mean, std


def _regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    err = y_pred - y_true
    mse = float((err.pow(2).mean()).item())
    mae = float((err.abs().mean()).item())

    y_true_centered = y_true - y_true.mean()
    ss_tot = float((y_true_centered.pow(2).sum()).item())
    ss_res = float((err.pow(2).sum()).item())
    r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
    return {"mse": mse, "mae": mae, "r2": r2}


def _metrics_by_target(y_true: torch.Tensor, y_pred: torch.Tensor, target_names: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for t, name in enumerate(target_names):
        out[name] = _regression_metrics(y_true[:, t], y_pred[:, t])
    return out


def build_probe_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    mlp_hidden_dim: int,
    mlp_layers: int,
    mlp_dropout: float,
    tfm_dim: int = 256,
    tfm_layers: int = 2,
    tfm_heads: int = 8,
    tfm_ff_mult: int = 4,
    tfm_dropout: float = 0.0,
) -> nn.Module:
    if model_type == "linear":
        return nn.Linear(int(input_dim), int(output_dim), bias=True)
    if model_type not in ("mlp", "deepset", "transformer"):
        raise ValueError(f"Unknown model_type: {model_type}")

    hidden_dim = int(mlp_hidden_dim)
    if hidden_dim <= 0:
        raise ValueError("--mlp_hidden_dim must be >= 1")
    layers = int(mlp_layers)
    if layers <= 0:
        raise ValueError("--mlp_layers must be >= 1")
    dropout = float(mlp_dropout)
    if not (0.0 <= dropout < 1.0):
        raise ValueError("--mlp_dropout must be in [0,1)")

    def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
        parts: List[nn.Module] = []
        cur = int(in_dim)
        for _ in range(layers):
            parts.append(nn.Linear(cur, hidden_dim, bias=True))
            parts.append(nn.ReLU())
            if dropout > 0.0:
                parts.append(nn.Dropout(p=dropout))
            cur = hidden_dim
        parts.append(nn.Linear(cur, int(out_dim), bias=True))
        return nn.Sequential(*parts)

    if model_type == "mlp":
        return _make_mlp(int(input_dim), int(output_dim))

    if model_type == "deepset":
        class DeepSetsNodeScorer(nn.Module):
            def __init__(self, in_dim: int, out_dim: int):
                super().__init__()
                self.phi = _make_mlp(int(in_dim), hidden_dim)
                self.rho = _make_mlp(hidden_dim * 2, int(out_dim))

            def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                h = self.phi(x)
                if key_padding_mask is not None:
                    valid = (~key_padding_mask).to(h.dtype).unsqueeze(-1)
                    pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
                else:
                    pooled = h.mean(dim=1)
                pooled = pooled.unsqueeze(1).expand(-1, h.shape[1], -1)
                return self.rho(torch.cat([h, pooled], dim=-1))

        return DeepSetsNodeScorer(in_dim=int(input_dim), out_dim=int(output_dim))

    # Transformer: set/sequence model over nodes. No positional embeddings -> permutation equivariant.
    model_dim = int(tfm_dim)
    if model_dim <= 0:
        raise ValueError("--tfm_dim must be >= 1")
    num_layers = int(tfm_layers)
    if num_layers <= 0:
        raise ValueError("--tfm_layers must be >= 1")
    num_heads = int(tfm_heads)
    if num_heads <= 0:
        raise ValueError("--tfm_heads must be >= 1")
    ff_mult = int(tfm_ff_mult)
    if ff_mult <= 0:
        raise ValueError("--tfm_ff_mult must be >= 1")
    dropout = float(tfm_dropout)
    if not (0.0 <= dropout < 1.0):
        raise ValueError("--tfm_dropout must be in [0,1)")

    class TransformerNodeScorer(nn.Module):
        def __init__(self, in_dim: int, d_model: int, n_layers: int, n_heads: int, ff_mult: int, dropout: float):
            super().__init__()
            self.in_proj = nn.Identity() if in_dim == d_model else nn.Linear(in_dim, d_model, bias=True)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * ff_mult,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.out = nn.Linear(d_model, int(output_dim), bias=True)

        def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            h = self.in_proj(x)
            if key_padding_mask is not None:
                h = self.encoder(h, src_key_padding_mask=key_padding_mask)
            else:
                h = self.encoder(h)
            return self.out(h)

    return TransformerNodeScorer(
        in_dim=int(input_dim),
        d_model=model_dim,
        n_layers=num_layers,
        n_heads=num_heads,
        ff_mult=ff_mult,
        dropout=dropout,
    )


def _reshape_by_instance_node(
    X: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    instance_id: torch.Tensor,
    node_id: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = X.detach().cpu()
    y = y.detach().cpu()
    valid = valid.detach().cpu()
    instance_id = instance_id.detach().cpu().to(torch.int64)
    node_id = node_id.detach().cpu().to(torch.int64)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be [N,d], got {tuple(X.shape)}")
    if y.ndim != 2:
        raise ValueError(f"Expected y to be [N,T], got {tuple(y.shape)}")
    if valid.ndim != 1:
        raise ValueError(f"Expected valid to be [N], got {tuple(valid.shape)}")

    N = int(X.shape[0])
    B = int(instance_id.max().item()) + 1
    n = int(node_id.max().item()) + 1
    if N != B * n:
        raise ValueError(f"Expected N == B*n, got N={N}, B={B}, n={n}")

    expected_instance = torch.arange(B, dtype=torch.int64).repeat_interleave(n)
    expected_node = torch.arange(n, dtype=torch.int64).repeat(B)
    if torch.equal(instance_id, expected_instance) and torch.equal(node_id, expected_node):
        idx = None
    else:
        key = instance_id * n + node_id
        idx = torch.argsort(key)

    if idx is not None:
        X = X[idx]
        y = y[idx]
        valid = valid[idx]

    X_inst = X.view(B, n, X.shape[1])
    y_inst = y.view(B, n, y.shape[1])
    valid_inst = valid.view(B, n)
    return X_inst, y_inst, valid_inst


def _rank_metrics_by_instance(
    instance_ids: torch.Tensor,
    node_ids: torch.Tensor,
    valid: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    instance_mask: torch.Tensor,
    target_names: List[str],
) -> Dict[str, float]:
    instance_ids = instance_ids.detach().cpu()
    node_ids = node_ids.detach().cpu()
    valid = valid.detach().cpu()
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()
    instance_mask = instance_mask.detach().cpu()

    inst_set = sorted(set(int(i) for i in instance_ids[instance_mask].tolist()))
    if not inst_set:
        return {}

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true and y_pred must be [N,T] tensors")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {tuple(y_true.shape)} vs {tuple(y_pred.shape)}")
    if y_true.shape[1] != len(target_names):
        raise ValueError("target_names length must match y columns")

    spearman_by_t: List[List[float]] = [[] for _ in target_names]
    regret_by_t: List[List[float]] = [[] for _ in target_names]
    top1_by_t: List[List[float]] = [[] for _ in target_names]
    top5_by_t: List[List[float]] = [[] for _ in target_names]

    for inst in inst_set:
        rows = (instance_ids == inst) & instance_mask & valid
        if rows.sum().item() < 2:
            continue

        yt = y_true[rows]
        yp = y_pred[rows]

        for t, _name in enumerate(target_names):
            spearman_by_t[t].append(spearman_corr(yp[:, t], yt[:, t]))

            pred_best = int(torch.argmax(yp[:, t]).item())
            true_best_idx = int(torch.argmax(yt[:, t]).item())
            true_best = float(torch.max(yt[:, t]).item())
            chosen_true = float(yt[pred_best, t].item())
            regret_by_t[t].append(true_best - chosen_true)
            top1_by_t[t].append(float(pred_best == true_best_idx))
            k = min(5, int(yt.shape[0]))
            topk = torch.topk(yp[:, t], k=k).indices
            top5_by_t[t].append(float(true_best_idx in set(int(i) for i in topk.tolist())))

    def _mean(xs: List[float]) -> float:
        xs2 = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs2)) if xs2 else float("nan")

    out: Dict[str, float] = {"num_instances_eval": float(len(inst_set))}
    for t, name in enumerate(target_names):
        out[f"spearman_{name}_mean"] = _mean(spearman_by_t[t])
        out[f"top1_regret_{name}_mean"] = _mean(regret_by_t[t])
        out[f"top1_acc_{name}"] = _mean(top1_by_t[t])
        out[f"top5_acc_{name}"] = _mean(top5_by_t[t])
    return out


def train_set_regression(
    X_inst: torch.Tensor,
    y_inst: torch.Tensor,
    valid_inst: torch.Tensor,
    instance_id: torch.Tensor,
    splits: SplitMasks,
    target_names: List[str],
    model_type: str,
    mlp_hidden_dim: int,
    mlp_layers: int,
    mlp_dropout: float,
    tfm_dim: int,
    tfm_layers: int,
    tfm_heads: int,
    tfm_ff_mult: int,
    tfm_dropout: float,
    device: torch.device,
    batch_size_instances: int,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    l1_lambda: float,
    standardize_x: bool,
    instance_standardize_x: bool,
    node_standardize_x: bool,
    seed: int,
) -> Tuple[nn.Module, Dict, Dict]:
    if y_inst.ndim != 2 or X_inst.ndim != 3 or valid_inst.ndim != 2:
        raise ValueError("Expected X_inst [B,n,d], y_inst [B,n], valid_inst [B,n]")
    B, n, d = X_inst.shape
    if y_inst.shape != (B, n):
        raise ValueError(f"y_inst shape mismatch: expected {(B, n)}, got {tuple(y_inst.shape)}")
    if valid_inst.shape != (B, n):
        raise ValueError(f"valid_inst shape mismatch: expected {(B, n)}, got {tuple(valid_inst.shape)}")

    X_inst = X_inst.to(torch.float32)
    y_inst = y_inst.to(torch.float32)
    valid_inst = valid_inst.to(torch.bool)

    if node_standardize_x:
        mean = X_inst.mean(dim=2, keepdim=True)
        std = X_inst.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
        X_inst = (X_inst - mean) / std

    if instance_standardize_x:
        m = valid_inst.to(torch.float32).unsqueeze(-1)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (X_inst * m).sum(dim=1, keepdim=True) / denom
        var = ((X_inst - mean) * m).pow(2).sum(dim=1, keepdim=True) / denom
        std = var.sqrt().clamp_min(1e-6)
        X_inst = (X_inst - mean) / std

    X_flat = X_inst.reshape(B * n, d)
    y_flat = y_inst.reshape(B * n)
    valid_flat = valid_inst.reshape(B * n)
    train_row_mask = splits.train & valid_flat

    y_mean = y_flat[train_row_mask].mean()
    y_std = y_flat[train_row_mask].std(unbiased=False).clamp_min(1e-6)
    y_norm_flat = (y_flat - y_mean) / y_std
    y_norm_inst = y_norm_flat.reshape(B, n)

    x_mean = x_std = None
    if standardize_x:
        x_mean = X_flat[train_row_mask].mean(dim=0)
        x_std = X_flat[train_row_mask].std(dim=0, unbiased=False).clamp_min(1e-6)
        X_inst = (X_inst - x_mean) / x_std

    model = build_probe_model(
        model_type=model_type,
        input_dim=int(d),
        output_dim=1,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_layers=mlp_layers,
        mlp_dropout=mlp_dropout,
        tfm_dim=tfm_dim,
        tfm_layers=tfm_layers,
        tfm_heads=tfm_heads,
        tfm_ff_mult=tfm_ff_mult,
        tfm_dropout=tfm_dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_inst_mask = torch.zeros(B, dtype=torch.bool)
    val_inst_mask = torch.zeros(B, dtype=torch.bool)
    test_inst_mask = torch.zeros(B, dtype=torch.bool)
    train_inst_mask[instance_id[splits.train]] = True
    val_inst_mask[instance_id[splits.val]] = True
    test_inst_mask[instance_id[splits.test]] = True

    train_idx = torch.nonzero(train_inst_mask, as_tuple=False).squeeze(1)
    val_idx = torch.nonzero(val_inst_mask, as_tuple=False).squeeze(1)
    test_idx = torch.nonzero(test_inst_mask, as_tuple=False).squeeze(1)

    if int(train_idx.numel()) == 0 or int(val_idx.numel()) == 0 or int(test_idx.numel()) == 0:
        raise ValueError(
            f"Empty split for set regression: train={int(train_idx.numel())}, val={int(val_idx.numel())}, test={int(test_idx.numel())}"
        )

    batch_size_instances = int(batch_size_instances)
    if batch_size_instances <= 0:
        raise ValueError("--batch_size must be >= 1 for set regression")

    best_val = float("inf")
    best_state = None

    g = torch.Generator().manual_seed(int(seed))
    for _epoch in range(int(num_epochs)):
        model.train()
        perm = train_idx[torch.randperm(int(train_idx.numel()), generator=g)]
        for start in range(0, int(perm.numel()), batch_size_instances):
            batch = perm[start : start + batch_size_instances]
            xb = X_inst[batch].to(device)
            vb = valid_inst[batch].to(device)
            yb = y_norm_inst[batch].to(device)

            if model_type in ("transformer", "deepset"):
                pred = model(xb, key_padding_mask=(~vb)).squeeze(-1)
            else:
                pred = model(xb).squeeze(-1)

            mask = vb.to(pred.dtype)
            diff = (pred - yb) * mask
            denom = mask.sum().clamp_min(1.0)
            loss = (diff * diff).sum() / denom
            if l1_lambda > 0:
                l1_pen = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_pen
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss_sum = 0.0
        val_total = 0.0
        with torch.no_grad():
            for start in range(0, int(val_idx.numel()), batch_size_instances):
                batch = val_idx[start : start + batch_size_instances]
                xb = X_inst[batch].to(device)
                vb = valid_inst[batch].to(device)
                yb = y_norm_inst[batch].to(device)
                if model_type in ("transformer", "deepset"):
                    pred = model(xb, key_padding_mask=(~vb)).squeeze(-1)
                else:
                    pred = model(xb).squeeze(-1)
                mask = vb.to(pred.dtype)
                diff = (pred - yb) * mask
                denom = float(mask.sum().item())
                denom = max(1.0, denom)
                val_loss_sum += float((diff * diff).sum().item())
                val_total += denom
        val_loss = float(val_loss_sum / val_total) if val_total > 0 else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    def _predict_inst(idx: torch.Tensor) -> torch.Tensor:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, int(idx.numel()), batch_size_instances):
                batch = idx[start : start + batch_size_instances]
                xb = X_inst[batch].to(device)
                vb = valid_inst[batch].to(device)
                if model_type in ("transformer", "deepset"):
                    out = model(xb, key_padding_mask=(~vb)).squeeze(-1)
                else:
                    out = model(xb).squeeze(-1)
                preds.append(out.cpu())
        return torch.cat(preds, dim=0) if preds else torch.empty((0, n), dtype=torch.float32)

    pred_train_norm = _predict_inst(train_idx)
    pred_val_norm = _predict_inst(val_idx)
    pred_test_norm = _predict_inst(test_idx)
    pred_full_norm = _predict_inst(torch.arange(B, dtype=torch.int64))

    pred_train = pred_train_norm * y_std + y_mean
    pred_val = pred_val_norm * y_std + y_mean
    pred_test = pred_test_norm * y_std + y_mean
    pred_full = pred_full_norm * y_std + y_mean

    preds = {
        "train": pred_train,
        "val": pred_val,
        "test": pred_test,
        "full": pred_full,
    }

    results = {
        "y_mean": [float(y_mean.item())],
        "y_std": [float(y_std.item())],
        "x_mean": x_mean.tolist() if x_mean is not None else None,
        "x_std": x_std.tolist() if x_std is not None else None,
        "best_val_loss": best_val,
    }

    return model, results, preds


def train_linear(
    X: torch.Tensor,
    y: torch.Tensor,
    splits: SplitMasks,
    target_names: List[str],
    model_type: str,
    mlp_hidden_dim: int,
    mlp_layers: int,
    mlp_dropout: float,
    device: torch.device,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    l1_lambda: float,
    standardize_x: bool,
) -> Tuple[nn.Module, Dict, Dict]:
    X = X.to(torch.float32)
    y = y.to(torch.float32)

    train_mask = splits.train
    val_mask = splits.val
    test_mask = splits.test

    y_norm, y_mean, y_std = _standardize_targets(y, train_mask)

    x_mean = x_std = None
    if standardize_x:
        X, x_mean, x_std = _standardize_x(X, train_mask)

    out_dim = int(y.shape[1])
    if out_dim != len(target_names):
        raise ValueError(f"target_names length must match y columns ({len(target_names)} vs {out_dim})")
    model = build_probe_model(
        model_type=model_type,
        input_dim=int(X.shape[1]),
        output_dim=out_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_layers=mlp_layers,
        mlp_dropout=mlp_dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(X[train_mask], y_norm[train_mask])
    val_ds = TensorDataset(X[val_mask], y_norm[val_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    best_val = float("inf")
    best_state = None

    for _epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if l1_lambda > 0:
                l1_pen = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_pen
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    def _predict(mask: torch.Tensor) -> torch.Tensor:
        model.eval()
        if int(mask.sum().item()) == 0:
            return torch.empty((0, out_dim), dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for xb in DataLoader(TensorDataset(X[mask]), batch_size=batch_size, shuffle=False):
                xb0 = xb[0].to(device)
                preds.append(model(xb0).cpu())
        pred_norm = torch.cat(preds, dim=0) if preds else torch.empty((0, out_dim), dtype=torch.float32)
        return pred_norm * y_std + y_mean

    pred_train = _predict(train_mask)
    pred_val = _predict(val_mask)
    pred_test = _predict(test_mask)

    results = {
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "x_mean": x_mean.tolist() if x_mean is not None else None,
        "x_std": x_std.tolist() if x_std is not None else None,
        "best_val_loss": float(best_val),
        "train": _metrics_by_target(y[train_mask], pred_train, target_names=target_names),
        "val": _metrics_by_target(y[val_mask], pred_val, target_names=target_names),
        "test": _metrics_by_target(y[test_mask], pred_test, target_names=target_names),
    }
    preds = {"train": pred_train, "val": pred_val, "test": pred_test}
    return model, results, preds


def train_best_node_ce(
    X_inst: torch.Tensor,
    y_inst: torch.Tensor,
    valid_inst: torch.Tensor,
    splits: SplitMasks,
    objective: str,
    soft_ce_tau: float,
    pairwise_pairs_per_instance: int,
    pairwise_margin: float,
    model_type: str,
    mlp_hidden_dim: int,
    mlp_layers: int,
    mlp_dropout: float,
    tfm_dim: int,
    tfm_layers: int,
    tfm_heads: int,
    tfm_ff_mult: int,
    tfm_dropout: float,
    device: torch.device,
    batch_size_instances: int,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    l1_lambda: float,
    standardize_x: bool,
    instance_standardize_x: bool,
    node_standardize_x: bool,
    seed: int,
) -> Tuple[nn.Module, Dict]:
    if y_inst.ndim != 2:
        raise ValueError(f"Expected y_inst to be [B,n], got {tuple(y_inst.shape)}")
    if X_inst.ndim != 3:
        raise ValueError(f"Expected X_inst to be [B,n,d], got {tuple(X_inst.shape)}")
    if valid_inst.ndim != 2:
        raise ValueError(f"Expected valid_inst to be [B,n], got {tuple(valid_inst.shape)}")

    B, n, d = X_inst.shape
    if y_inst.shape != (B, n):
        raise ValueError(f"y_inst shape mismatch: expected {(B, n)}, got {tuple(y_inst.shape)}")
    if valid_inst.shape != (B, n):
        raise ValueError(f"valid_inst shape mismatch: expected {(B, n)}, got {tuple(valid_inst.shape)}")

    objective = str(objective)
    if objective not in ("best_node_ce", "soft_ce", "pairwise_rank"):
        raise ValueError(f"Unsupported objective for best-node training: {objective}")
    soft_ce_tau = float(soft_ce_tau)
    if objective == "soft_ce" and not (soft_ce_tau > 0):
        raise ValueError("--soft_ce_tau must be > 0 for --objective soft_ce")
    pairwise_pairs_per_instance = int(pairwise_pairs_per_instance)
    if objective == "pairwise_rank" and pairwise_pairs_per_instance <= 0:
        raise ValueError("--pairwise_pairs_per_instance must be >= 1 for --objective pairwise_rank")
    pairwise_margin = float(pairwise_margin)
    if objective == "pairwise_rank" and pairwise_margin < 0:
        raise ValueError("--pairwise_margin must be >= 0 for --objective pairwise_rank")

    # Label = argmax over valid nodes of the scalar target.
    y_masked = y_inst.clone()
    y_masked[~valid_inst] = float("-inf")
    has_any = valid_inst.any(dim=1)
    labels = torch.argmax(y_masked, dim=1).to(torch.int64)

    splits = SplitMasks(
        train=splits.train & has_any,
        val=splits.val & has_any,
        test=splits.test & has_any,
    )

    X_inst = X_inst.to(torch.float32)
    y_inst = y_inst.to(torch.float32)
    valid_inst = valid_inst.to(torch.bool)
    labels = labels.to(torch.int64)

    if node_standardize_x:
        mean = X_inst.mean(dim=2, keepdim=True)
        std = X_inst.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
        X_inst = (X_inst - mean) / std

    if instance_standardize_x:
        # Per-instance per-feature normalization over nodes.
        # This is intended to remove pool-node identity shortcuts that rely on absolute feature scales.
        m = valid_inst.to(torch.float32).unsqueeze(-1)  # [B,n,1]
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (X_inst * m).sum(dim=1, keepdim=True) / denom
        var = ((X_inst - mean) * m).pow(2).sum(dim=1, keepdim=True) / denom
        std = var.sqrt().clamp_min(1e-6)
        X_inst = (X_inst - mean) / std

    x_mean = x_std = None
    if standardize_x:
        X_train = X_inst[splits.train].reshape(-1, d)
        x_mean = X_train.mean(dim=0)
        x_std = X_train.std(dim=0, unbiased=False).clamp_min(1e-6)
        X_inst = (X_inst - x_mean) / x_std

    model = build_probe_model(
        model_type=model_type,
        input_dim=int(d),
        output_dim=1,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_layers=mlp_layers,
        mlp_dropout=mlp_dropout,
        tfm_dim=tfm_dim,
        tfm_layers=tfm_layers,
        tfm_heads=tfm_heads,
        tfm_ff_mult=tfm_ff_mult,
        tfm_dropout=tfm_dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_idx = torch.nonzero(splits.train, as_tuple=False).squeeze(1)
    val_idx = torch.nonzero(splits.val, as_tuple=False).squeeze(1)
    test_idx = torch.nonzero(splits.test, as_tuple=False).squeeze(1)

    if int(train_idx.numel()) == 0 or int(val_idx.numel()) == 0 or int(test_idx.numel()) == 0:
        raise ValueError(
            f"Empty split for best_node_ce: train={int(train_idx.numel())}, val={int(val_idx.numel())}, test={int(test_idx.numel())}"
        )

    batch_size_instances = int(batch_size_instances)
    if batch_size_instances <= 0:
        raise ValueError("--batch_size must be >= 1 for best_node_ce")

    best_val = float("inf")
    best_state = None

    def _batch_loss(logits: torch.Tensor, vb: torch.Tensor, yb_hard: torch.Tensor, yb_soft: torch.Tensor) -> torch.Tensor:
        logits = logits.masked_fill(~vb, -1e9)
        if objective == "best_node_ce":
            return F.cross_entropy(logits, yb_hard)

        # soft_ce: target distribution derived from target values.
        if objective == "soft_ce":
            tgt = yb_soft.masked_fill(~vb, float("-inf"))
            tgt = tgt / soft_ce_tau
            tgt = tgt - tgt.max(dim=1, keepdim=True).values
            tgt_probs = torch.softmax(tgt, dim=1)
            logp = torch.log_softmax(logits, dim=1)
            return -(tgt_probs * logp).sum(dim=1).mean()

        # pairwise_rank: sample node pairs and train on ordering.
        # Target = 1 if y_i > y_j, else 0; loss = BCE(logit_i - logit_j).
        bsz, nnodes = logits.shape
        i_idx = torch.randint(0, nnodes, (bsz, pairwise_pairs_per_instance), device=logits.device)
        j_idx = torch.randint(0, nnodes, (bsz, pairwise_pairs_per_instance), device=logits.device)
        # ensure i != j
        j_idx = torch.where(j_idx == i_idx, (j_idx + 1) % nnodes, j_idx)

        vi = vb.gather(1, i_idx)
        vj = vb.gather(1, j_idx)
        vpair = vi & vj

        yi = yb_soft.gather(1, i_idx)
        yj = yb_soft.gather(1, j_idx)
        if pairwise_margin > 0:
            vpair = vpair & ((yi - yj).abs() >= pairwise_margin)

        li = logits.gather(1, i_idx)
        lj = logits.gather(1, j_idx)
        diff = li - lj
        target = (yi > yj).to(diff.dtype)
        loss = F.binary_cross_entropy_with_logits(diff, target, reduction="none")
        loss = loss.masked_fill(~vpair, 0.0)
        denom = vpair.to(diff.dtype).sum().clamp_min(1.0)
        return loss.sum() / denom

    g = torch.Generator().manual_seed(int(seed))
    for _epoch in range(int(num_epochs)):
        model.train()
        perm = train_idx[torch.randperm(int(train_idx.numel()), generator=g)]
        for start in range(0, int(perm.numel()), batch_size_instances):
            batch = perm[start : start + batch_size_instances]
            xb = X_inst[batch].to(device)
            vb = valid_inst[batch].to(device)
            yb_hard = labels[batch].to(device)
            yb_soft = y_inst[batch].to(device)

            if model_type in ("transformer", "deepset"):
                logits = model(xb, key_padding_mask=(~vb)).squeeze(-1)  # [b,n]
            else:
                logits = model(xb).squeeze(-1)  # [b,n]
            loss = _batch_loss(logits, vb, yb_hard, yb_soft)
            if l1_lambda > 0:
                l1_pen = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_pen
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        with torch.no_grad():
            for start in range(0, int(val_idx.numel()), batch_size_instances):
                batch = val_idx[start : start + batch_size_instances]
                xb = X_inst[batch].to(device)
                vb = valid_inst[batch].to(device)
                yb_hard = labels[batch].to(device)
                yb_soft = y_inst[batch].to(device)
                if model_type in ("transformer", "deepset"):
                    logits = model(xb, key_padding_mask=(~vb)).squeeze(-1)
                else:
                    logits = model(xb).squeeze(-1)
                loss = float(_batch_loss(logits, vb, yb_hard, yb_soft).item())
                bs = int(batch.numel())
                val_loss_sum += loss * bs
                val_total += bs
        val_loss = float(val_loss_sum / val_total) if val_total > 0 else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    def _eval(split_idx: torch.Tensor) -> Dict[str, float]:
        model.eval()
        spearmans = []
        loss_sum = 0.0
        top1_correct = 0
        top5_correct = 0
        regret_sum = 0.0
        total = 0

        with torch.no_grad():
            for start in range(0, int(split_idx.numel()), batch_size_instances):
                batch = split_idx[start : start + batch_size_instances]
                xb = X_inst[batch].to(device)
                vb = valid_inst[batch].to(device)
                yb = labels[batch].to(device)

                if model_type in ("transformer", "deepset"):
                    logits = model(xb, key_padding_mask=(~vb)).squeeze(-1)
                else:
                    logits = model(xb).squeeze(-1)
                logits = logits.masked_fill(~vb, -1e9)

                if objective == "best_node_ce":
                    loss = float(F.cross_entropy(logits, yb).item())
                elif objective == "soft_ce":
                    tgt = y_inst[batch].to(device).masked_fill(~vb, float("-inf"))
                    tgt = tgt / soft_ce_tau
                    tgt = tgt - tgt.max(dim=1, keepdim=True).values
                    tgt_probs = torch.softmax(tgt, dim=1)
                    logp = torch.log_softmax(logits, dim=1)
                    loss = float((-(tgt_probs * logp).sum(dim=1).mean()).item())
                else:
                    loss = float(_batch_loss(logits, vb, yb, y_inst[batch].to(device)).item())
                bs = int(batch.numel())
                loss_sum += loss * bs
                total += bs

                pred = torch.argmax(logits, dim=1).to(torch.int64)  # [b]
                top1_correct += int((pred == yb).sum().item())

                k = min(5, n)
                topk = torch.topk(logits, k=k, dim=1).indices  # [b,k]
                top5_correct += int((topk == yb.unsqueeze(1)).any(dim=1).sum().item())

                yt = y_inst[batch]  # [b,n] on cpu
                vt = valid_inst[batch]
                yt_masked = yt.clone()
                yt_masked[~vt] = float("-inf")
                best = torch.max(yt_masked, dim=1).values
                chosen = yt.gather(dim=1, index=pred.cpu().unsqueeze(1)).squeeze(1)
                regret_sum += float((best - chosen).sum().item())

                # Spearman per instance (on CPU)
                logits_cpu = logits.detach().cpu()
                for bi in range(int(batch.numel())):
                    vmask = vt[bi]
                    if int(vmask.sum().item()) < 2:
                        continue
                    spearmans.append(spearman_corr(logits_cpu[bi][vmask], yt[bi][vmask]))

        def _mean(xs: List[float]) -> float:
            xs2 = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
            return float(np.mean(xs2)) if xs2 else float("nan")

        return {
            "loss": float(loss_sum / total) if total > 0 else float("nan"),
            "top1_acc": float(top1_correct / total) if total > 0 else float("nan"),
            "top5_acc": float(top5_correct / total) if total > 0 else float("nan"),
            "top1_regret_mean": float(regret_sum / total) if total > 0 else float("nan"),
            "spearman_mean": _mean(spearmans),
            "num_instances": float(total),
        }

    results = {
        "x_mean": x_mean.tolist() if x_mean is not None else None,
        "x_std": x_std.tolist() if x_std is not None else None,
        "best_val_loss": float(best_val),
        "objective": objective,
        "soft_ce_tau": float(soft_ce_tau) if objective == "soft_ce" else None,
        "pairwise_pairs_per_instance": int(pairwise_pairs_per_instance) if objective == "pairwise_rank" else None,
        "pairwise_margin": float(pairwise_margin) if objective == "pairwise_rank" else None,
        "instance_standardize_x": bool(instance_standardize_x),
        "node_standardize_x": bool(node_standardize_x),
        "train": _eval(train_idx),
        "val": _eval(val_idx),
        "test": _eval(test_idx),
    }
    return model, results


def main() -> None:
    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    args = build_arg_parser().parse_args()
    set_seed(int(args.seed))

    reps_path = Path(args.reps_path).expanduser().resolve()
    reps = torch.load(reps_path, weights_only=False)

    instance_id = reps.get("instance_id")
    node_id = reps.get("node_id")
    valid = reps.get("valid")
    y = reps.get("y")
    if not torch.is_tensor(instance_id) or not torch.is_tensor(node_id) or not torch.is_tensor(valid) or not torch.is_tensor(y):
        raise ValueError("reps file missing required keys: instance_id, node_id, valid, y")

    instance_id = instance_id.to(torch.int64)
    node_id = node_id.to(torch.int64)
    valid = valid.to(torch.bool)
    y = y.to(torch.float32)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"Expected y to be [N,2], got {tuple(y.shape)}")

    y_names = reps.get("y_names")
    if not (isinstance(y_names, list) and len(y_names) == 2 and all(isinstance(x, str) for x in y_names)):
        meta = reps.get("meta", {})
        meta_names = meta.get("label_names") if isinstance(meta, dict) else None
        if isinstance(meta_names, list) and len(meta_names) == 2 and all(isinstance(x, str) for x in meta_names):
            y_names = meta_names
        else:
            y_names = ["y0", "y1"]

    target = str(args.target)
    if target == "length":
        y = y[:, :1]
        target_names = [y_names[0]]
    elif target == "time":
        y = y[:, 1:2]
        target_names = [y_names[1]]
    elif target == "both":
        target_names = [y_names[0], y_names[1]]
    else:
        raise ValueError(f"Unknown target: {target}")

    objective = str(args.objective)
    if objective != "regression" and target == "both":
        raise ValueError("--objective best_node_ce/soft_ce/pairwise_rank requires a scalar --target (length or time), not both")
    if objective == "regression" and target == "both" and str(args.model) in ("transformer", "deepset"):
        raise ValueError("--objective regression with --target both is not supported for transformer/deepset")

    print(f"[train] objective: {objective}")
    print(f"[train] target: {target} ({', '.join(target_names)})")
    print(f"[train] model: {str(args.model)}")

    rows = int(instance_id.numel())
    if rows == 0:
        raise ValueError("Empty reps file")

    splits = make_instance_splits(instance_id, float(args.train_frac), float(args.val_frac), seed=int(args.seed))
    splits = SplitMasks(
        train=splits.train & valid,
        val=splits.val & valid,
        test=splits.test & valid,
    )

    device = _as_device(args.device)
    default_out = "probe_artifacts" if objective == "regression" else f"probe_artifacts_{objective}"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (reps_path.parent / default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = reps.get("meta", {}) if isinstance(reps, dict) else {}
    common_cfg = {
        "reps_path": str(reps_path),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "device": str(device),
        "target": target,
        "target_names": target_names,
        "objective": objective,
        "soft_ce_tau": float(args.soft_ce_tau),
        "pairwise_pairs_per_instance": int(args.pairwise_pairs_per_instance),
        "pairwise_margin": float(args.pairwise_margin),
        "model": str(args.model),
        "mlp_hidden_dim": int(args.mlp_hidden_dim),
        "mlp_layers": int(args.mlp_layers),
        "mlp_dropout": float(args.mlp_dropout),
        "tfm_dim": int(args.tfm_dim),
        "tfm_layers": int(args.tfm_layers),
        "tfm_heads": int(args.tfm_heads),
        "tfm_ff_mult": int(args.tfm_ff_mult),
        "tfm_dropout": float(args.tfm_dropout),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_epochs": int(args.num_epochs),
        "l1_lambda": float(args.l1_lambda),
        "standardize_x": bool(args.standardize_x),
        "instance_standardize_x": bool(args.instance_standardize_x),
        "node_standardize_x": bool(args.node_standardize_x),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "use_extra_features": bool(args.use_extra_features),
        "extra_feature_names": reps.get("extra_feature_names") if isinstance(reps, dict) else None,
        "meta": meta,
    }

    results_all: Dict[str, Dict] = {"config": common_cfg}

    def _train_one(key: str) -> None:
        X = reps.get(key)
        if not torch.is_tensor(X):
            raise ValueError(f"reps file missing tensor '{key}'")

        tag = "resid"
        X_used = X
        extra_names = reps.get("extra_feature_names")
        X_extra = reps.get("X_extra")
        if bool(args.use_extra_features):
            if torch.is_tensor(X_extra):
                if X_extra.ndim != 2 or X_extra.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"X_extra has shape {tuple(X_extra.shape)} but expected [rows,k] with rows={int(X.shape[0])}"
                    )
                X_used = torch.cat([X, X_extra.to(dtype=X.dtype)], dim=1)
            else:
                raise ValueError("--use_extra_features set but reps file has no X_extra")
    
        if objective == "regression":
            model_type = str(args.model)
            if model_type in ("transformer", "deepset"):
                X_inst, y_inst, valid_inst = _reshape_by_instance_node(
                    X=X_used,
                    y=y,
                    valid=valid,
                    instance_id=instance_id,
                    node_id=node_id,
                )
                model, results, preds = train_set_regression(
                    X_inst=X_inst,
                    y_inst=y_inst[:, :, 0],
                    valid_inst=valid_inst,
                    instance_id=instance_id,
                    splits=splits,
                    target_names=target_names,
                    model_type=model_type,
                    mlp_hidden_dim=int(args.mlp_hidden_dim),
                    mlp_layers=int(args.mlp_layers),
                    mlp_dropout=float(args.mlp_dropout),
                    tfm_dim=int(args.tfm_dim),
                    tfm_layers=int(args.tfm_layers),
                    tfm_heads=int(args.tfm_heads),
                    tfm_ff_mult=int(args.tfm_ff_mult),
                    tfm_dropout=float(args.tfm_dropout),
                    device=device,
                    batch_size_instances=int(args.batch_size),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    num_epochs=int(args.num_epochs),
                    l1_lambda=float(args.l1_lambda),
                    standardize_x=bool(args.standardize_x),
                    instance_standardize_x=bool(args.instance_standardize_x),
                    node_standardize_x=bool(args.node_standardize_x),
                    seed=int(args.seed),
                )
    
                pred_full = preds["full"]
                B, n = pred_full.shape
                pred_full_flat = pred_full.reshape(B * n, 1)
                test_rank = _rank_metrics_by_instance(
                    instance_ids=instance_id,
                    node_ids=node_id,
                    valid=valid,
                    y_true=y,
                    y_pred=pred_full_flat,
                    instance_mask=splits.test,
                    target_names=target_names,
                )
                val_rank = _rank_metrics_by_instance(
                    instance_ids=instance_id,
                    node_ids=node_id,
                    valid=valid,
                    y_true=y,
                    y_pred=pred_full_flat,
                    instance_mask=splits.val,
                    target_names=target_names,
                )
                train_rank = _rank_metrics_by_instance(
                    instance_ids=instance_id,
                    node_ids=node_id,
                    valid=valid,
                    y_true=y,
                    y_pred=pred_full_flat,
                    instance_mask=splits.train,
                    target_names=target_names,
                )
    
                results["target_names"] = target_names
                results.setdefault("train", {})["ranking"] = train_rank
                results.setdefault("val", {})["ranking"] = val_rank
                results.setdefault("test", {})["ranking"] = test_rank
            else:
                model, results, preds = train_linear(
                    X=X_used,
                    y=y,
                    splits=splits,
                    target_names=target_names,
                    model_type=model_type,
                    mlp_hidden_dim=int(args.mlp_hidden_dim),
                    mlp_layers=int(args.mlp_layers),
                    mlp_dropout=float(args.mlp_dropout),
                    device=device,
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    num_epochs=int(args.num_epochs),
                    l1_lambda=float(args.l1_lambda),
                    standardize_x=bool(args.standardize_x),
                )
    
                pred_test = preds["test"]
                test_rank = _rank_metrics_by_instance(
                    instance_ids=instance_id,
                    node_ids=node_id,
                    valid=valid,
                    y_true=y,
                    y_pred=torch.zeros_like(y).index_copy(
                        0,
                        torch.nonzero(splits.test, as_tuple=False).squeeze(1),
                        pred_test,
                    ),
                    instance_mask=splits.test,
                    target_names=target_names,
                )
    
                results["target_names"] = target_names
                results.setdefault("test", {})["ranking"] = test_rank
    
            if bool(args.use_extra_features):
                results["extra_feature_names"] = extra_names if isinstance(extra_names, list) else None
            results_all[tag] = results
    
            if bool(args.save_models):
                model_path = out_dir / f"{str(args.model)}_{tag}.pt"
                torch.save(
                    {
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "objective": objective,
                        "model": str(args.model),
                        "mlp_hidden_dim": int(args.mlp_hidden_dim),
                        "mlp_layers": int(args.mlp_layers),
                        "mlp_dropout": float(args.mlp_dropout),
                        "input_key": key,
                        "target": target,
                        "target_names": target_names,
                        "y_mean": results["y_mean"],
                        "y_std": results["y_std"],
                        "x_mean": results.get("x_mean"),
                        "x_std": results.get("x_std"),
                        "meta": meta,
                    },
                    model_path,
                )
                print(f"[train] wrote model: {model_path}")
            return
    
        if objective in ("best_node_ce", "soft_ce", "pairwise_rank"):
            X_inst, y_inst, valid_inst = _reshape_by_instance_node(
                X=X_used,
                y=y,
                valid=valid,
                instance_id=instance_id,
                node_id=node_id,
            )
            # y is [B,n,1] here; squeeze to scalar per node.
            y_inst = y_inst.squeeze(-1)

            splits_inst = make_instance_splits(
                torch.arange(X_inst.shape[0], dtype=torch.int64),
                float(args.train_frac),
                float(args.val_frac),
                seed=int(args.seed),
            )

            model, results = train_best_node_ce(
                X_inst=X_inst,
                y_inst=y_inst,
                valid_inst=valid_inst,
                splits=splits_inst,
                objective=objective,
                soft_ce_tau=float(args.soft_ce_tau),
                pairwise_pairs_per_instance=int(args.pairwise_pairs_per_instance),
                pairwise_margin=float(args.pairwise_margin),
                model_type=str(args.model),
                mlp_hidden_dim=int(args.mlp_hidden_dim),
                mlp_layers=int(args.mlp_layers),
                mlp_dropout=float(args.mlp_dropout),
                tfm_dim=int(args.tfm_dim),
                tfm_layers=int(args.tfm_layers),
                tfm_heads=int(args.tfm_heads),
                tfm_ff_mult=int(args.tfm_ff_mult),
                tfm_dropout=float(args.tfm_dropout),
                device=device,
                batch_size_instances=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                num_epochs=int(args.num_epochs),
                l1_lambda=float(args.l1_lambda),
                standardize_x=bool(args.standardize_x),
                instance_standardize_x=bool(args.instance_standardize_x),
                node_standardize_x=bool(args.node_standardize_x),
                seed=int(args.seed),
            )

            if bool(args.use_extra_features):
                results["extra_feature_names"] = extra_names if isinstance(extra_names, list) else None
            results_all[tag] = results

            if bool(args.save_models):
                model_path = out_dir / f"{str(args.model)}_{tag}.pt"
                torch.save(
                    {
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "objective": objective,
                        "model": str(args.model),
                        "mlp_hidden_dim": int(args.mlp_hidden_dim),
                        "mlp_layers": int(args.mlp_layers),
                        "mlp_dropout": float(args.mlp_dropout),
                        "tfm_dim": int(args.tfm_dim),
                        "tfm_layers": int(args.tfm_layers),
                        "tfm_heads": int(args.tfm_heads),
                        "tfm_ff_mult": int(args.tfm_ff_mult),
                        "tfm_dropout": float(args.tfm_dropout),
                        "soft_ce_tau": float(args.soft_ce_tau),
                        "pairwise_pairs_per_instance": int(args.pairwise_pairs_per_instance),
                        "pairwise_margin": float(args.pairwise_margin),
                        "input_key": key,
                        "target": target,
                        "target_names": target_names,
                        "x_mean": results.get("x_mean"),
                        "x_std": results.get("x_std"),
                        "instance_standardize_x": bool(args.instance_standardize_x),
                        "node_standardize_x": bool(args.node_standardize_x),
                        "meta": meta,
                    },
                    model_path,
                )
                print(f"[train] wrote model: {model_path}")
            return

        raise ValueError(f"Unknown objective: {objective}")

    _train_one("X_resid")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as fp:
        json.dump(results_all, fp, indent=2)
    print(f"[train] wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
