#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train a probe on what-if labels (delta_length_pct per node) and evaluate top-k argmax accuracy.\n"
            "Supports set-transformer and deepset models, and multiple losses (mse/huber/listwise/ranknet)."
        )
    )
    p.add_argument(
        "--reps_path",
        type=str,
        default="tasks/node_removal/tmp/tuning/probe_reps_expdecay_layers0-4_plus_output.pt",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="tasks/node_removal/tmp/tuning/probe_multiseed",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--repeats", type=int, default=1)

    p.add_argument("--train_size", type=int, default=2500)
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--standardize_x", action="store_true")
    p.add_argument("--standardize_y", action="store_true")
    p.add_argument(
        "--add_nn_dist",
        action="store_true",
        help="Concatenate per-node nearest-neighbor distance (from dataset locs) to the input features.",
    )
    p.add_argument(
        "--add_knn_mean_dist",
        action="store_true",
        help="Concatenate per-node mean distance to the k nearest neighbors (from dataset locs) to the input features.",
    )
    p.add_argument(
        "--knn_k",
        type=int,
        default=5,
        help="k for --add_knn_mean_dist (default: 5). Uses k nearest neighbors excluding self.",
    )
    p.add_argument(
        "--add_centroid_dist",
        action="store_true",
        help="Concatenate per-node distance to the instance centroid (from dataset locs) to the input features.",
    )
    p.add_argument(
        "--add_oracle_splice_contrib",
        action="store_true",
        help=(
            "Concatenate an oracle-ish feature computed from a Concorde optimal tour: "
            "per-node splice improvement if you remove that node from the optimal tour without re-optimizing."
        ),
    )
    p.add_argument(
        "--oracle_splice_cache",
        type=str,
        default=None,
        help="Optional path to cache/load oracle splice features (default: <data_dir>/oracle_splice_contrib.pt).",
    )
    p.add_argument(
        "--concorde_timeout_sec",
        type=float,
        default=60.0,
        help="Timeout per Concorde solve when computing oracle splice features (default: 60).",
    )
    p.add_argument(
        "--nn_dist_variant",
        type=str,
        default="raw",
        choices=["raw", "zscore", "raw+zscore", "rank", "raw+rank", "raw+zscore+rank"],
        help="Featureization for nn_dist when --add_nn_dist is set (default: raw).",
    )

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad_norm", type=float, default=0.0)

    p.add_argument("--model", type=str, default="set_transformer", choices=["set_transformer", "deepset", "linear"])
    p.add_argument("--model_dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--head_mlp_layers", type=int, default=0, help="0 for linear head; 2 for 2-layer MLP head.")

    p.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "huber", "listwise_ce", "listwise_kl", "ranknet", "hard_ce"],
    )
    p.add_argument(
        "--loss2",
        type=str,
        default=None,
        choices=["mse", "huber", "listwise_ce", "listwise_kl", "ranknet", "hard_ce"],
    )
    p.add_argument("--loss2_weight", type=float, default=0.0)

    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0, help="Temperature for listwise losses (softmax(y/T)).")
    p.add_argument("--ranknet_pairs", type=int, default=256, help="Pairs per instance per step for ranknet loss.")
    p.add_argument("--select_metric", type=str, default="val_top1", choices=["val_loss", "val_top1"])
    return p


def as_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_int_list(csv: str) -> List[int]:
    out = []
    for part in csv.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


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
class InstanceTensors:
    X: torch.Tensor  # [B,n,d]
    y: torch.Tensor  # [B,n] (delta_length_pct)
    valid: torch.Tensor  # [B,n]
    labels: torch.Tensor  # [B] (argmax delta over valid)
    meta: Dict


def _oracle_script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "baselines" / "compute_oracle_splice_contrib.py"


def _compute_nn_dist(meta: Dict) -> torch.Tensor:
    data_dir = meta.get("data_dir")
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError("Cannot compute nn_dist: reps meta['data_dir'] missing")
    ds = torch.load(Path(data_dir) / "dataset.pt", weights_only=False, map_location="cpu")
    locs = ds.get("locs")
    if not torch.is_tensor(locs) or locs.ndim != 3 or locs.shape[-1] != 2:
        raise ValueError("dataset.pt missing locs [B,n,2]")
    B, n, _ = locs.shape

    out = torch.empty((B, n), dtype=torch.float32)
    bs = 128
    for start in range(0, B, bs):
        end = min(B, start + bs)
        chunk = locs[start:end].to(torch.float32)
        d = torch.cdist(chunk, chunk, p=2)  # [b,n,n]
        eye = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(end - start, n, n)
        d = d.masked_fill(eye, float("inf"))
        out[start:end] = d.min(dim=-1).values
    return out


def _compute_knn_mean_dist(meta: Dict, *, k: int) -> torch.Tensor:
    data_dir = meta.get("data_dir")
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError("Cannot compute knn_mean_dist: reps meta['data_dir'] missing")
    if k < 1:
        raise ValueError("--knn_k must be >= 1")
    ds = torch.load(Path(data_dir) / "dataset.pt", weights_only=False, map_location="cpu")
    locs = ds.get("locs")
    if not torch.is_tensor(locs) or locs.ndim != 3 or locs.shape[-1] != 2:
        raise ValueError("dataset.pt missing locs [B,n,2]")
    B, n, _ = locs.shape
    k_eff = min(int(k), int(n) - 1)
    if k_eff < 1:
        raise ValueError(f"Cannot compute kNN distances with n={n}")

    out = torch.empty((B, n), dtype=torch.float32)
    bs = 128
    for start in range(0, B, bs):
        end = min(B, start + bs)
        chunk = locs[start:end].to(torch.float32)
        d = torch.cdist(chunk, chunk, p=2)  # [b,n,n]
        eye = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(end - start, n, n)
        d = d.masked_fill(eye, float("inf"))
        knn = torch.topk(d, k=k_eff, dim=-1, largest=False).values  # [b,n,k]
        out[start:end] = knn.mean(dim=-1)
    return out


def _compute_centroid_dist(meta: Dict) -> torch.Tensor:
    data_dir = meta.get("data_dir")
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError("Cannot compute centroid_dist: reps meta['data_dir'] missing")
    ds = torch.load(Path(data_dir) / "dataset.pt", weights_only=False, map_location="cpu")
    locs = ds.get("locs")
    if not torch.is_tensor(locs) or locs.ndim != 3 or locs.shape[-1] != 2:
        raise ValueError("dataset.pt missing locs [B,n,2]")
    locs = locs.to(torch.float32)
    centroid = locs.mean(dim=1, keepdim=True)  # [B,1,2]
    return torch.norm(locs - centroid, dim=-1)  # [B,n]


def _compute_oracle_splice_contrib(
    meta: Dict, *, cache_path: Optional[str], timeout_sec: float
) -> torch.Tensor:
    data_dir = meta.get("data_dir")
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError("Cannot compute oracle splice contrib: reps meta['data_dir'] missing")
    data_dir_p = Path(data_dir)

    cache_p = Path(cache_path).expanduser().resolve() if cache_path else (data_dir_p / "oracle_splice_contrib.pt")
    if cache_p.exists():
        obj = torch.load(cache_p, weights_only=False, map_location="cpu")
        if isinstance(obj, dict) and torch.is_tensor(obj.get("oracle_splice_contrib_pct")):
            feat = obj["oracle_splice_contrib_pct"].to(torch.float32)
            return feat
        if torch.is_tensor(obj):
            return obj.to(torch.float32)
        raise TypeError(f"Unexpected oracle splice cache format at {cache_p}")

    script = _oracle_script_path()
    if not script.exists():
        raise FileNotFoundError(f"Oracle splice script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--data_dir",
        str(data_dir_p),
        "--out_path",
        str(cache_p),
        "--concorde_timeout_sec",
        str(timeout_sec),
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)

    obj = torch.load(cache_p, weights_only=False, map_location="cpu")
    if isinstance(obj, dict) and torch.is_tensor(obj.get("oracle_splice_contrib_pct")):
        return obj["oracle_splice_contrib_pct"].to(torch.float32)
    if torch.is_tensor(obj):
        return obj.to(torch.float32)
    raise TypeError(f"Unexpected oracle splice cache format at {cache_p}")


def _zscore_per_instance(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # x, valid: [B,n]
    out = torch.zeros_like(x, dtype=torch.float32)
    B, n = x.shape
    for i in range(B):
        v = valid[i]
        if int(v.sum().item()) < 2:
            continue
        vals = x[i][v]
        mean = vals.mean()
        std = vals.std(unbiased=False).clamp_min(1e-6)
        out[i][v] = (vals - mean) / std
    return out


def _rank01_per_instance(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # Rank-percentile in [0,1] over valid nodes: 0=smallest, 1=largest.
    out = torch.zeros_like(x, dtype=torch.float32)
    B, n = x.shape
    for i in range(B):
        v = valid[i]
        k = int(v.sum().item())
        if k < 2:
            continue
        vals = x[i][v]
        order = torch.argsort(vals)  # ascending
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(k, dtype=torch.float32)
        denom = float(k - 1)
        out[i][v] = ranks / denom
    return out


def load_instance_tensors(
    reps_path: Path,
    *,
    add_nn_dist: bool,
    nn_dist_variant: str,
    add_knn_mean_dist: bool,
    knn_k: int,
    add_centroid_dist: bool,
    add_oracle_splice_contrib: bool,
    oracle_splice_cache: Optional[str],
    concorde_timeout_sec: float,
) -> InstanceTensors:
    import os

    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    reps = torch.load(reps_path, weights_only=False)
    if not isinstance(reps, dict):
        raise TypeError(f"Unexpected reps format at {reps_path}: {type(reps)}")

    X = reps["X_resid"].to(torch.float32)
    y = reps["y"][:, 0].to(torch.float32)
    valid = reps["valid"].to(torch.bool)
    instance_id = reps["instance_id"].to(torch.int64)
    node_id = reps["node_id"].to(torch.int64)

    B = int(instance_id.max().item()) + 1
    n = int(node_id.max().item()) + 1
    if int(X.shape[0]) != B * n:
        raise ValueError("Expected flattened N=B*n layout")

    key = instance_id * n + node_id
    idx = torch.argsort(key)
    X = X[idx]
    y = y[idx]
    valid = valid[idx]

    X_inst = X.view(B, n, X.shape[1]).contiguous()
    y_inst = y.view(B, n).contiguous()
    valid_inst = valid.view(B, n).contiguous()

    y_masked = y_inst.clone()
    y_masked[~valid_inst] = float("-inf")
    labels = torch.argmax(y_masked, dim=1).to(torch.int64)

    meta = reps.get("meta", {}) if isinstance(reps.get("meta", {}), dict) else {}

    feats = []
    feat_names: List[str] = []

    if add_nn_dist:
        nn_dist = _compute_nn_dist(meta)  # [B,n]
        variant = str(nn_dist_variant)
        if variant in ("raw", "raw+zscore", "raw+rank", "raw+zscore+rank"):
            feats.append(nn_dist)
            feat_names.append("nn_dist")
        if variant in ("zscore", "raw+zscore", "raw+zscore+rank"):
            feats.append(_zscore_per_instance(nn_dist, valid_inst))
            feat_names.append("nn_dist_z")
        if variant in ("rank", "raw+rank", "raw+zscore+rank"):
            feats.append(_rank01_per_instance(nn_dist, valid_inst))
            feat_names.append("nn_dist_rank01")
        if not feats:
            raise ValueError(f"Unexpected --nn_dist_variant: {variant}")

    if add_knn_mean_dist:
        knn_mean = _compute_knn_mean_dist(meta, k=int(knn_k))  # [B,n]
        feats.append(knn_mean)
        feat_names.append(f"knn_mean_dist_k{int(knn_k)}")

    if add_centroid_dist:
        centroid = _compute_centroid_dist(meta)  # [B,n]
        feats.append(centroid)
        feat_names.append("centroid_dist")

    if add_oracle_splice_contrib:
        oracle = _compute_oracle_splice_contrib(
            meta, cache_path=oracle_splice_cache, timeout_sec=float(concorde_timeout_sec)
        )  # [B,n]
        feats.append(oracle)
        feat_names.append("oracle_splice_contrib_pct")

    if feats:
        extra = torch.stack(feats, dim=-1)  # [B,n,k]
        X_inst = torch.cat([X_inst, extra], dim=-1)
        meta = dict(meta)
        meta["extra_feature_names"] = feat_names

    return InstanceTensors(X=X_inst, y=y_inst, valid=valid_inst, labels=labels, meta=meta)


def standardize_X(X_inst: torch.Tensor, train_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X_train = X_inst[train_ids].reshape(-1, X_inst.shape[-1])
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (X_inst - mean) / std, mean, std


def standardize_y(
    y_inst: torch.Tensor, valid_inst: torch.Tensor, train_ids: torch.Tensor
) -> Tuple[torch.Tensor, float, float]:
    y_train = y_inst[train_ids]
    v_train = valid_inst[train_ids]
    vals = y_train[v_train]
    mean = float(vals.mean().item()) if vals.numel() else 0.0
    std = float(vals.std(unbiased=False).clamp_min(1e-6).item()) if vals.numel() else 1.0
    return (y_inst - mean) / std, mean, std


class SetTransformerProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        head_mlp_layers: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Identity() if input_dim == model_dim else nn.Linear(input_dim, model_dim, bias=True)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        if head_mlp_layers == 0:
            self.head = nn.Linear(model_dim, 1, bias=True)
        elif head_mlp_layers == 2:
            self.head = nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, 1, bias=True),
            )
        else:
            raise ValueError("head_mlp_layers must be 0 or 2")

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.head(h).squeeze(-1)


class DeepSetProbe(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, ff_dim: int, dropout: float, head_mlp_layers: int) -> None:
        super().__init__()
        self.in_proj = nn.Identity() if input_dim == model_dim else nn.Linear(input_dim, model_dim, bias=True)
        self.phi = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if head_mlp_layers == 0:
            self.rho = nn.Linear(model_dim * 2, 1, bias=True)
        elif head_mlp_layers == 2:
            self.rho = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, 1, bias=True),
            )
        else:
            raise ValueError("head_mlp_layers must be 0 or 2")

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.phi(h)
        mask = valid.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        ctx = (h * mask).sum(dim=1) / denom  # [B,d]
        ctx = ctx.unsqueeze(1).expand_as(h)
        out = self.rho(torch.cat([h, ctx], dim=-1)).squeeze(-1)
        return out


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,n,d] -> [B,n]
        return self.proj(x).squeeze(-1)


def score(model: nn.Module, xb: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    if isinstance(model, SetTransformerProbe):
        return model(xb, key_padding_mask=~vb)
    if isinstance(model, DeepSetProbe):
        return model(xb, vb)
    if isinstance(model, LinearProbe):
        return model(xb)
    raise TypeError(f"Unsupported model type: {type(model)}")


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err2 = (pred - target).pow(2)
    denom = mask.float().sum().clamp_min(1.0)
    return (err2 * mask.float()).sum() / denom


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    err = pred - target
    abs_err = err.abs()
    quad = torch.minimum(abs_err, torch.tensor(float(delta), device=err.device))
    lin = abs_err - quad
    loss = 0.5 * quad.pow(2) + float(delta) * lin
    denom = mask.float().sum().clamp_min(1.0)
    return (loss * mask.float()).sum() / denom


def listwise_loss(
    preds: torch.Tensor,
    target_y: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
    mode: str,
) -> torch.Tensor:
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0")
    logits = preds.masked_fill(~valid, float("-inf"))
    log_probs = F.log_softmax(logits, dim=1)

    tgt_logits = (target_y / float(temperature)).masked_fill(~valid, float("-inf"))
    tgt_probs = F.softmax(tgt_logits, dim=1)

    if mode == "ce":
        return -(tgt_probs * log_probs).sum(dim=1).mean()
    if mode == "kl":
        return F.kl_div(log_probs, tgt_probs, reduction="batchmean")
    raise ValueError(f"Unknown listwise mode: {mode}")


def ranknet_loss(
    preds: torch.Tensor,
    target_y: torch.Tensor,
    valid: torch.Tensor,
    num_pairs: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    B, n = preds.shape
    losses: List[torch.Tensor] = []
    for bi in range(B):
        idxs = torch.nonzero(valid[bi], as_tuple=False).flatten()
        if int(idxs.numel()) < 2:
            continue
        m = int(idxs.numel())
        # Sample pairs uniformly among valid indices.
        ii = torch.randint(0, m, size=(int(num_pairs),), generator=generator, device=preds.device)
        jj = torch.randint(0, m, size=(int(num_pairs),), generator=generator, device=preds.device)
        neq = ii != jj
        if not bool(neq.any()):
            continue
        ii = ii[neq]
        jj = jj[neq]
        a = idxs[ii]
        b = idxs[jj]
        ya = target_y[bi, a]
        yb = target_y[bi, b]
        t = (ya > yb).to(torch.float32)
        keep = (ya != yb)
        if not bool(keep.any()):
            continue
        diff = preds[bi, a] - preds[bi, b]
        losses.append(F.binary_cross_entropy_with_logits(diff[keep], t[keep], reduction="mean"))
    if not losses:
        return torch.tensor(0.0, device=preds.device)
    return torch.stack(losses, dim=0).mean()


def compute_loss(
    preds: torch.Tensor,
    y_for_loss: torch.Tensor,
    valid: torch.Tensor,
    loss_name: str,
    huber_delta: float,
    temperature: float,
    ranknet_pairs: int,
    pair_generator: Optional[torch.Generator],
) -> torch.Tensor:
    if loss_name == "mse":
        return masked_mse(preds, y_for_loss, valid)
    if loss_name == "huber":
        return masked_huber(preds, y_for_loss, valid, delta=huber_delta)
    if loss_name == "listwise_ce":
        return listwise_loss(preds, y_for_loss, valid, temperature=temperature, mode="ce")
    if loss_name == "listwise_kl":
        return listwise_loss(preds, y_for_loss, valid, temperature=temperature, mode="kl")
    if loss_name == "ranknet":
        return ranknet_loss(preds, y_for_loss, valid, num_pairs=ranknet_pairs, generator=pair_generator)
    if loss_name == "hard_ce":
        logits = preds.masked_fill(~valid, -1e9)
        y_masked = y_for_loss.masked_fill(~valid, float("-inf"))
        labels = torch.argmax(y_masked, dim=1)
        return F.cross_entropy(logits, labels)
    raise ValueError(f"Unknown loss: {loss_name}")


@torch.no_grad()
def eval_split(
    model: nn.Module,
    tensors: InstanceTensors,
    instance_ids: torch.Tensor,
    device: torch.device,
    x_mean: Optional[torch.Tensor],
    x_std: Optional[torch.Tensor],
    y_mean: Optional[float],
    y_std: Optional[float],
    loss_name: str,
    loss2_name: Optional[str],
    loss2_weight: float,
    huber_delta: float,
    temperature: float,
    ranknet_pairs: int,
    seed: int,
) -> Dict[str, float]:
    model.eval()
    X = tensors.X[instance_ids].to(device)
    valid = tensors.valid[instance_ids].to(device)
    labels = tensors.labels[instance_ids].to(device)
    y_raw = tensors.y[instance_ids].to(device)

    if x_mean is not None and x_std is not None:
        X = (X - x_mean.to(device)) / x_std.to(device)

    y_for_loss = y_raw
    if y_mean is not None and y_std is not None:
        y_for_loss = (y_raw - float(y_mean)) / float(y_std)

    preds = score(model, X, valid)

    g = torch.Generator(device=device.type).manual_seed(int(seed))
    loss1 = compute_loss(
        preds,
        y_for_loss,
        valid,
        loss_name=loss_name,
        huber_delta=huber_delta,
        temperature=temperature,
        ranknet_pairs=ranknet_pairs,
        pair_generator=g,
    )
    loss = loss1
    if loss2_name is not None and float(loss2_weight) != 0.0:
        loss = loss + float(loss2_weight) * compute_loss(
            preds,
            y_for_loss,
            valid,
            loss_name=loss2_name,
            huber_delta=huber_delta,
            temperature=temperature,
            ranknet_pairs=ranknet_pairs,
            pair_generator=g,
        )

    pred_best = torch.argmax(preds.masked_fill(~valid, float("-inf")), dim=1).to(torch.int64)
    top1 = float((pred_best == labels).float().mean().item())
    k = min(5, int(preds.shape[1]))
    topk = torch.topk(preds.masked_fill(~valid, float("-inf")), k=k, dim=1).indices
    top5 = float((topk == labels.unsqueeze(1)).any(dim=1).float().mean().item())

    preds_cpu = preds.detach().cpu()
    y_cpu = y_raw.detach().cpu()
    valid_cpu = valid.detach().cpu()

    y_masked = y_cpu.clone()
    y_masked[~valid_cpu] = float("-inf")
    best_val = torch.max(y_masked, dim=1).values
    chosen = y_cpu.gather(dim=1, index=pred_best.detach().cpu().unsqueeze(1)).squeeze(1)
    regret = float((best_val - chosen).mean().item())

    spearmans: List[float] = []
    for bi in range(int(y_cpu.shape[0])):
        mask = valid_cpu[bi]
        if int(mask.sum().item()) < 2:
            continue
        spearmans.append(spearman_corr(preds_cpu[bi][mask], y_cpu[bi][mask]))
    xs = [x for x in spearmans if not (isinstance(x, float) and math.isnan(x))]
    spearman = float(np.mean(xs)) if xs else float("nan")

    return {
        "loss": float(loss.detach().cpu().item()),
        "top1_acc": top1,
        "top5_acc": top5,
        "spearman_mean": spearman,
        "top1_regret_mean": regret,
    }


def make_model(args: argparse.Namespace, input_dim: int) -> nn.Module:
    if args.model == "set_transformer":
        return SetTransformerProbe(
            input_dim=input_dim,
            model_dim=int(args.model_dim),
            num_layers=int(args.layers),
            num_heads=int(args.heads),
            ff_dim=int(args.ff_dim),
            dropout=float(args.dropout),
            head_mlp_layers=int(args.head_mlp_layers),
        )
    if args.model == "deepset":
        return DeepSetProbe(
            input_dim=input_dim,
            model_dim=int(args.model_dim),
            ff_dim=int(args.ff_dim),
            dropout=float(args.dropout),
            head_mlp_layers=int(args.head_mlp_layers),
        )
    if args.model == "linear":
        return LinearProbe(input_dim=input_dim)
    raise ValueError(f"Unknown model: {args.model}")


def train_one(
    tensors: InstanceTensors,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    run_seed: int,
) -> Tuple[nn.Module, Optional[torch.Tensor], Optional[torch.Tensor], Optional[float], Optional[float], Dict[str, float]]:
    set_seed(run_seed)

    X_inst = tensors.X
    x_mean = x_std = None
    if bool(args.standardize_x):
        X_inst, x_mean, x_std = standardize_X(X_inst, train_ids=train_ids)

    y_inst = tensors.y
    y_mean = y_std = None
    if bool(args.standardize_y):
        y_inst, y_mean, y_std = standardize_y(y_inst, tensors.valid, train_ids=train_ids)

    model = make_model(args, input_dim=int(X_inst.shape[-1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_ids = train_ids.to(torch.int64)
    g = torch.Generator().manual_seed(int(run_seed))

    best_state = None
    best_metric = None

    for epoch in range(int(args.num_epochs)):
        model.train()
        perm = train_ids[torch.randperm(int(train_ids.numel()), generator=g)]
        for start in range(0, int(perm.numel()), int(args.batch_size)):
            batch = perm[start : start + int(args.batch_size)]
            xb = X_inst[batch].to(device)
            vb = tensors.valid[batch].to(device)
            yb = y_inst[batch].to(device)
            preds = score(model, xb, vb)

            pair_gen = torch.Generator(device=device.type).manual_seed(int(run_seed) * 100000 + epoch * 131 + start)
            loss = compute_loss(
                preds,
                yb,
                vb,
                loss_name=str(args.loss),
                huber_delta=float(args.huber_delta),
                temperature=float(args.temperature),
                ranknet_pairs=int(args.ranknet_pairs),
                pair_generator=pair_gen,
            )
            if args.loss2 is not None and float(args.loss2_weight) != 0.0:
                loss = loss + float(args.loss2_weight) * compute_loss(
                    preds,
                    yb,
                    vb,
                    loss_name=str(args.loss2),
                    huber_delta=float(args.huber_delta),
                    temperature=float(args.temperature),
                    ranknet_pairs=int(args.ranknet_pairs),
                    pair_generator=pair_gen,
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.clip_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad_norm))
            opt.step()

        # Selection on val split (loss or top1).
        model.eval()
        with torch.no_grad():
            xb = X_inst[val_ids].to(device)
            vb = tensors.valid[val_ids].to(device)
            yb = y_inst[val_ids].to(device)
            preds = score(model, xb, vb)
            pair_gen = torch.Generator(device=device.type).manual_seed(int(run_seed) * 777 + epoch * 17)
            val_loss = float(
                (
                    compute_loss(
                        preds,
                        yb,
                        vb,
                        loss_name=str(args.loss),
                        huber_delta=float(args.huber_delta),
                        temperature=float(args.temperature),
                        ranknet_pairs=int(args.ranknet_pairs),
                        pair_generator=pair_gen,
                    )
                    + (
                        float(args.loss2_weight)
                        * compute_loss(
                            preds,
                            yb,
                            vb,
                            loss_name=str(args.loss2),
                            huber_delta=float(args.huber_delta),
                            temperature=float(args.temperature),
                            ranknet_pairs=int(args.ranknet_pairs),
                            pair_generator=pair_gen,
                        )
                        if args.loss2 is not None and float(args.loss2_weight) != 0.0
                        else 0.0
                    )
                )
                .detach()
                .cpu()
                .item()
            )

            pred_best = torch.argmax(preds.masked_fill(~vb, float("-inf")), dim=1).to(torch.int64)
            labels = tensors.labels[val_ids].to(device)
            val_top1 = float((pred_best == labels).float().mean().item())

        metric = val_loss if str(args.select_metric) == "val_loss" else val_top1
        better = best_metric is None or (
            metric < best_metric if str(args.select_metric) == "val_loss" else metric > best_metric
        )
        if better:
            best_metric = float(metric)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    best_val = {"best_val_metric": float(best_metric) if best_metric is not None else float("nan")}
    return model, x_mean, x_std, y_mean, y_std, best_val


def _mean_std(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "std": float("nan")}
    if len(xs) == 1:
        return {"mean": float(xs[0]), "std": 0.0}
    mean = float(np.mean(xs))
    std = float(np.std(xs))
    return {"mean": mean, "std": std}


def main() -> None:
    args = build_arg_parser().parse_args()
    reps_path = Path(args.reps_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = as_device(args.device)
    tensors = load_instance_tensors(
        reps_path,
        add_nn_dist=bool(args.add_nn_dist),
        nn_dist_variant=str(args.nn_dist_variant),
        add_knn_mean_dist=bool(args.add_knn_mean_dist),
        knn_k=int(args.knn_k),
        add_centroid_dist=bool(args.add_centroid_dist),
        add_oracle_splice_contrib=bool(args.add_oracle_splice_contrib),
        oracle_splice_cache=str(args.oracle_splice_cache) if args.oracle_splice_cache else None,
        concorde_timeout_sec=float(args.concorde_timeout_sec),
    )
    B = int(tensors.X.shape[0])

    seeds = parse_int_list(args.seeds)
    repeats = int(args.repeats)
    train_size = int(args.train_size)
    val_size = int(args.val_size)
    test_size = int(args.test_size)
    if train_size + val_size + test_size > B:
        raise ValueError("Split sizes exceed available instances")

    cfg = vars(args)
    cfg["reps_path"] = str(reps_path)
    cfg["device"] = str(device)
    with open(out_dir / "config.json", "w") as fp:
        json.dump(cfg, fp, indent=2)

    per_seed_best: List[Dict] = []
    all_runs: List[Dict] = []

    for seed in seeds:
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(B, generator=g)
        test_ids = perm[:test_size]
        val_ids = perm[test_size : test_size + val_size]
        train_pool = perm[test_size + val_size :]
        train_ids = train_pool[:train_size]

        best = None
        for rep in range(repeats):
            run_seed = int(seed) * 1000 + rep * 31 + int(args.model_dim) * 7 + int(args.layers) * 11
            model, x_mean, x_std, y_mean, y_std, best_val = train_one(
                tensors=tensors,
                train_ids=train_ids,
                val_ids=val_ids,
                device=device,
                args=args,
                run_seed=run_seed,
            )

            train_m = eval_split(
                model,
                tensors,
                train_ids,
                device=device,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                loss_name=str(args.loss),
                loss2_name=str(args.loss2) if args.loss2 is not None else None,
                loss2_weight=float(args.loss2_weight),
                huber_delta=float(args.huber_delta),
                temperature=float(args.temperature),
                ranknet_pairs=int(args.ranknet_pairs),
                seed=run_seed + 1,
            )
            val_m = eval_split(
                model,
                tensors,
                val_ids,
                device=device,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                loss_name=str(args.loss),
                loss2_name=str(args.loss2) if args.loss2 is not None else None,
                loss2_weight=float(args.loss2_weight),
                huber_delta=float(args.huber_delta),
                temperature=float(args.temperature),
                ranknet_pairs=int(args.ranknet_pairs),
                seed=run_seed + 2,
            )
            test_m = eval_split(
                model,
                tensors,
                test_ids,
                device=device,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                loss_name=str(args.loss),
                loss2_name=str(args.loss2) if args.loss2 is not None else None,
                loss2_weight=float(args.loss2_weight),
                huber_delta=float(args.huber_delta),
                temperature=float(args.temperature),
                ranknet_pairs=int(args.ranknet_pairs),
                seed=run_seed + 3,
            )

            rec = {
                "seed": int(seed),
                "repeat": int(rep),
                **best_val,
                "train": train_m,
                "val": val_m,
                "test": test_m,
            }
            all_runs.append(rec)

            # Choose best per-seed by the same selection metric used in training.
            score_key = "loss" if str(args.select_metric) == "val_loss" else "top1_acc"
            better = best is None or (
                float(rec["val"][score_key]) < float(best["val"][score_key])
                if str(args.select_metric) == "val_loss"
                else float(rec["val"][score_key]) > float(best["val"][score_key])
            )
            if better:
                best = rec

            print(
                f"[probe] seed={seed} rep={rep} "
                f"val top1={val_m['top1_acc']:.3f} test top1={test_m['top1_acc']:.3f} "
                f"top5={test_m['top5_acc']:.3f} loss={test_m['loss']:.3f}"
            )

        if best is not None:
            per_seed_best.append(best)

    agg = {
        "test_top1_acc": _mean_std([float(r["test"]["top1_acc"]) for r in per_seed_best]),
        "test_top5_acc": _mean_std([float(r["test"]["top5_acc"]) for r in per_seed_best]),
        "test_spearman_mean": _mean_std([float(r["test"]["spearman_mean"]) for r in per_seed_best]),
        "test_top1_regret_mean": _mean_std([float(r["test"]["top1_regret_mean"]) for r in per_seed_best]),
        "test_loss": _mean_std([float(r["test"]["loss"]) for r in per_seed_best]),
        "num_seeds": int(len(per_seed_best)),
    }

    payload = {"config": cfg, "per_seed_best": per_seed_best, "all_runs": all_runs, "aggregate": agg}
    with open(out_dir / "results.json", "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"[probe] aggregate test top1 mean={agg['test_top1_acc']['mean']:.3f} std={agg['test_top1_acc']['std']:.3f}")
    print(f"[probe] wrote: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
