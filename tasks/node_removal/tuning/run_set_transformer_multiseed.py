#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train/evaluate a fixed set-transformer probe config across multiple random splits (seeds)."
    )
    p.add_argument(
        "--reps_path",
        type=str,
        default=(
            "tasks/node_removal/data/processed/"
            "TSP100_uniform_expdecay_12-24_00:09:50__n3000__seed0__n3000_s20/probe_reps.pt"
        ),
        help="Path to probe_reps.pt from extract_representations.py.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="tasks/node_removal/tmp/tuning/set_transformer_multiseed",
        help="Output directory.",
    )
    p.add_argument("--device", type=str, default=None, help="Device string (cpu/cuda). Default: auto.")

    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated list of seeds.")
    p.add_argument("--repeats", type=int, default=1, help="Training restarts per seed.")

    p.add_argument("--train_size", type=int, default=2500)
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--standardize_x", action="store_true")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--model_dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
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


def parse_int_list(csv: str) -> List[int]:
    out = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


@dataclass(frozen=True)
class InstanceTensors:
    X: torch.Tensor  # [B,n,d]
    y: torch.Tensor  # [B,n]
    valid: torch.Tensor  # [B,n]
    labels: torch.Tensor  # [B]
    meta: Dict


def load_instance_tensors(reps_path: Path) -> InstanceTensors:
    import os

    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    reps = torch.load(reps_path, weights_only=False)
    if not isinstance(reps, dict):
        raise TypeError(f"Unexpected reps format at {reps_path}: {type(reps)}")

    X = reps.get("X_resid")
    y = reps.get("y")
    valid = reps.get("valid")
    instance_id = reps.get("instance_id")
    node_id = reps.get("node_id")

    if not torch.is_tensor(X) or X.ndim != 2:
        raise ValueError("reps missing X_resid [N,d]")
    if not torch.is_tensor(y) or y.ndim != 2 or y.shape[1] < 1:
        raise ValueError("reps missing y [N,>=1]")
    if not torch.is_tensor(valid) or valid.ndim != 1:
        raise ValueError("reps missing valid [N]")
    if not torch.is_tensor(instance_id) or not torch.is_tensor(node_id):
        raise ValueError("reps missing instance_id/node_id")

    y = y[:, 0].to(torch.float32)  # delta_length_pct

    instance_id = instance_id.to(torch.int64)
    node_id = node_id.to(torch.int64)
    valid = valid.to(torch.bool)
    X = X.to(torch.float32)

    B = int(instance_id.max().item()) + 1
    n = int(node_id.max().item()) + 1
    if int(X.shape[0]) != B * n:
        raise ValueError(f"Expected N=B*n, got N={int(X.shape[0])}, B={B}, n={n}")

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
    if not bool(valid_inst.any(dim=1).all().item()):
        raise ValueError("Found instance(s) with no valid nodes.")
    labels = torch.argmax(y_masked, dim=1).to(torch.int64)

    meta = reps.get("meta", {}) if isinstance(reps.get("meta", {}), dict) else {}
    return InstanceTensors(X=X_inst, y=y_inst, valid=valid_inst, labels=labels, meta=meta)


def standardize_X(
    X_inst: torch.Tensor,
    train_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X_train = X_inst[train_ids].reshape(-1, X_inst.shape[-1])
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (X_inst - mean) / std, mean, std


class SetTransformerProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
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
        self.out_proj = nn.Linear(model_dim, 1, bias=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.out_proj(h).squeeze(-1)


def score(model: SetTransformerProbe, xb: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    return model(xb, key_padding_mask=~vb)


@torch.no_grad()
def eval_split(
    model: SetTransformerProbe,
    tensors: InstanceTensors,
    instance_ids: torch.Tensor,
    device: torch.device,
    x_mean: Optional[torch.Tensor],
    x_std: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.eval()
    X = tensors.X[instance_ids].to(device)
    valid = tensors.valid[instance_ids].to(device)
    labels = tensors.labels[instance_ids].to(device)
    y = tensors.y[instance_ids].detach().cpu()

    if x_mean is not None and x_std is not None:
        X = (X - x_mean.to(device)) / x_std.to(device)

    logits = score(model, X, valid).masked_fill(~valid, -1e9)
    loss = float(F.cross_entropy(logits, labels).item())

    pred = torch.argmax(logits, dim=1).to(torch.int64)
    top1 = float((pred == labels).float().mean().item())
    k = min(5, int(logits.shape[1]))
    topk = torch.topk(logits, k=k, dim=1).indices
    top5 = float((topk == labels.unsqueeze(1)).any(dim=1).float().mean().item())

    logits_cpu = logits.detach().cpu()
    valid_cpu = valid.detach().cpu()
    y_masked = y.clone()
    y_masked[~valid_cpu] = float("-inf")
    best = torch.max(y_masked, dim=1).values
    chosen = y.gather(dim=1, index=pred.cpu().unsqueeze(1)).squeeze(1)
    regret = float((best - chosen).mean().item())

    spearmans: List[float] = []
    for bi in range(int(y.shape[0])):
        mask = valid_cpu[bi]
        if int(mask.sum().item()) < 2:
            continue
        spearmans.append(spearman_corr(logits_cpu[bi][mask], y[bi][mask]))

    xs = [x for x in spearmans if not (isinstance(x, float) and math.isnan(x))]
    spearman = float(np.mean(xs)) if xs else float("nan")

    return {
        "loss": loss,
        "top1_acc": top1,
        "top5_acc": top5,
        "spearman_mean": spearman,
        "top1_regret_mean": regret,
        "num_instances": float(instance_ids.numel()),
    }


def train_one(
    tensors: InstanceTensors,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    device: torch.device,
    run_seed: int,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    standardize_x: bool,
    model_dim: int,
    layers: int,
    heads: int,
    ff_dim: int,
    dropout: float,
) -> Tuple[SetTransformerProbe, Optional[torch.Tensor], Optional[torch.Tensor], float]:
    set_seed(run_seed)

    X_inst = tensors.X
    x_mean = x_std = None
    if standardize_x:
        X_inst, x_mean, x_std = standardize_X(X_inst, train_ids=train_ids)

    d = int(X_inst.shape[-1])
    model = SetTransformerProbe(
        input_dim=d,
        model_dim=model_dim,
        num_layers=layers,
        num_heads=heads,
        ff_dim=ff_dim,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ids = train_ids.to(torch.int64)
    g = torch.Generator().manual_seed(run_seed)

    best_val = float("inf")
    best_state = None

    for _ in range(int(num_epochs)):
        model.train()
        perm = train_ids[torch.randperm(int(train_ids.numel()), generator=g)]
        for start in range(0, int(perm.numel()), int(batch_size)):
            batch = perm[start : start + int(batch_size)]
            xb = X_inst[batch].to(device)
            vb = tensors.valid[batch].to(device)
            yb = tensors.labels[batch].to(device)
            logits = score(model, xb, vb).masked_fill(~vb, -1e9)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # val loss checkpointing
        model.eval()
        with torch.no_grad():
            xb = X_inst[val_ids].to(device)
            vb = tensors.valid[val_ids].to(device)
            yb = tensors.labels[val_ids].to(device)
            logits = score(model, xb, vb).masked_fill(~vb, -1e9)
            val_loss = float(F.cross_entropy(logits, yb).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, x_mean, x_std, float(best_val)


def _mean_std(xs: List[float]) -> Dict[str, float]:
    xs2 = [x for x in xs if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]
    if not xs2:
        return {"mean": float("nan"), "std": float("nan")}
    if len(xs2) == 1:
        return {"mean": float(xs2[0]), "std": 0.0}
    return {"mean": float(np.mean(xs2)), "std": float(np.std(xs2))}


def main() -> None:
    args = build_arg_parser().parse_args()
    reps_path = Path(args.reps_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = as_device(args.device)
    tensors = load_instance_tensors(reps_path)
    B = int(tensors.X.shape[0])

    seeds = parse_int_list(args.seeds)
    repeats = int(args.repeats)
    train_size = int(args.train_size)
    val_size = int(args.val_size)
    test_size = int(args.test_size)
    if train_size + val_size + test_size > B:
        raise ValueError(f"Split sizes exceed available instances: {train_size}+{val_size}+{test_size} > {B}")

    cfg = {
        "reps_path": str(reps_path),
        "run_dir": tensors.meta.get("run_dir"),
        "device": str(device),
        "seeds": seeds,
        "repeats": repeats,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "standardize_x": bool(args.standardize_x),
        "batch_size": int(args.batch_size),
        "num_epochs": int(args.num_epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "model_dim": int(args.model_dim),
        "layers": int(args.layers),
        "heads": int(args.heads),
        "ff_dim": int(args.ff_dim),
        "dropout": float(args.dropout),
    }
    with open(out_dir / "config.json", "w") as fp:
        json.dump(cfg, fp, indent=2)

    print(f"[multiseed] reps: {reps_path}")
    print(f"[multiseed] device={device} B={B} train/val/test={train_size}/{val_size}/{test_size}")
    print(
        f"[multiseed] model d={args.model_dim} L={args.layers} h={args.heads} ff={args.ff_dim} do={args.dropout} "
        f"lr={args.lr:g} wd={args.weight_decay:g} epochs={args.num_epochs} bs={args.batch_size} stdx={bool(args.standardize_x)}"
    )
    print(f"[multiseed] seeds={seeds} repeats={repeats}")

    rows: List[Dict] = []
    per_seed_best: List[Dict] = []

    for seed in seeds:
        g = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(B, generator=g)
        test_ids = perm[:test_size]
        val_ids = perm[test_size : test_size + val_size]
        train_pool = perm[test_size + val_size :]
        train_ids = train_pool[:train_size]

        best = None
        for rep in range(repeats):
            run_seed = int(seed) * 1000 + rep * 17 + int(args.model_dim) * 3
            model, x_mean, x_std, best_val = train_one(
                tensors=tensors,
                train_ids=train_ids,
                val_ids=val_ids,
                device=device,
                run_seed=run_seed,
                batch_size=int(args.batch_size),
                num_epochs=int(args.num_epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                standardize_x=bool(args.standardize_x),
                model_dim=int(args.model_dim),
                layers=int(args.layers),
                heads=int(args.heads),
                ff_dim=int(args.ff_dim),
                dropout=float(args.dropout),
            )
            train_m = eval_split(model, tensors, train_ids, device=device, x_mean=x_mean, x_std=x_std)
            val_m = eval_split(model, tensors, val_ids, device=device, x_mean=x_mean, x_std=x_std)
            test_m = eval_split(model, tensors, test_ids, device=device, x_mean=x_mean, x_std=x_std)

            rec = {
                "seed": int(seed),
                "repeat": int(rep),
                "best_val_loss": float(best_val),
                "train": train_m,
                "val": val_m,
                "test": test_m,
            }
            rows.append(rec)
            if best is None or float(best_val) < float(best["best_val_loss"]):
                best = rec

            print(
                f"[multiseed] seed={seed} rep={rep} "
                f"val top1={val_m['top1_acc']:.3f} test top1={test_m['top1_acc']:.3f} top5={test_m['top5_acc']:.3f}"
            )

        if best is not None:
            per_seed_best.append(best)

    # Aggregate (best-by-val per seed).
    agg = {
        "test_top1_acc": _mean_std([float(r["test"]["top1_acc"]) for r in per_seed_best]),
        "test_top5_acc": _mean_std([float(r["test"]["top5_acc"]) for r in per_seed_best]),
        "test_spearman_mean": _mean_std([float(r["test"]["spearman_mean"]) for r in per_seed_best]),
        "test_top1_regret_mean": _mean_std([float(r["test"]["top1_regret_mean"]) for r in per_seed_best]),
        "num_seeds": int(len(per_seed_best)),
    }

    payload = {
        "config": cfg,
        "per_seed_best": per_seed_best,
        "all_runs": rows,
        "aggregate": agg,
    }
    with open(out_dir / "results.json", "w") as fp:
        json.dump(payload, fp, indent=2)

    # Tiny printout
    m = agg["test_top1_acc"]["mean"]
    s = agg["test_top1_acc"]["std"]
    print(f"[multiseed] aggregate test top1: mean={m:.3f} std={s:.3f} over {agg['num_seeds']} seeds")
    print(f"[multiseed] wrote: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

