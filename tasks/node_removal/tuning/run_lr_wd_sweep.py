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
        description="Grid sweep lr/weight_decay for the best-node CE probe at a fixed train size."
    )
    p.add_argument(
        "--reps_path",
        type=str,
        default=(
            "tasks/node_removal/data/processed/"
            "TSP100_uniform_expdecay_12-24_00:09:50__n3000__seed0__n3000_s20/probe_reps.pt"
        ),
        help="Path to probe_reps.pt (from extract_representations.py).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="tasks/node_removal/tmp/tuning/lr_wd_sweep_train2500",
        help="Output directory.",
    )

    p.add_argument("--seed", type=int, default=0, help="Seed for the fixed split and training order.")
    p.add_argument("--device", type=str, default=None, help="Device string (cpu/cuda). Default: auto.")

    p.add_argument("--test_size", type=int, default=300, help="Held-out test instances (fixed).")
    p.add_argument("--val_size", type=int, default=200, help="Held-out val instances (fixed).")
    p.add_argument("--train_size", type=int, default=2500, help="Train instances.")

    p.add_argument("--batch_size", type=int, default=128, help="Instances per training batch.")
    p.add_argument("--num_epochs", type=int, default=200, help="Training epochs.")
    p.add_argument("--standardize_x", action="store_true", help="Standardize X using train mean/std.")
    p.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=["linear", "set_transformer"],
        help="Probe model family.",
    )
    p.add_argument("--st_model_dim", type=int, default=128, help="Model dim for --model set_transformer.")
    p.add_argument("--st_layers", type=int, default=1, help="Transformer layers for --model set_transformer.")
    p.add_argument("--st_heads", type=int, default=4, help="Attention heads for --model set_transformer.")
    p.add_argument("--st_ff_dim", type=int, default=256, help="Feedforward dim for --model set_transformer.")
    p.add_argument("--st_dropout", type=float, default=0.1, help="Dropout for --model set_transformer.")
    p.add_argument(
        "--lrs",
        type=str,
        default="1e-3,3e-3,1e-2",
        help="Comma-separated learning rates to try.",
    )
    p.add_argument(
        "--wds",
        type=str,
        default="0,1e-5,1e-4,1e-3",
        help="Comma-separated weight decays to try.",
    )
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


@dataclass(frozen=True)
class InstanceTensors:
    X: torch.Tensor  # [B,n,d]
    y: torch.Tensor  # [B,n]
    valid: torch.Tensor  # [B,n]
    labels: torch.Tensor  # [B]
    meta: Dict


class SetTransformerProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be >= 1")
        if model_dim <= 0:
            raise ValueError("model_dim must be >= 1")
        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1")
        if num_heads <= 0:
            raise ValueError("num_heads must be >= 1")
        if ff_dim <= 0:
            raise ValueError("ff_dim must be >= 1")
        if output_dim <= 0:
            raise ValueError("output_dim must be >= 1")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0,1)")

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
        self.out_proj = nn.Linear(model_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.out_proj(h)


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

    # Use delta_length_pct only (column 0).
    y = y[:, 0].to(torch.float32)

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


@torch.no_grad()
def eval_probe(
    model: nn.Module,
    tensors: InstanceTensors,
    instance_ids: torch.Tensor,
    device: torch.device,
    x_mean: Optional[torch.Tensor],
    x_std: Optional[torch.Tensor],
    model_type: str,
) -> Dict[str, float]:
    model.eval()
    X = tensors.X[instance_ids].to(device)
    valid = tensors.valid[instance_ids].to(device)
    labels = tensors.labels[instance_ids].to(device)
    y = tensors.y[instance_ids].detach().cpu()

    if x_mean is not None and x_std is not None:
        X = (X - x_mean.to(device)) / x_std.to(device)

    if model_type == "set_transformer":
        logits = model(X, key_padding_mask=~valid).squeeze(-1)
    else:
        logits = model(X).squeeze(-1)  # [B,n]
    logits = logits.masked_fill(~valid, -1e9)
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


def train_probe(
    tensors: InstanceTensors,
    train_ids: torch.Tensor,
    device: torch.device,
    seed: int,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    standardize_x: bool,
    model_type: str,
    st_model_dim: int,
    st_layers: int,
    st_heads: int,
    st_ff_dim: int,
    st_dropout: float,
) -> Tuple[nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]:
    set_seed(seed)

    X_inst = tensors.X
    x_mean = x_std = None
    if standardize_x:
        X_inst, x_mean, x_std = standardize_X(X_inst, train_ids=train_ids)

    d = int(X_inst.shape[-1])
    if model_type == "set_transformer":
        model = SetTransformerProbe(
            input_dim=d,
            model_dim=int(st_model_dim),
            num_layers=int(st_layers),
            num_heads=int(st_heads),
            ff_dim=int(st_ff_dim),
            dropout=float(st_dropout),
            output_dim=1,
        ).to(device)
    else:
        model = nn.Linear(d, 1, bias=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ids = train_ids.to(torch.int64)
    g = torch.Generator().manual_seed(seed)
    for _ in range(int(num_epochs)):
        model.train()
        perm = train_ids[torch.randperm(int(train_ids.numel()), generator=g)]
        for start in range(0, int(perm.numel()), int(batch_size)):
            batch = perm[start : start + int(batch_size)]
            xb = X_inst[batch].to(device)
            vb = tensors.valid[batch].to(device)
            yb = tensors.labels[batch].to(device)
            if model_type == "set_transformer":
                logits = model(xb, key_padding_mask=~vb).squeeze(-1)
            else:
                logits = model(xb).squeeze(-1)
            logits = logits.masked_fill(~vb, -1e9)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model, x_mean, x_std


def parse_floats(csv: str) -> List[float]:
    out = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    with open(path, "w") as fp:
        fp.write("lr,weight_decay,split,top1_acc,top5_acc,loss,spearman_mean,top1_regret_mean\n")
        for row in rows:
            m = row["metrics"]
            fp.write(
                ",".join(
                    [
                        str(row["lr"]),
                        str(row["weight_decay"]),
                        str(row["split"]),
                        str(m["top1_acc"]),
                        str(m["top5_acc"]),
                        str(m["loss"]),
                        str(m["spearman_mean"]),
                        str(m["top1_regret_mean"]),
                    ]
                )
                + "\n"
            )


def plot_heatmap(out_dir: Path, grid: Dict[Tuple[float, float], Dict[str, Dict[str, float]]], lrs: List[float], wds: List[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mat(split: str, key: str) -> np.ndarray:
        arr = np.full((len(lrs), len(wds)), np.nan, dtype=float)
        for i, lr in enumerate(lrs):
            for j, wd in enumerate(wds):
                arr[i, j] = float(grid[(lr, wd)][split][key])
        return arr

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, split in zip(axes, ["val", "test"]):
        arr = mat(split, "top1_acc")
        im = ax.imshow(arr, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{split} top1_acc")
        ax.set_xticks(range(len(wds)))
        ax.set_xticklabels([f"{wd:g}" for wd in wds], rotation=45, ha="right")
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{lr:g}" for lr in lrs])
        ax.set_xlabel("weight_decay")
        ax.set_ylabel("lr")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_top1.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    reps_path = Path(args.reps_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_instance_tensors(reps_path)
    device = as_device(args.device)

    B = int(tensors.X.shape[0])
    test_size = int(args.test_size)
    val_size = int(args.val_size)
    train_size = int(args.train_size)

    if test_size <= 0 or val_size <= 0 or train_size <= 0:
        raise ValueError("test_size, val_size, train_size must be >= 1")
    if test_size + val_size + train_size > B:
        raise ValueError(f"Split sizes exceed available instances: {test_size}+{val_size}+{train_size} > {B}")

    g = torch.Generator().manual_seed(int(args.seed))
    perm = torch.randperm(B, generator=g)
    test_ids = perm[:test_size]
    val_ids = perm[test_size : test_size + val_size]
    train_pool = perm[test_size + val_size :]
    train_ids = train_pool[:train_size]

    lrs = parse_floats(str(args.lrs))
    wds = parse_floats(str(args.wds))
    lrs_sorted = sorted(lrs)
    wds_sorted = sorted(wds)

    config = {
        "reps_path": str(reps_path),
        "run_dir": tensors.meta.get("run_dir"),
        "seed": int(args.seed),
        "device": str(device),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "batch_size": int(args.batch_size),
        "num_epochs": int(args.num_epochs),
        "standardize_x": bool(args.standardize_x),
        "model": str(args.model),
        "st_model_dim": int(args.st_model_dim),
        "st_layers": int(args.st_layers),
        "st_heads": int(args.st_heads),
        "st_ff_dim": int(args.st_ff_dim),
        "st_dropout": float(args.st_dropout),
        "lrs": lrs_sorted,
        "wds": wds_sorted,
    }
    with open(out_dir / "config.json", "w") as fp:
        json.dump(config, fp, indent=2)

    print(f"[tune] reps: {reps_path}")
    print(f"[tune] train/val/test: {train_size}/{val_size}/{test_size} (total {B})")
    print(f"[tune] lrs: {lrs_sorted}")
    print(f"[tune] wds: {wds_sorted}")
    print(
        f"[tune] epochs={args.num_epochs} bs={args.batch_size} standardize_x={bool(args.standardize_x)} "
        f"model={args.model} device={device}"
    )

    rows: List[Dict] = []
    grid: Dict[Tuple[float, float], Dict[str, Dict[str, float]]] = {}

    for lr in lrs_sorted:
        for wd in wds_sorted:
            run_seed = int(args.seed) * 10000 + int(round(lr * 1e6)) + int(round(wd * 1e8))
            model, x_mean, x_std = train_probe(
                tensors=tensors,
                train_ids=train_ids,
                device=device,
                seed=run_seed,
                batch_size=int(args.batch_size),
                num_epochs=int(args.num_epochs),
                lr=float(lr),
                weight_decay=float(wd),
                standardize_x=bool(args.standardize_x),
                model_type=str(args.model),
                st_model_dim=int(args.st_model_dim),
                st_layers=int(args.st_layers),
                st_heads=int(args.st_heads),
                st_ff_dim=int(args.st_ff_dim),
                st_dropout=float(args.st_dropout),
            )

            model_type = str(args.model)
            train_m = eval_probe(model, tensors, train_ids, device=device, x_mean=x_mean, x_std=x_std, model_type=model_type)
            val_m = eval_probe(model, tensors, val_ids, device=device, x_mean=x_mean, x_std=x_std, model_type=model_type)
            test_m = eval_probe(model, tensors, test_ids, device=device, x_mean=x_mean, x_std=x_std, model_type=model_type)

            grid[(float(lr), float(wd))] = {"train": train_m, "val": val_m, "test": test_m}

            for split, metrics in [("train", train_m), ("val", val_m), ("test", test_m)]:
                rows.append({"lr": float(lr), "weight_decay": float(wd), "split": split, "metrics": metrics})

            print(
                f"[tune] lr={lr:g} wd={wd:g} "
                f"val top1={val_m['top1_acc']:.3f} test top1={test_m['top1_acc']:.3f} top5={test_m['top5_acc']:.3f}"
            )

    write_csv(out_dir / "results.csv", rows)
    with open(out_dir / "results.json", "w") as fp:
        json.dump({"config": config, "grid": {f"{k[0]}__{k[1]}": v for k, v in grid.items()}}, fp, indent=2)

    plot_heatmap(out_dir, grid, lrs_sorted, wds_sorted)

    # Report best by val top1_acc.
    best = max(((lr, wd, grid[(lr, wd)]["val"]["top1_acc"]) for lr in lrs_sorted for wd in wds_sorted), key=lambda x: x[2])
    best_lr, best_wd, best_val = best
    best_test = grid[(best_lr, best_wd)]["test"]["top1_acc"]
    print(f"[tune] best by val top1: lr={best_lr:g} wd={best_wd:g} val_top1={best_val:.3f} test_top1={best_test:.3f}")
    print(f"[tune] wrote: {out_dir / 'results.csv'}")

if __name__ == "__main__":
    main()
