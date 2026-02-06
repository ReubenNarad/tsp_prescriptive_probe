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
        description="Small hyperparameter sweep for the set-transformer what-if probe (best_node_ce)."
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
        default="tasks/node_removal/tmp/tuning/set_transformer_sweep_train2500_seed0",
        help="Output directory.",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for fixed split and per-run seeds.")
    p.add_argument("--device", type=str, default=None, help="Device string (cpu/cuda). Default: auto.")

    p.add_argument("--train_size", type=int, default=2500)
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument("--test_size", type=int, default=300)

    p.add_argument("--batch_size", type=int, default=64, help="Instances per training batch.")
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--standardize_x", action="store_true", help="Standardize X using train mean/std.")

    p.add_argument("--model_dims", type=str, default="128,256", help="Comma-separated model dims.")
    p.add_argument("--layers", type=str, default="1,2", help="Comma-separated layer counts.")
    p.add_argument("--heads", type=str, default="4", help="Comma-separated head counts.")
    p.add_argument("--dropouts", type=str, default="0.0,0.1", help="Comma-separated dropouts.")
    p.add_argument("--ff_mult", type=float, default=2.0, help="ff_dim = round(ff_mult * model_dim).")

    p.add_argument("--lrs", type=str, default="1e-3", help="Comma-separated learning rates.")
    p.add_argument("--wds", type=str, default="1e-4", help="Comma-separated weight decays.")
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


def parse_ints(csv: str) -> List[int]:
    out: List[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_floats(csv: str) -> List[float]:
    out: List[float] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
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
        return self.out_proj(h).squeeze(-1)  # [B,n]


def score(model: SetTransformerProbe, xb: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    return model(xb, key_padding_mask=~vb)


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
) -> Tuple[SetTransformerProbe, Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        # Simple val loss checkpointing.
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

    return model, x_mean, x_std


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


def write_csv(path: Path, rows: List[Dict]) -> None:
    with open(path, "w") as fp:
        fp.write(
            "model_dim,layers,heads,ff_dim,dropout,lr,weight_decay,split,top1_acc,top5_acc,loss,spearman_mean,top1_regret_mean\n"
        )
        for row in rows:
            m = row["metrics"]
            fp.write(
                ",".join(
                    [
                        str(row["model_dim"]),
                        str(row["layers"]),
                        str(row["heads"]),
                        str(row["ff_dim"]),
                        str(row["dropout"]),
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


def plot_top1(out_dir: Path, records: List[Dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records_sorted = sorted(records, key=lambda r: float(r["test"]["top1_acc"]), reverse=True)
    top = records_sorted[:12]
    labels = [
        f"d{r['model_dim']}-L{r['layers']}-h{r['heads']}-do{r['dropout']}-lr{r['lr']}-wd{r['weight_decay']}"
        for r in top
    ]
    vals = [float(r["test"]["top1_acc"]) for r in top]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("test top1_acc")
    ax.set_title("Set-transformer sweep (top configs by test top1)")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "top1_bar.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    reps_path = Path(args.reps_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_instance_tensors(reps_path)
    device = as_device(args.device)

    B = int(tensors.X.shape[0])
    train_size = int(args.train_size)
    val_size = int(args.val_size)
    test_size = int(args.test_size)
    if train_size + val_size + test_size > B:
        raise ValueError(f"Split sizes exceed available instances: {train_size}+{val_size}+{test_size} > {B}")

    g = torch.Generator().manual_seed(int(args.seed))
    perm = torch.randperm(B, generator=g)
    test_ids = perm[:test_size]
    val_ids = perm[test_size : test_size + val_size]
    train_pool = perm[test_size + val_size :]
    train_ids = train_pool[:train_size]

    model_dims = parse_ints(args.model_dims)
    layers_list = parse_ints(args.layers)
    heads_list = parse_ints(args.heads)
    dropouts = parse_floats(args.dropouts)
    lrs = parse_floats(args.lrs)
    wds = parse_floats(args.wds)

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
        "model_dims": model_dims,
        "layers": layers_list,
        "heads": heads_list,
        "dropouts": dropouts,
        "ff_mult": float(args.ff_mult),
        "lrs": lrs,
        "wds": wds,
    }
    with open(out_dir / "config.json", "w") as fp:
        json.dump(config, fp, indent=2)

    print(f"[st-sweep] reps: {reps_path}")
    print(f"[st-sweep] train/val/test: {train_size}/{val_size}/{test_size} (total {B})")
    print(f"[st-sweep] device: {device} epochs={args.num_epochs} bs={args.batch_size} standardize_x={bool(args.standardize_x)}")

    rows: List[Dict] = []
    records: List[Dict] = []

    for model_dim in model_dims:
        for layers in layers_list:
            for heads in heads_list:
                for dropout in dropouts:
                    ff_dim = max(1, int(round(float(args.ff_mult) * model_dim)))
                    for lr in lrs:
                        for wd in wds:
                            run_seed = int(args.seed) * 10_000 + model_dim * 31 + layers * 7 + heads * 13 + int(dropout * 1000) + int(lr * 1e6) + int(wd * 1e8)
                            model, x_mean, x_std = train_one(
                                tensors=tensors,
                                train_ids=train_ids,
                                val_ids=val_ids,
                                device=device,
                                run_seed=run_seed,
                                batch_size=int(args.batch_size),
                                num_epochs=int(args.num_epochs),
                                lr=float(lr),
                                weight_decay=float(wd),
                                standardize_x=bool(args.standardize_x),
                                model_dim=int(model_dim),
                                layers=int(layers),
                                heads=int(heads),
                                ff_dim=int(ff_dim),
                                dropout=float(dropout),
                            )

                            train_m = eval_split(model, tensors, train_ids, device=device, x_mean=x_mean, x_std=x_std)
                            val_m = eval_split(model, tensors, val_ids, device=device, x_mean=x_mean, x_std=x_std)
                            test_m = eval_split(model, tensors, test_ids, device=device, x_mean=x_mean, x_std=x_std)

                            record = {
                                "model_dim": int(model_dim),
                                "layers": int(layers),
                                "heads": int(heads),
                                "ff_dim": int(ff_dim),
                                "dropout": float(dropout),
                                "lr": float(lr),
                                "weight_decay": float(wd),
                                "train": train_m,
                                "val": val_m,
                                "test": test_m,
                            }
                            records.append(record)

                            for split, metrics in [("train", train_m), ("val", val_m), ("test", test_m)]:
                                rows.append(
                                    {
                                        "model_dim": int(model_dim),
                                        "layers": int(layers),
                                        "heads": int(heads),
                                        "ff_dim": int(ff_dim),
                                        "dropout": float(dropout),
                                        "lr": float(lr),
                                        "weight_decay": float(wd),
                                        "split": split,
                                        "metrics": metrics,
                                    }
                                )

                            print(
                                f"[st-sweep] d={model_dim} L={layers} h={heads} do={dropout} lr={lr:g} wd={wd:g} "
                                f"val top1={val_m['top1_acc']:.3f} test top1={test_m['top1_acc']:.3f} top5={test_m['top5_acc']:.3f}"
                            )

    write_csv(out_dir / "results.csv", rows)
    with open(out_dir / "results.json", "w") as fp:
        json.dump({"config": config, "records": records}, fp, indent=2)
    plot_top1(out_dir, records)

    best_val = max(records, key=lambda r: float(r["val"]["top1_acc"]))
    best_test = max(records, key=lambda r: float(r["test"]["top1_acc"]))
    summary = {
        "best_by_val_top1": best_val,
        "best_by_test_top1": best_test,
    }
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(
        "[st-sweep] best val top1:",
        f"d={best_val['model_dim']} L={best_val['layers']} h={best_val['heads']} do={best_val['dropout']}",
        f"lr={best_val['lr']:g} wd={best_val['weight_decay']:g} val_top1={best_val['val']['top1_acc']:.3f} test_top1={best_val['test']['top1_acc']:.3f}",
    )
    print(
        "[st-sweep] best test top1:",
        f"d={best_test['model_dim']} L={best_test['layers']} h={best_test['heads']} do={best_test['dropout']}",
        f"lr={best_test['lr']:g} wd={best_test['weight_decay']:g} test_top1={best_test['test']['top1_acc']:.3f}",
    )
    print(f"[st-sweep] wrote: {out_dir}")


if __name__ == "__main__":
    main()
