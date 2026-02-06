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
        description=(
            "Sweep best-node CE probe performance vs training-set size while keeping a fixed held-out test set."
        )
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
    p.add_argument("--out_dir", type=str, default="tasks/node_removal/tmp/tuning", help="Output directory.")

    p.add_argument("--seed", type=int, default=0, help="Seed for fixed test split and training order.")
    p.add_argument("--device", type=str, default=None, help="Device string (cuda/cpu). Default: auto.")

    p.add_argument("--test_size", type=int, default=300, help="Number of held-out test instances (fixed).")
    p.add_argument(
        "--train_step",
        type=int,
        default=200,
        help="Train size increment for the sweep (e.g. 200 gives 200,400,...).",
    )
    p.add_argument(
        "--max_train",
        type=int,
        default=3000,
        help="Requested max train instances (capped by available instances after holding out test_size).",
    )
    p.add_argument("--repeats", type=int, default=1, help="Number of random train-subset draws per train_size.")

    p.add_argument("--batch_size", type=int, default=1024, help="Instances per batch during training.")
    p.add_argument("--num_epochs", type=int, default=25, help="Training epochs per sweep point.")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--standardize_x", action="store_true", help="Standardize X using train-subset mean/std.")
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


def load_instance_tensors(reps_path: Path) -> InstanceTensors:
    os_environ = getattr(__import__("os"), "environ")
    os_environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")

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
    has_any = valid_inst.any(dim=1)
    if not bool(has_any.all().item()):
        # Keep behavior explicit; missing labels would complicate the sweep.
        bad = int((~has_any).sum().item())
        raise ValueError(f"Found {bad} instances with no valid nodes.")
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
def eval_ce_probe(
    model: nn.Module,
    tensors: InstanceTensors,
    instance_ids: torch.Tensor,
    device: torch.device,
    x_mean: Optional[torch.Tensor],
    x_std: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.eval()
    X = tensors.X[instance_ids].to(device)
    y = tensors.y[instance_ids]
    valid = tensors.valid[instance_ids]
    labels = tensors.labels[instance_ids].to(device)

    if x_mean is not None and x_std is not None:
        X = (X - x_mean.to(device)) / x_std.to(device)

    logits = model(X).squeeze(-1)  # [B,n]
    logits = logits.masked_fill(~valid.to(device), -1e9)

    loss = float(F.cross_entropy(logits, labels).item())

    pred = torch.argmax(logits, dim=1).to(torch.int64)  # [B]
    top1 = float((pred == labels).float().mean().item())

    k = min(5, int(logits.shape[1]))
    topk = torch.topk(logits, k=k, dim=1).indices
    top5 = float((topk == labels.unsqueeze(1)).any(dim=1).float().mean().item())

    y_cpu = y.detach().cpu()
    valid_cpu = valid.detach().cpu()
    logits_cpu = logits.detach().cpu()

    y_masked = y_cpu.clone()
    y_masked[~valid_cpu] = float("-inf")
    best = torch.max(y_masked, dim=1).values
    chosen = y_cpu.gather(dim=1, index=pred.cpu().unsqueeze(1)).squeeze(1)
    regret = float((best - chosen).mean().item())

    spearmans: List[float] = []
    for bi in range(int(y_cpu.shape[0])):
        mask = valid_cpu[bi]
        if int(mask.sum().item()) < 2:
            continue
        spearmans.append(spearman_corr(logits_cpu[bi][mask], y_cpu[bi][mask]))

    def _mean(xs: List[float]) -> float:
        xs2 = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs2)) if xs2 else float("nan")

    return {
        "loss": loss,
        "top1_acc": top1,
        "top5_acc": top5,
        "top1_regret_mean": regret,
        "spearman_mean": _mean(spearmans),
        "num_instances": float(instance_ids.numel()),
    }


def train_ce_probe(
    tensors: InstanceTensors,
    train_ids: torch.Tensor,
    device: torch.device,
    seed: int,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    standardize_x: bool,
) -> Tuple[nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]:
    set_seed(seed)

    X_inst = tensors.X
    x_mean = x_std = None
    if standardize_x:
        X_inst, x_mean, x_std = standardize_X(X_inst, train_ids=train_ids)

    B, n, d = X_inst.shape
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

            logits = model(xb).squeeze(-1)
            logits = logits.masked_fill(~vb, -1e9)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return model, x_mean, x_std


def write_outputs(out_dir: Path, payload: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as fp:
        json.dump(payload, fp, indent=2)

    rows = payload["rows"]
    with open(out_dir / "results.csv", "w") as fp:
        fp.write(
            "train_size,repeat,split,top1_acc,top5_acc,loss,spearman_mean,top1_regret_mean\n"
        )
        for row in rows:
            fp.write(
                ",".join(
                    [
                        str(row["train_size"]),
                        str(row["repeat"]),
                        str(row["split"]),
                        str(row["metrics"]["top1_acc"]),
                        str(row["metrics"]["top5_acc"]),
                        str(row["metrics"]["loss"]),
                        str(row["metrics"]["spearman_mean"]),
                        str(row["metrics"]["top1_regret_mean"]),
                    ]
                )
                + "\n"
            )


def plot_results(out_dir: Path, summary: List[Dict], test_size: int, dataset_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_sizes = [row["train_size"] for row in summary]
    top1 = [row["test"]["top1_acc"]["mean"] for row in summary]
    top1_std = [row["test"]["top1_acc"]["std"] for row in summary]
    top5 = [row["test"]["top5_acc"]["mean"] for row in summary]
    top5_std = [row["test"]["top5_acc"]["std"] for row in summary]
    spearman = [row["test"]["spearman_mean"]["mean"] for row in summary]
    spearman_std = [row["test"]["spearman_mean"]["std"] for row in summary]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(train_sizes, top1, yerr=top1_std, label="Top-1 (best node)", linewidth=2, capsize=3)
    ax.errorbar(train_sizes, top5, yerr=top5_std, label="Top-5 contains best", linewidth=2, capsize=3)
    ax.errorbar(train_sizes, spearman, yerr=spearman_std, label="Spearman (scores)", linewidth=2, capsize=3)
    ax.axhline(1.0 / 100.0, linestyle="--", color="gray", linewidth=1, label="Chance top-1 (1/100)")
    ax.axhline(5.0 / 100.0, linestyle=":", color="gray", linewidth=1, label="Chance top-5 (5/100)")

    ax.set_title(f"Overfitting check: probe vs train size ({dataset_name}, held-out test={test_size})")
    ax.set_xlabel("Train instances")
    ax.set_ylabel("Metric (higher is better)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "plot.png", dpi=200)
    plt.close(fig)


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

    tensors = load_instance_tensors(reps_path)
    device = as_device(args.device)

    B = int(tensors.X.shape[0])
    test_size = int(args.test_size)
    if test_size <= 0 or test_size >= B:
        raise ValueError(f"--test_size must be in [1, {B-1}], got {test_size}")

    g = torch.Generator().manual_seed(int(args.seed))
    perm = torch.randperm(B, generator=g)
    test_ids = perm[:test_size]
    train_pool = perm[test_size:]

    max_train_avail = int(train_pool.numel())
    requested_max = int(args.max_train)
    max_train = min(max_train_avail, requested_max)
    if max_train < int(args.train_step):
        raise ValueError(f"Not enough training instances after holdout: {max_train_avail}")

    train_sizes = list(range(int(args.train_step), max_train + 1, int(args.train_step)))
    if train_sizes[-1] != max_train:
        train_sizes.append(max_train)

    dataset_name = reps_path.parent.name
    run_dir = tensors.meta.get("run_dir")
    payload = {
        "config": {
            "reps_path": str(reps_path),
            "dataset_name": dataset_name,
            "run_dir": run_dir,
            "seed": int(args.seed),
            "device": str(device),
            "test_size": test_size,
            "train_step": int(args.train_step),
            "max_train_requested": requested_max,
            "max_train_used": max_train,
            "repeats": int(args.repeats),
            "batch_size": int(args.batch_size),
            "num_epochs": int(args.num_epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "standardize_x": bool(args.standardize_x),
        },
        "rows": [],
        "summary": [],
    }

    print(f"[sweep] reps: {reps_path}")
    print(f"[sweep] run_dir: {run_dir}")
    print(f"[sweep] instances: {B} (test={test_size}, train_pool={max_train_avail})")
    print(f"[sweep] train sizes: {train_sizes[0]}..{train_sizes[-1]} (step={args.train_step}), repeats={args.repeats}")
    print(f"[sweep] device: {device}")

    for train_size in train_sizes:
        test_metrics_all: List[Dict[str, float]] = []
        train_metrics_all: List[Dict[str, float]] = []

        for rep in range(int(args.repeats)):
            rep_seed = int(args.seed) * 1000 + rep * 17 + train_size
            gg = torch.Generator().manual_seed(rep_seed)
            pool_perm = train_pool[torch.randperm(int(train_pool.numel()), generator=gg)]
            train_ids = pool_perm[: int(train_size)]

            model, x_mean, x_std = train_ce_probe(
                tensors=tensors,
                train_ids=train_ids,
                device=device,
                seed=rep_seed,
                batch_size=int(args.batch_size),
                num_epochs=int(args.num_epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                standardize_x=bool(args.standardize_x),
            )

            train_m = eval_ce_probe(model, tensors, train_ids, device=device, x_mean=x_mean, x_std=x_std)
            test_m = eval_ce_probe(model, tensors, test_ids, device=device, x_mean=x_mean, x_std=x_std)

            payload["rows"].append({"train_size": int(train_size), "repeat": int(rep), "split": "train", "metrics": train_m})
            payload["rows"].append({"train_size": int(train_size), "repeat": int(rep), "split": "test", "metrics": test_m})
            train_metrics_all.append(train_m)
            test_metrics_all.append(test_m)

            print(
                f"[sweep] train_size={train_size:4d} rep={rep} "
                f"test top1={test_m['top1_acc']:.3f} top5={test_m['top5_acc']:.3f} spearman={test_m['spearman_mean']:.3f}"
            )

        def summarize(metrics: List[Dict[str, float]], key: str) -> Dict[str, float]:
            return _mean_std([float(m.get(key, float("nan"))) for m in metrics])

        summary_row = {
            "train_size": int(train_size),
            "test": {
                "top1_acc": summarize(test_metrics_all, "top1_acc"),
                "top5_acc": summarize(test_metrics_all, "top5_acc"),
                "spearman_mean": summarize(test_metrics_all, "spearman_mean"),
                "loss": summarize(test_metrics_all, "loss"),
                "top1_regret_mean": summarize(test_metrics_all, "top1_regret_mean"),
            },
            "train": {
                "top1_acc": summarize(train_metrics_all, "top1_acc"),
                "top5_acc": summarize(train_metrics_all, "top5_acc"),
                "spearman_mean": summarize(train_metrics_all, "spearman_mean"),
                "loss": summarize(train_metrics_all, "loss"),
                "top1_regret_mean": summarize(train_metrics_all, "top1_regret_mean"),
            },
        }
        payload["summary"].append(summary_row)

        write_outputs(out_dir, payload)
        plot_results(out_dir, payload["summary"], test_size=test_size, dataset_name=dataset_name)

    print(f"[sweep] wrote: {out_dir / 'results.json'}")
    print(f"[sweep] wrote: {out_dir / 'results.csv'}")
    print(f"[sweep] wrote: {out_dir / 'plot.png'}")


if __name__ == "__main__":
    main()
