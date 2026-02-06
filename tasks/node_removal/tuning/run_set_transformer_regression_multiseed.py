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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train set-transformer to regress delta_length_pct per node; evaluate top-k argmax accuracy."
    )
    p.add_argument(
        "--reps_path",
        type=str,
        default="tasks/node_removal/tmp/tuning/probe_reps_expdecay_layers0-4_plus_output.pt",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="tasks/node_removal/tmp/tuning/st_regression_multiseed",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--repeats", type=int, default=1)

    p.add_argument("--train_size", type=int, default=2500)
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--standardize_x", action="store_true")
    p.add_argument("--standardize_y", action="store_true", help="Standardize y using train mean/std.")
    p.add_argument(
        "--use_extra_features",
        action="store_true",
        help="If reps contain X_extra, concatenate it to X_resid before training.",
    )

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--save_best_model_dir",
        type=str,
        default=None,
        help="Optional directory to save best model checkpoints per seed.",
    )

    p.add_argument("--model_dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
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
        if part:
            out.append(int(part))
    return out


@dataclass(frozen=True)
class InstanceTensors:
    X: torch.Tensor  # [B,n,d]
    y: torch.Tensor  # [B,n] (delta_length_pct)
    valid: torch.Tensor  # [B,n]
    labels: torch.Tensor  # [B] (argmax delta over valid)
    meta: Dict


def load_instance_tensors(reps_path: Path, *, use_extra_features: bool) -> InstanceTensors:
    import os

    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    reps = torch.load(reps_path, weights_only=False)
    if not isinstance(reps, dict):
        raise TypeError(f"Unexpected reps format at {reps_path}: {type(reps)}")

    X = reps["X_resid"].to(torch.float32)
    if use_extra_features and "X_extra" in reps:
        X_extra = reps["X_extra"].to(torch.float32)
        if X_extra.shape[0] != X.shape[0]:
            raise ValueError(f"X_extra shape mismatch: {tuple(X_extra.shape)} vs {tuple(X.shape)}")
        X = torch.cat([X, X_extra], dim=1)
    y = reps["y"][:, 0].to(torch.float32)
    valid = reps["valid"].to(torch.bool)
    instance_id = reps["instance_id"].to(torch.int64)
    node_id = reps["node_id"].to(torch.int64)

    B = int(instance_id.max().item()) + 1
    n = int(node_id.max().item()) + 1
    if int(X.shape[0]) != B * n:
        raise ValueError("Expected N=B*n")

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
    if use_extra_features:
        meta = dict(meta)
        meta["use_extra_features"] = True
    return InstanceTensors(X=X_inst, y=y_inst, valid=valid_inst, labels=labels, meta=meta)


def standardize_X(
    X_inst: torch.Tensor,
    train_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X_train = X_inst[train_ids].reshape(-1, X_inst.shape[-1])
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (X_inst - mean) / std, mean, std


def standardize_y(
    y_inst: torch.Tensor,
    valid_inst: torch.Tensor,
    train_ids: torch.Tensor,
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


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = pred - target
    err2 = err.pow(2)
    denom = mask.float().sum().clamp_min(1.0)
    return (err2 * mask.float()).sum() / denom


@torch.no_grad()
def eval_split(
    model: SetTransformerProbe,
    tensors: InstanceTensors,
    instance_ids: torch.Tensor,
    device: torch.device,
    x_mean: Optional[torch.Tensor],
    x_std: Optional[torch.Tensor],
    y_mean: Optional[float],
    y_std: Optional[float],
) -> Dict[str, float]:
    model.eval()
    X = tensors.X[instance_ids].to(device)
    valid = tensors.valid[instance_ids].to(device)
    labels = tensors.labels[instance_ids].to(device)
    y = tensors.y[instance_ids].to(device)

    if x_mean is not None and x_std is not None:
        X = (X - x_mean.to(device)) / x_std.to(device)

    y_for_loss = y
    if y_mean is not None and y_std is not None:
        y_for_loss = (y - float(y_mean)) / float(y_std)

    preds = score(model, X, valid)
    loss = float(masked_mse(preds, y_for_loss, valid).item())

    pred_best = torch.argmax(preds.masked_fill(~valid, float("-inf")), dim=1).to(torch.int64)
    top1 = float((pred_best == labels).float().mean().item())
    k = min(5, int(preds.shape[1]))
    topk = torch.topk(preds.masked_fill(~valid, float("-inf")), k=k, dim=1).indices
    top5 = float((topk == labels.unsqueeze(1)).any(dim=1).float().mean().item())

    preds_cpu = preds.detach().cpu()
    y_cpu = tensors.y[instance_ids].detach().cpu()
    valid_cpu = tensors.valid[instance_ids].detach().cpu()

    y_masked = y_cpu.clone()
    y_masked[~valid_cpu] = float("-inf")
    best_val = torch.max(y_masked, dim=1).values
    chosen = y_cpu.gather(dim=1, index=pred_best.cpu().unsqueeze(1)).squeeze(1)
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
        "mse": loss,
        "top1_acc": top1,
        "top5_acc": top5,
        "spearman_mean": spearman,
        "top1_regret_mean": regret,
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
    standardize_x_flag: bool,
    standardize_y_flag: bool,
    model_dim: int,
    layers: int,
    heads: int,
    ff_dim: int,
    dropout: float,
) -> Tuple[SetTransformerProbe, Optional[torch.Tensor], Optional[torch.Tensor], Optional[float], Optional[float], float]:
    set_seed(run_seed)

    X_inst = tensors.X
    x_mean = x_std = None
    if standardize_x_flag:
        X_inst, x_mean, x_std = standardize_X(X_inst, train_ids=train_ids)

    y_inst = tensors.y
    y_mean = y_std = None
    if standardize_y_flag:
        y_inst, y_mean, y_std = standardize_y(y_inst, tensors.valid, train_ids=train_ids)

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
            yb = y_inst[batch].to(device)
            preds = score(model, xb, vb)
            loss = masked_mse(preds, yb, vb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xb = X_inst[val_ids].to(device)
            vb = tensors.valid[val_ids].to(device)
            yb = y_inst[val_ids].to(device)
            preds = score(model, xb, vb)
            val_loss = float(masked_mse(preds, yb, vb).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, x_mean, x_std, y_mean, y_std, float(best_val)


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
    tensors = load_instance_tensors(reps_path, use_extra_features=bool(args.use_extra_features))
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
        best_snapshot = None
        for rep in range(repeats):
            run_seed = int(seed) * 1000 + rep * 31 + int(args.model_dim) * 7
            model, x_mean, x_std, y_mean, y_std, best_val = train_one(
                tensors=tensors,
                train_ids=train_ids,
                val_ids=val_ids,
                device=device,
                run_seed=run_seed,
                batch_size=int(args.batch_size),
                num_epochs=int(args.num_epochs),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                standardize_x_flag=bool(args.standardize_x),
                standardize_y_flag=bool(args.standardize_y),
                model_dim=int(args.model_dim),
                layers=int(args.layers),
                heads=int(args.heads),
                ff_dim=int(args.ff_dim),
                dropout=float(args.dropout),
            )

            train_m = eval_split(model, tensors, train_ids, device=device, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
            val_m = eval_split(model, tensors, val_ids, device=device, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
            test_m = eval_split(model, tensors, test_ids, device=device, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

            rec = {
                "seed": int(seed),
                "repeat": int(rep),
                "best_val_mse": float(best_val),
                "train": train_m,
                "val": val_m,
                "test": test_m,
            }
            all_runs.append(rec)
            if best is None or float(best_val) < float(best["best_val_mse"]):
                best = rec
                best_snapshot = {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "model_dim": int(args.model_dim),
                    "layers": int(args.layers),
                    "heads": int(args.heads),
                    "ff_dim": int(args.ff_dim),
                    "dropout": float(args.dropout),
                    "x_mean": x_mean.tolist() if x_mean is not None else None,
                    "x_std": x_std.tolist() if x_std is not None else None,
                    "y_mean": float(y_mean) if y_mean is not None else None,
                    "y_std": float(y_std) if y_std is not None else None,
                    "standardize_x": bool(args.standardize_x),
                    "standardize_y": bool(args.standardize_y),
                    "reps_path": str(reps_path),
                    "seed": int(seed),
                    "train_size": int(train_size),
                    "val_size": int(val_size),
                    "test_size": int(test_size),
                    "best_val_mse": float(best_val),
                    "train_metrics": train_m,
                    "val_metrics": val_m,
                    "test_metrics": test_m,
                }

            print(
                f"[reg] seed={seed} rep={rep} "
                f"val top1={val_m['top1_acc']:.3f} test top1={test_m['top1_acc']:.3f} top5={test_m['top5_acc']:.3f} mse={test_m['mse']:.3f}"
            )

        if best is not None:
            per_seed_best.append(best)
            if args.save_best_model_dir and best_snapshot is not None:
                out_dir_seed = Path(args.save_best_model_dir).expanduser().resolve()
                out_dir_seed.mkdir(parents=True, exist_ok=True)
                out_path = out_dir_seed / f"best_model_seed{int(seed)}.pt"
                torch.save(best_snapshot, out_path)
                print(f"[reg] wrote best model: {out_path}")

    agg = {
        "test_top1_acc": _mean_std([float(r["test"]["top1_acc"]) for r in per_seed_best]),
        "test_top5_acc": _mean_std([float(r["test"]["top5_acc"]) for r in per_seed_best]),
        "test_spearman_mean": _mean_std([float(r["test"]["spearman_mean"]) for r in per_seed_best]),
        "test_top1_regret_mean": _mean_std([float(r["test"]["top1_regret_mean"]) for r in per_seed_best]),
        "test_mse": _mean_std([float(r["test"]["mse"]) for r in per_seed_best]),
        "num_seeds": int(len(per_seed_best)),
    }

    payload = {"config": cfg, "per_seed_best": per_seed_best, "all_runs": all_runs, "aggregate": agg}
    with open(out_dir / "results.json", "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"[reg] aggregate test top1 mean={agg['test_top1_acc']['mean']:.3f} std={agg['test_top1_acc']['std']:.3f}")
    print(f"[reg] wrote: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
