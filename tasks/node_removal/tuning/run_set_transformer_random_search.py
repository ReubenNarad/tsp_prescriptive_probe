#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Random search over set-transformer probe hyperparameters (best_node_ce).")
    p.add_argument(
        "--reps_path",
        type=str,
        default="tasks/node_removal/tmp/tuning/probe_reps_expdecay_layers0-4_plus_output.pt",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="tasks/node_removal/tmp/tuning/st_random_search_layers0-4_plus_output_seed0",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0, help="Seed for fixed split + search RNG.")
    p.add_argument("--num_trials", type=int, default=30)

    p.add_argument("--train_size", type=int, default=2500)
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--standardize_x", action="store_true")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=50)

    p.add_argument("--model_dims", type=str, default="128,256")
    p.add_argument("--layers", type=str, default="1,2")
    p.add_argument("--heads", type=str, default="4,8")
    p.add_argument("--dropouts", type=str, default="0.0,0.05,0.1,0.2")
    p.add_argument("--ff_mults", type=str, default="2.0,4.0")

    p.add_argument("--lr_min", type=float, default=3e-4)
    p.add_argument("--lr_max", type=float, default=3e-3)
    p.add_argument("--wd_values", type=str, default="0,1e-5,1e-4,1e-3")
    return p


def as_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_ints(csv: str) -> List[int]:
    out = []
    for part in csv.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_floats(csv: str) -> List[float]:
    out = []
    for part in csv.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    if lo <= 0 or hi <= 0 or hi < lo:
        raise ValueError("log_uniform expects 0 < lo <= hi")
    x = rng.random()
    return float(math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))))


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


def load_instance_tensors(reps_path: Path) -> InstanceTensors:
    import os

    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    reps = torch.load(reps_path, weights_only=False)
    X = reps["X_resid"].to(torch.float32)
    y = reps["y"][:, 0].to(torch.float32)  # delta_length_pct
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
    return InstanceTensors(X=X_inst, y=y_inst, valid=valid_inst, labels=labels)


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
    spearman = float(np.mean([x for x in spearmans if not (isinstance(x, float) and math.isnan(x))])) if spearmans else float("nan")

    return {
        "loss": loss,
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


def main() -> None:
    args = build_arg_parser().parse_args()
    reps_path = Path(args.reps_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = as_device(args.device)
    tensors = load_instance_tensors(reps_path)
    B = int(tensors.X.shape[0])

    seed = int(args.seed)
    rng = random.Random(seed)
    set_seed(seed)

    train_size = int(args.train_size)
    val_size = int(args.val_size)
    test_size = int(args.test_size)
    if train_size + val_size + test_size > B:
        raise ValueError("Split sizes exceed available instances")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(B, generator=g)
    test_ids = perm[:test_size]
    val_ids = perm[test_size : test_size + val_size]
    train_pool = perm[test_size + val_size :]
    train_ids = train_pool[:train_size]

    model_dims = parse_ints(args.model_dims)
    layers_list = parse_ints(args.layers)
    heads_list = parse_ints(args.heads)
    dropouts = parse_floats(args.dropouts)
    ff_mults = parse_floats(args.ff_mults)
    wd_values = parse_floats(args.wd_values)

    config = {
        "reps_path": str(reps_path),
        "seed": seed,
        "device": str(device),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "standardize_x": bool(args.standardize_x),
        "batch_size": int(args.batch_size),
        "num_epochs": int(args.num_epochs),
        "model_dims": model_dims,
        "layers": layers_list,
        "heads": heads_list,
        "dropouts": dropouts,
        "ff_mults": ff_mults,
        "lr_min": float(args.lr_min),
        "lr_max": float(args.lr_max),
        "wd_values": wd_values,
        "num_trials": int(args.num_trials),
    }
    with open(out_dir / "config.json", "w") as fp:
        json.dump(config, fp, indent=2)

    rows: List[Dict] = []
    best = None

    for t in range(int(args.num_trials)):
        model_dim = rng.choice(model_dims)
        layers = rng.choice(layers_list)
        heads = rng.choice(heads_list)
        dropout = rng.choice(dropouts)
        ff_mult = rng.choice(ff_mults)
        ff_dim = max(1, int(round(ff_mult * model_dim)))
        lr = log_uniform(rng, float(args.lr_min), float(args.lr_max))
        wd = rng.choice(wd_values)

        if model_dim % heads != 0:
            continue

        run_seed = seed * 10000 + t * 31 + model_dim * 3 + layers * 7 + heads
        try:
            model, x_mean, x_std, best_val = train_one(
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
        except Exception as exc:
            rows.append(
                {
                    "trial": int(t),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "model_dim": int(model_dim),
                    "layers": int(layers),
                    "heads": int(heads),
                    "ff_dim": int(ff_dim),
                    "dropout": float(dropout),
                    "lr": float(lr),
                    "weight_decay": float(wd),
                }
            )
            continue

        train_m = eval_split(model, tensors, train_ids, device=device, x_mean=x_mean, x_std=x_std)
        val_m = eval_split(model, tensors, val_ids, device=device, x_mean=x_mean, x_std=x_std)
        test_m = eval_split(model, tensors, test_ids, device=device, x_mean=x_mean, x_std=x_std)

        rec = {
            "trial": int(t),
            "status": "ok",
            "best_val_loss": float(best_val),
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
        rows.append(rec)
        if best is None or rec["val"]["loss"] < best["val"]["loss"]:
            best = rec

        print(
            f"[rs] trial={t} d={model_dim} L={layers} h={heads} do={dropout} ff={ff_dim} "
            f"lr={lr:.2e} wd={wd:g} val top1={val_m['top1_acc']:.3f} test top1={test_m['top1_acc']:.3f}"
        )

    payload = {"config": config, "rows": rows, "best_by_val": best}
    with open(out_dir / "results.json", "w") as fp:
        json.dump(payload, fp, indent=2)

    ok = [r for r in rows if r.get("status") == "ok"]
    ok_sorted = sorted(ok, key=lambda r: float(r["test"]["top1_acc"]), reverse=True)
    with open(out_dir / "top_test.json", "w") as fp:
        json.dump(ok_sorted[:10], fp, indent=2)

    print(f"[rs] wrote: {out_dir / 'results.json'}")
    if best is not None:
        print(
            f"[rs] best_by_val: test top1={best['test']['top1_acc']:.3f} "
            f"d={best['model_dim']} L={best['layers']} h={best['heads']} do={best['dropout']} "
            f"ff={best['ff_dim']} lr={best['lr']:.2e} wd={best['weight_decay']:g}"
        )


if __name__ == "__main__":
    main()
