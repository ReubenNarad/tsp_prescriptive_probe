#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "2-opt repair baseline for edge-what-if: "
            "given the optimal tour, score each tour edge by the cheapest 2-opt move that removes it."
        )
    )
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing merged dataset.pt")
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <data_dir>/baseline_2opt)",
    )
    p.add_argument(
        "--label_key",
        type=str,
        default="delta_length_pct",
        help="Label key to evaluate against (default: delta_length_pct).",
    )
    p.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Optional cap on number of instances to process (debug).",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (only affects tie-breaking).")
    p.add_argument(
        "--save_scores",
        action="store_true",
        help="Save per-instance per-edge scores as scores.pt in out_dir.",
    )
    return p


def coords_to_ceil2d_int_matrix(coords: np.ndarray, coord_scale: float, coord_decimals: int) -> np.ndarray:
    """Return an NxN int64 matrix matching TSPLIB CEIL_2D on scaled/rounded coords."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be [n,2], got {coords.shape}")

    factor = 10**int(coord_decimals)
    coords_scaled = np.round(coords * float(coord_scale) * factor) / factor
    diff = coords_scaled[:, None, :] - coords_scaled[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    dist_int = np.ceil(dist - 1e-9).astype(np.int64)
    np.fill_diagonal(dist_int, 0)
    return dist_int


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n < 2:
        return float("nan")

    # Rank data (ties broken arbitrarily, consistent with argsort argsort).
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    xr = x_rank.astype(np.float64)
    yr = y_rank.astype(np.float64)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = math.sqrt(float(np.sum(xr * xr) * np.sum(yr * yr)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(xr * yr) / denom)


def compute_2opt_min_delta(D: np.ndarray, tour: np.ndarray) -> np.ndarray:
    """For each tour edge e=(v_i,v_{i+1}), return min length delta among valid 2-opt moves that remove it."""
    tour = np.asarray(tour, dtype=np.int64)
    n = int(tour.size)
    if n < 4:
        raise ValueError("2-opt requires n>=4")

    A = tour
    B = np.roll(tour, -1)

    dab = D[A, B].astype(np.int64)  # [n]
    base_sub = dab[:, None] + dab[None, :]  # [n,n]

    # For edge i=(A[i],B[i]) and edge j=(A[j],B[j]):
    # Δ1 = d(Ai,Aj)+d(Bi,Bj) - d(Ai,Bi) - d(Aj,Bj)
    # Δ2 = d(Ai,Bj)+d(Bi,Aj) - d(Ai,Bi) - d(Aj,Bj)
    D_AA = D[np.ix_(A, A)]
    D_BB = D[np.ix_(B, B)]
    D_AB = D[np.ix_(A, B)]
    D_BA = D[np.ix_(B, A)]

    delta1 = (D_AA + D_BB - base_sub).astype(np.float64)
    delta2 = (D_AB + D_BA - base_sub).astype(np.float64)
    delta = np.minimum(delta1, delta2)

    # Disallow adjacent/same edges (share a vertex).
    valid_pair = np.ones((n, n), dtype=bool)
    idx = np.arange(n)
    valid_pair[idx, idx] = False
    valid_pair[idx, (idx - 1) % n] = False
    valid_pair[idx, (idx + 1) % n] = False

    delta[~valid_pair] = np.inf
    min_delta = np.min(delta, axis=1)
    # Base tour is optimal, but there can be alternative optimal tours -> clamp negative noise to 0.
    min_delta = np.maximum(min_delta, 0.0)
    return min_delta.astype(np.float32)


def evaluate_instance(
    scores: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    seed: int,
) -> Optional[Tuple[int, int, float, float]]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if scores.shape != y.shape or scores.shape != valid.shape:
        raise ValueError("shape mismatch in evaluate_instance")

    mask = valid & np.isfinite(y) & np.isfinite(scores)
    if int(mask.sum()) == 0:
        return None

    # Mask invalid edges.
    y_masked = np.where(mask, y, -np.inf)
    s_masked = np.where(mask, scores, -np.inf)

    # Tie-breaking: add tiny deterministic noise based on seed.
    if int(mask.sum()) > 1:
        rng = np.random.default_rng(int(seed))
        noise = rng.standard_normal(size=s_masked.shape) * 1e-12
        s_masked = s_masked + noise

    true_idx = int(np.argmax(y_masked))
    pred_idx = int(np.argmax(s_masked))

    top1 = int(pred_idx == true_idx)

    k = min(5, int(mask.sum()))
    topk_idx = np.argsort(s_masked)[-k:]
    top5 = int(true_idx in set(int(i) for i in topk_idx.tolist()))

    best = float(np.max(y_masked))
    chosen = float(y[pred_idx])
    regret = float(best - chosen)

    sp = spearman_corr(scores[mask], y[mask])
    return top1, top5, regret, sp


def main() -> None:
    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    args = build_arg_parser().parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    ds_path = data_dir / "dataset.pt"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {ds_path}")
    ds: Dict = torch.load(ds_path, weights_only=False)

    locs = ds.get("locs")
    base_tour = ds.get("base_tour")
    valid_base = ds.get("valid_base")
    valid_forbid = ds.get("valid_forbid")
    y_full = ds.get(str(args.label_key))

    if not torch.is_tensor(locs) or locs.ndim != 3 or locs.shape[2] != 2:
        raise ValueError("dataset.pt missing 'locs' tensor with shape [B,n,2]")
    if not torch.is_tensor(base_tour) or base_tour.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError("dataset.pt missing 'base_tour' tensor with shape [B,n]")
    if not torch.is_tensor(valid_base) or valid_base.shape != (locs.shape[0],):
        raise ValueError("dataset.pt missing 'valid_base' bool tensor [B]")
    if not torch.is_tensor(valid_forbid) or valid_forbid.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError("dataset.pt missing 'valid_forbid' bool tensor [B,n]")
    if not torch.is_tensor(y_full) or y_full.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError(f"dataset.pt missing '{args.label_key}' tensor with shape [B,n]")

    B, n, _ = locs.shape
    max_instances = int(args.max_instances) if args.max_instances is not None else None
    B_eval = min(B, max_instances) if max_instances is not None else B

    meta = ds.get("meta", {}) if isinstance(ds, dict) else {}
    coord_scale = float(meta.get("coord_scale", 100.0))
    coord_decimals = int(meta.get("coord_decimals", 4))

    scores_all = np.full((B_eval, n), np.nan, dtype=np.float32)

    top1_sum = 0
    top5_sum = 0
    regret_sum = 0.0
    spearmans = []
    num_instances = 0

    for i in range(B_eval):
        if not bool(valid_base[i].item()):
            continue

        coords = locs[i].detach().cpu().numpy()
        tour = base_tour[i].detach().cpu().numpy()

        if (tour < 0).any():
            continue
        if int(np.unique(tour).size) != int(n):
            continue

        D = coords_to_ceil2d_int_matrix(coords, coord_scale=coord_scale, coord_decimals=coord_decimals)
        s = compute_2opt_min_delta(D, tour)
        scores_all[i] = s

        y = y_full[i].detach().cpu().numpy()
        valid_edges = (valid_forbid[i].detach().cpu().numpy().astype(bool)) & True
        ev = evaluate_instance(scores=s, y=y, valid=valid_edges, seed=int(args.seed) + i)
        if ev is None:
            continue
        top1, top5, regret, sp = ev
        top1_sum += int(top1)
        top5_sum += int(top5)
        regret_sum += float(regret)
        if not (isinstance(sp, float) and math.isnan(sp)):
            spearmans.append(float(sp))
        num_instances += 1

    metrics = {
        "config": {
            "data_dir": str(data_dir),
            "label_key": str(args.label_key),
            "coord_scale": coord_scale,
            "coord_decimals": coord_decimals,
            "max_instances": max_instances,
            "seed": int(args.seed),
        },
        "metrics": {
            "num_instances": int(num_instances),
            "top1_acc": float(top1_sum / num_instances) if num_instances > 0 else float("nan"),
            "top5_acc": float(top5_sum / num_instances) if num_instances > 0 else float("nan"),
            "top1_regret_mean": float(regret_sum / num_instances) if num_instances > 0 else float("nan"),
            "spearman_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
        },
    }

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (data_dir / "baseline_2opt")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[2opt] wrote: {out_dir / 'metrics.json'}")

    if bool(args.save_scores):
        torch.save(
            {
                "scores": torch.from_numpy(scores_all),
                "meta": {
                    "description": "2-opt min-delta score per tour edge (higher = more harmful to forbid)",
                    "label_key": str(args.label_key),
                    "coord_scale": coord_scale,
                    "coord_decimals": coord_decimals,
                },
            },
            out_dir / "scores.pt",
        )
        print(f"[2opt] wrote: {out_dir / 'scores.pt'}")


if __name__ == "__main__":
    main()

