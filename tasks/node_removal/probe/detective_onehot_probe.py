#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detective analysis for surprising probe metrics (onehot extraction).")
    p.add_argument("--dataset_pt", type=str, required=True, help="Path to dataset.pt (merged what-if dataset)")
    p.add_argument("--reps_pt", type=str, required=True, help="Path to probe reps .pt (from extract_representations.py)")
    p.add_argument("--probe_ckpt", type=str, default=None, help="Optional trained probe .pt (from train_probes.py)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for plots + summary.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid_instances", type=int, default=16, help="Number of instances for montage (default: 16)")
    p.add_argument("--max_scatter_points", type=int, default=50000, help="Max points for scatter plots (subsample).")
    return p


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_py_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
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


def _reshape_rows(
    X: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    instance_id: torch.Tensor,
    node_id: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    instance_id = instance_id.to(torch.int64)
    node_id = node_id.to(torch.int64)
    B = int(instance_id.max().item()) + 1
    n = int(node_id.max().item()) + 1
    key = instance_id * n + node_id
    idx = torch.argsort(key)
    X_inst = X[idx].view(B, n, X.shape[1])
    y_inst = y[idx].view(B, n)
    valid_inst = valid[idx].view(B, n)
    return X_inst, y_inst, valid_inst


def _subsample_pairs(coords: np.ndarray, weights: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if coords.shape[0] <= max_points:
        return coords, weights
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(coords.shape[0], size=int(max_points), replace=False)
    return coords[idx], weights[idx]


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    dataset_pt = Path(args.dataset_pt).expanduser().resolve()
    reps_pt = Path(args.reps_pt).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds: Dict[str, Any] = torch.load(dataset_pt, weights_only=False)
    reps: Dict[str, Any] = torch.load(reps_pt, weights_only=False)

    locs = ds["locs"].detach().cpu().to(torch.float32)  # [B,n,2] lon/lat
    cost_matrix = ds["cost_matrix"].detach().cpu().to(torch.float32)  # [B,n,n]
    delta = ds["delta_length_pct"].detach().cpu().to(torch.float32)  # [B,n]
    B, n, _ = locs.shape

    # Cost heuristics
    eye = torch.eye(n, dtype=torch.bool).unsqueeze(0)
    cm_inf = cost_matrix.masked_fill(eye, float("inf"))
    min_out = cm_inf.min(dim=2).values  # [B,n] (symmetric => also min_in)

    best = delta.argmax(dim=1)  # [B]
    heur = {
        "min_out_top1_acc": float((min_out.argmax(dim=1) == best).float().mean().item()),
        "min_out_top5_hit": float((min_out.topk(5, dim=1).indices == best.unsqueeze(1)).any(dim=1).float().mean().item()),
        "min_out_top10_hit": float((min_out.topk(10, dim=1).indices == best.unsqueeze(1)).any(dim=1).float().mean().item()),
        "pooled_corr_min_out_delta": float(np.corrcoef(min_out.reshape(-1).numpy(), delta.reshape(-1).numpy())[0, 1]),
    }

    # Best-node geo stats
    cent = locs.mean(dim=1, keepdim=True)
    dist_cent = ((locs - cent) ** 2).sum(dim=2).sqrt()
    dist_best = dist_cent[torch.arange(B), best]
    dist_rank = (dist_cent.argsort(dim=1).argsort(dim=1).float() + 1) / float(n)
    dist_rank_best = dist_rank[torch.arange(B), best]
    heur.update(
        {
            "dist_to_centroid_mean_best": float(dist_best.mean().item()),
            "dist_to_centroid_mean_all": float(dist_cent.mean().item()),
            "dist_to_centroid_rank_mean_best": float(dist_rank_best.mean().item()),
        }
    )

    # Plot: best nodes scatter colored by delta
    best_coords = locs[torch.arange(B), best].numpy()
    best_vals = delta[torch.arange(B), best].numpy()
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(best_coords[:, 0], best_coords[:, 1], c=best_vals, s=10, cmap="viridis", alpha=0.8, linewidths=0)
    ax.set_title("Best node locations across instances (color = delta_length_pct)")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    fig.colorbar(sc, ax=ax, label="delta_length_pct (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "best_nodes_scatter.png", dpi=200)
    plt.close(fig)

    # Plot: delta vs min_out scatter (subsample)
    coords_flat = locs.reshape(-1, 2).numpy()
    delta_flat = delta.reshape(-1).numpy()
    min_out_flat = min_out.reshape(-1).numpy()
    _, delta_s = _subsample_pairs(coords_flat, delta_flat, max_points=int(args.max_scatter_points), seed=int(args.seed))
    _, min_out_s = _subsample_pairs(coords_flat, min_out_flat, max_points=int(args.max_scatter_points), seed=int(args.seed) + 1)
    # Independent subsamples are fine for distribution visualization; keep sizes equal.
    m = min(delta_s.shape[0], min_out_s.shape[0])
    delta_s = delta_s[:m]
    min_out_s = min_out_s[:m]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(min_out_s, delta_s, s=2, alpha=0.15)
    ax.set_title("Node delta vs nearest-neighbor cost proxy")
    ax.set_xlabel("min_out (min edge cost from node)")
    ax.set_ylabel("delta_length_pct (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_vs_min_out.png", dpi=200)
    plt.close(fig)

    # Montage: random instances with best + top-min_out marked
    g = np.random.default_rng(int(args.seed))
    grid_k = int(args.grid_instances)
    grid_k = max(1, grid_k)
    picks = g.choice(B, size=min(grid_k, B), replace=False)
    cols = int(round(math.sqrt(len(picks))))
    cols = max(1, cols)
    rows = int(math.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for j, bi in enumerate(picks.tolist()):
        ax = axes[j // cols][j % cols]
        ax.set_visible(True)
        xy = locs[bi].numpy()
        dv = delta[bi].numpy()
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=dv, s=18, cmap="viridis", linewidths=0.4, edgecolors="k")
        bidx = int(best[bi].item())
        midx = int(min_out[bi].argmax().item())
        ax.scatter([xy[bidx, 0]], [xy[bidx, 1]], s=80, facecolors="none", edgecolors="red", linewidths=2.0)
        ax.scatter([xy[midx, 0]], [xy[midx, 1]], s=60, facecolors="none", edgecolors="white", linewidths=2.0)
        ax.set_title(f"inst {bi} (best=red, max-min_out=white)")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.colorbar(sc, ax=axes, shrink=0.85, label="delta_length_pct (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_montage.png", dpi=200)
    plt.close(fig)

    summary: Dict[str, Any] = {
        "dataset_pt": str(dataset_pt),
        "reps_pt": str(reps_pt),
        "B": int(B),
        "n": int(n),
        "heuristics": heur,
        "probe": None,
    }

    # If a trained probe is provided, evaluate it and generate diagnostics.
    if args.probe_ckpt:
        train_mod = _load_py_module(_repo_root() / "core" / "probe" / "train_probes.py", "train_probes_mod")
        ckpt = torch.load(Path(args.probe_ckpt).expanduser().resolve(), weights_only=False)
        state = ckpt["model_state_dict"]
        model_type = str(ckpt["model"])
        X = reps["X_resid"].detach().cpu().to(torch.float32)
        y_rows = reps["y"][:, 0].detach().cpu().to(torch.float32)
        valid_rows = reps["valid"].detach().cpu().to(torch.bool)
        instance_id = reps["instance_id"].detach().cpu()
        node_id = reps["node_id"].detach().cpu()
        X_inst, y_inst, valid_inst = _reshape_rows(X, y_rows, valid_rows, instance_id, node_id)
        labels = torch.argmax(y_inst.masked_fill(~valid_inst, float("-inf")), dim=1)

        model = train_mod.build_probe_model(
            model_type=model_type,
            input_dim=int(X_inst.shape[-1]),
            output_dim=1,
            mlp_hidden_dim=int(ckpt.get("mlp_hidden_dim", 256)),
            mlp_layers=int(ckpt.get("mlp_layers", 1)),
            mlp_dropout=float(ckpt.get("mlp_dropout", 0.0)),
            tfm_dim=int(ckpt.get("tfm_dim", 256)),
            tfm_layers=int(ckpt.get("tfm_layers", 2)),
            tfm_heads=int(ckpt.get("tfm_heads", 8)),
            tfm_ff_mult=int(ckpt.get("tfm_ff_mult", 4)),
            tfm_dropout=float(ckpt.get("tfm_dropout", 0.0)),
        )
        model.load_state_dict(state)
        model.eval()

        # Standardization used during training (optional)
        x_mean = ckpt.get("x_mean")
        x_std = ckpt.get("x_std")
        if x_mean is not None and x_std is not None:
            x_mean_t = torch.tensor(x_mean, dtype=torch.float32)
            x_std_t = torch.tensor(x_std, dtype=torch.float32).clamp_min(1e-6)
            X_inst = (X_inst - x_mean_t) / x_std_t

        with torch.inference_mode():
            logits = model(X_inst, key_padding_mask=(~valid_inst)).squeeze(-1) if model_type == "transformer" else model(X_inst).squeeze(-1)
            logits = logits.masked_fill(~valid_inst, -1e9)
            pred = torch.argmax(logits, dim=1)
            top1_acc = float((pred == labels).float().mean().item())
            # Spearman mean (diagnostic; CE doesn't train for full ranking)
            spearmans = []
            for i in range(B):
                v = valid_inst[i]
                if int(v.sum().item()) < 2:
                    continue
                spearmans.append(_spearman_corr(logits[i][v], y_inst[i][v]))
            spearman_mean = float(np.nanmean(np.array(spearmans, dtype=np.float64))) if spearmans else float("nan")
            # Cross-entropy
            ce = float(F.cross_entropy(logits, labels).item())

        # Diagnostics: does the probe rely on feature-dim alignment?
        # Randomly permute feature dimensions independently per instance (should break if alignment matters).
        g2 = torch.Generator().manual_seed(int(args.seed))
        perm = torch.stack([torch.randperm(X_inst.shape[-1], generator=g2) for _ in range(B)], dim=0)
        X_perm = torch.empty_like(X_inst)
        for i in range(B):
            X_perm[i] = X_inst[i, :, perm[i]]
        with torch.inference_mode():
            logits_perm = (
                model(X_perm, key_padding_mask=(~valid_inst)).squeeze(-1)
                if model_type == "transformer"
                else model(X_perm).squeeze(-1)
            )
            logits_perm = logits_perm.masked_fill(~valid_inst, -1e9)
            pred_perm = torch.argmax(logits_perm, dim=1)
            top1_perm = float((pred_perm == labels).float().mean().item())

        # Spearman histogram
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist([s for s in spearmans if not (isinstance(s, float) and math.isnan(s))], bins=40)
        ax.set_title("Per-instance Spearman(logits, delta_length_pct)")
        ax.set_xlabel("Spearman")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "spearman_hist.png", dpi=200)
        plt.close(fig)

        summary["probe"] = {
            "probe_ckpt": str(Path(args.probe_ckpt).expanduser().resolve()),
            "model_type": model_type,
            "top1_acc": top1_acc,
            "cross_entropy": ce,
            "spearman_mean": spearman_mean,
            "top1_acc_after_feature_dim_permutation": top1_perm,
            "spearman_count": int(len(spearmans)),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[detective] wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
