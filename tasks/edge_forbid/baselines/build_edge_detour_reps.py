#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build detour-only reps for edge-forbid task.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_path", type=str, default=None)
    return p


def _load_dataset(data_dir: Path) -> Dict:
    ds_path = data_dir / "dataset.pt"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {ds_path}")
    return torch.load(ds_path, weights_only=False)


def _load_edge_collect_module():
    module_path = Path(__file__).resolve().parents[1] / "collect" / "collect_dataset.py"
    spec = importlib.util.spec_from_file_location("edge_collect", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _coords_to_matrix(coords: torch.Tensor) -> np.ndarray:
    mod = _load_edge_collect_module()
    return mod.coords_to_ceil2d_int_matrix(coords)


def _edge_detour_scores(coords: torch.Tensor, tour: torch.Tensor) -> np.ndarray:
    D = _coords_to_matrix(coords)
    n = int(tour.numel())
    scores = np.full((n,), np.nan, dtype=np.float32)
    tour_idx = tour.detach().cpu().numpy().astype(np.int64)
    for e in range(n):
        u = int(tour_idx[e])
        v = int(tour_idx[(e + 1) % n])
        sums = D[u].astype(np.float64) + D[v].astype(np.float64)
        sums[u] = np.inf
        sums[v] = np.inf
        best = float(np.min(sums))
        scores[e] = float(best - D[u, v])
    return scores


def main() -> None:
    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    args = build_arg_parser().parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    ds = _load_dataset(data_dir)

    locs = ds.get("locs")
    base_tour = ds.get("base_tour")
    valid_base = ds.get("valid_base")
    valid_forbid = ds.get("valid_forbid")
    delta_length_pct = ds.get("delta_length_pct")
    delta_time_pct = ds.get("delta_time_pct")

    if not torch.is_tensor(locs) or locs.ndim != 3:
        raise ValueError("dataset.pt missing 'locs' tensor [B,n,2]")
    if not torch.is_tensor(base_tour) or base_tour.ndim != 2:
        raise ValueError("dataset.pt missing 'base_tour' tensor [B,n]")
    if not torch.is_tensor(valid_base) or valid_base.shape != (locs.shape[0],):
        raise ValueError("dataset.pt missing 'valid_base' tensor [B]")
    if not torch.is_tensor(valid_forbid) or valid_forbid.shape != base_tour.shape:
        raise ValueError("dataset.pt missing 'valid_forbid' tensor [B,n]")

    B, n, _ = locs.shape
    edge_detour = np.zeros((B, n), dtype=np.float32)
    for b in range(B):
        edge_detour[b] = _edge_detour_scores(locs[b], base_tour[b])

    X = torch.from_numpy(edge_detour).reshape(-1, 1).to(torch.float32)

    pair_valid = valid_base.unsqueeze(1) & valid_forbid
    y = torch.stack(
        [
            delta_length_pct.reshape(-1),
            delta_time_pct.reshape(-1),
        ],
        dim=1,
    ).to(torch.float32)

    inst_ids = torch.arange(B, dtype=torch.int64).repeat_interleave(n)
    edge_ids = torch.arange(n, dtype=torch.int64).repeat(B)

    out = {
        "X_resid": X,
        "y": y,
        "valid": pair_valid.reshape(-1).to(torch.bool),
        "instance_id": inst_ids,
        "node_id": edge_ids,
        "meta": {
            "feature_kind": "edge_detour",
            "data_dir": str(data_dir),
            "num_instances": int(B),
            "num_loc": int(n),
        },
    }

    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else (data_dir / "probe_reps_edge_detour.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"[edge-detour] wrote {out_path}")


if __name__ == "__main__":
    main()
