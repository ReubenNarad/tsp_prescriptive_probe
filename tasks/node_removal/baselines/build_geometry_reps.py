#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict
import torch


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build geometry-only probe reps from dataset.pt (node removal).")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_path", type=str, default=None)
    p.add_argument("--use_scaled_coords", action="store_true", help="Apply coord_scale/coord_decimals from meta.")
    p.add_argument("--include_nn_dist", action="store_true", help="Append nearest-neighbor distance as a feature.")
    return p


def _load_dataset(data_dir: Path) -> Dict:
    ds_path = data_dir / "dataset.pt"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {ds_path}")
    return torch.load(ds_path, weights_only=False)


def _scale_coords(coords: torch.Tensor, meta: Dict) -> torch.Tensor:
    coord_scale = float(meta.get("coord_scale", 1.0))
    coord_decimals = int(meta.get("coord_decimals", 2))
    factor = 10 ** coord_decimals
    coords_scaled = torch.round(coords * coord_scale * factor) / factor
    return coords_scaled


def main() -> None:
    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    args = build_arg_parser().parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    ds = _load_dataset(data_dir)

    locs = ds.get("locs")
    if not torch.is_tensor(locs) or locs.ndim != 3 or locs.shape[2] != 2:
        raise ValueError("dataset.pt missing 'locs' tensor [B,n,2]")

    valid_base = ds.get("valid_base")
    valid_minus = ds.get("valid_minus")
    delta_length_pct = ds.get("delta_length_pct")
    delta_time_pct = ds.get("delta_time_pct")

    if not torch.is_tensor(valid_base) or valid_base.shape != (locs.shape[0],):
        raise ValueError("dataset.pt missing 'valid_base' tensor [B]")
    if not torch.is_tensor(valid_minus) or valid_minus.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError("dataset.pt missing 'valid_minus' tensor [B,n]")

    coords = locs.to(torch.float32)
    if args.use_scaled_coords:
        coords = _scale_coords(coords, ds.get("meta", {}))

    B, n, _ = coords.shape
    X = coords.reshape(-1, 2)

    if args.include_nn_dist:
        dists = torch.cdist(coords, coords, p=2)
        eye = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(B, n, n)
        dists = dists.masked_fill(eye, float("inf"))
        nn = dists.min(dim=-1).values
        X = torch.cat([X, nn.reshape(-1, 1)], dim=1)

    pair_valid = valid_base.unsqueeze(1) & valid_minus
    y = torch.stack(
        [
            delta_length_pct.reshape(-1),
            delta_time_pct.reshape(-1),
        ],
        dim=1,
    ).to(torch.float32)

    inst_ids = torch.arange(B, dtype=torch.int64).repeat_interleave(n)
    node_ids = torch.arange(n, dtype=torch.int64).repeat(B)

    out = {
        "X_resid": X.to(torch.float32),
        "y": y,
        "valid": pair_valid.reshape(-1).to(torch.bool),
        "instance_id": inst_ids,
        "node_id": node_ids,
        "meta": {
            "feature_kind": "coords_scaled" if args.use_scaled_coords else "coords_raw",
            "include_nn_dist": bool(args.include_nn_dist),
            "data_dir": str(data_dir),
            "num_instances": int(B),
            "num_loc": int(n),
        },
    }

    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else (data_dir / "probe_reps_geo.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"[geometry] wrote {out_path}")


if __name__ == "__main__":
    main()
