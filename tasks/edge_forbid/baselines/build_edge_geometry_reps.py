#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import torch


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build geometry-only reps for edge-forbid task.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_path", type=str, default=None)
    return p


def _load_dataset(data_dir: Path) -> Dict:
    ds_path = data_dir / "dataset.pt"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {ds_path}")
    return torch.load(ds_path, weights_only=False)


def main() -> None:
    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    args = build_arg_parser().parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    ds = _load_dataset(data_dir)

    locs = ds.get("locs")
    edge_u = ds.get("edge_u")
    edge_v = ds.get("edge_v")
    valid_base = ds.get("valid_base")
    valid_forbid = ds.get("valid_forbid")
    delta_length_pct = ds.get("delta_length_pct")
    delta_time_pct = ds.get("delta_time_pct")

    if not torch.is_tensor(locs) or locs.ndim != 3 or locs.shape[-1] != 2:
        raise ValueError("dataset.pt missing 'locs' tensor [B,n,2]")
    if not torch.is_tensor(edge_u) or not torch.is_tensor(edge_v):
        raise ValueError("dataset.pt missing 'edge_u'/'edge_v' tensors [B,n]")
    if edge_u.shape != edge_v.shape:
        raise ValueError("edge_u and edge_v shapes mismatch")
    if not torch.is_tensor(valid_base) or valid_base.shape != (edge_u.shape[0],):
        raise ValueError("dataset.pt missing 'valid_base' tensor [B]")
    if not torch.is_tensor(valid_forbid) or valid_forbid.shape != edge_u.shape:
        raise ValueError("dataset.pt missing 'valid_forbid' tensor [B,n]")

    B, n = edge_u.shape
    batch_idx = torch.arange(B, dtype=torch.int64).unsqueeze(1).expand(B, n)

    u = edge_u.long()
    v = edge_v.long()
    u_xy = locs[batch_idx, u]
    v_xy = locs[batch_idx, v]
    dx = v_xy[..., 0] - u_xy[..., 0]
    dy = v_xy[..., 1] - u_xy[..., 1]
    dist = torch.sqrt(dx * dx + dy * dy)

    feats = torch.stack(
        [
            u_xy[..., 0],
            u_xy[..., 1],
            v_xy[..., 0],
            v_xy[..., 1],
            dx,
            dy,
            dist,
            dx.abs(),
            dy.abs(),
        ],
        dim=-1,
    )
    X = feats.reshape(-1, feats.shape[-1]).to(torch.float32)

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
            "feature_kind": "edge_geometry",
            "feature_names": [
                "x_u",
                "y_u",
                "x_v",
                "y_v",
                "dx",
                "dy",
                "dist",
                "abs_dx",
                "abs_dy",
            ],
            "data_dir": str(data_dir),
            "num_instances": int(B),
            "num_loc": int(n),
        },
    }

    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else (data_dir / "probe_reps_edge_geometry.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"[edge-geometry] wrote {out_path}")


if __name__ == "__main__":
    main()
