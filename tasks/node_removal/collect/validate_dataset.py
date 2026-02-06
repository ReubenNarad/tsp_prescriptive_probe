#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate invariants for a merged what-if dataset")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset.pt")
    p.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Allowed epsilon for monotonic checks (default: 0 for exact non-negativity).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    ds_path = data_dir / "dataset.pt"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {ds_path}")

    ds: Dict[str, Any] = torch.load(ds_path, weights_only=False)
    locs = ds.get("locs")
    if not torch.is_tensor(locs) or locs.ndim != 3 or locs.shape[2] != 2:
        raise ValueError("dataset.pt missing 'locs' tensor with shape [B,n,2]")
    B, n, _ = locs.shape

    valid_base = ds.get("valid_base")
    valid_minus = ds.get("valid_minus")
    base_length = ds.get("base_length")
    minus_length = ds.get("minus_length")
    delta_length_pct = ds.get("delta_length_pct")

    if not torch.is_tensor(valid_base) or valid_base.shape != (B,):
        raise ValueError("dataset.pt missing 'valid_base' bool tensor [B]")
    if not torch.is_tensor(valid_minus) or valid_minus.shape != (B, n):
        raise ValueError("dataset.pt missing 'valid_minus' bool tensor [B,n]")
    if not torch.is_tensor(base_length) or base_length.shape != (B,):
        raise ValueError("dataset.pt missing 'base_length' tensor [B]")
    if not torch.is_tensor(minus_length) or minus_length.shape != (B, n):
        raise ValueError("dataset.pt missing 'minus_length' tensor [B,n]")
    if not torch.is_tensor(delta_length_pct) or delta_length_pct.shape != (B, n):
        raise ValueError("dataset.pt missing 'delta_length_pct' tensor [B,n]")

    pair_valid = valid_base.unsqueeze(1) & valid_minus
    if not pair_valid.any():
        raise ValueError("No valid base/minus pairs to validate")

    eps = float(args.eps)

    base_rep = base_length.unsqueeze(1).expand(B, n)
    delta_len = (base_rep - minus_length)[pair_valid]
    if (delta_len < -eps).any():
        min_val = float(delta_len.min().item())
        idx = (base_rep - minus_length).masked_fill(~pair_valid, float("inf")).argmin()
        j = int((idx // n).item())
        i = int((idx % n).item())
        raise AssertionError(
            "Length monotonicity violated: found base_length < minus_length\n"
            f"  min(base-minus)={min_val}\n"
            f"  example: instance={j} node={i} base={float(base_length[j].item())} minus={float(minus_length[j, i].item())}"
        )

    delta_pct = delta_length_pct[pair_valid]
    if (delta_pct < -eps).any():
        min_pct = float(delta_pct.min().item())
        raise AssertionError(f"delta_length_pct has negatives: min={min_pct}")

    print("[validate] OK")
    print(f"[validate] instances: {B}, num_loc: {n}, valid_pairs: {int(pair_valid.sum().item())}")
    print(f"[validate] min(base-minus): {float(delta_len.min().item())}")
    print(f"[validate] min(delta_length_pct): {float(delta_pct.min().item())}")


if __name__ == "__main__":
    main()

