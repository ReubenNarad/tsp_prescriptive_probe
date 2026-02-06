import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize a merged what-if dataset")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset.pt")
    p.add_argument("--out_json", type=str, default=None, help="Optional path to write summary JSON")
    return p


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    dataset_path = data_dir / "dataset.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {dataset_path}")

    ds: Dict[str, Any] = torch.load(dataset_path, weights_only=False)

    locs = ds.get("locs")
    if not torch.is_tensor(locs) or locs.ndim != 3:
        raise ValueError("dataset.pt missing tensor key 'locs' with shape [B,n,2]")
    B, n, _ = locs.shape

    valid_base = ds.get("valid_base")
    valid_minus = ds.get("valid_minus")
    base_length = ds.get("base_length")
    base_time_wall = ds.get("base_time_wall")
    delta_length_pct = ds.get("delta_length_pct")
    delta_time_pct = ds.get("delta_time_pct")

    if not torch.is_tensor(valid_base) or valid_base.shape != (B,):
        raise ValueError("dataset.pt missing 'valid_base' bool tensor [B]")
    if not torch.is_tensor(valid_minus) or valid_minus.shape != (B, n):
        raise ValueError("dataset.pt missing 'valid_minus' bool tensor [B,n]")

    base_valid_frac = float(valid_base.float().mean().item()) if B > 0 else 0.0
    minus_valid_frac = float(valid_minus.float().mean().item()) if (B > 0 and n > 0) else 0.0

    pair_valid = valid_base.unsqueeze(1) & valid_minus

    def _mask_to_np(t: torch.Tensor, m: torch.Tensor) -> np.ndarray:
        return t[m].detach().cpu().numpy().astype(np.float64)

    summary: Dict[str, Any] = {
        "num_instances": int(B),
        "num_loc": int(n),
        "valid_base_frac": base_valid_frac,
        "valid_minus_frac": minus_valid_frac,
        "meta": ds.get("meta", {}),
    }

    if torch.is_tensor(base_length):
        summary["base_length"] = _stats(_mask_to_np(base_length, valid_base))
    if torch.is_tensor(base_time_wall):
        summary["base_time_wall"] = _stats(_mask_to_np(base_time_wall, valid_base))

    if torch.is_tensor(delta_length_pct):
        summary["delta_length_pct"] = _stats(_mask_to_np(delta_length_pct, pair_valid))
    if torch.is_tensor(delta_time_pct):
        summary["delta_time_pct"] = _stats(_mask_to_np(delta_time_pct, pair_valid))

    print("[summary] instances:", summary["num_instances"])
    print("[summary] num_loc:", summary["num_loc"])
    print("[summary] valid_base_frac:", f"{summary['valid_base_frac']:.3f}")
    print("[summary] valid_minus_frac:", f"{summary['valid_minus_frac']:.3f}")
    if "delta_length_pct" in summary:
        print("[summary] delta_length_pct p50:", _safe_float(summary["delta_length_pct"].get("p50")))
    if "delta_time_pct" in summary:
        print("[summary] delta_time_pct p50:", _safe_float(summary["delta_time_pct"].get("p50")))

    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (data_dir / "summary.json")
    with open(out_json, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[summary] wrote: {out_json}")


if __name__ == "__main__":
    main()

