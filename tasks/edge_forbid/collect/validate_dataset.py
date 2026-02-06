import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate invariants for an edge-what-if dataset")
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset.pt")
    p.add_argument("--max_print", type=int, default=20, help="Max violations to print before raising.")
    return p


def _is_perm_0n(tour: torch.Tensor) -> bool:
    n = int(tour.numel())
    if n <= 0:
        return False
    tour = tour.detach().cpu().to(torch.int64)
    if tour.min().item() != 0 or tour.max().item() != n - 1:
        return False
    return int(torch.unique(tour).numel()) == n


def _collect_violations(ds: Dict[str, Any], max_print: int) -> Tuple[int, List[str]]:
    locs = ds.get("locs")
    base_tour = ds.get("base_tour")
    valid_base = ds.get("valid_base")
    valid_forbid = ds.get("valid_forbid")
    base_length = ds.get("base_length")
    forbid_length = ds.get("forbid_length")
    delta_length_pct = ds.get("delta_length_pct")

    if not torch.is_tensor(locs) or locs.ndim != 3:
        raise ValueError("dataset.pt missing 'locs' tensor [B,n,2]")
    B, n, _ = locs.shape

    required = {
        "base_tour": base_tour,
        "valid_base": valid_base,
        "valid_forbid": valid_forbid,
        "base_length": base_length,
        "forbid_length": forbid_length,
        "delta_length_pct": delta_length_pct,
    }
    for k, v in required.items():
        if not torch.is_tensor(v):
            raise ValueError(f"dataset.pt missing required tensor '{k}'")

    if base_tour.shape != (B, n):
        raise ValueError(f"base_tour shape mismatch: expected {(B, n)}, got {tuple(base_tour.shape)}")
    if valid_base.shape != (B,):
        raise ValueError(f"valid_base shape mismatch: expected {(B,)}, got {tuple(valid_base.shape)}")
    if valid_forbid.shape != (B, n):
        raise ValueError(f"valid_forbid shape mismatch: expected {(B, n)}, got {tuple(valid_forbid.shape)}")
    if base_length.shape != (B,):
        raise ValueError(f"base_length shape mismatch: expected {(B,)}, got {tuple(base_length.shape)}")
    if forbid_length.shape != (B, n):
        raise ValueError(f"forbid_length shape mismatch: expected {(B, n)}, got {tuple(forbid_length.shape)}")
    if delta_length_pct.shape != (B, n):
        raise ValueError(f"delta_length_pct shape mismatch: expected {(B, n)}, got {tuple(delta_length_pct.shape)}")

    eps = 1e-6
    violations: List[str] = []
    count = 0

    vb = valid_base.to(torch.bool)
    vf = valid_forbid.to(torch.bool)
    pair_valid = vb.unsqueeze(1) & vf

    # Check tour permutations for valid base instances.
    for b in range(B):
        if not bool(vb[b].item()):
            continue
        if not _is_perm_0n(base_tour[b]):
            count += 1
            if len(violations) < max_print:
                violations.append(f"[tour] instance {b}: base_tour is not a permutation of 0..{n-1}")

    # Check monotonicity: forbid >= base.
    base_len = base_length.unsqueeze(1).expand(-1, n)
    forbid_len = forbid_length
    bad = pair_valid & (forbid_len + eps < base_len)
    if bad.any():
        bad_idx = torch.nonzero(bad, as_tuple=False)
        for k in range(min(int(bad_idx.shape[0]), max_print - len(violations))):
            b, e = int(bad_idx[k, 0].item()), int(bad_idx[k, 1].item())
            violations.append(
                f"[mono] instance {b} edge_idx {e}: forbid_length {float(forbid_len[b,e].item())} < base_length {float(base_len[b,e].item())}"
            )
        count += int(bad_idx.shape[0])

    # Check pct deltas nonnegative.
    bad2 = pair_valid & (delta_length_pct < -1e-4)
    if bad2.any():
        bad_idx = torch.nonzero(bad2, as_tuple=False)
        for k in range(min(int(bad_idx.shape[0]), max_print - len(violations))):
            b, e = int(bad_idx[k, 0].item()), int(bad_idx[k, 1].item())
            violations.append(
                f"[pct] instance {b} edge_idx {e}: delta_length_pct {float(delta_length_pct[b,e].item())} < 0"
            )
        count += int(bad_idx.shape[0])

    return count, violations


def main() -> None:
    args = build_arg_parser().parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    dataset_path = data_dir / "dataset.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {dataset_path}")

    ds: Dict[str, Any] = torch.load(dataset_path, weights_only=False)
    total, violations = _collect_violations(ds, max_print=int(args.max_print))

    meta = ds.get("meta", {})
    B = int(ds["locs"].shape[0]) if torch.is_tensor(ds.get("locs")) else 0
    n = int(ds["locs"].shape[1]) if torch.is_tensor(ds.get("locs")) and ds["locs"].ndim == 3 else 0
    valid_base_frac = float(ds["valid_base"].float().mean().item()) if B else 0.0
    valid_forbid_frac = float(ds["valid_forbid"].float().mean().item()) if (B and n) else 0.0

    print("[validate] data_dir:", str(data_dir))
    print("[validate] instances:", B)
    print("[validate] num_loc:", n)
    print("[validate] valid_base_frac:", f"{valid_base_frac:.3f}")
    print("[validate] valid_forbid_frac:", f"{valid_forbid_frac:.3f}")
    if isinstance(meta, dict) and meta:
        print("[validate] meta keys:", ", ".join(sorted(meta.keys())))

    if total > 0:
        for line in violations:
            print("[validate]", line)
        raise SystemExit(f"Found {total} violations")

    print("[validate] OK")


if __name__ == "__main__":
    main()

