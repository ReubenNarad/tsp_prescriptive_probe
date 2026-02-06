#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import torch


EDGE_WEIGHT_TYPE = "CEIL_2D"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute oracle splice contributions for node-removal.\n"
            "For each instance, solve the optimal tour with Concorde, then score each node by\n"
            "the immediate splice improvement from removing it (no re-optimization).\n"
            "Writes oracle_splice_contrib.pt with key oracle_splice_contrib_pct."
        )
    )
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset.pt")
    p.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output path (default: <data_dir>/oracle_splice_contrib.pt)",
    )
    p.add_argument(
        "--concorde_timeout_sec",
        type=float,
        default=60.0,
        help="Timeout per Concorde solve (seconds).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite out_path if it exists.",
    )
    return p


def _load_dataset(data_dir: Path) -> Dict:
    ds_path = data_dir / "dataset.pt"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {ds_path}")
    return torch.load(ds_path, weights_only=False, map_location="cpu")


def _write_tsplib_file(path: Path, coords: torch.Tensor, coord_scale: float, coord_decimals: int) -> None:
    coords = coords.detach().cpu()
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be [n,2], got {tuple(coords.shape)}")

    with open(path, "w") as fp:
        fp.write("NAME: td_tsp\n")
        fp.write("TYPE: TSP\n")
        fp.write(f"DIMENSION: {coords.shape[0]}\n")
        fp.write(f"EDGE_WEIGHT_TYPE: {EDGE_WEIGHT_TYPE}\n")
        fp.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(coords.tolist()):
            fp.write(f"{idx + 1} {float(x * coord_scale):.{coord_decimals}f} {float(y * coord_scale):.{coord_decimals}f}\n")
        fp.write("EOF\n")


def _read_solution_file(path: Path) -> list[int]:
    with open(path, "r") as fp:
        lines = fp.readlines()

    tour: list[int] = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        for tok in line.strip().split():
            val = int(tok)
            if val == -1:
                return tour
            tour.append(val)
    return tour


def _normalize_tour_indices(route: list[int], n: int) -> Optional[list[int]]:
    if len(route) != n:
        return None

    mn, mx = min(route), max(route)
    if mn == 0 and mx == n - 1:
        route0 = route
    elif mn == 1 and mx == n:
        route0 = [x - 1 for x in route]
    elif mn >= 0 and mx < n:
        route0 = route
    elif mn >= 1 and mx <= n:
        route0 = [x - 1 for x in route]
    else:
        return None

    if any((x < 0 or x >= n) for x in route0):
        return None
    if len(set(route0)) != n:
        return None
    return route0


def _dist_matrix_ceil(coords: torch.Tensor, coord_scale: float, coord_decimals: int) -> torch.Tensor:
    coords = coords.detach().cpu().to(torch.float64)
    factor = 10 ** int(coord_decimals)
    coords_scaled = torch.round(coords * float(coord_scale) * factor) / factor
    d = torch.cdist(coords_scaled, coords_scaled, p=2)
    if EDGE_WEIGHT_TYPE != "CEIL_2D":
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {EDGE_WEIGHT_TYPE}")
    return torch.ceil(d - 1e-9)


def main() -> None:
    args = build_arg_parser().parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else (data_dir / "oracle_splice_contrib.pt")

    if out_path.exists() and not args.overwrite:
        print(f"[oracle] Output exists, skipping (use --overwrite): {out_path}")
        return

    ds = _load_dataset(data_dir)
    locs = ds.get("locs")
    valid_base = ds.get("valid_base")
    meta = ds.get("meta", {})

    if not torch.is_tensor(locs) or locs.ndim != 3 or locs.shape[-1] != 2:
        raise ValueError("dataset.pt missing locs [B,n,2]")
    if not torch.is_tensor(valid_base) or valid_base.ndim != 1 or valid_base.shape[0] != locs.shape[0]:
        raise ValueError("dataset.pt missing valid_base [B]")

    coord_scale = float(meta.get("coord_scale", 100.0))
    coord_decimals = int(meta.get("coord_decimals", 4))

    B, n, _ = locs.shape
    out = torch.zeros((B, n), dtype=torch.float32)

    num_valid = int(valid_base.sum().item())
    print(f"[oracle] computing Concorde tours for {num_valid}/{B} instances (n={n})...")

    with tempfile.TemporaryDirectory(prefix="oracle_splice_") as tmp:
        scratch = Path(tmp)
        for i in range(B):
            if not bool(valid_base[i].item()):
                continue
            tsp_path = scratch / f"inst_{i}.tsp"
            sol_path = scratch / f"inst_{i}.sol"
            _write_tsplib_file(tsp_path, locs[i], coord_scale, coord_decimals)
            try:
                subprocess.run(
                    ["concorde", "-o", sol_path.name, tsp_path.name],
                    cwd=str(scratch),
                    check=True,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=args.concorde_timeout_sec if args.concorde_timeout_sec and args.concorde_timeout_sec > 0 else None,
                )
            except Exception:
                if tsp_path.exists():
                    tsp_path.unlink()
                if sol_path.exists():
                    sol_path.unlink()
                continue

            route = _read_solution_file(sol_path)
            route0 = _normalize_tour_indices(route, n=int(n))
            if tsp_path.exists():
                tsp_path.unlink()
            if sol_path.exists():
                sol_path.unlink()
            if route0 is None:
                continue

            dmat = _dist_matrix_ceil(locs[i], coord_scale, coord_decimals)
            route_t = torch.tensor(route0, dtype=torch.long)
            base_len = float(dmat[route_t, route_t.roll(-1)].sum().item())
            if base_len <= 0:
                continue

            pred = torch.empty((n,), dtype=torch.long)
            succ = torch.empty((n,), dtype=torch.long)
            for pos, node in enumerate(route0):
                pred[node] = route0[pos - 1]
                succ[node] = route0[(pos + 1) % n]

            prev = pred
            nxt = succ
            improv = dmat[prev, torch.arange(n)] + dmat[torch.arange(n), nxt] - dmat[prev, nxt]
            out[i] = (improv / base_len) * 100.0

            if (i + 1) % 100 == 0:
                print(f"[oracle] solved {i+1}/{B}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "oracle_splice_contrib_pct": out,
            "coord_scale": coord_scale,
            "coord_decimals": coord_decimals,
            "edge_weight_type": EDGE_WEIGHT_TYPE,
            "timeout_sec": float(args.concorde_timeout_sec),
            "data_dir": str(data_dir),
        },
        out_path,
    )
    print(f"[oracle] wrote cache: {out_path}")


if __name__ == "__main__":
    main()
