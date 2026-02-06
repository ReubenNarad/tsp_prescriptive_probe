import argparse
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torchrl.data import Composite


STATUS_OK = 0
STATUS_TIMEOUT = 1
STATUS_CALLED_PROCESS_ERROR = 2
STATUS_NO_SOLUTION_FILE = 3
STATUS_BAD_TOUR = 4
STATUS_EXCEPTION = 5


CONCORDE_INTERMEDIATE_EXTS = [".pul", ".sav", ".res", ".mas", ".pix", ".sol"]

COORD_SCALE = 100.0
COORD_DECIMALS = 4
# Use a metric distance to guarantee triangle inequality; then removing a node cannot increase the optimal tour length.
# TSPLIB CEIL_2D uses ceil(EuclideanDistance) on the coordinates provided in NODE_COORD_SECTION.
EDGE_WEIGHT_TYPE = "CEIL_2D"


@dataclass(frozen=True)
class SolveResult:
    valid: bool
    status: int
    length: float
    time_wall: float
    time_reported: float
    bb_nodes: int
    lp_rows: int
    lp_cols: int
    lp_nonzeros: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def patch_env_specs(env) -> None:
    def _patch(spec):
        if isinstance(spec, Composite):
            if not hasattr(spec, "data_cls"):
                spec.data_cls = None
            if not hasattr(spec, "step_mdp_static"):
                spec.step_mdp_static = False
            for child in spec.values():
                if child is not None:
                    _patch(child)

    for spec_name in ["input_spec", "output_spec", "observation_spec", "reward_spec"]:
        spec = getattr(env, spec_name, None)
        if spec is not None:
            _patch(spec)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def shard_bounds(num_instances_total: int, num_shards: int, shard_idx: int) -> Tuple[int, int]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"shard_idx must be in [0, {num_shards}), got {shard_idx}")

    base = num_instances_total // num_shards
    rem = num_instances_total % num_shards
    start = shard_idx * base + min(shard_idx, rem)
    size = base + (1 if shard_idx < rem else 0)
    end = start + size
    return start, end


def write_tsplib_file(path: Path, coords: torch.Tensor) -> None:
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
            fp.write(f"{idx + 1} {float(x * COORD_SCALE):.{COORD_DECIMALS}f} {float(y * COORD_SCALE):.{COORD_DECIMALS}f}\n")
        fp.write("EOF\n")


def read_solution_file(path: Path) -> list[int]:
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


def normalize_tour_indices(route: list[int], n: int) -> Optional[list[int]]:
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


def compute_tour_length(coords: torch.Tensor, route0: list[int]) -> float:
    coords = coords.detach().cpu().to(torch.float64)
    route_t = torch.tensor(route0, dtype=torch.long)

    # Mirror exactly what we write into TSPLIB: scale and round to COORD_DECIMALS.
    factor = 10**COORD_DECIMALS
    coords_scaled = torch.round(coords * COORD_SCALE * factor) / factor

    route_coords = coords_scaled[route_t]
    dist = (route_coords - route_coords.roll(-1, dims=0)).pow(2).sum(-1).sqrt()

    if EDGE_WEIGHT_TYPE != "CEIL_2D":
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE for length computation: {EDGE_WEIGHT_TYPE}")

    # Avoid floating point jitter pushing exact integers over the boundary.
    dist_int = torch.ceil(dist - 1e-9)
    return float(dist_int.sum().item())


def _parse_int(pattern: str, text: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else -1


def _parse_float(pattern: str, text: str) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else -1.0


def parse_concorde_stats(stdout: str, stderr: str) -> Tuple[int, float, int, int, int]:
    combined = "\n".join([stdout or "", stderr or ""])
    bb_nodes = _parse_int(r"Number of bbnodes:\s*(\d+)", combined)
    time_reported = _parse_float(r"Total Running Time:\s*([0-9.]+)", combined)

    lp_rows = lp_cols = lp_nonzeros = -1
    m = re.search(r"Final LP has (\d+) rows, (\d+) columns, (\d+) nonzeros", combined)
    if m:
        lp_rows, lp_cols, lp_nonzeros = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return bb_nodes, time_reported, lp_rows, lp_cols, lp_nonzeros


def cleanup_concorde_files(scratch_dir: Path, base_stub: str, keep_solution: bool = False) -> None:
    for ext in CONCORDE_INTERMEDIATE_EXTS:
        for prefix in ("", "O"):
            path = scratch_dir / f"{prefix}{base_stub}{ext}"
            if path.exists():
                path.unlink()

    tsp = scratch_dir / f"{base_stub}.tsp"
    if tsp.exists():
        tsp.unlink()

    if not keep_solution:
        sol = scratch_dir / f"{base_stub}.sol"
        if sol.exists():
            sol.unlink()


def solve_with_concorde(
    coords: torch.Tensor,
    scratch_dir: Path,
    base_stub: str,
    timeout_sec: Optional[float],
) -> SolveResult:
    tsp_path = scratch_dir / f"{base_stub}.tsp"
    sol_path = scratch_dir / f"{base_stub}.sol"

    write_tsplib_file(tsp_path, coords)

    stdout = ""
    stderr = ""
    status = STATUS_OK
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            ["concorde", "-o", sol_path.name, tsp_path.name],
            cwd=str(scratch_dir),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec if timeout_sec and timeout_sec > 0 else None,
        )
        stdout, stderr = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        status = STATUS_TIMEOUT
        stdout = (e.stdout.decode() if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")) or ""
        stderr = (e.stderr.decode() if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")) or ""
    except subprocess.CalledProcessError as e:
        status = STATUS_CALLED_PROCESS_ERROR
        stdout = e.stdout or ""
        stderr = e.stderr or ""
    except Exception as e:
        status = STATUS_EXCEPTION
        stderr = f"{type(e).__name__}: {e}"
    time_wall = time.perf_counter() - start

    bb_nodes, time_reported, lp_rows, lp_cols, lp_nonzeros = parse_concorde_stats(stdout, stderr)

    length = -1.0
    valid = False
    if sol_path.exists():
        route = read_solution_file(sol_path)
        route0 = normalize_tour_indices(route, n=int(coords.shape[0]))
        if route0 is not None:
            length = compute_tour_length(coords, route0)
            valid = True
        else:
            status = STATUS_BAD_TOUR
    else:
        if status == STATUS_OK:
            status = STATUS_NO_SOLUTION_FILE

    cleanup_concorde_files(scratch_dir, base_stub)
    return SolveResult(
        valid=valid,
        status=status,
        length=length,
        time_wall=float(time_wall),
        time_reported=float(time_reported),
        bb_nodes=int(bb_nodes),
        lp_rows=int(lp_rows),
        lp_cols=int(lp_cols),
        lp_nonzeros=int(lp_nonzeros),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect node-removal what-if dataset using Concorde.")
    p.add_argument("--run_dir", type=str, required=True, help="Path to policy run dir containing env.pkl")
    p.add_argument("--num_instances", type=int, required=True, help="Total instances across all shards")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--out_path", type=str, required=True, help="Output .pt shard path")
    p.add_argument("--tmp_root", type=str, required=True, help="Root directory for concorde scratch dirs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concorde_timeout_sec", type=float, default=None)
    p.add_argument(
        "--log_every_sec",
        type=float,
        default=15.0,
        help="Print progress at most every N seconds (<=0 prints every solve).",
    )
    p.add_argument(
        "--max_removed_nodes",
        type=int,
        default=None,
        help="Optional cap on how many node removals to solve per instance (debug/smoke tests).",
    )
    p.add_argument("--assert_num_loc", type=int, default=None, help="Optional sanity check on num_loc")
    p.add_argument("--overwrite", action="store_true", help="Overwrite out_path if it exists")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    tmp_root = Path(args.tmp_root).expanduser().resolve()

    if out_path.exists() and not args.overwrite:
        print(f"[collect] Output exists, skipping (use --overwrite): {out_path}")
        return

    if not (run_dir / "env.pkl").exists():
        raise FileNotFoundError(f"env.pkl not found under run_dir: {run_dir}")

    tmp_root.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

    with open(run_dir / "env.pkl", "rb") as fp:
        env = pickle.load(fp)
    patch_env_specs(env)

    start_idx, end_idx = shard_bounds(args.num_instances, args.num_shards, args.shard_idx)
    shard_size = end_idx - start_idx

    shard_seed = int(args.seed) + int(args.shard_idx)
    set_seed(shard_seed)

    if shard_size == 0:
        payload = {
            "meta": {
                "run_dir": str(run_dir),
                "seed": int(args.seed),
                "shard_seed": int(shard_seed),
                "num_instances_total": int(args.num_instances),
                "num_shards": int(args.num_shards),
                "shard_idx": int(args.shard_idx),
                "global_instance_start": int(start_idx),
                "global_instance_end": int(end_idx),
                "num_loc": 0,
                "concorde_timeout_sec": float(args.concorde_timeout_sec) if args.concorde_timeout_sec else None,
                "created_at_unix": float(time.time()),
            }
        }
        torch.save(payload, out_path)
        print(f"[collect] Wrote empty shard: {out_path} (no instances)")
        return

    td = env.reset(batch_size=[shard_size]).to("cpu")
    locs = td["locs"].detach().cpu().to(torch.float32)
    if locs.ndim != 3 or locs.shape[2] != 2:
        raise ValueError(f"Expected locs [B,n,2], got {tuple(locs.shape)}")

    num_loc = int(locs.shape[1])
    if args.assert_num_loc is not None and num_loc != int(args.assert_num_loc):
        raise ValueError(f"assert_num_loc failed: expected {args.assert_num_loc}, got {num_loc}")

    B, n, _ = locs.shape

    base_length = torch.full((B,), float("nan"), dtype=torch.float32)
    base_time_wall = torch.full((B,), -1.0, dtype=torch.float32)
    base_time_reported = torch.full((B,), -1.0, dtype=torch.float32)
    base_bb_nodes = torch.full((B,), -1, dtype=torch.int64)
    base_lp_rows = torch.full((B,), -1, dtype=torch.int64)
    base_lp_cols = torch.full((B,), -1, dtype=torch.int64)
    base_lp_nonzeros = torch.full((B,), -1, dtype=torch.int64)
    base_status = torch.full((B,), -1, dtype=torch.int16)
    valid_base = torch.zeros((B,), dtype=torch.bool)

    minus_length = torch.full((B, n), float("nan"), dtype=torch.float32)
    minus_time_wall = torch.full((B, n), -1.0, dtype=torch.float32)
    minus_time_reported = torch.full((B, n), -1.0, dtype=torch.float32)
    minus_bb_nodes = torch.full((B, n), -1, dtype=torch.int64)
    minus_lp_rows = torch.full((B, n), -1, dtype=torch.int64)
    minus_lp_cols = torch.full((B, n), -1, dtype=torch.int64)
    minus_lp_nonzeros = torch.full((B, n), -1, dtype=torch.int64)
    minus_status = torch.full((B, n), -1, dtype=torch.int16)
    valid_minus = torch.zeros((B, n), dtype=torch.bool)

    scratch_dir = Path(
        tempfile.mkdtemp(
            dir=str(tmp_root),
            prefix=f"whatif_shard{args.shard_idx:04d}_",
        )
    )

    max_removed_nodes = None
    if args.max_removed_nodes is not None:
        if args.max_removed_nodes <= 0:
            raise ValueError(f"--max_removed_nodes must be >= 1, got {args.max_removed_nodes}")
        max_removed_nodes = min(int(args.max_removed_nodes), n)

    shard_start_t = time.perf_counter()
    last_log_t = shard_start_t
    solves_done = 0
    solves_per_instance = 1 + (n if max_removed_nodes is None else max_removed_nodes)
    total_solves = int(B) * int(solves_per_instance)

    def _format_eta(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "?"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds/60:.1f}m"
        return f"{seconds/3600:.2f}h"

    def _maybe_log(j: int, global_id: int, phase: str, node_i: Optional[int]) -> None:
        nonlocal last_log_t
        now = time.perf_counter()
        if args.log_every_sec is not None and args.log_every_sec <= 0:
            last_log_t = now
        else:
            if args.log_every_sec is not None and args.log_every_sec > 0 and (now - last_log_t) < args.log_every_sec:
                return
            last_log_t = now

        elapsed = now - shard_start_t
        rate = (solves_done / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, total_solves - solves_done)
        eta = (remaining / rate) if rate > 0 else float("nan")

        node_str = ""
        if node_i is not None:
            node_str = f" node {node_i+1}/{n if max_removed_nodes is None else max_removed_nodes}"

        pct = (100.0 * solves_done / total_solves) if total_solves > 0 else 0.0
        print(
            f"[collect] shard {args.shard_idx}/{args.num_shards}"
            f" inst {j+1}/{B} (global {global_id})"
            f"{node_str}"
            f" {phase}"
            f" solves {solves_done}/{total_solves} ({pct:.1f}%)"
            f" elapsed {_format_eta(elapsed)} eta {_format_eta(eta)}",
            flush=True,
        )

    try:
        for j in range(B):
            global_id = start_idx + j
            coords = locs[j]

            _maybe_log(j=j, global_id=global_id, phase="base", node_i=None)
            base_stub = f"inst{global_id:06d}__base"
            res_base = solve_with_concorde(
                coords=coords,
                scratch_dir=scratch_dir,
                base_stub=base_stub,
                timeout_sec=args.concorde_timeout_sec,
            )
            solves_done += 1
            _maybe_log(j=j, global_id=global_id, phase="base", node_i=None)
            base_length[j] = res_base.length if res_base.valid else float("nan")
            base_time_wall[j] = res_base.time_wall
            base_time_reported[j] = res_base.time_reported
            base_bb_nodes[j] = res_base.bb_nodes
            base_lp_rows[j] = res_base.lp_rows
            base_lp_cols[j] = res_base.lp_cols
            base_lp_nonzeros[j] = res_base.lp_nonzeros
            base_status[j] = res_base.status
            valid_base[j] = res_base.valid

            if not res_base.valid:
                continue

            for i in range(n if max_removed_nodes is None else max_removed_nodes):
                coords_minus = torch.cat([coords[:i], coords[i + 1 :]], dim=0)
                minus_stub = f"inst{global_id:06d}__minus{i:03d}"
                _maybe_log(j=j, global_id=global_id, phase="minus", node_i=i)
                res_minus = solve_with_concorde(
                    coords=coords_minus,
                    scratch_dir=scratch_dir,
                    base_stub=minus_stub,
                    timeout_sec=args.concorde_timeout_sec,
                )
                solves_done += 1
                _maybe_log(j=j, global_id=global_id, phase="minus", node_i=i)
                minus_length[j, i] = res_minus.length if res_minus.valid else float("nan")
                minus_time_wall[j, i] = res_minus.time_wall
                minus_time_reported[j, i] = res_minus.time_reported
                minus_bb_nodes[j, i] = res_minus.bb_nodes
                minus_lp_rows[j, i] = res_minus.lp_rows
                minus_lp_cols[j, i] = res_minus.lp_cols
                minus_lp_nonzeros[j, i] = res_minus.lp_nonzeros
                minus_status[j, i] = res_minus.status
                valid_minus[j, i] = res_minus.valid
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    eps = 1e-6
    delta_length_pct = torch.full((B, n), float("nan"), dtype=torch.float32)
    delta_time_pct = torch.full((B, n), float("nan"), dtype=torch.float32)

    pair_valid = valid_base.unsqueeze(1) & valid_minus
    if pair_valid.any():
        base_len = base_length.unsqueeze(1).expand(-1, n).clamp_min(eps)
        base_time = base_time_wall.unsqueeze(1).expand(-1, n).clamp_min(eps)
        delta_length_pct[pair_valid] = 100.0 * (base_len[pair_valid] - minus_length[pair_valid]) / base_len[pair_valid]
        delta_time_pct[pair_valid] = 100.0 * (base_time[pair_valid] - minus_time_wall[pair_valid]) / base_time[pair_valid]

    payload = {
        "meta": {
            "run_dir": str(run_dir),
            "seed": int(args.seed),
            "shard_seed": int(shard_seed),
            "num_instances_total": int(args.num_instances),
            "num_shards": int(args.num_shards),
            "shard_idx": int(args.shard_idx),
            "global_instance_start": int(start_idx),
            "global_instance_end": int(end_idx),
            "num_loc": int(num_loc),
            "concorde_timeout_sec": float(args.concorde_timeout_sec) if args.concorde_timeout_sec else None,
            "max_removed_nodes": int(max_removed_nodes) if max_removed_nodes is not None else None,
            "edge_weight_type": EDGE_WEIGHT_TYPE,
            "coord_scale": float(COORD_SCALE),
            "coord_decimals": int(COORD_DECIMALS),
            "created_at_unix": float(time.time()),
        },
        "locs": locs,
        "base_length": base_length,
        "base_time_wall": base_time_wall,
        "base_time_reported": base_time_reported,
        "base_bb_nodes": base_bb_nodes,
        "base_lp_rows": base_lp_rows,
        "base_lp_cols": base_lp_cols,
        "base_lp_nonzeros": base_lp_nonzeros,
        "base_status": base_status,
        "valid_base": valid_base,
        "minus_length": minus_length,
        "minus_time_wall": minus_time_wall,
        "minus_time_reported": minus_time_reported,
        "minus_bb_nodes": minus_bb_nodes,
        "minus_lp_rows": minus_lp_rows,
        "minus_lp_cols": minus_lp_cols,
        "minus_lp_nonzeros": minus_lp_nonzeros,
        "minus_status": minus_status,
        "valid_minus": valid_minus,
        "delta_length_pct": delta_length_pct,
        "delta_time_pct": delta_time_pct,
    }

    torch.save(payload, out_path)
    print(f"[collect] Wrote shard: {out_path} (instances {start_idx}..{end_idx-1}, n={num_loc})")


if __name__ == "__main__":
    main()
