import argparse
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
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

# Mirror TSPLIB CEIL_2D: we will compute integer distances by scaling + rounding coords, then ceil(euclid).
COORD_SCALE = 100.0
COORD_DECIMALS = 4


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
    tour0: Optional[list[int]]


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


def coords_to_ceil2d_int_matrix(coords: torch.Tensor) -> np.ndarray:
    """Return an NxN int64 matrix matching TSPLIB CEIL_2D on scaled/rounded coords."""
    coords = coords.detach().cpu().to(torch.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be [n,2], got {tuple(coords.shape)}")

    factor = 10**COORD_DECIMALS
    coords_scaled = torch.round(coords * COORD_SCALE * factor) / factor
    diff = coords_scaled[:, None, :] - coords_scaled[None, :, :]
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1))
    dist_int = torch.ceil(dist - 1e-9).to(torch.int64)
    dist_int.fill_diagonal_(0)
    return dist_int.numpy()


def write_tsplib_explicit_full_matrix(path: Path, D: np.ndarray) -> None:
    n = int(D.shape[0])
    if D.shape != (n, n):
        raise ValueError(f"D must be square, got {D.shape}")
    with open(path, "w") as fp:
        fp.write("NAME: td_tsp\n")
        fp.write("TYPE: TSP\n")
        fp.write(f"DIMENSION: {n}\n")
        fp.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        fp.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        fp.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            fp.write(" ".join(str(int(x)) for x in D[i].tolist()))
            fp.write("\n")
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


def compute_tour_length_from_matrix(D: np.ndarray, route0: list[int]) -> float:
    n = int(D.shape[0])
    if len(route0) != n:
        raise ValueError("route0 length mismatch")
    total = 0
    for i in range(n):
        a = int(route0[i])
        b = int(route0[(i + 1) % n])
        total += int(D[a, b])
    return float(total)


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


def solve_with_concorde_matrix(
    D: np.ndarray,
    scratch_dir: Path,
    base_stub: str,
    timeout_sec: Optional[float],
) -> SolveResult:
    tsp_path = scratch_dir / f"{base_stub}.tsp"
    sol_path = scratch_dir / f"{base_stub}.sol"

    write_tsplib_explicit_full_matrix(tsp_path, D)

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
    tour0: Optional[list[int]] = None
    if sol_path.exists():
        route = read_solution_file(sol_path)
        tour0 = normalize_tour_indices(route, n=int(D.shape[0]))
        if tour0 is not None:
            length = compute_tour_length_from_matrix(D, tour0)
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
        length=float(length),
        time_wall=float(time_wall),
        time_reported=float(time_reported),
        bb_nodes=int(bb_nodes),
        lp_rows=int(lp_rows),
        lp_cols=int(lp_cols),
        lp_nonzeros=int(lp_nonzeros),
        tour0=tour0,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect edge-what-if dataset using Concorde (forbid tour edges).")
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
        "--forbid_cost",
        type=int,
        default=10_000_000,
        help="Integer distance used to 'forbid' a single undirected edge (i,j) by inflating d(i,j)=d(j,i).",
    )
    p.add_argument(
        "--max_edges",
        type=int,
        default=None,
        help="Optional cap on how many tour edges to solve per instance (debug/smoke tests).",
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
                "forbid_cost": int(args.forbid_cost),
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

    base_tour = torch.full((B, n), -1, dtype=torch.int64)
    edge_u = torch.full((B, n), -1, dtype=torch.int64)
    edge_v = torch.full((B, n), -1, dtype=torch.int64)
    edge_length = torch.full((B, n), float("nan"), dtype=torch.float32)

    base_length = torch.full((B,), float("nan"), dtype=torch.float32)
    base_time_wall = torch.full((B,), -1.0, dtype=torch.float32)
    base_time_reported = torch.full((B,), -1.0, dtype=torch.float32)
    base_bb_nodes = torch.full((B,), -1, dtype=torch.int64)
    base_lp_rows = torch.full((B,), -1, dtype=torch.int64)
    base_lp_cols = torch.full((B,), -1, dtype=torch.int64)
    base_lp_nonzeros = torch.full((B,), -1, dtype=torch.int64)
    base_status = torch.full((B,), -1, dtype=torch.int16)
    valid_base = torch.zeros((B,), dtype=torch.bool)

    forbid_length = torch.full((B, n), float("nan"), dtype=torch.float32)
    forbid_time_wall = torch.full((B, n), -1.0, dtype=torch.float32)
    forbid_time_reported = torch.full((B, n), -1.0, dtype=torch.float32)
    forbid_bb_nodes = torch.full((B, n), -1, dtype=torch.int64)
    forbid_lp_rows = torch.full((B, n), -1, dtype=torch.int64)
    forbid_lp_cols = torch.full((B, n), -1, dtype=torch.int64)
    forbid_lp_nonzeros = torch.full((B, n), -1, dtype=torch.int64)
    forbid_status = torch.full((B, n), -1, dtype=torch.int16)
    valid_forbid = torch.zeros((B, n), dtype=torch.bool)

    scratch_dir = Path(
        tempfile.mkdtemp(
            dir=str(tmp_root),
            prefix=f"edgewhatif_shard{args.shard_idx:04d}_",
        )
    )

    max_edges = None
    if args.max_edges is not None:
        if args.max_edges <= 0:
            raise ValueError(f"--max_edges must be >= 1, got {args.max_edges}")
        max_edges = min(int(args.max_edges), n)

    shard_start_t = time.perf_counter()
    last_log_t = shard_start_t
    solves_done = 0
    solves_per_instance = 1 + (n if max_edges is None else max_edges)
    total_solves = int(B) * int(solves_per_instance)

    def _format_eta(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "?"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds/60:.1f}m"
        return f"{seconds/3600:.2f}h"

    def _maybe_log(j: int, global_id: int, phase: str, edge_i: Optional[int]) -> None:
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

        edge_str = ""
        if edge_i is not None:
            edge_str = f" edge {edge_i+1}/{n if max_edges is None else max_edges}"

        pct = (100.0 * solves_done / total_solves) if total_solves > 0 else 0.0
        print(
            f"[collect] shard {args.shard_idx}/{args.num_shards}"
            f" inst {j+1}/{B} (global {global_id})"
            f"{edge_str}"
            f" {phase}"
            f" solves {solves_done}/{total_solves} ({pct:.1f}%)"
            f" elapsed {_format_eta(elapsed)} eta {_format_eta(eta)}",
            flush=True,
        )

    try:
        for j in range(B):
            global_id = start_idx + j
            coords = locs[j]

            D = coords_to_ceil2d_int_matrix(coords)

            _maybe_log(j=j, global_id=global_id, phase="base", edge_i=None)
            base_stub = f"inst{global_id:06d}__base"
            res_base = solve_with_concorde_matrix(
                D=D,
                scratch_dir=scratch_dir,
                base_stub=base_stub,
                timeout_sec=args.concorde_timeout_sec,
            )
            solves_done += 1
            _maybe_log(j=j, global_id=global_id, phase="base", edge_i=None)

            base_length[j] = res_base.length if res_base.valid else float("nan")
            base_time_wall[j] = res_base.time_wall
            base_time_reported[j] = res_base.time_reported
            base_bb_nodes[j] = res_base.bb_nodes
            base_lp_rows[j] = res_base.lp_rows
            base_lp_cols[j] = res_base.lp_cols
            base_lp_nonzeros[j] = res_base.lp_nonzeros
            base_status[j] = res_base.status
            valid_base[j] = res_base.valid

            if not res_base.valid or res_base.tour0 is None:
                continue

            tour0 = res_base.tour0
            base_tour[j] = torch.tensor(tour0, dtype=torch.int64)

            # Tour edges are adjacent pairs along tour0.
            # Edge index e corresponds to (tour0[e], tour0[(e+1)%n]) and forbids the UNDIRECTED edge.
            edges = [(tour0[e], tour0[(e + 1) % n]) for e in range(n)]
            for e, (i, k) in enumerate(edges):
                edge_u[j, e] = int(i)
                edge_v[j, e] = int(k)
                edge_length[j, e] = float(D[int(i), int(k)])

            num_edges_to_solve = n if max_edges is None else max_edges
            forbid_cost = int(args.forbid_cost)
            if forbid_cost <= 0:
                raise ValueError(f"--forbid_cost must be >= 1, got {forbid_cost}")

            for e in range(num_edges_to_solve):
                i, k = int(edges[e][0]), int(edges[e][1])

                # Modify in-place to avoid copying O(n^2) per edge.
                old_ik = int(D[i, k])
                old_ki = int(D[k, i])
                D[i, k] = forbid_cost
                D[k, i] = forbid_cost

                forbid_stub = f"inst{global_id:06d}__forbid{e:03d}"
                _maybe_log(j=j, global_id=global_id, phase="forbid", edge_i=e)
                res_forbid = solve_with_concorde_matrix(
                    D=D,
                    scratch_dir=scratch_dir,
                    base_stub=forbid_stub,
                    timeout_sec=args.concorde_timeout_sec,
                )
                solves_done += 1
                _maybe_log(j=j, global_id=global_id, phase="forbid", edge_i=e)

                forbid_length[j, e] = res_forbid.length if res_forbid.valid else float("nan")
                forbid_time_wall[j, e] = res_forbid.time_wall
                forbid_time_reported[j, e] = res_forbid.time_reported
                forbid_bb_nodes[j, e] = res_forbid.bb_nodes
                forbid_lp_rows[j, e] = res_forbid.lp_rows
                forbid_lp_cols[j, e] = res_forbid.lp_cols
                forbid_lp_nonzeros[j, e] = res_forbid.lp_nonzeros
                forbid_status[j, e] = res_forbid.status
                valid_forbid[j, e] = res_forbid.valid

                # Restore
                D[i, k] = old_ik
                D[k, i] = old_ki
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    eps = 1e-6
    delta_length_pct = torch.full((B, n), float("nan"), dtype=torch.float32)
    delta_time_pct = torch.full((B, n), float("nan"), dtype=torch.float32)
    delta_length = torch.full((B, n), float("nan"), dtype=torch.float32)
    delta_time = torch.full((B, n), float("nan"), dtype=torch.float32)

    pair_valid = valid_base.unsqueeze(1) & valid_forbid
    if pair_valid.any():
        base_len = base_length.unsqueeze(1).expand(-1, n).clamp_min(eps)
        base_time = base_time_wall.unsqueeze(1).expand(-1, n).clamp_min(eps)
        delta_length[pair_valid] = forbid_length[pair_valid] - base_len[pair_valid]
        delta_time[pair_valid] = forbid_time_wall[pair_valid] - base_time[pair_valid]
        delta_length_pct[pair_valid] = 100.0 * delta_length[pair_valid] / base_len[pair_valid]
        delta_time_pct[pair_valid] = 100.0 * delta_time[pair_valid] / base_time[pair_valid]

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
            "forbid_cost": int(args.forbid_cost),
            "max_edges": int(max_edges) if max_edges is not None else None,
            "coord_scale": float(COORD_SCALE),
            "coord_decimals": int(COORD_DECIMALS),
            "created_at_unix": float(time.time()),
        },
        "locs": locs,
        "base_tour": base_tour,
        "edge_u": edge_u,
        "edge_v": edge_v,
        "edge_length": edge_length,
        "base_length": base_length,
        "base_time_wall": base_time_wall,
        "base_time_reported": base_time_reported,
        "base_bb_nodes": base_bb_nodes,
        "base_lp_rows": base_lp_rows,
        "base_lp_cols": base_lp_cols,
        "base_lp_nonzeros": base_lp_nonzeros,
        "base_status": base_status,
        "valid_base": valid_base,
        "forbid_length": forbid_length,
        "forbid_time_wall": forbid_time_wall,
        "forbid_time_reported": forbid_time_reported,
        "forbid_bb_nodes": forbid_bb_nodes,
        "forbid_lp_rows": forbid_lp_rows,
        "forbid_lp_cols": forbid_lp_cols,
        "forbid_lp_nonzeros": forbid_lp_nonzeros,
        "forbid_status": forbid_status,
        "valid_forbid": valid_forbid,
        "delta_length": delta_length,
        "delta_time": delta_time,
        "delta_length_pct": delta_length_pct,
        "delta_time_pct": delta_time_pct,
    }

    torch.save(payload, out_path)
    print(f"[collect] Wrote shard: {out_path} (instances {start_idx}..{end_idx-1}, n={num_loc})")


if __name__ == "__main__":
    main()
