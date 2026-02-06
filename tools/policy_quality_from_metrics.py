#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract policy quality metric from RL4CO metrics.csv.")
    p.add_argument("--run_dir", action="append", default=[], help="Run directory under runs/ (repeatable).")
    p.add_argument("--metric", type=str, default=None, help="Metric column to read (e.g., val/reward, val/cost_bsf).")
    p.add_argument("--out_csv", type=str, default=None, help="Optional CSV output.")
    return p


def _find_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "logs").glob("version_*/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"metrics.csv not found under {run_dir / 'logs'}")
    return candidates[0]


def _read_last_value(path: Path, metric: str) -> Dict:
    last_val = None
    last_epoch = None
    last_step = None
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if metric not in row or row[metric] in (None, ""):
                continue
            try:
                last_val = float(row[metric])
                last_epoch = int(float(row.get("epoch", "0")))
                last_step = int(float(row.get("step", "0")))
            except Exception:
                continue
    return {"value": last_val, "epoch": last_epoch, "step": last_step}


def _infer_metric(path: Path) -> str:
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames or []
    for cand in ("val/reward", "val/cost_bsf", "val/cost_init"):
        if cand in fieldnames:
            return cand
    raise ValueError(f"Could not infer metric from {path}. Provide --metric explicitly.")


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dirs = [Path(r).expanduser().resolve() for r in args.run_dir]
    if not run_dirs:
        raise SystemExit("Provide at least one --run_dir")

    rows: List[Dict] = []
    for run_dir in run_dirs:
        metrics_csv = _find_metrics_csv(run_dir)
        metric = args.metric or _infer_metric(metrics_csv)
        out = _read_last_value(metrics_csv, metric)
        rows.append(
            {
                "run_dir": str(run_dir),
                "metric": metric,
                "value": out["value"],
                "epoch": out["epoch"],
                "step": out["step"],
            }
        )

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["run_dir", "metric", "value", "epoch", "step"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[policy_quality] wrote {out_path}")

    for r in rows:
        print(f"{r['run_dir']}: {r['metric']} = {r['value']} (epoch {r['epoch']}, step {r['step']})")


if __name__ == "__main__":
    main()
