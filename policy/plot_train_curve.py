import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _latest_metrics_csv(run_dir: Path) -> Path:
    logs = run_dir / "logs"
    if not logs.exists():
        raise FileNotFoundError(f"Missing logs dir: {logs}")
    versions = []
    for p in logs.glob("version_*"):
        try:
            versions.append((int(p.name.split("_", 1)[1]), p))
        except Exception:
            continue
    if not versions:
        raise FileNotFoundError(f"No logs/version_*/ found under {logs}")
    _, latest = max(versions, key=lambda x: x[0])
    csv_path = latest / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics.csv: {csv_path}")
    return csv_path


def _load_baseline_cost(run_dir: Path, baseline_name: str) -> float:
    path = run_dir / baseline_name
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline file: {path}")
    baseline = pickle.load(open(path, "rb"))
    rewards = baseline.get("rewards")
    if not isinstance(rewards, list) or not rewards:
        raise ValueError(f"Unexpected baseline format in {path}: expected dict with non-empty list rewards")
    r0 = rewards[0]
    if not hasattr(r0, "mean"):
        raise ValueError(f"Unexpected rewards[0] type in {path}: {type(r0)}")
    return float((-r0).mean().item())


def _read_val_costs(metrics_csv: Path, step: int) -> tuple[np.ndarray, np.ndarray]:
    by_epoch: dict[int, float] = {}
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "epoch" not in reader.fieldnames or "val/reward" not in reader.fieldnames:
            raise ValueError(f"{metrics_csv} must contain columns 'epoch' and 'val/reward'")
        for row in reader:
            e = row.get("epoch")
            vr = row.get("val/reward")
            if e is None or vr is None or vr == "" or vr.lower() == "nan":
                continue
            try:
                epoch = int(float(e))
                val_reward = float(vr)
            except Exception:
                continue
            by_epoch[epoch] = val_reward

    if not by_epoch:
        raise ValueError(f"No val/reward entries found in {metrics_csv}")

    epochs = np.array(sorted(by_epoch.keys()), dtype=int)
    epochs = epochs[epochs % int(step) == 0]
    vals = np.array([by_epoch[int(e)] for e in epochs], dtype=float)
    costs = -vals
    return epochs, costs


def main(args: argparse.Namespace) -> None:
    run_dir = Path("runs") / args.run_name
    metrics_csv = _latest_metrics_csv(run_dir)
    epochs, costs = _read_val_costs(metrics_csv, step=int(args.step))

    # If training used cost scaling, convert plotted costs back to the original units.
    cost_scale = 1.0
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            import json

            cfg = json.loads(cfg_path.read_text())
            cost_scale = float(cfg.get("cost_scale", 1.0) or 1.0)
        except Exception:
            cost_scale = 1.0
    if cost_scale != 1.0:
        costs = costs * cost_scale

    baseline_cost = None
    if args.baseline is not None:
        baseline_cost = _load_baseline_cost(run_dir, args.baseline)
        if cost_scale != 1.0:
            baseline_cost = baseline_cost * cost_scale

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, costs, marker="o", markersize=2, label="Avg Distance per Epoch")

    if baseline_cost is not None:
        plt.axhline(y=baseline_cost, color="red", linestyle="--", label="Baseline (Optimal)")

    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.title(f"Average Distance per Training Epoch vs. Baseline (every {int(args.step)} epochs)")
    plt.legend()

    y_min = float(np.min(costs))
    y_max = float(np.max(costs))
    if baseline_cost is not None:
        y_min = min(y_min, float(baseline_cost))
    pad = 0.02 * (y_max - y_min) if y_max > y_min else 1.0
    plt.ylim(y_min - pad, y_max + pad)

    out = run_dir / "train_plot.png"
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--step", type=int, default=5)
    p.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline pickle filename in the run dir (e.g. baseline_concorde.pkl). If unset, no baseline/fit is drawn.",
    )
    main(p.parse_args())
