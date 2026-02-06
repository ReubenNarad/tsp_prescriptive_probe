#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _load_metrics(path: Path) -> Dict[str, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    if "step" not in df.columns or "epoch" not in df.columns:
        raise ValueError(f"Missing step/epoch in metrics.csv: {path}")

    reward_col = "val/reward" if "val/reward" in df.columns else "train/reward"
    if reward_col not in df.columns:
        raise ValueError(f"Missing reward column in {path} (need val/reward or train/reward)")

    step = df["step"].to_numpy(dtype=float)
    epoch = df["epoch"].to_numpy(dtype=float)
    reward = df[reward_col].to_numpy(dtype=float)

    mask = np.isfinite(step) & np.isfinite(epoch) & np.isfinite(reward)
    step = step[mask]
    epoch = epoch[mask]
    reward = reward[mask]

    # RL4CO rewards for TSP are negative tour lengths.
    tour_len = -reward
    return {"step": step, "epoch": epoch, "tour_len": tour_len}


def _load_baseline_mean_len(path: Path) -> float:
    with path.open("rb") as fp:
        obj = pickle.load(fp)
    if not isinstance(obj, dict) or "rewards" not in obj:
        raise ValueError(f"Unexpected baseline.pkl format: {path}")
    rewards = obj["rewards"]
    if isinstance(rewards, list):
        if not rewards:
            raise ValueError(f"baseline.pkl rewards list empty: {path}")
        rewards = rewards[0]
    rewards = np.asarray(rewards, dtype=float)
    mean_reward = float(np.mean(rewards))
    return float(-mean_reward)


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float)
    return np.convolve(y_pad, kernel, mode="valid") / float(window)


def _downsample_xy(x: np.ndarray, y: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if stride <= 1:
        return x, y
    return x[::stride], y[::stride]


def _load_probe_summary(path: Path) -> Dict[str, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"Missing epoch in probe summary: {path}")
    out = {"epoch": df["epoch"].to_numpy(dtype=float)}
    for key in ("top1_acc", "top5_acc", "spearman_mean"):
        if key in df.columns:
            out[key] = df[key].to_numpy(dtype=float)
    return out


def _interp_steps(metrics: Dict[str, np.ndarray], epochs: np.ndarray) -> np.ndarray:
    order = np.argsort(metrics["epoch"])
    x = metrics["epoch"][order]
    y = metrics["step"][order]
    if x.size < 2:
        return np.full_like(epochs, np.nan, dtype=float)
    return np.interp(epochs, x, y, left=np.nan, right=np.nan)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot training vs probe dynamics for multiple runs.")
    p.add_argument("--config", type=str, required=True, help="JSON config listing runs + paths.")
    p.add_argument(
        "--out_path",
        type=str,
        default="tasks/node_removal/tmp/scale_dynamics/train_vs_probe.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--probe_metric",
        type=str,
        default="top1_acc",
        choices=["top1_acc", "top5_acc", "spearman_mean"],
        help="Probe metric to plot on the right panel.",
    )
    p.add_argument(
        "--train_title",
        type=str,
        default="Avg tour length vs train step",
        help="Title for the left (training) panel.",
    )
    p.add_argument(
        "--probe_title",
        type=str,
        default="Linear probe vs train step ({probe_metric})",
        help="Title for the right (probe) panel; supports {probe_metric}.",
    )
    p.add_argument(
        "--train_logy",
        action="store_true",
        help="Use log scale for the training tour-length axis.",
    )
    p.add_argument(
        "--train_ylim_max",
        type=float,
        default=None,
        help="Optional upper limit for the training tour-length axis.",
    )
    p.add_argument(
        "--train_suboptimality",
        action="store_true",
        help="Plot percent suboptimality relative to baseline.pkl mean tour length.",
    )
    p.add_argument(
        "--probe_logy",
        action="store_true",
        help="Use log scale for the probe metric axis.",
    )
    p.add_argument("--smooth_train", type=int, default=1, help="Moving-average window for training curve.")
    p.add_argument("--smooth_probe", type=int, default=1, help="Moving-average window for probe curve.")
    p.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name for run colors (set to 'none' to use config colors).",
    )
    p.add_argument("--title", type=str, default="Scaling: training vs probe accuracy")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = _read_json(Path(args.config).expanduser().resolve())
    runs = cfg.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Config must contain a non-empty 'runs' list.")

    import matplotlib.pyplot as plt

    fig, (ax_train, ax_probe) = plt.subplots(1, 2, figsize=(12, 4.8))

    min_train_y: Optional[float] = None
    train_label = "Tour length (val)"
    colors = None
    if str(args.colormap).lower() != "none":
        cmap = plt.get_cmap(str(args.colormap))
        colors = cmap(np.linspace(0.15, 0.85, len(runs)))

    for idx, run in enumerate(runs):
        label = str(run.get("label", "run"))
        color = colors[idx] if colors is not None else run.get("color")
        step_scale = float(run.get("step_scale", 1.0))
        downsample_train = int(run.get("downsample_train", run.get("downsample", 1)))
        downsample_probe = int(run.get("downsample_probe", run.get("downsample", 1)))
        if not np.isfinite(step_scale) or step_scale <= 0:
            raise ValueError(f"Invalid step_scale for {label}: {step_scale}")
        if downsample_train < 1:
            raise ValueError(f"Invalid downsample_train for {label}: {downsample_train}")
        if downsample_probe < 1:
            raise ValueError(f"Invalid downsample_probe for {label}: {downsample_probe}")
        metrics_path = Path(run.get("metrics_csv", "")).expanduser()
        probe_path_raw = run.get("probe_summary")
        probe_path = Path(probe_path_raw).expanduser() if probe_path_raw else None
        baseline_path_raw = run.get("baseline_pkl")
        baseline_path = Path(baseline_path_raw).expanduser() if baseline_path_raw else None

        if not metrics_path.exists():
            print(f"[warn] missing metrics.csv for {label}: {metrics_path}")
            continue

        metrics = _load_metrics(metrics_path)
        if step_scale != 1.0:
            metrics = dict(metrics)
            metrics["step"] = metrics["step"] * step_scale
        if args.train_suboptimality:
            if baseline_path is None or not baseline_path.exists():
                print(f"[warn] missing baseline.pkl for {label}; falling back to tour length")
                train_y = metrics["tour_len"]
                train_label = "Tour length (val)"
            else:
                opt_len = _load_baseline_mean_len(baseline_path)
                train_y = (metrics["tour_len"] / opt_len - 1.0) * 100.0
                train_label = "% suboptimality (val)"
        else:
            train_y = metrics["tour_len"]
            train_label = "Tour length (val)"

        train_y = _smooth(train_y, int(args.smooth_train))
        train_x = metrics["step"]
        train_x_plot, train_y_plot = _downsample_xy(train_x, train_y, downsample_train)
        ax_train.plot(train_x_plot, train_y_plot, label=label, color=color, linewidth=2.0)
        positive = train_y[train_y > 0]
        if positive.size:
            min_pos = float(np.min(positive))
            if min_train_y is None or min_pos < min_train_y:
                min_train_y = min_pos

        if probe_path is not None and probe_path.exists():
            probe = _load_probe_summary(probe_path)
            steps = _interp_steps(metrics, probe["epoch"])
            y = probe.get(args.probe_metric)
            if y is None:
                print(f"[warn] missing {args.probe_metric} in {probe_path}")
            else:
                y = _smooth(y, int(args.smooth_probe))
                steps_plot, y_plot = _downsample_xy(steps, y, downsample_probe)
                ax_probe.plot(steps_plot, y_plot, label=label, color=color, linewidth=2.0)
        else:
            if probe_path is None:
                print(f"[warn] missing probe summary for {label}: (not set)")
            else:
                print(f"[warn] missing probe summary for {label}: {probe_path}")

    ax_train.set_title(args.train_title)
    ax_train.set_xlabel("TSP Policy Train step")
    ax_train.set_ylabel(train_label)
    if args.train_logy:
        ax_train.set_yscale("log", nonpositive="clip")
    if args.train_ylim_max is not None:
        ymin, _ymax = ax_train.get_ylim()
        if args.train_logy:
            ymin = min_train_y if min_train_y is not None else max(ymin, 1e-6)
        ax_train.set_ylim(ymin, float(args.train_ylim_max))
    ax_train.grid(True, alpha=0.25)
    ax_train.set_xlim(0, 590000)
    if args.train_suboptimality:
        from matplotlib.ticker import FuncFormatter

        ax_train.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:g}%"))
        ax_train.yaxis.set_minor_formatter(FuncFormatter(lambda *_args: ""))

    probe_title = str(args.probe_title).replace("{probe_metric}", args.probe_metric)
    ax_probe.set_title(probe_title)
    ax_probe.set_xlabel("TSP Policy Train step")
    ax_probe.set_xlim(0, 590000)
    
    ax_probe.set_ylabel(args.probe_metric)
    if args.probe_logy:
        ax_probe.set_yscale("log", nonpositive="clip")
    ax_probe.grid(True, alpha=0.25)

    ax_train.legend(loc="best", framealpha=0.9, title="model params")
    ax_probe.legend(loc="best", framealpha=0.9, title="model params")

    fig.suptitle(args.title)
    fig.tight_layout()
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
