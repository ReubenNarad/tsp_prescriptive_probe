#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def _downsample_xy(x: np.ndarray, y: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray]:
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
    return np.interp(epochs, x, y, left=np.nan, right=y[-1])


def _plot_probe(
    ax,
    probe_path: Optional[Path],
    metric: str,
    label: str,
    color,
    metrics: Dict[str, np.ndarray],
    smooth_window: int,
    downsample: int,
    warn_name: str,
) -> None:
    if probe_path is None:
        print(f"[warn] missing {warn_name} summary for {label}: (not set)")
        return
    if not probe_path.exists():
        print(f"[warn] missing {warn_name} summary for {label}: {probe_path}")
        return
    probe = _load_probe_summary(probe_path)
    steps = _interp_steps(metrics, probe["epoch"])
    y = probe.get(metric)
    if y is None:
        print(f"[warn] missing {metric} in {probe_path}")
        return
    y = _smooth(y, int(smooth_window))
    steps_plot, y_plot = _downsample_xy(steps, y, downsample)
    ax.plot(steps_plot, y_plot, label=label, color=color, linewidth=2.0)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot training vs probe dynamics in a 3x2 grid.")
    p.add_argument("--config", type=str, required=True, help="JSON config listing runs + paths.")
    p.add_argument(
        "--out_path",
        type=str,
        default="tasks/node_removal/tmp/scale_dynamics/train_vs_probe_grid.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--probe_metric",
        type=str,
        default="top1_acc",
        choices=["top1_acc", "top5_acc", "spearman_mean"],
        help="Linear probe metric for row 2 (A/B).",
    )
    p.add_argument(
        "--transformer_metric",
        type=str,
        default="top1_acc",
        choices=["top1_acc", "top5_acc", "spearman_mean"],
        help="Transformer probe metric for row 3 (A/B).",
    )
    p.add_argument("--train_title", type=str, default="TSP policy performance")
    p.add_argument("--probe_title_a", type=str, default="Linear probe: node removal")
    p.add_argument("--probe_title_b", type=str, default="Linear probe: edge forbid")
    p.add_argument("--transformer_title_a", type=str, default="Transformer probe: node removal")
    p.add_argument("--transformer_title_b", type=str, default="Transformer probe: edge forbid")
    p.add_argument("--title", type=str, default="Scaling: training vs probe accuracy")
    p.add_argument("--xlim_max", type=float, default=590000.0, help="Max x-limit for all panels.")
    p.add_argument("--train_logy", action="store_true", help="Use log scale for the training axis.")
    p.add_argument("--train_ylim_max", type=float, default=None, help="Optional upper limit for train axis.")
    p.add_argument(
        "--train_suboptimality",
        action="store_true",
        help="Plot percent suboptimality relative to baseline.pkl mean tour length.",
    )
    p.add_argument("--probe_logy", action="store_true", help="Use log scale for linear probe axes.")
    p.add_argument("--transformer_logy", action="store_true", help="Use log scale for transformer axes.")
    p.add_argument("--smooth_train", type=int, default=1, help="Moving-average window for training curve.")
    p.add_argument("--smooth_probe", type=int, default=1, help="Moving-average window for probe curves.")
    p.add_argument(
        "--smooth_transformer",
        type=int,
        default=1,
        help="Moving-average window for transformer probe curves.",
    )
    p.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name for run colors (set to 'none' to use config colors).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = _read_json(Path(args.config).expanduser().resolve())
    runs = cfg.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Config must contain a non-empty 'runs' list.")

    import matplotlib.pyplot as plt

    scale = 1.5
    base_sizes = {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "figure.titlesize": 14,
    }
    for key, base in base_sizes.items():
        plt.rcParams[key] = base * scale

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0])
    ax_train = fig.add_subplot(gs[0, :])
    ax_probe_a = fig.add_subplot(gs[1, 0])
    ax_probe_b = fig.add_subplot(gs[1, 1])
    ax_xf_a = fig.add_subplot(gs[2, 0])
    ax_xf_b = fig.add_subplot(gs[2, 1])

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
        downsample_transformer = int(run.get("downsample_transformer", run.get("downsample", 1)))
        if not np.isfinite(step_scale) or step_scale <= 0:
            raise ValueError(f"Invalid step_scale for {label}: {step_scale}")
        if downsample_train < 1:
            raise ValueError(f"Invalid downsample_train for {label}: {downsample_train}")
        if downsample_probe < 1:
            raise ValueError(f"Invalid downsample_probe for {label}: {downsample_probe}")
        if downsample_transformer < 1:
            raise ValueError(f"Invalid downsample_transformer for {label}: {downsample_transformer}")

        metrics_path = Path(run.get("metrics_csv", "")).expanduser()
        baseline_path_raw = run.get("baseline_pkl")
        baseline_path = Path(baseline_path_raw).expanduser() if baseline_path_raw else None

        probe_a_raw = run.get("probe_summary_a")
        probe_b_raw = run.get("probe_summary_b")
        xf_a_raw = run.get("transformer_summary_a")
        xf_b_raw = run.get("transformer_summary_b")
        probe_a = Path(probe_a_raw).expanduser() if probe_a_raw else None
        probe_b = Path(probe_b_raw).expanduser() if probe_b_raw else None
        xf_a = Path(xf_a_raw).expanduser() if xf_a_raw else None
        xf_b = Path(xf_b_raw).expanduser() if xf_b_raw else None

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

        _plot_probe(
            ax_probe_a,
            probe_a,
            args.probe_metric,
            label,
            color,
            metrics,
            args.smooth_probe,
            downsample_probe,
            "probe A",
        )
        _plot_probe(
            ax_probe_b,
            probe_b,
            args.probe_metric,
            label,
            color,
            metrics,
            args.smooth_probe,
            downsample_probe,
            "probe B",
        )
        _plot_probe(
            ax_xf_a,
            xf_a,
            args.transformer_metric,
            label,
            color,
            metrics,
            args.smooth_transformer,
            downsample_transformer,
            "transformer A",
        )
        _plot_probe(
            ax_xf_b,
            xf_b,
            args.transformer_metric,
            label,
            color,
            metrics,
            args.smooth_transformer,
            downsample_transformer,
            "transformer B",
        )

    ax_train.set_title(args.train_title)
    ax_train.set_xlabel("Train step")
    ax_train.set_ylabel("% Suboptimality" if args.train_suboptimality else train_label)
    ax_train.grid(True, alpha=0.25)
    ax_train.set_xlim(0, float(args.xlim_max))
    if args.train_logy:
        ax_train.set_yscale("log", nonpositive="clip")
    if args.train_ylim_max is not None:
        ymin, _ymax = ax_train.get_ylim()
        if args.train_logy:
            ymin = min_train_y if min_train_y is not None else max(ymin, 1e-6)
        ax_train.set_ylim(ymin, float(args.train_ylim_max))
    if args.train_suboptimality:
        from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

        ax_train.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:g}%"))
        if args.train_logy:
            ax_train.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=7))
        else:
            ax_train.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
        ax_train.yaxis.set_minor_formatter(FuncFormatter(lambda *_args: ""))

    probe_title_a = str(args.probe_title_a).replace("{probe_metric}", args.probe_metric)
    probe_title_b = str(args.probe_title_b).replace("{probe_metric}", args.probe_metric)
    xf_title_a = str(args.transformer_title_a).replace("{transformer_metric}", args.transformer_metric)
    xf_title_b = str(args.transformer_title_b).replace("{transformer_metric}", args.transformer_metric)

    ax_probe_a.set_title(probe_title_a)
    ax_probe_b.set_title(probe_title_b)
    ax_xf_a.set_title(xf_title_a)
    ax_xf_b.set_title(xf_title_b)

    for ax in (ax_probe_a, ax_probe_b, ax_xf_a, ax_xf_b):
        ax.set_xlabel("Train step")
        ax.set_xlim(0, float(args.xlim_max))
        ax.grid(True, alpha=0.25)

    from matplotlib.ticker import FuncFormatter

    def _sci_tick(x, _pos):
        if x == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(x))))
        mant = x / (10**exp)
        if abs(mant - round(mant)) < 1e-6:
            mant = int(round(mant))
            return f"{mant}e{exp}"
        return f"{mant:.1f}e{exp}"

    sci_fmt = FuncFormatter(_sci_tick)
    for ax in (ax_train, ax_probe_a, ax_probe_b, ax_xf_a, ax_xf_b):
        ax.xaxis.set_major_formatter(sci_fmt)

    def _metric_label(metric: str) -> str:
        metric = str(metric)
        if metric == "top1_acc":
            return "Top-1 Accuracy"
        if metric == "top5_acc":
            return "Top-5 Accuracy"
        if metric == "spearman_mean":
            return "Spearman"
        return metric

    ax_probe_a.set_ylabel(_metric_label(args.probe_metric))
    ax_probe_b.set_ylabel(_metric_label(args.probe_metric))
    ax_xf_a.set_ylabel(_metric_label(args.transformer_metric))
    ax_xf_b.set_ylabel(_metric_label(args.transformer_metric))

    if args.probe_logy:
        ax_probe_a.set_yscale("log", nonpositive="clip")
        ax_probe_b.set_yscale("log", nonpositive="clip")
    if args.transformer_logy:
        ax_xf_a.set_yscale("log", nonpositive="clip")
        ax_xf_b.set_yscale("log", nonpositive="clip")

    from matplotlib.ticker import MultipleLocator

    if not args.probe_logy:
        ax_probe_a.yaxis.set_major_locator(MultipleLocator(0.1))
        ax_probe_b.yaxis.set_major_locator(MultipleLocator(0.1))
    if not args.transformer_logy:
        ax_xf_a.yaxis.set_major_locator(MultipleLocator(0.1))
        ax_xf_b.yaxis.set_major_locator(MultipleLocator(0.1))

    ax_train.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
        framealpha=0.9,
        title="model params",
        ncol=max(1, len(runs)),
    )

    fig.suptitle(args.title)
    fig.tight_layout()
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
