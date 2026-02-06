#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _read_probe_summary(summary_csv: Path) -> Dict[str, List[float]]:
    rows: Dict[str, List[float]] = {}
    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for key in (reader.fieldnames or []):
            rows[key] = []
        for row in reader:
            for k, v in row.items():
                if v is None or v == "":
                    rows[k].append(float("nan"))
                else:
                    rows[k].append(float(v))
    return rows


def _nan_moving_average(y: List[float], window: int) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    window = int(window)
    if window <= 1:
        return y_arr
    if window % 2 == 0:
        window += 1
    pad = window // 2

    y_pad = np.pad(y_arr, (pad, pad), mode="edge")
    mask = np.isfinite(y_pad).astype(float)
    y0 = np.where(np.isfinite(y_pad), y_pad, 0.0)

    kernel = np.ones(window, dtype=float)
    num = np.convolve(y0, kernel, mode="valid")
    den = np.convolve(mask, kernel, mode="valid")
    return num / np.clip(den, 1e-9, None)


def _read_train_distances(results_dir: Path, step: int) -> Tuple[List[int], List[float]]:
    if step <= 0:
        raise ValueError("step must be >= 1")

    pat = re.compile(r"results_epoch_(\d+)\.pkl$")
    epochs: List[int] = []
    distances: List[float] = []

    files = []
    for p in results_dir.iterdir():
        m = pat.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])

    for epoch0, p in files[::step]:
        with p.open("rb") as f:
            d = pickle.load(f)
        rewards = d["rewards"]
        if isinstance(rewards, list):
            rewards = rewards[0]
        avg_dist = float((-rewards).mean().item())
        epochs.append(int(epoch0) + 1)  # results_epoch_0 corresponds to epoch 1
        distances.append(avg_dist)

    if not epochs:
        raise ValueError(f"No results found in {results_dir}")
    return epochs, distances


def _find_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "logs").glob("version_*/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"Could not find metrics.csv under {run_dir / 'logs'}")
    for c in candidates:
        if c.parent.name == "version_0":
            return c
    return candidates[0]


def _epoch_to_max_step(metrics_csv: Path) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with metrics_csv.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if not row:
                continue
            e_raw = (row.get("epoch") or "").strip()
            s_raw = (row.get("step") or "").strip()
            if not e_raw or not s_raw:
                continue
            try:
                e = int(float(e_raw))
                s = int(float(s_raw))
            except Exception:
                continue
            out[e] = max(out.get(e, -1), s)
    return out


def _steps_per_epoch_from_run(run_dir: Path) -> int:
    cfg = json.loads((run_dir / "config.json").read_text())
    train_data_size = int(cfg.get("num_instances", 0) or 0)
    batch_size = int(cfg.get("batch_size", 0) or 0)
    if batch_size <= 0:
        hparams = run_dir / "logs" / "version_0" / "hparams.yaml"
        if hparams.exists():
            for line in hparams.read_text().splitlines():
                if line.startswith("batch_size:"):
                    try:
                        batch_size = int(line.split(":", 1)[1].strip())
                    except Exception:
                        pass
                    break
    if train_data_size <= 0 or batch_size <= 0:
        raise ValueError(f"Could not infer steps/epoch from {run_dir} (num_instances/batch_size missing)")
    return (train_data_size + batch_size - 1) // batch_size


def _epoch1_to_step(epoch1: int, *, epoch_to_step: Dict[int, int], steps_per_epoch: int) -> int:
    # epoch1 is 1-based; metrics.csv uses 0-based epoch index.
    train_epoch = int(epoch1) - 1
    step = epoch_to_step.get(train_epoch)
    if step is None:
        step = (train_epoch + 1) * int(steps_per_epoch) - 1
    return int(step)


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay policy training distance on probe dynamics curves.")
    ap.add_argument("--summary_csv", type=Path, required=True)
    ap.add_argument("--results_dir", type=Path, required=True)
    ap.add_argument(
        "--x_axis",
        type=str,
        default="step",
        choices=["epoch", "step"],
        help="X-axis for both curves (default: step).",
    )
    ap.add_argument(
        "--x_scale",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="X-axis scale for both curves (default: linear).",
    )
    ap.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Policy run dir (defaults to parent of --results_dir). Needed to map epoch -> train step.",
    )
    ap.add_argument("--baseline_pkl", type=Path, required=False, default=None)
    ap.add_argument(
        "--baseline_label",
        type=str,
        default="Optimal",
        help="Legend label for --baseline_pkl horizontal line.",
    )
    ap.add_argument("--out_path", type=Path, required=True)
    ap.add_argument("--train_step", type=int, default=2, help="Read every Nth results_epoch_*.pkl (default: 2).")
    ap.add_argument("--smooth_probe", type=int, default=11, help="Centered moving-average window for probe curves.")
    ap.add_argument("--smooth_train", type=int, default=21, help="Centered moving-average window for train distance.")
    args = ap.parse_args()

    probe = _read_probe_summary(args.summary_csv)
    x_probe_epoch = [int(round(float(x))) for x in probe["epoch"]]

    x_train, y_train = _read_train_distances(args.results_dir, step=int(args.train_step))
    y_train_smooth = _nan_moving_average(y_train, window=int(args.smooth_train))

    if args.x_axis == "step":
        run_dir = args.run_dir
        if run_dir is None:
            run_dir = args.results_dir.parent
        metrics_csv = _find_metrics_csv(run_dir)
        epoch_to_step = _epoch_to_max_step(metrics_csv)
        steps_per_epoch = _steps_per_epoch_from_run(run_dir)
        x_probe = [_epoch1_to_step(e, epoch_to_step=epoch_to_step, steps_per_epoch=steps_per_epoch) for e in x_probe_epoch]
        x_train = [_epoch1_to_step(e, epoch_to_step=epoch_to_step, steps_per_epoch=steps_per_epoch) for e in x_train]
        x_label = "Policy train step"
    else:
        x_probe = x_probe_epoch
        x_label = "Epoch"

    if args.x_scale == "log":
        min_x = min(min(x_probe), min(x_train))
        x_offset = 1 - int(min_x) if min_x <= 0 else 0
        if x_offset:
            x_probe = [int(x) + x_offset for x in x_probe]
            x_train = [int(x) + x_offset for x in x_train]
            x_label = f"{x_label} (+{x_offset})"

    baseline_dist = None
    if args.baseline_pkl is not None:
        with args.baseline_pkl.open("rb") as f:
            baseline = pickle.load(f)
        baseline_dist = float((-baseline["rewards"][0]).mean().item())

    fig, ax = plt.subplots(figsize=(12, 6))

    # Blues for probe metrics, reds for policy distance/baseline (per user request).
    ax.plot(x_probe, _nan_moving_average(probe["top1_acc"], window=int(args.smooth_probe)), label="% identifies best node", color="#08306b")
    ax.plot(x_probe, _nan_moving_average(probe["top5_acc"], window=int(args.smooth_probe)), label="% top 5 contains best", color="#2171b5")
    ax.plot(
        x_probe,
        _nan_moving_average(probe["spearman_mean"], window=int(args.smooth_probe)),
        label="Spearman's ρ",
        color="#6baed6",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Probe metric")
    ax.set_ylim(0.0, 1.0)
    if args.x_scale == "log":
        ax.set_xscale("log")

    ax2 = ax.twinx()
    ax2.plot(x_train, y_train_smooth, label="Attention policy", color="#bd0b35", alpha=1, linewidth=1.25)
    if baseline_dist is not None:
        ax2.axhline(y=baseline_dist, color="#f05735", linestyle="--", linewidth=2.25, label=str(args.baseline_label))
        ax2.set_ylim(7.6, max(y_train_smooth))
    ax2.set_ylabel("Average route distance")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax2.ticklabel_format(style="plain", axis="y")

    # Add more tick marks on the right y-axis (range is narrow, so fixed ticks are clearer than log decades).
    y_min, y_max = ax2.get_ylim()
    tick_step = 0.2
    start = tick_step * np.floor(y_min / tick_step)
    stop = tick_step * np.ceil(y_max / tick_step)
    ticks = np.arange(start, stop + 1e-9, tick_step)
    # Include the Concorde baseline as its own tick, but offset its label slightly so it doesn't collide with nearby ticks.
    if baseline_dist is not None and float(baseline_dist) not in ticks:
        ticks = np.sort(np.unique(np.append(ticks, float(baseline_dist))))
    ax2.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax2.tick_params(axis="y", which="major", length=6)

    if baseline_dist is not None:
        # Nudge only the baseline tick label downward by a few points.
        from matplotlib.transforms import ScaledTranslation

        fig.canvas.draw()  # ensure tick labels exist
        for tick, value in zip(ax2.yaxis.get_major_ticks(), ax2.get_yticks()):
            if abs(float(value) - float(baseline_dist)) < 1e-9:
                # On a right y-axis, matplotlib may use label2; adjust both defensively.
                for label in (tick.label1, tick.label2):
                    label.set_transform(label.get_transform() + ScaledTranslation(0, -3 / 72, fig.dpi_scale_trans))
                break

    # Two legends (one per axis) makes it clearer which curves map to which y-axis.
    probe_handles, probe_labels = ax.get_legend_handles_labels()
    policy_handles, policy_labels = ax2.get_legend_handles_labels()
    legend_fontsize = 12
    legend_title_fontsize = 12
    probe_legend = ax.legend(
        probe_handles,
        probe_labels,
        loc="upper left",
        bbox_to_anchor=(0.10, 0.98),
        borderaxespad=0.0,
        title="Linear probe metrics ↑ (left)",
        framealpha=0.92,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )
    ax.add_artist(probe_legend)
    ax2.legend(
        policy_handles,
        policy_labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.0,
        title="TSP route distance ↓ (right)",
        framealpha=0.92,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )

    ax.set_title("TSP policy training dynamics (what-if probes)", fontsize=18)
    ax.grid(True, alpha=0.25)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
