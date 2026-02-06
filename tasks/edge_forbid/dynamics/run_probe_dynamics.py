#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_run_dir(run_arg: str) -> Path:
    p = Path(run_arg).expanduser()
    if p.is_dir():
        return p.resolve()
    candidate = _repo_root() / "runs" / run_arg
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve run dir from '{run_arg}'")


def _parse_epochs_arg(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    # Support "10,20,30" and "start:end:step" (end inclusive).
    if ":" in s and "," not in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid --epochs range: '{s}' (expected start:end[:step])")
        start = int(parts[0])
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step <= 0:
            raise ValueError("--epochs step must be >= 1")
        if end < start:
            raise ValueError("--epochs end must be >= start")
        return list(range(start, end + 1, step))
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def _find_checkpoints(run_dir: Path) -> Dict[int, Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints dir: {ckpt_dir}")
    pat = re.compile(r"checkpoint_epoch_(\d+)\.ckpt$")
    out: Dict[int, Path] = {}
    for p in ckpt_dir.iterdir():
        m = pat.match(p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise FileNotFoundError(f"No checkpoint_epoch_*.ckpt files found under {ckpt_dir}")
    return out


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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute probe-vs-training-epoch curves for edge-what-if.\n\n"
            "For each checkpoint_epoch_K.ckpt:\n"
            "  1) Extract edge representations aligned to optimal-tour edges.\n"
            "  2) Train a probe to predict which edge is most harmful to forbid.\n"
            "  3) Record test-set ranking metrics and plot vs epoch.\n"
        )
    )
    p.add_argument("--run_dir", type=str, required=True, help="Policy run dir (or runs/<name>) containing checkpoints/")
    p.add_argument("--data_dir", type=str, required=True, help="Processed edge-what-if dataset dir containing dataset.pt")
    p.add_argument(
        "--activation_key",
        type=str,
        default="encoder_output",
        help="Activation key to extract (ignored if --activation_keys is set).",
    )
    p.add_argument(
        "--activation_keys",
        type=str,
        default=None,
        help="Comma-separated activation keys to concatenate (overrides --activation_key).",
    )
    p.add_argument("--device", type=str, default="cuda", help="Device for extraction + probe training (default: cuda).")
    p.add_argument("--batch_size_extract", type=int, default=32, help="Instances per forward pass during extraction.")
    p.add_argument(
        "--resid_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for extracted X_resid (float16 saves disk).",
    )

    p.add_argument(
        "--epochs",
        type=str,
        default=None,
        help=(
            "Which checkpoint epochs to evaluate. "
            "Formats: '10,20,30' or 'start:end:step' (end inclusive). "
            "Default: all available checkpoints filtered by --epoch_step."
        ),
    )
    p.add_argument("--epoch_step", type=int, default=10, help="When --epochs is unset, keep epochs divisible by this.")
    p.add_argument("--min_epoch", type=int, default=None, help="Optional minimum epoch filter.")
    p.add_argument("--max_epoch", type=int, default=None, help="Optional maximum epoch filter.")

    p.add_argument("--objective", type=str, default="best_node_ce", choices=["best_node_ce", "soft_ce", "pairwise_rank"])
    p.add_argument("--model", type=str, default="linear", choices=["linear", "mlp", "transformer"])
    p.add_argument("--target", type=str, default="length", choices=["length", "time"])
    p.add_argument("--probe_batch_size", type=int, default=256, help="Probe batch size (instances) for ranking objectives.")
    p.add_argument("--probe_lr", type=float, default=1e-2)
    p.add_argument("--probe_weight_decay", type=float, default=0.0)
    p.add_argument("--probe_num_epochs", type=int, default=50)
    p.add_argument("--tfm_dim", type=int, default=256)
    p.add_argument("--tfm_layers", type=int, default=2)
    p.add_argument("--tfm_heads", type=int, default=8)
    p.add_argument("--tfm_ff_mult", type=int, default=4)
    p.add_argument("--tfm_dropout", type=float, default=0.0)
    p.add_argument("--standardize_x", action="store_true", help="Standardize X using train-set mean/std.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--soft_ce_tau", type=float, default=1.0)
    p.add_argument("--pairwise_pairs_per_instance", type=int, default=128)
    p.add_argument("--pairwise_margin", type=float, default=0.0)

    p.add_argument("--out_root", type=str, default=None, help="Output directory root (default: tasks/edge_forbid/tmp/dynamics/...)")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if per-epoch metrics already exist.")
    p.add_argument(
        "--keep_reps",
        action="store_true",
        help="Keep per-epoch probe_reps_epoch_*.pt files (default: delete after training to save disk).",
    )
    p.add_argument("--smooth", type=int, default=1, help="Centered moving-average window for plotted probe curves.")
    p.add_argument("--plot_only", action="store_true", help="Only plot from existing summary.csv (do not run extract/train).")
    p.add_argument("--no_plot", action="store_true", help="Skip plotting (only write summary.csv).")
    return p


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _read_csv(path: Path) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for key in (reader.fieldnames or []):
            out[key] = []
        for row in reader:
            for k, v in row.items():
                if v is None or v == "":
                    out[k].append(float("nan"))
                else:
                    out[k].append(float(v))
    return out


def _plot(summary_csv: Path, out_path: Path, *, n_edges: int, smooth: int) -> None:
    import matplotlib.pyplot as plt

    probe = _read_csv(summary_csv)
    x = [int(round(float(v))) for v in probe["epoch"]]
    top1 = _nan_moving_average(probe["top1_acc"], window=smooth)
    top5 = _nan_moving_average(probe["top5_acc"], window=smooth)
    spearman = _nan_moving_average(probe["spearman_mean"], window=smooth)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, top1, label="Top-1 (worst edge)", linewidth=2.0)
    ax.plot(x, top5, label="Top-5 contains worst", linewidth=2.0)
    ax.plot(x, spearman, label="Spearman (edge scores)", linewidth=2.0)

    chance_top1 = 1.0 / float(n_edges)
    chance_top5 = min(5, int(n_edges)) / float(n_edges)
    ax.axhline(y=chance_top1, color="0.4", linestyle="--", linewidth=1.5, label=f"Chance top-1 (1/{n_edges})")
    ax.axhline(y=chance_top5, color="0.4", linestyle=":", linewidth=1.5, label=f"Chance top-5 ({min(5,n_edges)}/{n_edges})")

    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel("Metric (higher is better)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.92)
    ax.set_title("Edge-what-if probe vs policy training epoch", fontsize=16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    args = build_arg_parser().parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    data_dir = Path(args.data_dir).expanduser().resolve()
    ds_path = data_dir / "dataset.pt"
    if not ds_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {ds_path}")
    ds = torch.load(ds_path, weights_only=False)
    locs = ds.get("locs")
    if not torch.is_tensor(locs) or locs.ndim != 3:
        raise ValueError("dataset.pt missing locs [B,n,2]")
    B, n, _ = locs.shape

    ckpts = _find_checkpoints(run_dir)
    avail_epochs = sorted(ckpts.keys())

    if args.epochs:
        epochs = _parse_epochs_arg(str(args.epochs))
        epochs = [e for e in epochs if e in ckpts]
    else:
        step = int(args.epoch_step)
        if step <= 0:
            raise ValueError("--epoch_step must be >= 1")
        epochs = [e for e in avail_epochs if (e % step) == 0]

    if args.min_epoch is not None:
        epochs = [e for e in epochs if e >= int(args.min_epoch)]
    if args.max_epoch is not None:
        epochs = [e for e in epochs if e <= int(args.max_epoch)]

    if not epochs:
        raise ValueError("No epochs selected (check --epochs/--epoch_step/--min_epoch/--max_epoch)")

    activation_keys = None
    if args.activation_keys:
        activation_keys = [k.strip() for k in str(args.activation_keys).split(",") if k.strip()]
    if not activation_keys:
        activation_keys = [str(args.activation_key)]
    act_tag = ",".join(activation_keys)

    if args.out_root:
        out_root = Path(args.out_root).expanduser().resolve()
    else:
        out_root = (
            _repo_root()
            / "tasks"
            / "edge_forbid"
            / "tmp"
            / "dynamics"
            / f"{run_dir.name}__{data_dir.name}"
            / act_tag
        )
    out_root.mkdir(parents=True, exist_ok=True)

    summary_csv = out_root / "summary.csv"
    plot_path = out_root / "dynamics.png"

    if args.plot_only:
        if not summary_csv.exists():
            raise FileNotFoundError(f"--plot_only set but missing {summary_csv}")
        if not args.no_plot:
            _plot(summary_csv, plot_path, n_edges=int(n), smooth=int(args.smooth))
            print(f"[edge-dynamics] wrote: {plot_path}")
        return

    extract_py = _repo_root() / "core" / "reps" / "extract_edge_reps.py"
    train_py = _repo_root() / "core" / "probe" / "train_probes.py"
    if not extract_py.exists():
        raise FileNotFoundError(f"Missing extractor: {extract_py}")
    if not train_py.exists():
        raise FileNotFoundError(f"Missing trainer: {train_py}")

    reps_dir = out_root / "reps"
    metrics_dir = out_root / "metrics"
    reps_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for epoch in epochs:
        metrics_path = metrics_dir / f"metrics_epoch_{epoch}.json"
        if metrics_path.exists() and not args.overwrite:
            m = _read_json(metrics_path)
        else:
            reps_path = reps_dir / f"probe_reps_epoch_{epoch}.pt"
            if (not reps_path.exists()) or args.overwrite:
                cmd = [
                    sys.executable,
                    str(extract_py),
                    "--data_dir",
                    str(data_dir),
                    "--run_dir",
                    str(run_dir),
                    "--checkpoint_epoch",
                    str(int(epoch)),
                    "--batch_size",
                    str(int(args.batch_size_extract)),
                    "--device",
                    str(args.device),
                    "--resid_dtype",
                    str(args.resid_dtype),
                    "--out_path",
                    str(reps_path),
                ]
                if args.activation_keys:
                    cmd += ["--activation_keys", str(args.activation_keys)]
                else:
                    cmd += ["--activation_key", str(args.activation_key)]
                print(f"[edge-dynamics] epoch {epoch}: extract -> {reps_path}", flush=True)
                _run(cmd)

            out_dir = out_root / f"probe_artifacts_epoch_{epoch}"
            out_dir.mkdir(parents=True, exist_ok=True)
            train_cmd = [
                sys.executable,
                str(train_py),
                "--reps_path",
                str(reps_path),
                "--out_dir",
                str(out_dir),
                "--target",
                str(args.target),
                "--objective",
                str(args.objective),
                "--model",
                str(args.model),
                "--batch_size",
                str(int(args.probe_batch_size)),
                "--lr",
                str(float(args.probe_lr)),
                "--weight_decay",
                str(float(args.probe_weight_decay)),
                "--num_epochs",
                str(int(args.probe_num_epochs)),
                "--seed",
                str(int(args.seed)),
                "--device",
                str(args.device),
                "--soft_ce_tau",
                str(float(args.soft_ce_tau)),
                "--pairwise_pairs_per_instance",
                str(int(args.pairwise_pairs_per_instance)),
                "--pairwise_margin",
                str(float(args.pairwise_margin)),
                "--tfm_dim",
                str(int(args.tfm_dim)),
                "--tfm_layers",
                str(int(args.tfm_layers)),
                "--tfm_heads",
                str(int(args.tfm_heads)),
                "--tfm_ff_mult",
                str(int(args.tfm_ff_mult)),
                "--tfm_dropout",
                str(float(args.tfm_dropout)),
            ]
            if args.standardize_x:
                train_cmd.append("--standardize_x")
            print(f"[edge-dynamics] epoch {epoch}: train -> {out_dir}", flush=True)
            _run(train_cmd)

            m = _read_json(out_dir / "metrics.json")
            metrics_path.write_text(json.dumps(m, indent=2) + "\n")

            if not args.keep_reps:
                try:
                    reps_path.unlink()
                except FileNotFoundError:
                    pass

        test = ((m.get("resid") or {}).get("test") or {})
        rows.append(
            {
                "epoch": float(epoch),
                "top1_acc": float(test.get("top1_acc", float("nan"))),
                "top5_acc": float(test.get("top5_acc", float("nan"))),
                "spearman_mean": float(test.get("spearman_mean", float("nan"))),
            }
        )
        _write_csv(summary_csv, rows)
        print(
            f"[edge-dynamics] epoch {epoch}: top1={rows[-1]['top1_acc']:.3f} top5={rows[-1]['top5_acc']:.3f} "
            f"spearman={rows[-1]['spearman_mean']:.3f}",
            flush=True,
        )

    print(f"[edge-dynamics] wrote: {summary_csv}")
    if not args.no_plot:
        _plot(summary_csv, plot_path, n_edges=int(n), smooth=int(args.smooth))
        print(f"[edge-dynamics] wrote: {plot_path}")


if __name__ == "__main__":
    main()
