#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _add_repo_to_path() -> None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ensemble a probe (train_probes.py checkpoint) with oracle splice scores (node-removal)."
    )
    p.add_argument("--reps_path", type=str, required=True, help="probe reps .pt (what-if)")
    p.add_argument("--model_path", type=str, required=True, help="Saved probe model .pt (train_probes.py --save_models)")
    p.add_argument("--oracle_scores_path", type=str, required=True, help="oracle_splice_contrib.pt")
    p.add_argument("--out_path", type=str, required=True, help="Output JSON path")
    p.add_argument("--seed", type=int, default=0, help="Split seed (matches training)")
    p.add_argument("--train_size", type=int, default=2500)
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument("--test_size", type=int, default=300)
    p.add_argument("--alpha_grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument(
        "--score_norm",
        type=str,
        default="raw",
        choices=["raw", "zscore"],
        help="Normalize scores per instance before mixing.",
    )
    p.add_argument("--batch_size_instances", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    return p


def _as_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_alpha_grid(csv: str) -> List[float]:
    out = []
    for part in csv.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("--alpha_grid produced no values")
    return out


def _make_fixed_splits(num_instances: int, train_size: int, val_size: int, test_size: int, seed: int) -> Tuple[set, set, set]:
    if train_size + val_size + test_size > num_instances:
        raise ValueError("Split sizes exceed available instances")
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_instances, generator=g)
    test_ids = perm[:test_size]
    val_ids = perm[test_size : test_size + val_size]
    train_pool = perm[test_size + val_size :]
    train_ids = train_pool[:train_size]
    return (
        set(int(i) for i in train_ids.tolist()),
        set(int(i) for i in val_ids.tolist()),
        set(int(i) for i in test_ids.tolist()),
    )


def _zscore_by_instance(scores: np.ndarray, instance_ids: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = scores.copy()
    for inst in np.unique(instance_ids):
        rows = instance_ids == inst
        v = valid[rows]
        vals = out[rows][v]
        if vals.size < 2:
            continue
        mean = float(vals.mean())
        std = float(vals.std())
        if std < 1e-6:
            std = 1.0
        out[rows] = (out[rows] - mean) / std
    return out


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n < 2:
        return float("nan")
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    xr = x_rank.astype(np.float64) - x_rank.mean()
    yr = y_rank.astype(np.float64) - y_rank.mean()
    denom = math.sqrt(float(np.sum(xr * xr) * np.sum(yr * yr)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(xr * yr) / denom)


def _eval_scores(
    scores: np.ndarray,
    y_true: np.ndarray,
    valid: np.ndarray,
    instance_ids: np.ndarray,
    instance_mask: np.ndarray,
) -> Dict[str, float]:
    inst_set = sorted(set(int(i) for i in instance_ids[instance_mask].tolist()))
    if not inst_set:
        return {}

    top1 = 0
    top5 = 0
    regret = []
    spears = []
    for inst in inst_set:
        rows = (instance_ids == inst) & instance_mask
        v = valid[rows]
        if v.sum() < 2:
            continue
        yt = y_true[rows]
        st = scores[rows]
        yt_masked = np.where(v, yt, -np.inf)
        st_masked = np.where(v, st, -np.inf)
        true_idx = int(np.argmax(yt_masked))
        pred_idx = int(np.argmax(st_masked))
        top1 += int(pred_idx == true_idx)
        k = min(5, int(v.sum()))
        topk_idx = np.argsort(st_masked)[-k:]
        top5 += int(true_idx in set(int(i) for i in topk_idx.tolist()))
        best = float(np.max(yt_masked))
        chosen = float(yt[pred_idx])
        regret.append(best - chosen)
        spears.append(_spearman_corr(st[v], yt[v]))

    def _mean(xs: List[float]) -> float:
        xs2 = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs2)) if xs2 else float("nan")

    return {
        "top1_acc": float(top1 / len(inst_set)),
        "top5_acc": float(top5 / len(inst_set)),
        "top1_regret_mean": float(np.mean(regret)) if regret else float("nan"),
        "spearman_mean": _mean(spears),
        "num_instances": int(len(inst_set)),
    }


def _load_probe_logits(
    reps_path: Path,
    model_path: Path,
    device: torch.device,
    batch_size_instances: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    os_env = __import__("os").environ
    os_env.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    reps = torch.load(reps_path, weights_only=False)
    X = reps["X_resid"].to(torch.float32)
    y = reps["y"][:, 0].to(torch.float32)
    valid = reps["valid"].to(torch.bool)
    instance_id = reps["instance_id"].to(torch.int64)
    node_id = reps["node_id"].to(torch.int64)

    B = int(instance_id.max().item()) + 1
    n = int(node_id.max().item()) + 1
    key = instance_id * n + node_id
    idx = torch.argsort(key)
    X = X[idx]
    y = y[idx]
    valid = valid[idx]
    instance_id = instance_id[idx]
    node_id = node_id[idx]

    expected_instance = torch.arange(B, dtype=torch.int64).repeat_interleave(n)
    expected_node = torch.arange(n, dtype=torch.int64).repeat(B)
    if not (torch.equal(instance_id, expected_instance) and torch.equal(node_id, expected_node)):
        raise ValueError("Unexpected instance/node ordering after sort; cannot align scores safely.")

    X_inst = X.view(B, n, X.shape[1])
    valid_inst = valid.view(B, n)

    _add_repo_to_path()
    import importlib.util

    train_path = _repo_root() / "core" / "probe" / "train_probes.py"
    spec = importlib.util.spec_from_file_location("train_probes", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load train_probes from {train_path}")
    train_probes = importlib.util.module_from_spec(spec)
    sys.modules["train_probes"] = train_probes
    spec.loader.exec_module(train_probes)

    model_data = torch.load(model_path, weights_only=False)
    model = train_probes.build_probe_model(
        model_type=model_data["model"],
        input_dim=X_inst.shape[2],
        output_dim=1,
        mlp_hidden_dim=model_data.get("mlp_hidden_dim", 256),
        mlp_layers=model_data.get("mlp_layers", 1),
        mlp_dropout=model_data.get("mlp_dropout", 0.0),
        tfm_dim=model_data.get("tfm_dim", 256),
        tfm_layers=model_data.get("tfm_layers", 2),
        tfm_heads=model_data.get("tfm_heads", 4),
        tfm_ff_mult=model_data.get("tfm_ff_mult", 4),
        tfm_dropout=model_data.get("tfm_dropout", 0.0),
    ).to(device)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()

    if model_data.get("node_standardize_x"):
        mean = X_inst.mean(dim=2, keepdim=True)
        std = X_inst.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
        X_inst = (X_inst - mean) / std
    if model_data.get("instance_standardize_x"):
        m = valid_inst.to(torch.float32).unsqueeze(-1)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (X_inst * m).sum(dim=1, keepdim=True) / denom
        var = ((X_inst - mean) * m).pow(2).sum(dim=1, keepdim=True) / denom
        std = var.sqrt().clamp_min(1e-6)
        X_inst = (X_inst - mean) / std

    x_mean = model_data.get("x_mean")
    x_std = model_data.get("x_std")
    if x_mean is not None and x_std is not None:
        x_mean_t = torch.tensor(x_mean, dtype=torch.float32)
        x_std_t = torch.tensor(x_std, dtype=torch.float32)
        X_inst = (X_inst - x_mean_t) / x_std_t

    logits = []
    with torch.no_grad():
        for start in range(0, B, batch_size_instances):
            batch = slice(start, min(B, start + batch_size_instances))
            xb = X_inst[batch].to(device)
            vb = valid_inst[batch].to(device)
            if model_data["model"] == "transformer" or model_data["model"] == "deepset":
                out = model(xb, key_padding_mask=(~vb)).squeeze(-1)
            else:
                out = model(xb).squeeze(-1)
            logits.append(out.cpu())
    logits_full = torch.cat(logits, dim=0).reshape(-1).numpy()
    return logits_full, y.numpy(), valid.numpy(), instance_id.numpy(), node_id.numpy(), B, n


def _load_oracle_scores(oracle_path: Path, B: int, n: int) -> np.ndarray:
    obj = torch.load(oracle_path, weights_only=False)
    if not isinstance(obj, dict) or "oracle_splice_contrib_pct" not in obj:
        raise ValueError("oracle_splice_contrib_pct not found in oracle file")
    scores = obj["oracle_splice_contrib_pct"]
    if not torch.is_tensor(scores) or scores.shape != (B, n):
        raise ValueError(f"oracle scores shape mismatch: expected {(B, n)}, got {getattr(scores, 'shape', None)}")
    return scores.reshape(-1).to(torch.float32).numpy()


def main() -> None:
    args = build_arg_parser().parse_args()
    reps_path = Path(args.reps_path).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    oracle_path = Path(args.oracle_scores_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()

    device = _as_device(args.device)
    logits, y_true, valid, instance_ids, node_ids, B, n = _load_probe_logits(
        reps_path=reps_path,
        model_path=model_path,
        device=device,
        batch_size_instances=int(args.batch_size_instances),
    )
    oracle_scores = _load_oracle_scores(oracle_path, B, n)

    if args.score_norm == "zscore":
        logits = _zscore_by_instance(logits, instance_ids, valid)
        oracle_scores = _zscore_by_instance(oracle_scores, instance_ids, valid)

    train_ids, val_ids, test_ids = _make_fixed_splits(
        num_instances=B,
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
        seed=int(args.seed),
    )
    inst_arr = instance_ids
    train_mask = np.array([int(i) in train_ids for i in inst_arr], dtype=bool)
    val_mask = np.array([int(i) in val_ids for i in inst_arr], dtype=bool)
    test_mask = np.array([int(i) in test_ids for i in inst_arr], dtype=bool)

    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    best_alpha = None
    best_val = None
    best_val_metrics = None

    for alpha in alpha_grid:
        mix = alpha * logits + (1.0 - alpha) * oracle_scores
        val_metrics = _eval_scores(mix, y_true, valid, instance_ids, val_mask)
        score = val_metrics.get("top1_acc", float("nan"))
        if best_val is None or (not math.isnan(score) and score > best_val):
            best_val = score
            best_alpha = alpha
            best_val_metrics = val_metrics

    if best_alpha is None:
        raise RuntimeError("No valid alpha produced metrics")

    mix = best_alpha * logits + (1.0 - best_alpha) * oracle_scores

    out = {
        "config": {
            "reps_path": str(reps_path),
            "model_path": str(model_path),
            "oracle_scores_path": str(oracle_path),
            "seed": int(args.seed),
            "train_size": int(args.train_size),
            "val_size": int(args.val_size),
            "test_size": int(args.test_size),
            "alpha_grid": alpha_grid,
            "score_norm": str(args.score_norm),
            "batch_size_instances": int(args.batch_size_instances),
        },
        "probe_only": {
            "train": _eval_scores(logits, y_true, valid, instance_ids, train_mask),
            "val": _eval_scores(logits, y_true, valid, instance_ids, val_mask),
            "test": _eval_scores(logits, y_true, valid, instance_ids, test_mask),
        },
        "oracle_only": {
            "train": _eval_scores(oracle_scores, y_true, valid, instance_ids, train_mask),
            "val": _eval_scores(oracle_scores, y_true, valid, instance_ids, val_mask),
            "test": _eval_scores(oracle_scores, y_true, valid, instance_ids, test_mask),
        },
        "ensemble": {
            "alpha": float(best_alpha),
            "val_selected": best_val_metrics,
            "train": _eval_scores(mix, y_true, valid, instance_ids, train_mask),
            "val": _eval_scores(mix, y_true, valid, instance_ids, val_mask),
            "test": _eval_scores(mix, y_true, valid, instance_ids, test_mask),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[ensemble] wrote {out_path}")


if __name__ == "__main__":
    main()
