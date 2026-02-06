#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect probe_tuning results.json files into a sortable table (CSV/MD).")
    p.add_argument("--root", type=str, default="tasks/node_removal/tmp/tuning")
    p.add_argument("--out_csv", type=str, default="tasks/node_removal/tmp/tuning/summary_table.csv")
    p.add_argument("--out_md", type=str, default="tasks/node_removal/tmp/tuning/summary_table.md")
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--include_patterns", type=str, default="", help="Comma-separated substrings to include (optional).")
    p.add_argument("--exclude_patterns", type=str, default="", help="Comma-separated substrings to exclude (optional).")
    p.add_argument(
        "--schema",
        type=str,
        default="",
        help="Comma-separated schemas to include (e.g. multiseed,grid,records). Empty = all.",
    )
    p.add_argument("--min_num_seeds", type=int, default=0, help="Only keep rows with >= this num_seeds (if present).")
    p.add_argument("--relative_paths", action="store_true", help="Use paths relative to --root in tables.")
    p.add_argument("--topk_md", type=int, default=40, help="Rows to show in markdown table.")
    return p


def _get(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _maybe_mean_std(x: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(x, (int, float)):
        return float(x), 0.0
    if isinstance(x, dict) and isinstance(x.get("mean"), (int, float)):
        mean = float(x["mean"])
        std = float(x.get("std", 0.0)) if isinstance(x.get("std", 0.0), (int, float)) else None
        return mean, std
    return None, None


def _extract_metrics_from_split(split: Any) -> Dict[str, Optional[float]]:
    if not isinstance(split, dict):
        return {}
    out: Dict[str, Optional[float]] = {}
    for k in ["top1_acc", "top5_acc", "spearman_mean", "top1_regret_mean", "mse", "loss"]:
        if k in split and isinstance(split[k], (int, float)):
            out[k] = float(split[k])
    return out


def _flatten_config(cfg: Any) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    keep = [
        "model",
        "loss",
        "loss2",
        "loss2_weight",
        "temperature",
        "ranknet_pairs",
        "train_size",
        "val_size",
        "test_size",
        "standardize_x",
        "standardize_y",
        "batch_size",
        "num_epochs",
        "lr",
        "weight_decay",
        "wd",
        "dropout",
        "model_dim",
        "layers",
        "heads",
        "ff_dim",
        "head_mlp_layers",
        "reps_path",
    ]
    out: Dict[str, Any] = {}
    for k in keep:
        if k in cfg:
            out[k] = cfg[k]
    return out


def _normalize_wd(row: Dict[str, Any]) -> None:
    if "wd" not in row and "weight_decay" in row:
        row["wd"] = row.get("weight_decay")


def _infer_model(row: Dict[str, Any]) -> Optional[str]:
    model = row.get("model")
    if isinstance(model, str) and model:
        return model

    run_dir = str(row.get("run_dir") or "")

    if "/st_" in run_dir or run_dir.startswith("st_") or "set_transformer" in run_dir:
        return "set_transformer"
    if "lr_wd_sweep" in run_dir or "sweep_bs" in run_dir:
        return "linear"

    # Heuristics based on config keys.
    if "lrs" in row and "wds" in row:
        return "linear"
    if ("heads" in row and "ff_dim" in row) or ("heads" in row and "model_dim" in row and "layers" in row):
        return "set_transformer"

    return None


def _infer_loss(row: Dict[str, Any]) -> Optional[str]:
    loss = row.get("loss")
    if isinstance(loss, str) and loss:
        return loss
    run_dir = str(row.get("run_dir") or "")
    if "reg_" in run_dir or "regression" in run_dir:
        return "mse"
    return None


def _load_input_dim_from_reps(reps_path: str) -> Optional[int]:
    try:
        import os

        os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
        import torch

        reps = torch.load(reps_path, weights_only=False, map_location="cpu")
        X = reps.get("X_resid")
        if X is None:
            return None
        return int(X.shape[-1])
    except Exception:
        return None


def _param_count_set_transformer(
    input_dim: int, model_dim: int, layers: int, ff_dim: int, head_mlp_layers: int
) -> int:
    d = int(model_dim)
    L = int(layers)
    f = int(ff_dim)
    total = 0

    if int(input_dim) != d:
        total += d * int(input_dim) + d

    per_layer = (3 * d * d + 3 * d) + (d * d + d) + (d * f + f) + (f * d + d) + (4 * d)
    total += L * per_layer

    if int(head_mlp_layers) == 0:
        total += d + 1
    elif int(head_mlp_layers) == 2:
        total += (d * d + d) + (d + 1)
    return int(total)


def _param_count_deepset(input_dim: int, model_dim: int, ff_dim: int, head_mlp_layers: int) -> int:
    d = int(model_dim)
    f = int(ff_dim)
    total = 0
    if int(input_dim) != d:
        total += d * int(input_dim) + d
    total += (d * f + f) + (f * d + d)
    if int(head_mlp_layers) == 0:
        total += 2 * d + 1
    elif int(head_mlp_layers) == 2:
        total += (2 * d * d + d) + (d + 1)
    return int(total)


def _compute_total_params(row: Dict[str, Any], reps_dim_cache: Dict[str, int]) -> Optional[int]:
    reps_path = row.get("reps_path")
    if not isinstance(reps_path, str) or not reps_path:
        return None
    if reps_path not in reps_dim_cache:
        d_in = _load_input_dim_from_reps(reps_path)
        if d_in is None:
            return None
        reps_dim_cache[reps_path] = int(d_in)
    input_dim = int(reps_dim_cache[reps_path])

    model = row.get("model")
    if not isinstance(model, str) or not model:
        return None

    if model == "set_transformer":
        if not all(isinstance(row.get(k), (int, float)) for k in ["model_dim", "layers", "ff_dim"]):
            return None
        head_layers = int(row.get("head_mlp_layers") or 0)
        return _param_count_set_transformer(
            input_dim=input_dim,
            model_dim=int(row["model_dim"]),
            layers=int(row["layers"]),
            ff_dim=int(row["ff_dim"]),
            head_mlp_layers=head_layers,
        )
    if model == "deepset":
        if not all(isinstance(row.get(k), (int, float)) for k in ["model_dim", "ff_dim"]):
            return None
        head_layers = int(row.get("head_mlp_layers") or 0)
        return _param_count_deepset(
            input_dim=input_dim, model_dim=int(row["model_dim"]), ff_dim=int(row["ff_dim"]), head_mlp_layers=head_layers
        )
    if model == "linear":
        return int(input_dim + 1)
    return None


def parse_results_json(path: Path, root: Path, relative_paths: bool) -> List[Dict[str, Any]]:
    d = json.loads(path.read_text())
    parent_abs = path.parent.as_posix()
    parent_rel = ""
    try:
        parent_rel = path.parent.relative_to(root).as_posix()
    except Exception:
        parent_rel = parent_abs
    parent = parent_rel if relative_paths else parent_abs

    # 1) Multiseed runner schema: {config, per_seed_best, all_runs, aggregate}
    if isinstance(d, dict) and "aggregate" in d and "per_seed_best" in d:
        agg = d.get("aggregate", {})
        top1_mean, top1_std = _maybe_mean_std(agg.get("test_top1_acc"))
        top5_mean, top5_std = _maybe_mean_std(agg.get("test_top5_acc"))
        spearman_mean, spearman_std = _maybe_mean_std(agg.get("test_spearman_mean"))
        regret_mean, regret_std = _maybe_mean_std(agg.get("test_top1_regret_mean"))
        loss_mean, loss_std = _maybe_mean_std(agg.get("test_loss") or agg.get("test_mse"))
        num_seeds = agg.get("num_seeds")

        row: Dict[str, Any] = {
            "run_dir": parent,
            "run_dir_rel": parent_rel,
            "schema": "multiseed",
            "num_seeds": int(num_seeds) if isinstance(num_seeds, int) else None,
            "test_top1_mean": top1_mean,
            "test_top1_std": top1_std,
            "test_top5_mean": top5_mean,
            "test_top5_std": top5_std,
            "test_spearman_mean": spearman_mean,
            "test_spearman_std": spearman_std,
            "test_regret_mean": regret_mean,
            "test_regret_std": regret_std,
            "test_loss_mean": loss_mean,
            "test_loss_std": loss_std,
        }
        row.update(_flatten_config(d.get("config")))
        _normalize_wd(row)
        row["model"] = _infer_model(row) or row.get("model")
        row["loss"] = _infer_loss(row) or row.get("loss")
        return [row]

    # 2) Arch sweep schema: {config, records:[{... train/val/test ...}]}
    if isinstance(d, dict) and "records" in d and isinstance(d["records"], list):
        base_cfg = _flatten_config(d.get("config"))
        out_rows: List[Dict[str, Any]] = []
        for i, rec in enumerate(d["records"]):
            if not isinstance(rec, dict):
                continue
            train = _extract_metrics_from_split(rec.get("train"))
            val = _extract_metrics_from_split(rec.get("val"))
            test = _extract_metrics_from_split(rec.get("test"))
            row: Dict[str, Any] = {
                "run_dir": parent,
                "run_dir_rel": parent_rel,
                "schema": "records",
                "record_idx": i,
                "test_top1_mean": test.get("top1_acc"),
                "test_top5_mean": test.get("top5_acc"),
                "test_spearman_mean": test.get("spearman_mean"),
                "test_regret_mean": test.get("top1_regret_mean"),
                "test_loss_mean": test.get("loss") or test.get("mse"),
                "val_top1": val.get("top1_acc"),
                "val_loss": val.get("loss") or val.get("mse"),
                "train_top1": train.get("top1_acc"),
                "train_loss": train.get("loss") or train.get("mse"),
            }
            # hyperparams are stored at top-level of each record
            for k in ["model_dim", "layers", "heads", "ff_dim", "dropout", "lr", "wd", "weight_decay"]:
                if k in rec:
                    row[k] = rec[k]
            row.update(base_cfg)
            _normalize_wd(row)
            row["model"] = _infer_model(row) or row.get("model")
            row["loss"] = _infer_loss(row) or row.get("loss")
            out_rows.append(row)
        return out_rows

    # 3) Grid sweep schema: {config, grid:{\"lr__wd\": {train/val/test}}}
    if isinstance(d, dict) and "grid" in d and isinstance(d["grid"], dict):
        base_cfg = _flatten_config(d.get("config"))
        out_rows: List[Dict[str, Any]] = []
        for key, rec in d["grid"].items():
            if not isinstance(rec, dict):
                continue
            train = _extract_metrics_from_split(rec.get("train"))
            val = _extract_metrics_from_split(rec.get("val"))
            test = _extract_metrics_from_split(rec.get("test"))

            lr = wd = None
            if isinstance(key, str) and "__" in key:
                a, b = key.split("__", 1)
                try:
                    lr = float(a)
                    wd = float(b)
                except Exception:
                    lr = wd = None

            row: Dict[str, Any] = {
                "run_dir": parent,
                "run_dir_rel": parent_rel,
                "schema": "grid",
                "grid_key": key,
                "lr": lr,
                "wd": wd,
                "test_top1_mean": test.get("top1_acc"),
                "test_top5_mean": test.get("top5_acc"),
                "test_spearman_mean": test.get("spearman_mean"),
                "test_regret_mean": test.get("top1_regret_mean"),
                "test_loss_mean": test.get("loss") or test.get("mse"),
                "val_top1": val.get("top1_acc"),
                "val_loss": val.get("loss") or val.get("mse"),
                "train_top1": train.get("top1_acc"),
                "train_loss": train.get("loss") or train.get("mse"),
            }
            row.update(base_cfg)
            _normalize_wd(row)
            row["model"] = _infer_model(row) or row.get("model")
            row["loss"] = _infer_loss(row) or row.get("loss")
            out_rows.append(row)
        return out_rows

    # 4) Legacy table schema: {config, rows, summary}
    if isinstance(d, dict) and "rows" in d and isinstance(d["rows"], list):
        base_cfg = _flatten_config(d.get("config"))
        out_rows: List[Dict[str, Any]] = []
        for i, r in enumerate(d["rows"]):
            if not isinstance(r, dict):
                continue
            row: Dict[str, Any] = {"run_dir": parent, "run_dir_rel": parent_rel, "schema": "rows", "row_idx": i}
            row.update(r)
            row.update(base_cfg)
            _normalize_wd(row)
            row["model"] = _infer_model(row) or row.get("model")
            row["loss"] = _infer_loss(row) or row.get("loss")
            out_rows.append(row)
        return out_rows

    return [{"run_dir": parent, "run_dir_rel": parent_rel, "schema": "unknown"}]


def find_results(root: Path, max_depth: int) -> List[Path]:
    # results.json appears at varying depths; limit depth to keep runtime small.
    out: List[Path] = []
    for p in root.rglob("results.json"):
        try:
            rel = p.relative_to(root)
        except Exception:
            continue
        if len(rel.parts) > max_depth + 1:
            continue
        out.append(p)
    return sorted(out)


def match_patterns(path: str, include: List[str], exclude: List[str]) -> bool:
    if include and not any(s in path for s in include):
        return False
    if exclude and any(s in path for s in exclude):
        return False
    return True


def to_markdown_table(rows: List[Dict[str, Any]], cols: List[str], topk: int) -> str:
    precisions = {"dropout": 2}

    def fmt(col: str, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            prec = int(precisions.get(col, 4))
            return f"{v:.{prec}f}"
        return str(v)

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows[:topk]:
        lines.append("| " + " | ".join(fmt(c, r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    include = [s.strip() for s in str(args.include_patterns).split(",") if s.strip()]
    exclude = [s.strip() for s in str(args.exclude_patterns).split(",") if s.strip()]
    schema_allow = [s.strip() for s in str(args.schema).split(",") if s.strip()]

    rows: List[Dict[str, Any]] = []
    for p in find_results(root, max_depth=int(args.max_depth)):
        if not match_patterns(p.as_posix(), include, exclude):
            continue
        try:
            rows.extend(parse_results_json(p, root=root, relative_paths=bool(args.relative_paths)))
        except Exception as e:
            rows.append({"run_dir": p.parent.as_posix(), "schema": "error", "error": str(e)})

    if schema_allow:
        rows = [r for r in rows if r.get("schema") in set(schema_allow)]
    if int(args.min_num_seeds) > 0:
        rows = [r for r in rows if (r.get("num_seeds") is None or int(r["num_seeds"]) >= int(args.min_num_seeds))]

    # Prefer ranking by test_top1_mean if present, else fall back to val_top1.
    def sort_key(r: Dict[str, Any]) -> Tuple[float, float]:
        t = r.get("test_top1_mean")
        v = r.get("val_top1")
        return (
            float(t) if isinstance(t, (int, float)) else -1.0,
            float(v) if isinstance(v, (int, float)) else -1.0,
        )

    rows_sorted = sorted(rows, key=sort_key, reverse=True)

    reps_dim_cache: Dict[str, int] = {}
    for r in rows_sorted:
        r["model"] = _infer_model(r) or r.get("model")
        r["loss"] = _infer_loss(r) or r.get("loss")
        r["total_params"] = _compute_total_params(r, reps_dim_cache=reps_dim_cache)

    # Choose a stable set of columns (extra keys will still be in CSV if we include them).
    preferred_cols = [
        "run_dir",
        "run_dir_rel",
        "schema",
        "record_idx",
        "grid_key",
        "num_seeds",
        "model",
        "loss",
        "loss2",
        "loss2_weight",
        "temperature",
        "model_dim",
        "layers",
        "heads",
        "ff_dim",
        "dropout",
        "head_mlp_layers",
        "lr",
        "wd",
        "total_params",
        "test_top1_mean",
        "test_top1_std",
        "test_top5_mean",
        "test_top5_std",
        "test_spearman_mean",
        "test_regret_mean",
        "test_loss_mean",
        "val_top1",
        "val_loss",
    ]
    # Add any other keys at end for CSV completeness.
    all_keys = set()
    for r in rows_sorted:
        all_keys.update(r.keys())
    cols_csv = preferred_cols + [k for k in sorted(all_keys) if k not in set(preferred_cols)]

    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols_csv)
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)

    md_cols = [
        "run_dir",
        "schema",
        "num_seeds",
        "model",
        "loss",
        "model_dim",
        "layers",
        "heads",
        "ff_dim",
        "dropout",
        "lr",
        "wd",
        "total_params",
        "test_top1_mean",
        "test_top1_std",
        "test_top5_mean",
    ]
    out_md.write_text(to_markdown_table(rows_sorted, cols=md_cols, topk=int(args.topk_md)))

    print(f"Wrote {out_csv} ({len(rows_sorted)} rows)")
    print(f"Wrote {out_md} (top {int(args.topk_md)})")


if __name__ == "__main__":
    main()
