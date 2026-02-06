#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _add_repo_to_path() -> None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


def _as_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract per-edge policy representations aligned to an edge-what-if dataset (tour-edge forbids).",
    )
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing merged dataset.pt")
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Policy run dir. If omitted, uses dataset meta['run_dir'] when present.",
    )
    p.add_argument(
        "--activation_key",
        type=str,
        default="encoder_output",
        help="Activation key to extract (e.g., encoder_output, encoder_layer_0). Ignored if --activation_keys is set.",
    )
    p.add_argument(
        "--activation_keys",
        type=str,
        default=None,
        help="Comma-separated list of activation keys to concatenate along feature dim (overrides --activation_key).",
    )
    p.add_argument("--batch_size", type=int, default=16, help="Number of instances per forward pass.")
    p.add_argument("--device", type=str, default=None, help="Device string, e.g. cuda, cpu.")
    p.add_argument("--random_init", action="store_true", help="Reinitialize policy weights after loading.")
    p.add_argument("--random_init_seed", type=int, default=0, help="Seed for --random_init.")
    p.add_argument("--out_path", type=str, default=None, help="Output .pt path (default: <data_dir>/probe_reps.pt)")
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional explicit policy checkpoint path (.ckpt). If set, overrides --checkpoint_epoch.",
    )
    p.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="Optional policy checkpoint epoch number (loads runs/<run>/checkpoints/checkpoint_epoch_<N>.ckpt).",
    )
    p.add_argument(
        "--resid_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Storage dtype for X_resid (float16 saves disk; training will cast back to float32).",
    )
    return p


def _load_dataset(data_dir: Path) -> Dict:
    dataset_path = data_dir / "dataset.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {dataset_path}")
    return torch.load(dataset_path, weights_only=False)


def _resolve_run_dir(ds: Dict, run_dir_arg: Optional[str]) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")
        return run_dir

    meta = ds.get("meta", {})
    run_dir_from_meta = meta.get("run_dir") if isinstance(meta, dict) else None
    if not run_dir_from_meta:
        raise ValueError("--run_dir not provided and dataset meta['run_dir'] missing")
    run_dir = Path(run_dir_from_meta).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir from dataset meta not found: {run_dir}")
    return run_dir


def _activation_to_tensor(activation, key: str) -> torch.Tensor:
    if torch.is_tensor(activation):
        return activation
    if isinstance(activation, tuple):
        if len(activation) == 2 and torch.is_tensor(activation[0]):
            return activation[0]
        raise TypeError(f"Activation '{key}' is a tuple of unsupported shape/types: {type(activation)} len={len(activation)}")
    raise TypeError(f"Activation '{key}' has unsupported type: {type(activation)}")


def _gather_nodes(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x: [B,n,d], idx: [B,n] -> out: [B,n,d]"""
    if x.ndim != 3:
        raise ValueError(f"x must be [B,n,d], got {tuple(x.shape)}")
    if idx.ndim != 2:
        raise ValueError(f"idx must be [B,n], got {tuple(idx.shape)}")
    B, n, d = x.shape
    if idx.shape != (B, n):
        raise ValueError(f"idx shape mismatch: expected {(B, n)}, got {tuple(idx.shape)}")
    return torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(B, n, d))


def main() -> None:
    os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
    args = build_arg_parser().parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    ds = _load_dataset(data_dir)

    locs = ds.get("locs")
    if not torch.is_tensor(locs) or locs.ndim != 3 or locs.shape[2] != 2:
        raise ValueError("dataset.pt missing 'locs' tensor with shape [B,n,2]")

    base_tour = ds.get("base_tour")
    if not torch.is_tensor(base_tour) or base_tour.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError("dataset.pt missing 'base_tour' tensor with shape [B,n]")

    valid_base = ds.get("valid_base")
    valid_forbid = ds.get("valid_forbid")
    delta_length_pct = ds.get("delta_length_pct")
    delta_time_pct = ds.get("delta_time_pct")
    if not torch.is_tensor(valid_base) or valid_base.shape != (locs.shape[0],):
        raise ValueError("dataset.pt missing 'valid_base' bool tensor [B]")
    if not torch.is_tensor(valid_forbid) or valid_forbid.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError("dataset.pt missing 'valid_forbid' bool tensor [B,n]")
    if not torch.is_tensor(delta_length_pct) or delta_length_pct.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError("dataset.pt missing 'delta_length_pct' float tensor [B,n]")
    if not torch.is_tensor(delta_time_pct) or delta_time_pct.shape != (locs.shape[0], locs.shape[1]):
        raise ValueError("dataset.pt missing 'delta_time_pct' float tensor [B,n]")

    run_dir = _resolve_run_dir(ds, args.run_dir)
    device = _as_device(args.device)

    _add_repo_to_path()
    from policy.utils import load_env_and_policy, patch_env_specs
    from policy.policy_hooked import EnhancedHookedPolicy

    ckpt_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else None
    if args.random_init:
        env_path = run_dir / "env.pkl"
        config_path = run_dir / "config.json"
        if not env_path.exists():
            raise FileNotFoundError(f"Environment pickle missing: {env_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Policy config missing: {config_path}")
        with open(env_path, "rb") as fp:
            env = pickle.load(fp)
        with open(config_path, "r") as fp:
            config = json.load(fp)
        patch_env_specs(env)
        torch.manual_seed(int(args.random_init_seed))
        policy = EnhancedHookedPolicy(
            env_name=env.name,
            embed_dim=config["embed_dim"],
            num_encoder_layers=config["n_encoder_layers"],
            num_heads=int(config.get("num_heads", 8)),
            temperature=config["temperature"],
            dropout=config.get("dropout", 0.0),
            attention_dropout=config.get("attention_dropout", 0.0),
        )
        policy.to(device)
        policy.eval()
        info = {"env": env, "config": config}
    else:
        info, policy = load_env_and_policy(
            run_dir=run_dir,
            device=device,
            checkpoint_path=ckpt_path,
            checkpoint_epoch=int(args.checkpoint_epoch) if args.checkpoint_epoch is not None else None,
        )
    env = info["env"]

    activation_keys = None
    if args.activation_keys:
        activation_keys = [k.strip() for k in str(args.activation_keys).split(",") if k.strip()]
    if not activation_keys:
        activation_keys = [str(args.activation_key)]

    resid_dtype = torch.float32 if str(args.resid_dtype) == "float32" else torch.float16

    B, n, _ = locs.shape
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")

    X_parts = []
    y_parts = []
    valid_parts = []
    inst_parts = []
    edge_parts = []
    u_parts = []
    v_parts = []

    with torch.inference_mode():
        offset = 0
        while offset < B:
            end = min(B, offset + batch_size)
            current = end - offset

            locs_batch = locs[offset:end].to(torch.float32)
            tour_batch = base_tour[offset:end].to(torch.int64)

            td = env.reset(batch_size=[int(current)]).to(device)
            if "locs" in td.keys(True, True):
                td["locs"] = locs_batch.to(device)

            if hasattr(policy, "clear_cache"):
                policy.clear_cache()
            _ = policy.encoder(td)

            act_parts = []
            for key in activation_keys:
                activation = policy.activation_cache.get(key)
                if activation is None:
                    available = sorted(list(policy.activation_cache.keys()))
                    raise KeyError(f"Activation '{key}' not found. Available: {available}")
                activation_t = _activation_to_tensor(activation, key)
                if activation_t.ndim != 3 or activation_t.shape[0] != current or activation_t.shape[1] != n:
                    raise ValueError(
                        f"Activation '{key}' expected [B,n,d] = [{current},{n},d], got {tuple(activation_t.shape)}"
                    )
                act_parts.append(activation_t)

            node_repr = torch.cat(act_parts, dim=-1) if len(act_parts) > 1 else act_parts[0]  # [B,n,d]

            u_idx = tour_batch.to(device)
            v_idx = torch.roll(tour_batch, shifts=-1, dims=1).to(device)
            u_parts.append(u_idx.detach().cpu())
            v_parts.append(v_idx.detach().cpu())

            h_u = _gather_nodes(node_repr, u_idx)
            h_v = _gather_nodes(node_repr, v_idx)
            edge_repr = torch.cat([h_u, h_v, (h_u - h_v).abs()], dim=-1)  # [B,n,3d]
            X_parts.append(edge_repr.reshape(-1, edge_repr.shape[-1]).detach().to("cpu", dtype=resid_dtype))

            pair_valid = valid_base[offset:end].unsqueeze(1) & valid_forbid[offset:end]
            y = torch.stack(
                [
                    delta_length_pct[offset:end].reshape(-1),
                    delta_time_pct[offset:end].reshape(-1),
                ],
                dim=1,
            ).to(torch.float32)
            y_parts.append(y)
            valid_parts.append(pair_valid.reshape(-1).to(torch.bool))

            inst_ids = torch.arange(offset, end, dtype=torch.int64).repeat_interleave(n)
            edge_ids = torch.arange(n, dtype=torch.int64).repeat(current)
            inst_parts.append(inst_ids)
            edge_parts.append(edge_ids)

            offset = end

    out = {
        "meta": {
            "run_dir": str(run_dir),
            "data_dir": str(data_dir),
            "activation_key": str(args.activation_key),
            "activation_keys": activation_keys,
            "checkpoint_path": str(ckpt_path) if ckpt_path is not None else None,
            "checkpoint_epoch": int(args.checkpoint_epoch) if args.checkpoint_epoch is not None else None,
            "num_instances": int(B),
            "num_loc": int(n),
            "repr_dim_node": int(node_repr.shape[-1]) if B > 0 else 0,
            "repr_dim_edge": int(X_parts[0].shape[1]) if X_parts else 0,
            "edge_feature_kind": "concat(hu,hv,abs(hu-hv))",
            "device_used": str(device),
            "resid_dtype": str(args.resid_dtype),
            "label_names": ["delta_length_pct", "delta_time_pct"],
        },
        "y_names": ["delta_length_pct", "delta_time_pct"],
        "instance_id": torch.cat(inst_parts, dim=0) if inst_parts else torch.empty((0,), dtype=torch.int64),
        "node_id": torch.cat(edge_parts, dim=0) if edge_parts else torch.empty((0,), dtype=torch.int64),
        "valid": torch.cat(valid_parts, dim=0) if valid_parts else torch.empty((0,), dtype=torch.bool),
        "y": torch.cat(y_parts, dim=0) if y_parts else torch.empty((0, 2), dtype=torch.float32),
        "edge_u": torch.cat(u_parts, dim=0).reshape(-1) if u_parts else torch.empty((0,), dtype=torch.int64),
        "edge_v": torch.cat(v_parts, dim=0).reshape(-1) if v_parts else torch.empty((0,), dtype=torch.int64),
        "X_resid": torch.cat(X_parts, dim=0) if X_parts else torch.empty((0, 0), dtype=torch.float32),
    }
    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else (data_dir / "probe_reps.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"[extract] wrote: {out_path}")


if __name__ == "__main__":
    main()
