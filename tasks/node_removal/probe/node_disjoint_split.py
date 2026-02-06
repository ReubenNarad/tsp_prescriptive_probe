#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train/eval a probe under a pool-node-disjoint split.\n\n"
            "We hold out a set of pool node IDs and exclude any instance containing them from training.\n"
            "Then we evaluate on instances that contain at least one held-out node.\n"
        )
    )
    p.add_argument("--reps_path", type=str, required=True, help="Path to probe_reps*.pt (must correspond to dataset).")
    p.add_argument("--dataset_path", type=str, required=True, help="Path to dataset.pt (must include pool_indices).")
    p.add_argument("--holdout_size", type=int, default=50, help="Number of pool nodes to hold out (default: 50).")
    p.add_argument("--holdout_seed", type=int, default=0, help="RNG seed for choosing held-out pool nodes.")
    p.add_argument("--train_frac", type=float, default=0.9, help="Fraction of non-heldout instances used for training.")
    p.add_argument("--objective", type=str, default="best_node_ce", choices=["best_node_ce", "soft_ce", "pairwise_rank"])
    p.add_argument("--soft_ce_tau", type=float, default=1.0, help="Temperature for --objective soft_ce.")
    p.add_argument("--pairwise_pairs_per_instance", type=int, default=128)
    p.add_argument("--pairwise_margin", type=float, default=0.0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--standardize_x", action="store_true")
    p.add_argument("--instance_standardize_x", action="store_true")
    p.add_argument("--node_standardize_x", action="store_true")
    p.add_argument("--tfm_dim", type=int, default=512)
    p.add_argument("--tfm_layers", type=int, default=2)
    p.add_argument("--tfm_heads", type=int, default=8)
    p.add_argument("--tfm_ff_mult", type=int, default=4)
    p.add_argument("--tfm_dropout", type=float, default=0.1)
    return p


def _as_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _import_train_probes():
    probe_dir = Path(__file__).resolve().parent
    if str(probe_dir) not in sys.path:
        sys.path.append(str(probe_dir))
    import train_probes  # type: ignore

    return train_probes


def _choose_holdout_nodes(pool_indices: torch.Tensor, holdout_size: int, seed: int) -> torch.Tensor:
    # Choose from nodes that actually appear in this dataset (keeps split meaningful).
    flat = pool_indices.reshape(-1)
    unique = torch.unique(flat).to(torch.int64)
    holdout_size = int(holdout_size)
    if holdout_size <= 0:
        raise ValueError("--holdout_size must be >= 1")
    if holdout_size > int(unique.numel()):
        raise ValueError(f"--holdout_size={holdout_size} > unique pool nodes in dataset ({int(unique.numel())})")
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(unique.numpy(), size=holdout_size, replace=False)
    return torch.from_numpy(chosen.astype(np.int64))


def _make_splits(
    pool_indices: torch.Tensor, holdout_nodes: torch.Tensor, train_frac: float, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train_mask, val_mask, test_mask) over instances [B]."""
    B = int(pool_indices.shape[0])
    holdout_nodes = holdout_nodes.to(torch.int64)

    contains_holdout = torch.isin(pool_indices.to(torch.int64), holdout_nodes).any(dim=1)
    test_mask = contains_holdout
    avail = torch.nonzero(~contains_holdout, as_tuple=False).squeeze(1).tolist()
    if not avail:
        raise ValueError("No trainable instances after filtering held-out nodes; reduce --holdout_size")

    rng = random.Random(int(seed))
    rng.shuffle(avail)
    train_frac = float(train_frac)
    if not (0.0 < train_frac < 1.0):
        raise ValueError("--train_frac must be in (0,1)")

    n_train = max(1, int(round(train_frac * len(avail))))
    n_train = min(n_train, len(avail) - 1) if len(avail) > 1 else n_train
    train_ids = set(avail[:n_train])
    val_ids = set(avail[n_train:])

    train_mask = torch.tensor([i in train_ids for i in range(B)], dtype=torch.bool)
    val_mask = torch.tensor([i in val_ids for i in range(B)], dtype=torch.bool)
    return train_mask, val_mask, test_mask


@torch.inference_mode()
def _conditional_test_metrics(
    model: torch.nn.Module,
    X_inst: torch.Tensor,
    y_inst: torch.Tensor,
    valid_inst: torch.Tensor,
    pool_indices: torch.Tensor,
    holdout_nodes: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Compute test accuracy conditioned on whether the true best node is held-out."""
    tp = _import_train_probes()

    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze(1)
    if int(test_idx.numel()) == 0:
        return {"num_test": 0.0}

    Xb = X_inst[test_idx].to(device)
    vb = valid_inst[test_idx].to(device)
    yt = y_inst[test_idx].to(torch.float32)
    pi = pool_indices[test_idx].to(torch.int64)

    # labels
    yt_masked = yt.clone()
    yt_masked[~vb.cpu()] = float("-inf")
    label = yt_masked.argmax(dim=1)  # [b]
    best_pool = pi.gather(1, label.unsqueeze(1)).squeeze(1)

    holdout_set = set(holdout_nodes.tolist())
    is_best_holdout = torch.tensor([int(x) in holdout_set for x in best_pool.tolist()], dtype=torch.bool)

    logits = model(Xb, key_padding_mask=(~vb)).squeeze(-1)
    logits = logits.masked_fill(~vb, float("-inf"))
    pred = logits.argmax(dim=1).detach().cpu()

    def _acc(mask: torch.Tensor) -> float:
        if int(mask.sum().item()) == 0:
            return float("nan")
        return float((pred[mask] == label[mask]).float().mean().item())

    def _top5(mask: torch.Tensor) -> float:
        if int(mask.sum().item()) == 0:
            return float("nan")
        top5 = logits.detach().cpu().topk(5, dim=1).indices
        return float((top5[mask] == label[mask].unsqueeze(1)).any(dim=1).float().mean().item())

    # Spearman mean for the full test set (for convenience)
    spearmans = []
    logits_cpu = logits.detach().cpu()
    yt_cpu = yt.detach().cpu()
    vb_cpu = vb.detach().cpu()
    for bi in range(int(test_idx.numel())):
        m = vb_cpu[bi]
        if int(m.sum().item()) < 2:
            continue
        spearmans.append(tp.spearman_corr(logits_cpu[bi][m], yt_cpu[bi][m]))
    sp = float(np.mean([s for s in spearmans if not (isinstance(s, float) and np.isnan(s))])) if spearmans else float("nan")

    return {
        "num_test": float(int(test_idx.numel())),
        "num_best_holdout": float(int(is_best_holdout.sum().item())),
        "num_best_nonholdout": float(int((~is_best_holdout).sum().item())),
        "top1_all": _acc(torch.ones_like(is_best_holdout, dtype=torch.bool)),
        "top5_all": _top5(torch.ones_like(is_best_holdout, dtype=torch.bool)),
        "top1_best_holdout": _acc(is_best_holdout),
        "top5_best_holdout": _top5(is_best_holdout),
        "top1_best_nonholdout": _acc(~is_best_holdout),
        "top5_best_nonholdout": _top5(~is_best_holdout),
        "spearman_all": sp,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    device = _as_device(args.device)
    tp = _import_train_probes()

    reps = torch.load(Path(args.reps_path).expanduser().resolve(), weights_only=False, map_location="cpu")
    ds = torch.load(Path(args.dataset_path).expanduser().resolve(), weights_only=False, map_location="cpu")

    if "pool_indices" not in ds:
        raise ValueError("dataset_path must contain pool_indices")
    pool_indices = ds["pool_indices"].to(torch.int64)
    if pool_indices.ndim != 2:
        raise ValueError(f"pool_indices must be [B,n], got {tuple(pool_indices.shape)}")

    X_inst, y_inst, valid_inst = tp._reshape_by_instance_node(
        reps["X_resid"], reps["y"], reps["valid"], reps["instance_id"], reps["node_id"]
    )
    # Use delta_length_pct by default (index 0) and squeeze target dimension.
    y_scalar = y_inst[:, :, 0].contiguous()

    holdout_nodes = _choose_holdout_nodes(pool_indices, int(args.holdout_size), int(args.holdout_seed))
    train_mask, val_mask, test_mask = _make_splits(pool_indices, holdout_nodes, float(args.train_frac), int(args.holdout_seed))

    print(
        f"[node-disjoint] B={int(pool_indices.shape[0])} N={int(pool_indices.shape[1])} | "
        f"holdout_nodes={int(holdout_nodes.numel())} | "
        f"train={int(train_mask.sum())} val={int(val_mask.sum())} test={int(test_mask.sum())}"
    )

    splits = tp.SplitMasks(train=train_mask, val=val_mask, test=test_mask)

    model, results = tp.train_best_node_ce(
        X_inst=X_inst,
        y_inst=y_scalar,
        valid_inst=valid_inst,
        splits=splits,
        objective=str(args.objective),
        soft_ce_tau=float(args.soft_ce_tau),
        pairwise_pairs_per_instance=int(args.pairwise_pairs_per_instance),
        pairwise_margin=float(args.pairwise_margin),
        model_type="transformer",
        mlp_hidden_dim=256,
        mlp_layers=1,
        mlp_dropout=0.0,
        tfm_dim=int(args.tfm_dim),
        tfm_layers=int(args.tfm_layers),
        tfm_heads=int(args.tfm_heads),
        tfm_ff_mult=int(args.tfm_ff_mult),
        tfm_dropout=float(args.tfm_dropout),
        device=device,
        batch_size_instances=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_epochs=int(args.epochs),
        l1_lambda=0.0,
        standardize_x=bool(args.standardize_x),
        instance_standardize_x=bool(args.instance_standardize_x),
        node_standardize_x=bool(args.node_standardize_x),
        seed=int(args.holdout_seed),
    )

    cond = _conditional_test_metrics(
        model=model,
        X_inst=X_inst,
        y_inst=y_scalar,
        valid_inst=valid_inst,
        pool_indices=pool_indices,
        holdout_nodes=holdout_nodes,
        test_mask=test_mask,
        device=device,
    )

    print("[node-disjoint] train metrics:", results["train"])
    print("[node-disjoint] val metrics:", results["val"])
    print("[node-disjoint] test metrics:", results["test"])
    print("[node-disjoint] conditional test:", cond)


if __name__ == "__main__":
    main()
