import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge what-if dataset shards into a single dataset.pt")
    p.add_argument("--raw_dir", type=str, required=True, help="Directory containing shard_*.pt files")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory to write dataset.pt")
    p.add_argument("--pattern", type=str, default="shard_*.pt")
    return p


def _load_shard(path: Path) -> Dict[str, Any]:
    return torch.load(path, weights_only=False)


def _sort_key(shard: Dict[str, Any], default: int) -> Tuple[int, int]:
    meta = shard.get("meta", {}) if isinstance(shard, dict) else {}
    start = meta.get("global_instance_start", default)
    end = meta.get("global_instance_end", default)
    return int(start), int(end)


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(raw_dir.glob(args.pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {raw_dir} matching {args.pattern}")

    shards: List[Dict[str, Any]] = []
    for path in shard_paths:
        shards.append(_load_shard(path))

    shards_sorted_all = sorted(
        enumerate(shards),
        key=lambda pair: _sort_key(pair[1], default=pair[0]),
    )

    def _num_instances(shard: Dict[str, Any]) -> int:
        locs = shard.get("locs")
        if torch.is_tensor(locs) and locs.ndim >= 1:
            return int(locs.shape[0])
        return 0

    shards_sorted = [(i, s) for (i, s) in shards_sorted_all if _num_instances(s) > 0]

    tensor_keys: List[str] = []
    if shards_sorted:
        first = shards_sorted[0][1]
        for k, v in first.items():
            if k == "meta":
                continue
            if torch.is_tensor(v):
                tensor_keys.append(k)

    merged: Dict[str, Any] = {}
    merged_meta: Dict[str, Any] = {
        "num_shards_found": len(shards_sorted_all),
        "num_nonempty_shards": len(shards_sorted),
        "raw_dir": str(raw_dir),
        "shards": [],
    }

    for shard_idx, shard in shards_sorted_all:
        meta = shard.get("meta", {})
        merged_meta["shards"].append(
            {
                "path": str(shard_paths[shard_idx].resolve()),
                "meta": meta,
                "num_instances": _num_instances(shard),
            }
        )

    # Concatenate tensor keys (missing keys become None and are skipped)
    for k in tensor_keys:
        parts = []
        for _, shard in shards_sorted:
            v = shard.get(k)
            if v is None:
                raise KeyError(f"Shard missing key '{k}'")
            if not torch.is_tensor(v):
                raise TypeError(f"Shard key '{k}' expected tensor, got {type(v)}")
            parts.append(v)
        merged[k] = torch.cat(parts, dim=0) if parts else torch.empty((0,))

    # Try to propagate common metadata fields when consistent.
    common_meta_fields = [
        "run_dir",
        "seed",
        "num_instances_total",
        "num_shards",
        "num_loc",
        "concorde_timeout_sec",
    ]
    for field in common_meta_fields:
        values = []
        for _, shard in shards_sorted:
            m = shard.get("meta", {})
            if field in m:
                values.append(m[field])
        if values and all(v == values[0] for v in values):
            merged_meta[field] = values[0]

    merged["meta"] = merged_meta

    out_path = out_dir / "dataset.pt"
    torch.save(merged, out_path)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as fp:
        json.dump(merged_meta, fp, indent=2)

    total_instances = int(merged[tensor_keys[0]].shape[0]) if tensor_keys else 0
    print(f"[merge] Wrote {out_path} with {total_instances} instances")
    print(f"[merge] Wrote {manifest_path}")


if __name__ == "__main__":
    main()
