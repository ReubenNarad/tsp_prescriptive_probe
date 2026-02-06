# What-if: Node Removal Impact Probing (Tour Length)

This directory implements the end-to-end “what-if” experiment: for each TSP instance and node `i`, label how much the **optimal** tour length decreases when removing node `i`, then probe policy internals to predict the best node to remove.

All generated datasets / artifacts live under `tasks/node_removal/data/` and are intentionally git-ignored.

## Pipeline (implemented)

1) **Collect Concorde what-if dataset** (sharded): `collect/collect_dataset.py`  
2) **Merge + summarize**: `collect/merge_shards.py`, `collect/summarize_dataset.py`  
3) **Validate invariants** (monotonicity): `collect/validate_dataset.py`  
4) **Extract per-node representations** from the policy: `core/reps/extract_node_reps.py`  
5) **Train probes**: `core/probe/train_probes.py`  

Bash entrypoints:
- `collect_dataset.sh`: collect → merge → summarize
- `long_run.sh`: collect → validate → extract → train

## Correctness note: why we saw “removing a node increases length”

Removing a node cannot increase the optimal tour length, but we previously observed negatives because we were mixing:

- Concorde’s TSPLIB distance model (integer edge weights), and
- a floating-point Euclidean length computation.

Fix: the collector now writes TSPLIB with `EDGE_WEIGHT_TYPE: CEIL_2D` and `compute_tour_length()` mirrors the exact scaling/rounding/ceiling used by TSPLIB, so `L_base - L_minus_i >= 0` holds up to exact integer arithmetic.

`collect/validate_dataset.py` enforces:
- `base_length >= minus_length` for all valid base/minus pairs
- `delta_length_pct >= 0` for all valid base/minus pairs

## Data format

Merged dataset: `tasks/node_removal/data/processed/<dataset_id>/dataset.pt`

Key tensors (shapes for `B` instances, `n` nodes):
- `locs`: `[B, n, 2]` float32
- `base_length`: `[B]` float32
- `minus_length`: `[B, n]` float32
- `delta_length_pct`: `[B, n]` float32  (`100*(base-minus)/base`)
- `valid_base`: `[B]` bool
- `valid_minus`: `[B, n]` bool

Timing fields are also stored (`base_time_wall`, `minus_time_wall`, …) but the main probe target is `delta_length_pct`.

Extracted reps: `tasks/node_removal/data/processed/<dataset_id>/probe_reps.pt`
- `X_resid`: `[(B*n), d]` (policy activations, flattened)
- `y`: `[(B*n), 2]` with columns `[delta_length_pct, delta_time_pct]`
- `instance_id`, `node_id`: `[(B*n)]`
- `valid`: `[(B*n)]` (valid base/minus pair)
- `meta`: run/activation metadata

## Running

Collect only:

```bash
bash tasks/node_removal/collect_dataset.sh runs/<run_name> 200 10
```

Full pipeline (collect → validate → extract → train):

```bash
bash tasks/node_removal/long_run.sh runs/<run_name> 1500 10
```

Useful knobs (env vars):
- `CONCORDE_TIMEOUT_SEC=60` (default)
- `LOG_EVERY_SEC=15` (progress logging during collection)
- `MAX_REMOVED_NODES=5` (debug: only solve first K removals per instance)
- `ACTIVATION_KEY=encoder_output` or `ACTIVATION_KEYS=encoder_layer_0,encoder_layer_1,...`
- `PROBE_TARGET=length` (default), `PROBE_OBJECTIVE=regression|best_node_ce`, `PROBE_MODEL=linear|mlp`
- `STANDARDIZE_X=1` (standardize features on train split)

Note: `collect_dataset.sh` currently runs shards sequentially; true parallelism is “run shard_idx in separate processes”.

## Multi-layer activations

You can probe multiple encoder layers at once by concatenating activations:

```bash
ACTIVATION_KEYS=encoder_layer_0,encoder_layer_1,encoder_layer_2,encoder_layer_3,encoder_layer_4 \
  bash tasks/node_removal/long_run.sh runs/<run_name> 200 1
```

## Results (current snapshot)

Dataset: `TSP100_uniform_02-06_12:17:31__n1500__seed0__overnight_n1500_s10` (`n=100`, `B=1500`, valid base/minus = 100%)

Regression objective (`PROBE_OBJECTIVE=regression`, `PROBE_TARGET=length`, linear):
- Residual probe (encoder_output): test `r2≈0.21`, spearman≈0.33, top1 regret≈0.57

Best-node classification objective (`PROBE_OBJECTIVE=best_node_ce`, linear, `STANDARDIZE_X=1`):
- Residual probe: test top-1 acc≈0.37, top-5 acc≈0.65 (random baseline at `n=100`: top-1=0.01, top-5=0.05)
