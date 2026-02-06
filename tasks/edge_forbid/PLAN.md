# Edge What-if: Forbid an Optimal-Tour Edge

This experiment labels **edge fragility**: for each TSP instance we compute the **optimal** tour with Concorde, then for each edge on that optimal tour we re-solve the TSP while **forbidding that edge** (implemented as inflating `d(i,j)=d(j,i)=BIG_M` in an explicit TSPLIB distance matrix).

We then probe policy representations to predict which tour edge is most harmful to remove/forbid.

All generated datasets / artifacts live under `tasks/edge_forbid/data/` and are git-ignored.

## Pipeline (implemented)

1) **Collect edge-forbid dataset** (sharded): `collect/collect_dataset.py`
2) **Merge + summarize**: `collect/merge_shards.py`, `collect/summarize_dataset.py`
3) **Validate invariants**: `collect/validate_dataset.py` (`forbid_length >= base_length`)
4) **Extract per-edge representations** aligned to the optimal tour edges: `core/reps/extract_edge_reps.py`
5) **Train probes** (listwise / CE / ranking): reuse `core/probe/train_probes.py`
6) **Baselines** (no policy reps): `baselines/run_2opt_baseline.py`

## Data format

Merged dataset: `tasks/edge_forbid/data/processed/<dataset_id>/dataset.pt`

Key tensors (for `B` instances, `n` nodes, and `n` tour-edges):
- `locs`: `[B, n, 2]` float32
- `base_tour`: `[B, n]` int64 (0-indexed tour order)
- `base_length`: `[B]` float32 (integer CEIL_2D length)
- `forbid_length`: `[B, n]` float32 (integer CEIL_2D length)
- `delta_length_pct`: `[B, n]` float32 (`100*(forbid-base)/base`)
- `valid_base`: `[B]` bool
- `valid_forbid`: `[B, n]` bool

Extracted reps: `tasks/edge_forbid/data/processed/<dataset_id>/probe_reps.pt`
- `X_resid`: `[(B*n), d_edge]` edge features built from per-node activations
- `y`: `[(B*n), 2]` with columns `[delta_length_pct, delta_time_pct]`
- `instance_id`, `node_id`: `[(B*n)]` where `node_id` indexes the edge position in `base_tour`
- `valid`: `[(B*n)]`

## Running

Collect only:

```bash
bash tasks/edge_forbid/collect_dataset.sh runs/<run_name> 50 4
```

Full pipeline (collect → validate → extract → train):

```bash
bash tasks/edge_forbid/long_run.sh runs/<run_name> 200 4
```

Probe training dynamics across checkpoints (writes `summary.csv` + `dynamics.png` under `tasks/edge_forbid/tmp/`):

```bash
python tasks/edge_forbid/dynamics/run_probe_dynamics.py \
  --run_dir runs/<run_name> \
  --data_dir tasks/edge_forbid/data/processed/<dataset_id> \
  --epoch_step 10
```

Useful knobs (env vars):
- `CONCORDE_TIMEOUT_SEC=60`
- `LOG_EVERY_SEC=15`
- `FORBID_COST=10000000`
- `MAX_EDGES=10` (debug: only solve the first K tour edges per instance)
- `PARALLEL_SHARDS=24` (run up to this many shards concurrently)
- `ACTIVATION_KEY=encoder_output` or `ACTIVATION_KEYS=encoder_layer_0,...`
- `PROBE_OBJECTIVE=soft_ce` with `SOFT_CE_TAU=2.0` (or `best_node_ce`, `pairwise_rank`)

## Baselines (paper-friendly)

### 2-opt repair score (no policy)

Goal: approximate “how harmful is forbidding this **tour edge**” without re-solving with Concorde.

Given the **optimal** tour edges `(A[i],B[i])` and the underlying distance matrix `d(·,·)`, define for each tour edge `i`
the cheapest single 2-opt move that removes it:

- Choose another (non-adjacent) tour edge `(A[j],B[j])`.
- Remove both tour edges and reconnect using the cheaper reconnection:
  - `Δ1 = d(Ai,Aj) + d(Bi,Bj) - d(Ai,Bi) - d(Aj,Bj)`
  - `Δ2 = d(Ai,Bj) + d(Bi,Aj) - d(Ai,Bi) - d(Aj,Bj)`
- Score edge `i` by `s_i = min_j min(Δ1,Δ2)` (clamped to `>=0`).

We then predict the “worst” edge as `argmax_i s_i` and evaluate against the Concorde label `argmax_i delta_length_pct[i]`
using the same metrics as probes (`top1_acc`, `top5_acc`, Spearman, regret).

Run it on a processed dataset:

```bash
python tasks/edge_forbid/baselines/run_2opt_baseline.py \
  --data_dir tasks/edge_forbid/data/processed/<dataset_id>
```
