# Run the Foundation-Model UQ Comparison

This workflow trains and compares independent MACE fine-tunes, warm snapshots,
and a multi-head bootstrap model on the same curated Li/F data. It evaluates
held-out DFT energy/force error and whether ensemble force disagreement ranks
force error. It also exports standalone models for subsequent Li-diffusion MD.

## Requirements

- Python environment synchronized with `uv sync`.
- A CUDA GPU recommended; the default runner requires CUDA.
- The `mace_omat_medium` foundation model available locally, or cluster access
  to download it.
- Network access on the first multi-head run to obtain MACE's OMAT replay
  data. Use `--skip-multihead` if replay data cannot be obtained.
- Curated extxyz files with `REF_energy` in frame info and `REF_forces` in
  arrays. Create these from the six core groups listed in
  [`research/UQ_TEST_PROTOCOL.md`](../research/UQ_TEST_PROTOCOL.md).

## Prepare Inputs

Prepare three files before submitting the job:

- `uq_train.extxyz`: training frames.
- `uq_valid.extxyz`: a block-held-out validation set for model selection.
- `uq_test.extxyz`: a locked block-held-out test set used only for final
  metrics.

Use the protocol's final LiF, Li-rich Li-F, and Li|LiF interface groups. Keep
NVT/NPT identity and group labels in `config_type`; do not randomly split
neighboring trajectory frames. The runner checks that every frame has DFT
energy and force labels. It does not parse QE output or determine thermalization
cutoffs for you.

## Launch on the Cluster

From the repository root, activate or create the project's environment:

```bash
uv sync
```

Use the site's normal GPU allocation. For an interactive allocation, keep the
job alive in `tmux`; for Slurm, place the same command in your site's GPU batch
script. Example command (replace paths and the run ID):

```bash
uv run python src/mlpdft/uq_compare.py \
  --train-file /path/to/uq_train.extxyz \
  --valid-file /path/to/uq_valid.extxyz \
  --test-file /path/to/uq_test.extxyz \
  --output-dir src/mlpdft/outputs/uq_diffusion_comparison \
  --run-id lif_uq_001
```

The default run trains three independent fine-tunes (seeds 123, 124, 125), one
100-epoch warm-restart run with three exported snapshot checkpoints, and one
multi-head run with three block-bootstrap target heads. It also evaluates the
zero-shot foundation model. To omit multi-head fine-tuning, add
`--skip-multihead`. The multi-head recipe adds up to 1,000 OMAT replay structures
by default; change with `--pt-samples` and report this extra data when comparing
training cost. The target heads share a trunk, so treat their disagreement as a
separate, potentially correlated committee method.

Do not reuse a `--run-id`: the script refuses to overwrite an existing run
folder. The run is sequential and may need a long wall-time allocation.

## Outputs

Everything is written under
`src/mlpdft/outputs/uq_diffusion_comparison/<run-id>/`:

- `independent_seed_*/`: each independent model, checkpoints, MACE test results,
  logs, and run config.
- `warm_snapshots/`: final model, checkpoints, logs, and exported
  `snapshot_models/*epoch_*.model` files.
- `multihead_bootstrap/`: shared multi-head model, logs, and one standalone
  model per target head.
- `multihead_bootstrap_data/`: the three block-bootstrap target training files.
- `evaluation/uq_summary.json`: aggregate and per-group energy/force accuracy,
  force-uncertainty/error Spearman correlation, and low-uncertainty risk/coverage.
- `evaluation/*_predictions.csv`: per-test-frame predictions, errors, and
  committee disagreement.
- `experiment_manifest.json`: inputs, seeds, run durations, and exported model
  paths.

The summary measures disagreement on held-out DFT configurations. It does not
by itself establish accurate diffusivity or activation energy.

## Li-Diffusion Validation

After reviewing the UQ results, run MACE MD for each exported committee member
and target head using the same starting structure, cell, temperatures, and
independent velocity seeds. For bulk LiF, the repository's Arrhenius utility
can be used with an explicit model and fixed-cell LiF start configuration:

```bash
uv run python src/mlpdft/utils/arrhenius.py \
  --model src/mlpdft/outputs/uq_diffusion_comparison/lif_uq_001/independent_seed_123/models/independent_seed_123.model \
  --config /path/to/fixed_cell_LIF64.extxyz \
  --temperatures 300 400 500 600 \
  --seed 123 \
  --output src/mlpdft/outputs/uq_diffusion_comparison/lif_uq_001/md/independent_seed_123_replica_123
```

Repeat for each independent model, snapshot model, and exported multi-head target
model; use multiple velocity seeds per model. The utility's default trajectory
length is only a starting point: extend it until MSD fitting windows, hop counts,
replica-to-replica diffusivity, and the Arrhenius slope are stable. Do not report
an activation energy from a single temperature or a visibly non-linear MSD
window. For interfaces, confirm the periodic geometry and interface normal;
the current Arrhenius utility reports total 3D diffusivity, not separate
interface-parallel and interface-normal coefficients.

## Package Results

```bash
tar czf uq_test_output.tar.gz \
  -C src/mlpdft/outputs/uq_diffusion_comparison lif_uq_001
```

Include the three input extxyz files and their frame manifest with the results
so each metric can be traced to the exact data split.
