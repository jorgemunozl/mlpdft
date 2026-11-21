# MHC-4D A40 run — `mhc4d_a40_001`

**Status:** completed 2026-11-07 (~01:40 UTC); 90/90 epochs, smoke gate passed.
**Code:** `b7ac48b` (`orx/four-head-disjoint-frozen-body-mhc`).
**Run command:** `ORX_RUN_ID=mhc4d_a40_001 bash scripts/run_mhc_4d.sh`
**Full outputs:** project artifacts directory `mhc_4d_2026-11-06/` (model
weights, checkpoints and per-head extxyz are not committed here).

## Configuration

- Frozen MACE-OMAT-medium body, four randomly initialised readout heads.
- Disjoint, environment-stratified split: 250 frames/head (125 bulk LiF +
  125 F-doped Li); 1,000 unique training frames.
- Force-only loss (`energy_weight=0`), batch 16, lr 1e-3, 90 epochs.
- Dataset: `jorgemunozl/li-f-uq-interim-two-environments`, revision
  `c2699e8e52889c18f6d29f657eb996ac3f78524f`.
- Model: `jorgemunozl/mace_omat_medium`, revision
  `d64b6faf564bd682ba1e9ca337408f6925aa908d`.

## Test results (200 frames)

| Metric | Overall | Bulk LiF | F-doped Li |
|---|---:|---:|---:|
| Force RMSE (meV/Å) | 33.25 | 39.75 | 23.30 |
| Frame Spearman | 0.837 | 0.589 | 0.106 |
| Frame Pearson | 0.911 | 0.645 | 0.156 |
| 1σ coverage | 0.782 | 0.826 | 0.730 |
| RLL | −0.394 | −0.263 | −0.571 |

`α_force = 12.84`; 2σ coverage 0.937. Per-head test force RMSE:
33.43 / 34.49 / 32.72 / 33.89 meV/Å; the ensemble (33.25) is slightly worse
than the best member (32.72), so there is no committee variance reduction.

Same-frame descriptive baselines: archived pilot **58.69 meV/Å**, zero-shot
**48.20 meV/Å**. The old committee's 8.66 meV/Å was measured on a different
split and is not a like-for-like comparison.

## Interpretation

The frozen four-head, 250-frame/head recipe improves on the failed warm-start
pilot and on zero-shot, but plateaus around 33 meV/Å, far from committee-grade
accuracy. The pooled Spearman (0.837) is mostly between-environment contrast:
within F-doped Li the ranking is essentially absent (0.106) and the
risk–coverage curve is flat. Global `α` brings 1σ coverage to 0.78, but RLL is
negative in both environments.

This is a development result. No newly trained committee or frozen probe was
evaluated on these frames, so the pre-registered H1/H4 comparisons remain open.
Full analysis: [`analysis/report.md`](analysis/report.md).

## Contents

- `analysis/` — report, figures, tidy CSVs, regeneration script and vendored
  figure style.
- `full/evaluation/uq_summary.json` — machine-readable metrics for the new MHC,
  archived pilot and zero-shot.
- `full/evaluation/*_predictions.csv` — per-frame predictions.
- `full/mhc_frozen_readouts/results/*_train.txt` — per-epoch training curve.
- `full/experiment_manifest.json`, `full/mhc_frozen_readouts/run_config.json`,
  `full/mhc_frozen_readouts/mace_config.json` — effective configuration.
- `preflight.json` — pinned input hashes and head-split check.
- `run.log` — complete setup, smoke, training and evaluation log.
- `smoke/evaluation/` — smoke-gate metrics.

## Reproduce the analysis

```bash
cd analysis
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache \
  uv run --no-project --with matplotlib --with numpy python make_analysis.py
```

The script reads `../full/evaluation/uq_summary.json`,
`../full/evaluation/frozen_mhc_predictions.csv` and
`../full/mhc_frozen_readouts/results/mhc_frozen_readouts_run-123_train.txt`,
then rewrites `data/` and `figures/`.
