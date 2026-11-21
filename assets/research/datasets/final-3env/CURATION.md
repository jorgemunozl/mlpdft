---
status: active
date: 2026-11-20
owner: jorge
---

# Three-environment UQ dataset — curation build

**Date:** 2026-11-20
**Status:** provisional build. Three environments, interface restored. One gate
still fails (F-doped frame count); everything else passes. Not yet a production
dataset.

**Builder:** `src/mlpdft/utils/build_three_environment_uq_dataset.py`
**Build:** `outputs/uq_dataset_three_env_needblocks/` (gitignored, 2,830 frames)
**Supersedes:** the two-environment candidate in
[`UQ_TWO_ENVIRONMENT_CURATION.md`](../../archive/2026-11-06-two-env-dataset.md), which
excluded the interface.

## Design

| Environment | Source run | Stride (MD steps) | Burn-in (steps) | Train | Val | Cal | Test |
|---|---|---|---:|---:|---:|---:|---:|
| Bulk LiF, NPT | `LIF64_KJPAW_NPT_V3` | 3 | 100 | 800 | 100 | 100 | 100 |
| F-doped Li, NVT | `LIWITHF_V3` | 3 | 2,400 | 800 | 100 | 100 | 100 |
| LiF/Li interface, NVT | `LIFINTERFACE_KJPAW_V2` | 1 | 586 | 400 | 100 | 100 | 100 |

- One source run per environment; no mixing of thermostats or QE settings.
- Time-blocked, ordered splits (train → validation → calibration → test) with a
  **60-step guard** on every boundary. With the new re-blocking mode the guard
  is physical rather than nominal: 120 steps separate adjacent blocks.
- Selection is deterministic and evenly spaced inside each block; output
  cadence is irregular.
- The test split is drawn from the same trajectories, so it is not an
  independent-trajectory or OOD benchmark.

## Achieved

| Environment | Train | Val | Cal | Test | Shortfall |
|---|---:|---:|---:|---:|---:|
| Bulk LiF | 800 | 100 | 100 | 100 | 0 |
| F-doped Li | **754** | **92** | **92** | **92** | **70 frames** |
| Interface | 400 | 100 | 100 | 100 | 0 |
| **Total** | **1,954** | **292** | **292** | **292** | |

2,830 frames, 0 duplicate rounded geometries, 0 duplicate source frames, 0 SCF
warnings, maximum force 3.98 eV/Å, energy-offset spread 1.15e-06 eV.

## Two findings

### 1. The interface shortfall was a split-block artifact, not missing data

Under the Phase 1 audit blocks, the interface calibration block is 264 frames
(1,816–2,079) at stride 1. Removing the 60-step guard on both sides and splitting
it into validation and calibration leaves **12 frames each** — which is why a
100/100 interface holdout looked impossible.

The trajectory itself is long enough: burn-in ends at 586 and the run reaches
2,343, so 1,758 usable frames remain. Re-blocking the usable range
(`--block-mode need`) gives the interface its full 400 / 100 / 100 / 100
**without any new data**. The same mode leaves bulk complete.

### 2. F-doped Li is genuinely short

`LIWITHF_V3` ends at 5,973 steps. After the 2,400-step burn-in the usable span is
3,574 steps. The target design needs 3,780 steps at stride 3 (2,520 for training
plus 1,260 for the three holdout blocks), so the build is **206 steps short**:

- train 754/800 (−46 frames)
- validation, calibration, test 92/100 each (−8 frames each)

Adding ~1,000 MD steps (≈330 more frames at stride 3) clears the deficit with
margin. This is the only data the final build actually needs.

## Gates

| Gate | Result | Detail |
|---|---|---|
| Target frame counts | ❌ | F-doped 754/800 train, 92/100 holdout |
| SCF warnings zero, offset spread < 0.05 eV | ✅ | 0 warnings, 1.15e-06 eV |
| No frame above 5 eV/Å | ✅ | max 3.98 eV/Å |
| Temporal dependence characterised | ✅ | ACF recorded; lag limit reached in 5 splits |
| Burn-in recorded per environment | ✅ | 100 / 2,400 / 586 steps |
| Calibration separate from validation and test | ✅ | disjoint blocks, 60-step guard |

The lag limit in `F_doped_Li` validation/test and all three interface splits
except test means the correlation time is truncated, so `ESS = n / tau` is an
**upper bound** on the effective sample size. Report it as optimistic and do not
treat suggested block lengths as calibrated.

## Correlation time and the correct frame stride

> **Corrected 2026-11-21.** The τ values below reproduce, but none of the three
> runs meets the method's assumptions: bulk is a barely damped cell
> oscillation, and F-doped and the interface drift under overdamped Andersen
> dynamics. The strides and independent counts in this section are not
> decorrelation results. See
> [`methods/autocorrelation.md`](../../methods/autocorrelation.md).

Full method, results and limitations:
[`methods/autocorrelation.md`](../../methods/autocorrelation.md).

The stride in the source filenames (3 / 3 / 1) is the cadence chosen when the
extxyz files were extracted. It is **not** a decorrelation choice, and the
builder only conditions on it. The physical correlation time comes from the QE
output itself: `estimate_qe_correlation.py` reads every converged SCF block,
removes a linear drift, and estimates the integrated autocorrelation time with
Sokal automatic windowing.

| Run | Usable steps | $$\\tau_E$$ | $$\\tau_V$$ | Slowest $$\\tau$$ | Stride $$2\\tau$$ | Independent configs |
|---|---:|---:|---:|---:|---:|---:|
| Bulk LiF | 9,900 | 6.9 | 14.6 | 14.6 | **30** | ~330 |
| F-doped Li | 3,576 | 108.6 | — | 108.6 | **218** | ~16 |
| Interface | 1,758 | 187.4* | — | 187.4* | **375** | ~5 |

\* window-capped: the true value is larger, so the count is optimistic.

The dense dataset is therefore oversampled by roughly 10× (bulk), 73×
(F-doped) and 375× (interface). At a decorrelated spacing these trajectories
hold about **330 / 16 / 5** configurations, not 800 / 800 / 400.

Two consequences:

1. **Training may keep dense frames** — redundancy costs storage, not
   correctness — but every split must report its independent count, and
   calibration/test should not use more frames than the trajectory can decorrelate.
2. **F-doped Li and the interface cannot support the planned dataset from one
   run.** The defensible fix is independent replicas (several short runs), not a
   longer single trajectory, because replicas also give a clean replica-level
   train/test split.

The interface run also drifts by **−1.48 eV per 1,000 steps** (about −2.6 eV over
the usable window against a standard deviation of 0.78 eV). That is not
stationary. Treat its correlation time as a lower bound and the trajectory
itself as suspect until the drift is explained.

## Unresolved scientific caveats

- **F-doped burn-in is not verified.** The 2,400-step cutoff is provisional and
  F-doped statistics still drift through the early trajectory. Extending the
  same run adds frames but does not establish stationarity; a longer
  equilibration or an independent trajectory is the defensible fix.
- **Temporal dependence remains.** Splits are adjacent time blocks from single
  trajectories, separated by 120 steps. That is a guard, not decorrelation.
- **The test split is previously observed.** No exact geometry hash overlap with
  the earlier pilot, but temporally adjacent frames are not excluded. Final
  paper claims need a later independent dataset.
- **Frame counts are not independent sample counts.** Every table must carry the
  ESS caveat.

## Rebuild

```bash
PYTHONPATH=src python3 src/mlpdft/utils/build_three_environment_uq_dataset.py \
  --data-root /path/to/dataset \
  --frames assets/research/datasets/phase1-audit/audit/frames.csv \
  --output-dir outputs/uq_dataset_three_env_needblocks \
  --block-mode need
```

Outputs: `uq_train.extxyz`, `uq_valid.extxyz`, `uq_calibration.extxyz`,
`uq_test.extxyz`, `frame_manifest.csv`, `temporal_diagnostics.json`,
`dataset_manifest.json` (achieved versus target, shortfalls, gates).

When the extended F-doped run lands, add its frames to the audit, point
`--data-root` at it, and the same command re-derives every split and gate. The
only expected change is F-doped reaching 800 / 100 / 100 / 100.
