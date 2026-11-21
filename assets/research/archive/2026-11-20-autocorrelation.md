---
status: superseded
date: 2026-11-20
phase: 2
owner: jorge
superseded_by: ../methods/autocorrelation.md
---

> **Archived.** Superseded by the corrected
> [`methods/autocorrelation.md`](../methods/autocorrelation.md) (2026-11-21).
> The correlation times below reproduce exactly, but all three trajectories fail
> the assumptions the method needs, so the strides and independent counts are
> not valid. Kept for provenance; do not edit.

# Autocorrelation and the frame stride

**Date:** 2026-11-20
**Scope:** how far apart two MD frames must be before they count as separate
samples, measured from the Quantum ESPRESSO output rather than from the
extracted extxyz files.
**Tool:** `src/mlpdft/utils/estimate_qe_correlation.py`
**Data:** `LIF64_KJPAW_NPT_V3`, `LIWITHF_V3`, `LIFINTERFACE_KJPAW_V2` (QE `pw.x`
output, one converged SCF block per ionic step).

## 1. The problem

Every frame in the UQ dataset carries a `stride` value taken from the source
file name (`..._3_3333.extxyz` → 3). That number records the cadence chosen
when the trajectory was dumped to extxyz. It is bookkeeping, not a physical
spacing: nothing forces it to exceed the correlation time of the trajectory.

The dataset builder conditions on it — it keeps only rows at the configured
cadence — and then treats the frame count as a sample count. Both steps are
wrong if the cadence is finer than the correlation time. The questions this
artifact answers are:

1. What is the correlation time of each source trajectory, measured at full
   resolution?
2. What spacing makes consecutive frames approximately independent?
3. How many such frames does each trajectory actually contain?

## 2. What already existed in this repository

Nothing answered those questions from the QE origin. Three related pieces
existed, each with a limitation:

| Path | What it does | Limitation |
|---|---|---|
| `src/mlpdft/utils/build_interim_two_environment.py:51` | ACF of energy and force components on source windows, first-nonpositive truncation | runs on already-extracted extxyz; no windowing correction; no drift handling |
| `src/mlpdft/utils/build_three_environment_uq_dataset.py:225` | ACF of selected frames per split | measures the *selected* series, not the trajectory; spacing is irregular |
| `src/mlpdft/utils/uq_analysis.py` | paired moving-block bootstrap for confidence intervals | block length is a user constant (`--block 20`), not derived from the data |

So the repo could *report* an ESS after the fact, but could not choose the
frame spacing, and never looked at `pw.x` output directly. This artifact and
`estimate_qe_correlation.py` fill that gap.

## 3. Method

1. **Parse the QE output.** One total energy per converged SCF block
   (`!    total energy`), one cell volume per ionic step
   (`unit-cell volume`). This is the trajectory at its native resolution —
   no extraction stride is involved.
2. **Drop the burn-in** recorded in the curation audit: 100 steps (bulk),
   2,400 (F-doped), 586 (interface).
3. **Normalized autocovariance** via FFT,
   `rho(k) = sum_t (x_t - mean)(x_{t+k} - mean) / ((n-k) * var)`.
4. **Integrated autocorrelation time** with Sokal automatic windowing:
   `tau(W) = 0.5 + sum_{k=1..W} rho(k)`, choose the smallest `W` with
   `W >= 5 * tau(W)`, capped at `n/4`. A capped window means the estimate is a
   lower bound on `tau`.
5. **Detrend and repeat.** A linear drift inflates `rho`. The tool fits and
   removes a first-order polynomial, then re-estimates `tau`. Both values are
   reported; the recommendation uses the slowest **detrended** `tau` over
   energy and volume.
6. **Drift test.** If `|slope| * n > 0.5 * std(x)`, the run is flagged as
   non-stationary.
7. **Recommended stride** = `ceil(2 * tau)`. Independent configurations =
   `usable_steps / stride`.

## 4. Results

| Run | Steps after burn-in | `tau` energy (raw / detrended) | `tau` volume | Slowest detrended | Stride = 2`tau` | Independent configs |
|---|---:|---:|---:|---:|---:|---:|
| Bulk LiF, NPT | 9,900 | 6.9 / 6.9 | 14.6 | **14.6** | **30** | ~330 |
| F-doped Li, NVT | 3,576 | 545.9 / **108.6** | — | **108.6** | **218** | ~16 |
| Interface, NVT | 1,758 | 290.1 / **187.4*** | — | **187.4*** | **375** | ~5 |

\* window-capped: the true `tau` is larger, so the count is optimistic.

Drift check:

| Run | Drift (eV / 1,000 steps) | Over the usable window | Flag |
|---|---:|---:|---|
| Bulk LiF | +0.042 | +0.42 eV vs a 3.78 eV std | ok |
| F-doped Li | −0.100 | −0.36 eV vs a 0.11 eV std | **non-stationary** |
| Interface | −1.475 | −2.59 eV vs a 0.78 eV std | **non-stationary** |

Two things stand out.

**The extraction stride is far too fine.** Relative to the decorrelation
spacing, the datasets are oversampled by about 10× (bulk, 30/3), 73× (F-doped,
218/3) and 375× (interface, 375/1). A 754-frame F-doped training split is not
754 samples; at best it is 16.

**Two trajectories drift enough to question their stationarity.** The interface
run loses 1.5 eV per 1,000 steps — about 2.6 eV across its usable window against
a 0.78 eV standard deviation. The F-doped run is milder but still flagged. For
those runs the raw `tau` is dominated by the trend, which is why detrending
changes the F-doped value by 5× (546 → 109).

## 5. What it implies for the dataset

- The design target of 800 / 800 / 400 training frames at decorrelated spacing
  is **not supported by these trajectories**. They hold roughly 330 / 16 / 5.
- Training may keep dense frames — redundancy costs storage, not correctness —
  but a dense split must report its independent count next to its frame count.
- Calibration and test must be decorrelated: at most 330 / 16 / 5 frames from
  these runs, which is too few for a 100-frame F-doped or interface holdout.
- The fix for F-doped and the interface is **independent replicas** — several
  short runs from different initial velocities — not one longer run. Replicas
  also give a clean replica-level train/test split, which no single-trajectory
  time block can provide.

Per-split ESS of the current provisional build (selected frames, irregular
spacing) makes the same point:

| Split | Bulk ESS | F-doped ESS | Interface ESS |
|---|---:|---:|---:|
| Train | 82 / 800 | 4 / 754 | 3 / 400 |
| Validation | 10 / 100 | 4 / 92 | 3 / 100 |
| Calibration | 11 / 100 | 5 / 92 | 6 / 100 |
| Test | 11 / 100 | 4 / 92 | 4 / 100 |

## 6. Limitations

- **Energy is a proxy.** The relevant observable for MLIP labels is the force
  vector, which may decorrelate on a different scale. A force-based `tau`
  should be added before the stride is frozen.
- **One realization per run.** The statistical error on `tau` is roughly
  `sqrt(2 * tau / n)`, so the F-doped and interface values carry large
  uncertainty.
- **Detrending is a diagnostic, not a correction.** It separates fluctuation
  correlation from drift; it does not make a drifting trajectory stationary.
- **Window-capped values are lower bounds**, so the independent counts are
  upper bounds.
- The step spacing is in MD ionic steps; converting to physical time needs the
  run's timestep, which the parser does not read.

## 7. Reproduction

```bash
PYTHONPATH=src python3 src/mlpdft/utils/estimate_qe_correlation.py \
  --out /path/to/LIF64_KJPAW_NPT_V3.out --burn-in 100 --label bulk_LiF --json

PYTHONPATH=src python3 src/mlpdft/utils/estimate_qe_correlation.py \
  --out /path/to/LIWITHF_V3.out --burn-in 2400 --label F_doped_Li --json

PYTHONPATH=src python3 src/mlpdft/utils/estimate_qe_correlation.py \
  --out /path/to/LIFINTERFACE_KJPAW_V2.out --burn-in 586 --label interface --json
```

Each call prints `tau` for energy and volume (raw and detrended), the ACF lags
at 0.05 and 1/e, the drift check, the recommended stride and the independent
count.

## 8. Next steps

1. Add a force-based `tau` so the stride is set by the label quantity, not by
   the energy proxy.
2. Wire the recommended stride into
   `build_three_environment_uq_dataset.py` as a per-environment cap, so each
   split reports `achieved <= floor(block_width / stride)`.
3. Replace single long trajectories with independent replicas for F-doped and
   the interface, and split train/test by replica.
4. Re-check the interface run's drift before using any of its frames.
