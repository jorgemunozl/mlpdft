---
status: active
date: 2026-11-21
phase: 2
owner: jorge
supersedes: ../archive/2026-11-20-autocorrelation.md
superseded_by: —
---

# Autocorrelation and the frame stride

**Date:** 2026-11-21. Corrects the 2026-11-20 version, archived at
[`archive/2026-11-20-autocorrelation.md`](../archive/2026-11-20-autocorrelation.md).
**Scope:** can the correlation time measured from the Quantum ESPRESSO output
set the frame stride and the independent-sample count of the UQ dataset?
**Tools:** `src/mlpdft/utils/estimate_qe_correlation.py` (now with `--forces`
and assumption checks), `src/mlpdft/utils/diagnose_qe_drift.py`.
**Data:** `LIF64_KJPAW_NPT_V3`, `LIWITHF_V3`, `LIFINTERFACE_KJPAW_V2` (QE
`pw.x` output, one converged SCF block per ionic step, Δt = 0.4838 fs).

## Verdict

- The estimator is implemented correctly. It reproduces every number in the
  2026-11-20 version.
- It is **not valid on these three trajectories**. The integrated
  autocorrelation time is a decorrelation time only for a series that is
  stationary, mixing (not dominated by a coherent oscillation) and much longer
  than τ. Each run fails at least one of these conditions (§4).
- So the strides 30 / 218 / 375 and the counts ~330 / 16 / 5 are not
  decorrelation results. As counts of independent configurations, all three
  are optimistic.
- Use τ as a **diagnostic**, not to set the stride. Report the effective
  sample size (ESS) as an upper bound, with the checks it failed. The tool now
  prints `stride_reliable: false` and the reasons for all three runs.

## 1. The question

Every frame carries a `stride` taken from the source file name
(`..._3_3333.extxyz` → 3). It records the cadence chosen when the trajectory
was dumped to extxyz. It is bookkeeping, not a physical spacing. The question
is what spacing makes frames approximately independent, and how many such
frames each run holds.

## 2. Method

1. Parse one total energy per converged SCF block, the cell volume per step
   and, with `--forces`, the total force table per step.
2. Drop the burn-in: 100 steps (bulk), 2,400 (F-doped), 586 (interface).
3. Compute the normalised ACF with a zero-padded FFT,
   `rho(k) = sum_t (x_t - mean)(x_{t+k} - mean) / ((n-k) * var)`.
   For forces, remove each atom's mean force and pool over atoms and
   Cartesian components.
4. Integrated τ with Sokal windowing: `tau(W) = 0.5 + sum_{k=1..W} rho(k)`,
   taking the smallest `W >= 5 tau(W)`, capped at `n/4`.
5. Repeat on a linearly detrended copy.
6. Stride = `ceil(2 tau)`, which is the statistical inefficiency
   `g = 1 + 2 sum rho`. Independent count = `n / stride`.

Checks added on 2026-11-21. Any failure sets `stride_reliable: false`:

| Check | Test | Why it matters |
|---|---|---|
| Stationarity | `abs(slope) * n > 0.5 * std(E)` | τ is not defined for a drifting series |
| Window | Sokal window reached `n/4` | τ is only a lower bound |
| Mixing | > 50 % of the detrended variance within ±20 % of one frequency (period ≤ n/10) | positive and negative ACF lobes cancel, so τ looks short while frames stay correlated |
| Run length | `n / tau < 50`; Madras–Sokal relative error `sqrt(2(2W+1)/n)` reported | τ is too uncertain to set a stride |
| Label quantity | force τ > energy/volume τ, or force \|ρ\| > 0.1 beyond 10 τ_F | the energy proxy understates how correlated the labels are |

## 3. What holds

- The FFT is zero-padded, so there is no circular wrap.
- The Sokal window uses `c = 5` in the `tau = 0.5 + sum rho` convention. That
  truncates about e⁻⁵ of an exponential ACF.
- Stride `2 tau` equals the statistical inefficiency `g`, so `n / stride` is
  the standard ESS.
- On a synthetic AR(1) series the tool recovers the analytic τ = 9.5 within
  the test tolerance (±1.5) and raises no warning. Oscillating and drifting series are flagged
  (`tests/test_estimate_qe_correlation.py`).

## 4. Where it fails

### 4.1 Bulk LiF: an undamped cell oscillation, not decorrelation

- The variable-cell run starts at **P = −406 kbar** with
  `ecutwfc = 30 Ry`. The cell collapses and then oscillates for the whole run.
  V swings between 366 and 543 Å³ (mean 453 Å³) with a 183-step period
  (≈ 89 fs), and P swings between −402 and +536 kbar.
- The oscillation is barely damped: the volume range narrows only from
  366–543 Å³ in the first fifth of the run to 381–531 Å³ in the last.
- 98 % of the volume variance sits at the 183-step period. 98 % of the energy
  variance sits at half that period (91 steps).
- τ_V = 14.6 and τ_E = 7.0 are short because positive and negative lobes
  cancel. The frames are not decorrelated:
  - ρ_V is +0.51 at lag 30, −1.00 at lag 90 and +0.81 at lag 200;
  - the detrended ρ_E is +0.97 at lag 90.
- Forces show the same pattern. τ_F = 12.0 by the same cancellation, but
  |ρ_F| stays above 0.1 until **lag 2,227**.
- τ_V is still the right variance-of-the-mean factor for the volume itself.
  It is not evidence that frames 30 steps apart are independent
  configurations.
- *Inference, not tested:* the −406 kbar is mostly basis-set (Pulay) stress
  from the 30 Ry cutoff. The interface at 40 Ry starts at −8 kbar.
  Recomputing one frame at 40 Ry and higher would settle it.

### 4.2 F-doped Li and interface: not stationary

- The energies drift by −0.10 and −1.47 eV per 1,000 steps, and both runs are
  flagged. τ is not defined for these series.
- The 2026-11-20 version still recommended the detrended τ. A linear detrend
  does not remove an exponential relaxation. The interface energy falls by
  0.85, 0.71, 0.28 and 0.24 eV between successive fifths of its usable window:
  still relaxing, only more slowly.

### 4.3 F-doped Li and interface: overdamped dynamics from the thermostat

- Both runs use the Andersen thermostat with a collision frequency of
  **1/timestep**. The output reports 54 (F-doped) and 122 (interface)
  collisions at every step, so every velocity is redrawn from the
  Maxwell–Boltzmann distribution every 0.48 fs.
- The short-lag MSD exponent is 1.00–1.01 for Li and F in both runs, instead
  of about 2 for inertial motion. Atoms take a random walk with steps of about
  `v_rms Δt ≈ 0.005 Å`.
- This is overdamped dynamics with an effective friction `γ = 2/Δt ≈ 4 fs⁻¹`.
  An overdamped harmonic mode relaxes in `γ/ω²`: about 10 vibrational periods
  for a 100 fs mode, and longer for slower modes such as the structural
  rearrangement at the interface.
- So the long τ values largely measure the thermostat, not the material:
  energy 109 / 187, forces ≥ 415 / ≥ 205 for F-doped / interface.
- Temperature is held (301.7 ± 33.2 K and 300.1 ± 22.9 K). SCF convergence is
  tight (estimated accuracy ≤ 3.4 × 10⁻⁶ eV). The drift is physical
  relaxation slowed down by this dynamics, not a numerical fault.

### 4.4 Runs too short for their τ

| Run | n / τ | Madras–Sokal relative error | 2026-11-20 formula `sqrt(2 tau / n)` |
|---|---:|---:|---:|
| Bulk (volume) | 679 | 17 % | 5 % |
| F-doped (energy, detrended) | 33 | 78 % | 25 % |
| Interface (energy, detrended) | 9 | 100 % (window capped) | 46 % |

Reliable estimates need `n ≳ 50 tau`. The earlier error formula understated
the uncertainty by 2–3×.

### 4.5 Energy is a proxy

- Forces are the label quantity. They decorrelate more slowly than energy for
  F-doped (τ_F ≥ 415 vs 109) and the interface (≥ 205 vs 187), and keep a
  coherent oscillation in bulk.
- For UQ statistics, the series that matters is the per-frame nonconformity
  score (error divided by predicted σ) on the calibration and test frames.
  Once a model exists, its autocorrelation, not the energy's, should size the
  block bootstrap.

## 5. Corrected results

| Run | Steps used | τ_E raw / detrended | τ_V | τ_F | Failed checks | 2026-11-20 stride → count | Reading |
|---|---:|---:|---:|---:|---|---|---|
| Bulk LiF, variable cell | 9,900 | 7.0 / 6.9 | 14.6 | 12.0, \|ρ\| > 0.1 until 2,227 | oscillation (E, V); force coherence | 30 → ~330 | not a decorrelation time |
| F-doped Li, NVT, Andersen 1/Δt | 3,576 | 545.9* / 108.6 | — | ≥ 415* | drift; n < 50 τ; force τ > energy τ; window cap | 218 → ~16 | upper bound at best |
| Interface, NVT, Andersen 1/Δt | 1,758 | 290.1* / 187.4* | — | ≥ 205* | drift; window cap; n < 50 τ; force τ > energy τ | 375 → ~5 | upper bound at best |

\* window-capped at `n/4`: a lower bound on τ.

## 6. What it implies

1. **Do not set the stride from τ.** A fixed extraction stride for training is
   acceptable, since redundancy costs storage, not correctness. Every split
   must still report its frame count with the ESS caveat.
2. **Calibration and test statistics.**
   - Bulk: bootstrap blocks must span at least the oscillation period
     (≥ ~200 steps).
   - F-doped and interface: each split is effectively a single block, so
     their coverage numbers are qualitative.
3. **New data must change the dynamics, not only the seed.** Replicas run
   with Andersen at 1/Δt would reproduce the same overdamped sampling. Use
   `ion_temperature = 'svr'` with `nraise ≈ 100` (≈ 48 fs coupling), or
   Andersen with `nraise ≥ 100`. Then re-measure with `--forces`.
4. **Bulk needs new data too:** a damped or fixed-cell run at the relaxed
   volume, with a converged cutoff.
5. **QE settings differ across environments.** Bulk and F-doped use
   `ecutwfc = 30 Ry` and the interface uses 40 Ry (all with
   `ecutrho = 320 Ry`). Labels pooled into one model should come from one
   setting.

## 7. Effect on the decision records

ADR-0003 (stride from the QE τ) and ADR-0005 (replicas) cite the 2026-11-20
numbers. Under the decision rules they stay in force until a superseding
record is written. What no longer holds:

- [ADR-0003](../decisions/ADR-0003-qe-decorrelation-stride.md): the rule
  `stride = 2 × max(tau_E, tau_V)` and the values 30 / 218 / 375 (§4).
- [ADR-0005](../decisions/ADR-0005-replicas-over-long-runs.md):
  - "same … thermostat; only the velocity seed changes" (§4.3);
  - the replica sizing derived from τ ≈ 109 (§4.4);
  - "Bulk needs no new data" (§4.1).

## 8. Limitations

- The force ACF is pooled over atoms and components. Per-species or
  per-region (interface) values may differ.
- The oscillation check is a heuristic: one spectral band holding more than
  half the variance.
- Each run is a single realization. The F-doped and interface values are lower
  bounds with 80–100 % relative errors.
- The Pulay-stress explanation for the bulk pressure is an inference.

## 9. Reproduction

```bash
# Correlation times, assumption checks, force tau
PYTHONPATH=src python3 src/mlpdft/utils/estimate_qe_correlation.py \
  --out /path/to/LIF64_KJPAW_NPT_V3.out --burn-in 100 --label bulk_LiF --forces --json
PYTHONPATH=src python3 src/mlpdft/utils/estimate_qe_correlation.py \
  --out /path/to/LIWITHF_V3.out --burn-in 2400 --label F_doped_Li --forces --json
PYTHONPATH=src python3 src/mlpdft/utils/estimate_qe_correlation.py \
  --out /path/to/LIFINTERFACE_KJPAW_V2.out --burn-in 586 --label interface --forces --json

# Thermostat, temperature, SCF, drift by window, MSD exponent (fixed-cell runs only)
PYTHONPATH=src python3 src/mlpdft/utils/diagnose_qe_drift.py \
  --out /path/to/LIWITHF_V3.out --burn-in 2400 --label F_doped_Li
PYTHONPATH=src python3 src/mlpdft/utils/diagnose_qe_drift.py \
  --out /path/to/LIFINTERFACE_KJPAW_V2.out --burn-in 586 --label interface

# Bulk cutoff, starting pressure, volume and pressure range after burn-in
OUT=/path/to/LIF64_KJPAW_NPT_V3.out
grep -m1 "kinetic-energy cutoff" $OUT
grep -m1 "P=" $OUT
grep "new unit-cell volume =.*Ang" $OUT | awk 'NR>100{v=$8; if(min==""||v<min)min=v; if(v>max)max=v} END{print min, max}'
grep "P=" $OUT | awk 'NR>100{p=$6; if(min==""||p<min)min=p; if(p>max)max=p} END{print min, max}'

# Tests for the checks
python3 -m unittest discover -s tests -p 'test_estimate*'
```

## 10. Next steps

1. Decide whether to supersede ADR-0003 with a fixed training stride and τ
   kept as a diagnostic.
2. Recompute one bulk frame at 40 Ry and above to test the Pulay-stress
   inference. Then rerun bulk with damped or fixed-cell dynamics at the
   relaxed volume.
3. Rerun F-doped and the interface with SVR (or Andersen `nraise ≥ 100`),
   re-measure τ with `--forces`, and size the replicas from that.
4. Once a model exists, compute the ACF of per-frame nonconformity scores and
   size the block bootstrap from it.
