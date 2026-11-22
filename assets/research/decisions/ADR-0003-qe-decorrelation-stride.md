---
status: accepted
date: 2026-11-20
revised: 2026-11-21
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

> **Revised 2026-11-21** with the owner's approval. The rule
> `stride = 2 × max(tau_E, tau_V)` and its values 30 / 218 / 375 are
> withdrawn: none of the source trajectories meets the assumptions that rule
> needs ([methods/autocorrelation.md](../methods/autocorrelation.md)). Original
> text: commit `402298a`.

# ADR-0003: Fixed stride for training, independent replicas for calibration and test

## Context

Every extxyz frame carries a `stride` parsed from its file name (3, 3, 1 for
bulk, F-doped and interface). That number is the cadence chosen when the
trajectory was extracted; it has no relation to the correlation time.

The first version of this record set the stride to twice the integrated
autocorrelation time measured from the QE output. The 2026-11-21 review found
that the estimator is correct but its inputs are not usable:
- bulk is a coherent cell oscillation, whose cancelling ACF lobes make τ look
  short;
- F-doped Li and the interface drift under overdamped Andersen dynamics.

So τ cannot set the stride, and no single trajectory can supply independent
calibration and test frames.

## Decision

- **Training:** frames at a **fixed stride of 10 MD steps** after burn-in.
  Redundancy is accepted: it costs storage, not correctness.
- **Calibration and test, in distribution:** frames come from **independent
  replicas, split by replica**
  ([ADR-0005](ADR-0005-replicas-over-long-runs.md)). No time-block split
  inside one trajectory is used for a calibrated or tested set.
- **Shift sets:** a fixed stride that gives 200–500 frames per set.
- **τ is a diagnostic only.** τ from `estimate_qe_correlation.py`, for energy,
  volume and forces, together with its failed checks, is reported for every
  set but never chooses the stride. Every table shows frames and ESS side by
  side.

## Evidence

[`methods/autocorrelation.md`](../methods/autocorrelation.md) (2026-11-21)
and [`datasets/shift-ladder-40ry/PLAN.md`](../datasets/shift-ladder-40ry/PLAN.md).

## Consequences

- No split may call its frame count a sample count.
- Confidence intervals come from a bootstrap over replicas (in distribution).
  For single-trajectory shift sets they come from a time-block bootstrap with
  blocks at least as long as the force envelope lag.
- The earlier strides (30 / 218 / 375) and counts (~330 / 16 / 5) must not be
  cited as decorrelation results.
