---
status: accepted
date: 2026-11-20
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

# ADR-0003: Frame stride from the QE correlation time

## Context

Every extxyz frame carries a `stride` parsed from its file name (3, 3, 1 for
bulk, F-doped and interface). That number is the cadence chosen when the
trajectory was extracted; it has no relation to the correlation time. The
dataset builder conditioned on it and then treated frame counts as sample
counts, which overstates the information by 10–375×.

## Decision

Estimate the integrated autocorrelation time **from the QE output**, one value
per environment, and set:

```
stride = 2 × max(tau_energy, tau_volume)     # linearly detrended
```

with Sokal automatic windowing and a cap of `n/4`. The values are **30 / 218 /
375 MD steps** for bulk, F-doped and interface. Splits are capped so that
`frames × stride` fits each time block, and every split reports an independent
count next to its frame count.

## Evidence

[`methods/autocorrelation.md`](../methods/autocorrelation.md):
τ = 14.6 / 108.6 / 187.4 steps, giving about **330 / 16 / 5** independent
configurations. Tool: `src/mlpdft/utils/estimate_qe_correlation.py`.

## Consequences

- The 800 / 800 / 400 design is not supported by these trajectories.
- Dense training frames may be kept — redundancy costs storage, not correctness
  — but no split may call its frame count a sample count.
- The interface and F-doped numbers rest on drifting runs; their τ is a lower
  bound and must be re-measured after the drift is fixed.
- Energy is a proxy for the label quantity; a force-based τ is still owed.
