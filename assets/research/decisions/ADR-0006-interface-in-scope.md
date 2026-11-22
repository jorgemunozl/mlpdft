---
status: accepted
date: 2026-11-20
revised: 2026-11-21
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

> **Revised 2026-11-21** with the owner's approval. The interface stays in
> scope as an evaluation-only stress test
> ([ADR-0007](ADR-0007-bulk-trained-shift-ladder.md)). The 400 / 100 / 100 /
> 100 re-blocking is retired. Original text: commit `402298a`.

# ADR-0006: The interface is in scope as a stress test

## Context

The interface was removed from the dataset during the two-environment pivot,
then restored by the project owner. The 2026-11-21 review then showed that its
only clean 40 Ry trajectory (`LIFINTERFACE_KJPAW_V2`) is still relaxing. The
energy falls by −1.47 eV per 1,000 steps after step 586, under overdamped
Andersen dynamics. `LIFINTERFACE_KJPAW_FINAL` is a copy of its steps
908–2,343.

New interface DFT costs about 246 s per MD step, so independent interface
replicas are out of budget.

## Decision

- Keep the LiF/Li film as the **structure-shift set** of the shift ladder:
  frames of `LIFINTERFACE_KJPAW_V2` after step 586, at a fixed stride.
- It is **never trained or calibrated on**. Its drift and ESS are reported
  next to every result.
- `LIFINTERFACE_KJPAW_FINAL` is excluded.
- No new interface DFT.

## Evidence

[`methods/autocorrelation.md`](../methods/autocorrelation.md) §4.2–4.3 and
[`datasets/final-3env/CURATION.md`](../datasets/final-3env/CURATION.md).

## Consequences

- Interface results are qualitative: one non-stationary trajectory, used to
  test whether uncertainty rises and calibration degrades under a structural
  shift.
- Claims must keep saying that the cell is a thin-film model, not a buried
  Li|LiF interface.
