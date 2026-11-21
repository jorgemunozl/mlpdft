---
status: accepted
date: 2026-11-20
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

# ADR-0006: The interface is in scope

## Context

The interface was removed from the dataset during the two-environment pivot,
then restored by the project owner. The Phase 1 audit's interface calibration
block (264 frames, stride 1) cannot supply a 100/100 validation/calibration
split once a 60-step guard is applied — only 12 frames remain on each side —
which is why the interface had looked unusable.

## Decision

Keep **three environments** — bulk LiF, F-doped Li, LiF/Li interface — and
re-block the interface's usable time range rather than shrinking its holdout.
With the usable range 586–2,343 the interface reaches 400 / 100 / 100 / 100
from existing frames.

## Evidence

[`datasets/final-3env/CURATION.md`](../datasets/final-3env/CURATION.md):
provisional build 1,954 / 292 / 292 / 292, interface complete, F-doped short by
70 frames; 0 duplicate geometries, 0 SCF warnings.

## Consequences

- The interface test split comes from the same trajectory and is **not** an
  independent or OOD benchmark; claims must say so.
- The interface run drifts by −1.48 eV per 1,000 steps; stationarity is still
  unproven and its correlation time is a lower bound.
- Re-blocking changes the split boundaries relative to the Phase 1 audit, so
  results are not comparable to the earlier two-environment build.
