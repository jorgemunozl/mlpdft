---
status: accepted
date: 2026-11-20
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

# ADR-0005: Independent replicas, not longer single trajectories

## Context

At decorrelated spacing the current runs hold about 330 (bulk), 16 (F-doped)
and 5 (interface) independent configurations. Two of those runs also drift
(−0.10 and −1.48 eV per 1,000 steps), so their correlation times are lower
bounds. Reaching a few hundred independent F-doped frames from one trajectory
would need ~22,000 steps, and would still leave one thermostat history and
time-block splits with residual leakage.

## Decision

Generate **short independent replicas** for the data-starved environments:

- same cell, composition and thermostat; **only the velocity seed changes**;
- each replica: 300–600 steps of burn-in plus ~10τ of production;
- splits are made **by replica** (e.g. 6 train / 2 val-cal / 2 test), never by
  time block within one trajectory.

Sizing: F-doped τ ≈ 109 → 12 replicas × 2,000 steps; interface after its drift
is fixed → ~20 replicas × 1,600 steps. Bulk needs no new data.

## Evidence

[`methods/autocorrelation.md`](../methods/autocorrelation.md) and
[`datasets/final-3env/CURATION.md`](../datasets/final-3env/CURATION.md).

## Consequences

- Replica-level splits remove temporal leakage and give honest confidence
  intervals, which no single-trajectory block can.
- Replicas sample dynamics, not composition: if F-doped Li has several F
  arrangements, separate starting structures are still required.
- The interface drift must be diagnosed before its replicas are generated;
  otherwise the same non-stationarity is multiplied.
