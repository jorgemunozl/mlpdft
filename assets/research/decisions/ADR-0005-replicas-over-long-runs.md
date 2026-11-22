---
status: accepted
date: 2026-11-20
revised: 2026-11-21
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

> **Revised 2026-11-21** with the owner's approval. The following are
> withdrawn:
> - "same thermostat; only the velocity seed changes";
> - the sizing of 12 × 2,000 F-doped and 20 × 1,600 interface steps;
> - "bulk needs no new data".
>
> Replicas are now bulk LiF runs under a corrected thermostat
> ([ADR-0007](ADR-0007-bulk-trained-shift-ladder.md),
> [ADR-0008](ADR-0008-40ry-labels-and-md-protocol.md)). Original text: commit
> `402298a`.

# ADR-0005: Independent replicas, not longer single trajectories

## Context

A single trajectory cannot give independent calibration and test frames, and
the existing runs decorrelate slowly. The NVT runs use Andersen at every
step, and the variable-cell runs oscillate coherently
([methods/autocorrelation.md](../methods/autocorrelation.md)). A longer run
repeats one history. Replicas give independent histories and a clean
replica-level split.

Measured cost per MD step on one 48-core node is 22 s for bulk LiF, 66 s for
F-doped Li and 246 s for the interface.

## Decision

- **Bulk LiF, 300 K, 40 Ry:**
  - two thermostat pilots (SVR and Andersen, `nraise = 100`), then five
    production replicas with the winner;
  - 3,000 MD steps each;
  - every run starts from the ideal lattice with independent Gaussian
    displacements (σ = 0.06 Å);
  - the winner's six runs split three for calibration and three for test; the
    other pilot is a spare.
- **F-doped Li (optional):** two replicas with the same protocol, used only
  as a shift set.
- **Interface:** no new data.
- **Splits are by replica,** never by time block within one trajectory.

Stages, decision rules and costs:
[`datasets/shift-ladder-40ry/PLAN.md`](../datasets/shift-ladder-40ry/PLAN.md).

## Evidence

[`methods/autocorrelation.md`](../methods/autocorrelation.md) and the cost per
step measured from the existing logs (plan, §2).

## Consequences

- **Cost:** about 5.6 node-days, or 10 with the F-doped replicas.
- **In distribution:** confidence intervals come from a bootstrap over
  replicas, and no temporal leakage exists between training, calibration and
  test.
- **Seeds:** pw.x exposes no seed for its random numbers, so replicas differ
  only through their starting displacements. Their independence is checked by
  correlating forces across replicas.
