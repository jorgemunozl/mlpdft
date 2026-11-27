---
status: active
date: 2026-11-21
phase: 2
owner: jorge
---

# Status

Current state of the UQ + active-learning project. Update this file when a
decision is taken or a gate changes; leave history to the ADRs and run folders.

## Settled

| Topic | Decision | Record |
|---|---|---|
| Study design | Train on bulk LiF; calibrate and test on independent bulk replicas; evaluate along a graded shift (≈ 162 K bulk → F-doped Li, Li metal → LiF/Li film); LLPR and NLL shallow-ensemble baselines | [ADR-0007](decisions/ADR-0007-bulk-trained-shift-ladder.md) |
| DFT settings and MD protocol | 40/320 Ry labels only; new MD with `nraise = 100`, thermostat chosen by a pilot, ideal-lattice starts; repository pseudopotentials pinned by checksum and verified by a preflight that must reproduce the old 40 Ry data | [ADR-0008](decisions/ADR-0008-40ry-labels-and-md-protocol.md) (rev. 2026-11-21) |
| Multi-head architecture | Four randomly initialised heads, frozen MACE-OMAT body, disjoint bulk LiF data (250 frames per head at 1,000 labels), force-only loss | [ADR-0001](decisions/ADR-0001-four-head-disjoint.md) (rev. 2026-11-21) |
| Role of MHC | Fixed-budget arm; not an acquisition model inside the AL loop | [ADR-0002](decisions/ADR-0002-mhc-outside-al-loop.md) |
| Frame spacing | Fixed stride of 10 steps for training; calibration and test from independent replicas; τ reported, never used to pick the stride | [ADR-0003](decisions/ADR-0003-qe-decorrelation-stride.md) (rev. 2026-11-21) |
| Validation split | No validation-based model selection; a small unreported monitor slice only | [ADR-0004](decisions/ADR-0004-monitor-not-validation.md) |
| New data | 2 thermostat pilots + 5 bulk LiF replicas (3,000 steps each), split by replica; 2 optional F-doped replicas; no interface DFT | [ADR-0005](decisions/ADR-0005-replicas-over-long-runs.md) (rev. 2026-11-21) |
| Interface | Evaluation-only structure-shift stress test | [ADR-0006](decisions/ADR-0006-interface-in-scope.md) (rev. 2026-11-21) |

## Measured so far

- **Correlation review (2026-11-21):** the estimator is correct, but none of
  the old source runs meets its assumptions.
  - Bulk at 30 Ry is a barely damped cell oscillation that starts at
    −406 kbar.
  - F-doped Li and the interface drift under Andersen collisions at every step
    (QE's default `nraise = 1`).
  - Document: [`methods/autocorrelation.md`](methods/autocorrelation.md).
- **Basis-set stress:** the same bulk LiF cell is at −406 kbar at 30 Ry and
  −35 kbar at 40 Ry. All 20–30 Ry runs are therefore excluded
  ([ADR-0008](decisions/ADR-0008-40ry-labels-and-md-protocol.md)).
- **MHC-4D at 1,000 labels** (retired interim dataset, 30 Ry labels; fails its
  own H3 calibration rule):
  - 33.25 meV/Å test force RMSE;
  - within-group Spearman 0.589 (bulk) / 0.106 (F-doped).

  Not comparable with the new design. Run:
  [`exp-03-mhc/runs/2026-11-06-mhc4d`](experiments/exp-03-mhc/runs/2026-11-06-mhc4d/README.md).

## Open

| Question | Why it is open | Next action |
|---|---|---|
| Which thermostat decorrelates bulk LiF faster at `nraise = 100`? | Stage 1 pilot not run yet | Submit stage 1; apply the decision rule in the [plan](datasets/shift-ladder-40ry/PLAN.md) |
| Is 40 Ry converged enough for forces? | No cutoff or k-point check exists | Submit stage 2; accept if the force RMSE vs 80 Ry is ≤ 10 meV/Å |
| Is the frozen body or the head split the bottleneck? | No committee or frozen probe trained yet | Train both on the 40 Ry bulk pool |
| Does MHC pass its pre-registered gate? | The gate compares against a committee that does not exist yet | Evaluate at 1,000 labels once the committee lands |
| Does uncertainty improve acquisition? | Not run on the new design | Retrospective loop with a pool that includes held-out shift-set frames |

## Next

1. Submit stage 1 (thermostat pilot) and stage 2 (cutoff and k-point check):
   [`assets/quantum_expresso/campaign-2026-11/`](../quantum_expresso/campaign-2026-11/README.md).
2. Meanwhile, re-extract full trajectories from the 40 Ry logs and build the
   training pool and shift sets.
3. Pick the thermostat from the pilots, then generate and submit stage 3
   (5 bulk replicas) and, optionally, stage 4 (2 F-doped replicas).
4. Implement the LLPR and NLL shallow-ensemble baselines on the frozen body.
5. Train all arms on the bulk pool (≥ 3 seeds per rung), then evaluate in
   distribution and along the shift.
