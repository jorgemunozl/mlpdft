---
status: active
date: 2026-11-20
phase: 2
owner: jorge
---

# Status

Current state of the UQ + active-learning project. Update this file when a
decision is taken or a gate changes; leave history to the ADRs and run folders.

## Settled

| Topic | Decision | Record |
|---|---|---|
| Multi-head architecture | Four randomly initialised heads, frozen MACE-OMAT body, disjoint environment-stratified data, force-only loss | [ADR-0001](decisions/ADR-0001-four-head-disjoint.md) |
| Role of MHC | Fixed-budget arm; not an acquisition model inside the AL loop | [ADR-0002](decisions/ADR-0002-mhc-outside-al-loop.md) |
| Frame spacing | Stride = 2 × correlation time measured from the QE output, per environment | [ADR-0003](decisions/ADR-0003-qe-decorrelation-stride.md) |
| Validation split | No validation-based model selection; a small unreported monitor slice only | [ADR-0004](decisions/ADR-0004-monitor-not-validation.md) |
| New data | Independent replicas, split by replica | [ADR-0005](decisions/ADR-0005-replicas-over-long-runs.md) |
| Dataset scope | Interface is back in scope; three environments | [ADR-0006](decisions/ADR-0006-interface-in-scope.md) |

## Measured so far

- **MHC-4D at 1,000 labels:** 33.25 meV/Å test force RMSE; within-group
  Spearman 0.589 (bulk) / 0.106 (F-doped); α = 12.84; RLL = −0.394; no
  variance reduction over the best member. Run:
  [`exp-03-mhc/runs/2026-11-06-mhc4d`](experiments/exp-03-mhc/runs/2026-11-06-mhc4d/README.md).
- **Correlation times from QE:** τ = 14.6 / 108.6 / 187.4 steps for bulk,
  F-doped and interface; decorrelated counts ≈ 330 / 16 / 5. Document:
  [`methods/autocorrelation.md`](methods/autocorrelation.md).
- **Provisional three-environment build:** 1,954 / 292 / 292 / 292 frames;
  F-doped short by 70 frames. Document:
  [`datasets/final-3env/CURATION.md`](datasets/final-3env/CURATION.md).

## Open

| Question | Why it is open | Next action |
|---|---|---|
| Is the frozen body or the head split the bottleneck? | No committee or frozen probe was trained on the current split | Train the probe and the committee on the same split |
| Does MHC pass its pre-registered gate? | Gate compares against a committee that does not exist yet | Evaluate at 1,000 labels once the committee lands |
| Can F-doped support a quantitative claim? | ~16 independent configurations | Generate replicas ([ADR-0005](decisions/ADR-0005-replicas-over-long-runs.md)) |
| Is the interface trajectory usable? | −1.48 eV per 1,000 steps energy drift | Diagnose thermostat/timestep before generating more |
| Does uncertainty improve acquisition? | Only committee/snapshot are eligible for the loop; neither has run on this split | Run the loop on bulk once the committee exists |

## Next

1. Fix the interface drift and re-estimate its correlation time.
2. Generate F-doped replicas (12 × 2,000 steps) and interface replicas after
   the fix.
3. Train the committee and frozen probe on the three-environment split; this
   closes the MHC gate and the attribution question.
4. Run the retrospective AL loop on bulk (committee vs snapshot vs random).
5. Scope the paper to what the data supports; keep F-doped and interface
   qualitative until replicas exist.
