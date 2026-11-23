---
status: accepted
date: 2026-11-21
revised: 2026-11-23
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

> **Revised 2026-11-23:** the first shift was described as "bulk LiF at
> ≈ 162 K". Its frames have larger forces than the 300 K run, so it is now
> described as a variable-cell, unthermostatted state shift. Nothing else
> changed.

# ADR-0007: Train on bulk LiF and evaluate along a graded shift

## Context

The correlation review
([methods/autocorrelation.md](../methods/autocorrelation.md)) showed that none
of the existing trajectories supports decorrelated calibration and test splits:
- bulk is a coherent cell oscillation at 30 Ry;
- F-doped Li and the interface drift, under overdamped Andersen dynamics.

Making F-doped Li and the interface in-distribution environments would need
about 110 node-days of new DFT, about 90 of them for the interface alone.

Recent work on uncertainty for foundation MLIPs identifies out-of-distribution
behaviour as the open question. It also uses cheap last-layer methods (LLPR,
shallow ensembles) as standard comparisons. Four designs were weighed:
- A: bulk only;
- B: bulk-trained plus a shift ladder from existing data;
- C: B plus targeted DFT;
- D: the original three-environment plan.

The project owner chose C on 2026-11-21.

## Decision

- **Training:** bulk LiF only, from `LIF64_KJPAW_FINAL` (40 Ry, NVT 300 K).
- **Calibration and test in distribution:** six new, independent bulk LiF
  replicas at 300 K (three each), split by replica
  ([ADR-0005](ADR-0005-replicas-over-long-runs.md)).
- **Evaluation-only shift ladder.** No training or calibration on these sets:
  1. thermodynamic state: bulk LiF from a variable-cell run without a
     thermostat (`LIF64_KJPAW_NPT_FINAL`; kinetic ≈ 160 K, 2 % denser, yet
     larger forces than the 300 K run);
  2. chemistry: F-doped Li (`LIWITHF_KJPAW_FINAL`, plus 2 optional replicas)
     and Li metal (`BCC_54_KJPAW_FINAL`);
  3. structure: the LiF/Li film (`LIFINTERFACE_KJPAW_V2`), as a stress test;
  4. appendix only: the boron systems.
- **Arms:**
  - independent fine-tuning committee;
  - snapshot ensemble;
  - frozen-body multi-head committee;
  - frozen probe;
  - two baselines: LLPR and a shallow ensemble trained with a Gaussian NLL
    loss.
- **Not included:**
  - training on F-doped Li or the interface;
  - new interface DFT;
  - any transport, SEI or physical-interface claim.

Data roles, stages and costs:
[`datasets/shift-ladder-40ry/PLAN.md`](../datasets/shift-ladder-40ry/PLAN.md).

## Evidence

- Correlation and drift checks of every candidate run:
  [methods/autocorrelation.md](../methods/autocorrelation.md) and the run
  survey in the plan.
- Shallow ensembles share the backbone and train only the last layer,
  jointly, with an NLL loss
  ([How to Train a Shallow Ensemble](https://www.alphaxiv.org/abs/2602.15747)).
  That is structurally close to the frozen-body multi-head arm.
- LLPR-driven active learning for foundation fine-tuning
  ([Full-data accuracy with fewer labels](https://www.alphaxiv.org/abs/2607.14486)).
- Most post-hoc uncertainty methods for MLIPs are evaluated in domain
  ([Towards OOD UQ via pre-training](https://www.alphaxiv.org/abs/2610.41801)).

## Consequences

- Cross-environment calibration is measured as degradation under shift, not
  as in-distribution calibration in each environment. The charter's questions
  are reworded accordingly.
- Every model result so far used bulk and F-doped labels at 20–30 Ry and must
  be rerun. This costs GPU time, not DFT.
- [ADR-0001](ADR-0001-four-head-disjoint.md) changes: the heads train on bulk
  only.
- [ADR-0006](ADR-0006-interface-in-scope.md) changes: the interface becomes a
  stress test.
- If the bulk replicas still decorrelate slowly, the in-distribution intervals
  stay wide. The shift results are less sensitive to this, because they
  compare sets rather than estimate coverage.
