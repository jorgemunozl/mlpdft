---
status: proposed
date: 2026-11-27
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

# ADR-0009: Fix the decision rules of design C before any result

## Context

Every earlier experiment stated its conclusions after seeing its numbers. The
2026-11-25 review found that this is where they went wrong:

- exp-02 called snapshots the "best calibrator" from a rank correlation, with
  no calibration metric computed;
- exp-02's p = 0.002 held only for one choice of bootstrap block length;
- exp-01 and exp-02 read inversions on tail blocks of training trajectories as
  out-of-distribution (OOD) behaviour;
- exp-03 MHC-4D had pre-registered gates, but its write-up and ADR-0001 used
  the run as evidence although it failed its own calibration gate (H3).

Design C ([ADR-0007](ADR-0007-bulk-trained-shift-ladder.md)) produces many
numbers: 7 arms, 4 label rungs, 3 seed sets, 1 in-distribution test set and 4
shift sets, each with accuracy, ranking, calibration and shift-detection
metrics. Without rules fixed in advance, some comparison will look significant
by chance. No design-C result exists yet: the baseline run has not started and
the bulk replicas are still being computed.

## Decision

The rules below are fixed before any design-C number is seen. Changing one
after a result is seen requires a new revision of this record, and the result
it touches is then labelled exploratory.

### Terms

- **Arms with an uncertainty:** committee (the reference), snapshots,
  frozen-body multi-head committee (MHC), shallow ensemble trained with a
  Gaussian negative log-likelihood (shallow-NLL), and last-layer prediction
  rigidity (LLPR). The frozen probe and the zero-shot model are accuracy
  controls only.
- **ID test:** the three bulk LiF test replicas. **Calibration set:** the three
  calibration replicas. **Shift sets:** state, chemistry (F-doped Li, Li
  metal) and structure (LiF/Li film). `test_timeblock` and the monitor slice
  are never reported as in-distribution performance.
- **Interval:** the interval defined in the Inference section. A comparison
  between two arms uses the paired version, on the same frames.
- **Primary family:** the 8 comparisons of the 4 uncertainty arms against the
  committee, on the 2 primary metrics, on the ID test, at 1,000 labels.
  Everything else is secondary.

### Rules

1. **Run validity.** A run enters any comparison only if:
   - the restored model is the last epoch of its fixed budget, checked in the
     log;
   - it passes the convergence check (to be fixed with checklist item 5);
   - no member, head or σ is non-finite, and no two members predict identical
     forces;
   - calibration constants come only from the calibration set, and no test or
     shift frame is used to fit anything;
   - every arm is evaluated on identical frames.

   A run that fails is repaired and rerun. It is never reported as evidence
   about a method.
2. **Accuracy.** Metric: force root-mean-square error (RMSE) on the ID test,
   as the ratio of an arm to the committee. Three verdicts:
   - comparable: the whole interval of the ratio is at or below 1.25;
   - worse: the whole interval is above 1.25;
   - inconclusive: otherwise.

   The same rule reports zero-shot against committee (value of fine-tuning),
   probe against committee (cost of freezing) and MHC against probe (cost of
   splitting the labels).
3. **Ranking.** Primary metric: per-frame Spearman ρ between the frame's RMS σ
   and its force RMSE, on the ID test. Per-atom ρ is secondary. An arm ranks
   errors if the lower bound of its ρ interval is above 0. Against the
   committee, on the paired difference Δρ: better if the lower bound is above
   0, worse if the upper bound is below 0, not separated otherwise. The same
   rule is reported on each shift set.
4. **Calibration.** α is fitted by Gaussian NLL on the calibration set and
   applied unchanged to every other set. Primary metric: miscalibration area on
   the ID test after α. Secondary: NLL, 1σ and 2σ coverage, and the
   split-conformal 90 % coverage.
   - Usable as an error bar only if the calibrated NLL is lower than the NLL of
     a constant σ equal to the calibration-set force RMSE, and the conformal
     90 % coverage on the ID test lies between 0.85 and 0.95. An arm that
     fails may still rank errors; it is then reported as not calibrated.
   - Against the committee: the paired difference in miscalibration area, with
     the verdicts of rule 3.
   - Under shift: the change in miscalibration area and the conformal coverage
     on each shift set, reported with intervals and no verdict.
5. **Shift detection.** Area under the receiver operating characteristic curve
   (AUROC) of per-frame σ, ID test against each shift set. An arm detects a
   shift if the lower bound of its AUROC interval is above 0.5.
6. **Cost.** GPU-hours for training and evaluation per arm, rung and seed set,
   measured. An arm is called cheaper only if it uses at most 50 % of the
   committee's GPU-hours and is neither worse in accuracy (rule 2) nor worse in
   ranking (rule 3).
7. **Seeds.** Every verdict uses the estimate pooled over the 3 seed sets. A
   verdict other than "inconclusive" or "not separated" also requires the
   point estimate to have the same sign, or for rule 2 the same side of 1.25,
   in all 3 seed sets. Otherwise it is reported as inconsistent across seeds.
8. **Label rungs.** Metrics against labels (250, 500, 1,000 and 1,536) are
   reported for every arm. A trend is stated only if it has the same direction
   in all 3 seed sets.
9. **Multiplicity.** The primary family is corrected with the Holm method at a
   family-wise level of 0.05. Secondary results are reported with unadjusted
   95 % intervals and labelled exploratory.
10. **Leakage and the interim baseline.** Before the replicas exist, the
    baseline run is a pipeline check, and its numbers are not interpreted.
    `test_timeblock` is used only for the leakage comparison: the same metrics
    on `test_timeblock` and on the ID test.
11. **Negative results.** If no arm separates from the committee in the
    primary family, the paper reports that. No new sweep, recipe change or
    tuning on the test or shift sets follows from it.

### Inference

The interval treats two sources of variation as random: the data (which
trajectory) and the training (which seed set).

1. **Cells.** Each metric is computed per cell: one data unit for one seed
   set. On the ID test a data unit is a test replica, so there are 3 × 3 = 9
   cells. A paired comparison uses the per-cell difference between two arms on
   the same frames. Ratios, such as the accuracy ratio of rule 2, are taken on
   the log scale and transformed back.
2. **Point estimate.** The mean over cells. A metric such as ρ is therefore the
   mean of within-replica values, not the value on the pooled frames, so
   differences between replicas cannot create a correlation.
3. **Interval.** A two-way decomposition of the cell values into data units
   (a = 3) and seed sets (b = 3), without replication. With mean squares MS_D
   for data units and MS_S for seed sets, the variance of the mean is
   estimated as (MS_D + MS_S) / (a b). The interval is a t interval with
   Satterthwaite degrees of freedom
   (MS_D + MS_S)² / (MS_D² / (a − 1) + MS_S² / (b − 1)). The p-value for the
   Holm correction comes from the same t statistic.
4. **Shift sets.** A shift set is one trajectory. Its data units are
   non-overlapping time blocks at least as long as that set's force-envelope
   lag, measured with `estimate_qe_correlation.py --forces`. With at least 3
   such blocks the same two-way interval applies (blocks × seed sets). With
   fewer, the interval uses the 3 seed sets only (a t interval, 2 degrees of
   freedom) and is reported as conditional on that trajectory.
5. **AUROC.** A cell is one test replica against the whole shift set, for one
   seed set. The interval is therefore conditional on the shift trajectory.
6. **Calibration constants.** α and the conformal multiplier are fitted per
   seed set on the 3 calibration replicas pooled, then applied to every cell
   of that seed set. Their own sampling error is not propagated.
7. **Not used for verdicts:** the percentile block bootstrap over replicas
   that `uq_metrics.summarize` reports. In the simulation below it covers the
   true value in 26–72 % of cases. It stays in the outputs as a descriptive
   spread.

Coverage of a nominal 95 % interval on a simulated 3 × 3 grid of cell values
(5,000 draws per row; `src/mlpdft/utils/simulate_interval_coverage.py`). Each
row sets the standard deviation of the replica, seed and cell effects:

| Scenario (replica, seed, cell) | Two-way, (MS_D + MS_S) | t over 9 cells | t over replica means | Bootstrap over 3 replicas |
|---|---:|---:|---:|---:|
| Replica-dominated (1, 0.2, 0.3) | 0.948 | 0.684 | 0.949 | 0.663 |
| Seed-dominated (0.2, 1, 0.3) | 0.947 | 0.683 | 0.620 | 0.261 |
| Both (1, 1, 0.3) | 0.969 | 0.706 | 0.910 | 0.641 |
| Cell noise (0.3, 0.3, 1) | 0.988 | 0.920 | 0.942 | 0.723 |
| Both, heavy tails (Student t, 3 degrees of freedom) | 0.976 | 0.726 | 0.901 | 0.560 |

The chosen interval is the only one near nominal in every row. The simulation
works on cell values, so frame-level effects within a cell are folded into the
cell term.

### Not included

- The retrospective active-learning loop (charter question 3). It gets its own
  pre-registration in `experiments/exp-04-active-learning/` before it runs.
- Energy metrics, pending checklist item 8.
- Ensemble-diversity diagnostics. They are reported and discussed, not used as
  gates (ADR-0007, revision of 2026-11-27, on the design-C branch).

## Evidence

- Review of exp-01 to exp-03 (2026-11-25), project artifacts
  `reviews/past-experiments-review.md`, findings 4–7 and 9.
- [`exp-03-mhc/DESIGN.md`](../experiments/exp-03-mhc/DESIGN.md) §6, the gates
  this record generalises (the 1.25 accuracy margin and the 50 % cost bar come
  from there).
- The coverage simulation above.

## Consequences

- With three replicas and three seed sets the intervals are wide (about 2
  standard deviations of the cell effects either side), so many verdicts will
  be "inconclusive" or "not separated". That is the honest result of this
  design, and the paper says so.
- The paper's claims are limited to the primary family and the rules above;
  any other pattern is exploratory.
- The evaluator's verdicts need a post-processing step across the seed-set
  runs, since each run holds one seed set.
- The cost of being wrong in the other direction is small: a rule that turns
  out too strict only costs a weaker claim, never a false one.
