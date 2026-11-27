---
status: active
date: 2026-11-27
phase: 2
owner: jorge
---

# Ensemble diversity of the design-C arms

**Question.** Does each arm of design C have enough diversity for its spread to
measure its own error?

**Answer.** Not as the arms stood before 2026-11-27. Only the snapshot arm
reliably produced enough spread, and part of that spread was a convergence
trend rather than diversity. The probe and LLPR are not ensembles. Every
frozen-body arm shares one body and cannot see errors that come from it. The
changes in §5 fix the snapshot artefact and make diversity measured instead of
assumed; they do not make the committee or the frozen-body arms more diverse,
because that weakness is part of what the paper compares.

| Arm | Where its spread comes from | Evidence so far | Verdict |
|---|---|---|---|
| Committee | Bootstrap draw and data order. All members start from the same MACE-OMAT-0 weights and anneal to the same point. | exp-01 without bootstrap collapsed; exp-02 with bootstrap: median error/σ 4.2–6.4 | Probably insufficient: a local ensemble by construction |
| Snapshots | Learning-rate restarts, plus a convergence trend between cycles | exp-02: median error/σ 2.7–3.7; ensemble mean worse than the last member | Enough spread, partly the wrong kind |
| Frozen-body MHC | Random head initialisation and disjoint data quarters; nothing from the body | Pilot (identical heads) α = 51.6; MHC-4D (random heads) α = 12.8 | Insufficient scale; blind to body error |
| Frozen probe | None: one head | — | Not applicable (accuracy control, LLPR host) |
| LLPR | Not an ensemble: distance in last-layer feature space | Not run | Not applicable; blind to body error |
| Shallow NLL ensemble | Spread trained by the NLL loss to match training residuals | Not run | Scale fitted by construction in distribution; blind to body error |

MHC is the multi-head committee, LLPR last-layer prediction rigidity, NLL the
Gaussian negative log-likelihood, and α the factor that rescales σ to fit the
errors of the calibration set (α = 1 means the raw spread already has the right
size).

## 1. What "enough diversity" means here

The spread identity (literature review §13.1) says that the squared spread
equals the mean squared error of the members minus the squared error of their
mean. A spread therefore sees only what the members do differently. An error
that all members share is invisible to it.

Too little diversity shows up in four ways:

1. the spread is much smaller than the error (α ≫ 1). For reference, Beck et
   al. needed α = 1.32 for a naive committee and 2.00 for disjoint heads on
   3BPA (review §13.5);
2. the ensemble mean is no more accurate than its best member;
3. the members' errors are almost perfectly correlated;
4. along the shift ladder, the spread grows more slowly than the error.

The wrong kind of diversity shows up as members that are not exchangeable: one
member is systematically worse, so the spread contains that member's offset and
the ensemble mean is worse than the best member.

## 2. Per-arm review

Settings are those of the baseline node (1,000 labels) in
`configs/shift_ladder/node.env` on the design-C run branch.

### 2.1 Independent fine-tuning committee

- **Construction.** Three full fine-tunes of MACE-OMAT-0 medium (seeds
  123/124/125; the shift-ladder plan says four members), 51 epochs, each on a
  block bootstrap of the training pool (block 5, on by default), last epoch kept.
- **Diversity channels.** The bootstrap draw and the data order. Not the
  initialisation: with `foundation_filter_elements` true (MACE's default, which
  the committee does not override), mace-torch copies every weight, readouts
  included, from the foundation model. Not the hyperparameters or the schedule.
- **Evidence.** exp-01 had no bootstrap; its members' force RMSE was
  0.0151 / 0.0150 / 0.0150 eV/Å. exp-02 added the bootstrap. Its spread was
  still 4.2–6.4 times smaller than the error (median per frame), and its bulk
  ranking fell from ρ = 0.74 to 0.30 once a linear time trend was removed.
- **Literature.** Models fine-tuned from one checkpoint stay in one basin and
  form a local deep ensemble; a tuned snapshot ensemble can match it (review
  §14.2).
- **Verdict.** Probably insufficient, by construction. This is the committee
  practitioners actually build, so it stays as it is. The paper should present
  its weakness as a property of fine-tuning from one checkpoint, with the
  diversity measured (§4).

### 2.2 Warm-restart snapshot ensemble

- **Construction.** One fine-tune with cosine warm restarts (minimum learning
  rate 1e-6), three cycles of 17 epochs, one member at the end of each cycle.
  §5 adds a discarded burn-in cycle in front of these three.
- **Diversity channels.** Exploration of the basin after each restart, and the
  convergence trend: an early member is a less-trained model.
- **Evidence.** In exp-02 the snapshot spread was 1.2–2.8 times the committee's
  and ranked the committee's own errors better than the committee's spread did
  (review §14.3). The members were not exchangeable: energy RMSE 0.188 / 0.099 /
  0.066 meV/atom over the three cycles, and an ensemble-mean force RMSE of
  0.0101 eV/Å against 0.0082 for the last member alone.
- **Verdict.** Enough spread, partly a trend. The trend ranks errors in
  distribution, but it is not uncertainty about the model, it costs accuracy
  when the mean is used, and nothing shows that it survives a shift.
- **Adopted (§5).** Discard the first cycle as burn-in; report each member's
  error; hold the optimiser steps per cycle fixed across label rungs.

### 2.3 Frozen-body multi-head committee

- **Construction.** The foundation body frozen (`freeze = 6`); four randomly
  initialised readout heads, each trained on a disjoint quarter of the
  training set (250 frames at 1,000 labels), 60 epochs.
- **Diversity channels.** Random head initialisation and disjoint data. Nothing
  comes from the body.
- **Evidence.** The pilot's heads started identical (a mace-torch default
  copied the foundation readout into every head): α = 51.6, and the ensemble was
  no better than its average member. MHC-4D used random heads and disjoint
  quarters: α = 12.8, and the ensemble was no better than its best head. MHC-4D
  ran on 30 Ry compressed cells, so its numbers do not transfer as they stand.
- **Verdict.** The spread is an order of magnitude too small, which a fitted α
  can repair in distribution. The blindness to body error cannot be repaired:
  with the body frozen, its error is common to all heads (review §13.5).

### 2.4 Frozen probe and LLPR

The probe is one head on the frozen body, trained on all frames. It has no
spread; it is the accuracy control and the model LLPR is fitted on.

LLPR is not an ensemble. Its variance is α² jᵀ(C + ς²I)⁻¹j, where j holds the
force derivatives of the 144 last-layer features and C their covariance over the
training frames; α and ς² are fitted by NLL on the calibration replicas. The
diversity question becomes whether the last-layer features of a shifted
structure leave the span of the training features. Because the body is frozen,
LLPR shares the frozen-body blindness.

### 2.5 Shallow ensemble trained with NLL

Four randomly initialised heads on the frozen body, all trained on every frame,
jointly, with a Gaussian NLL after a 20% MSE warm-up. The loss trains the spread
to match the training residuals, so its scale cannot collapse in distribution
the way the MHC's did. Two limits: training residuals are smaller than test
residuals, so α > 1 is still expected; and the body is frozen, so it shares the
frozen-body blindness.

## 3. Diversity changes with the label rung

Design C trains every arm at 250, 500, 1,000 and about 1,850 labels.

- **Committee.** Bootstrap members differ more when the pool is small. This is
  a legitimate epistemic effect and needs no correction, but it should be
  measured.
- **MHC.** Each head sees a quarter of the labels: about 62 frames at the
  250-label rung.
- **Snapshots.** A cycle counted in epochs has optimiser steps proportional to
  the number of labels. At 250 labels a 17-epoch cycle has a quarter of the
  steps it has at 1,000, so the convergence trend grows exactly where the
  uncertainty is supposed to grow.

So a curve of uncertainty against labels mixes knowledge with diversity. The
per-rung diagnostics in §4 separate them.

## 4. Diagnostics reported for every ensemble arm and every set

`uq_metrics.ensemble_diversity` computes them from each member's force errors,
and `uq_evaluate` writes them to `metrics.json` (key `diversity`) and the member
errors to the per-set `.npz` files (key `member_err`).

| Quantity | Definition | Reading |
|---|---|---|
| `member_rmse` | Force RMSE of each member | A spread between members, or a trend with cycle index, means the members are not exchangeable |
| `ensemble_rmse`, `best_member_rmse`, `mean_gain` | RMSE of the mean, of the best member, and their ratio | `mean_gain` ≥ 1: averaging does not help |
| `spread_fraction` | Mean squared spread over mean member squared error | The share of member error that averaging removes; 1 − ρ for exchangeable members |
| `error_correlation` | Mean pairwise Pearson correlation of the members' force errors | Close to 1: the members err together |
| `error_to_spread` | Force RMSE over the RMS spread | The raw scale gap; α is its NLL-fitted counterpart |
| `sigma_growth`, `error_growth`, `common_mode_index` | RMS σ and RMSE of a set over those of the in-distribution test set, and their ratio | Index below 1: along this shift, the spread grows more slowly than the error |

These are reported and discussed, not used as gates: no arm is dropped or
retuned because of them.

## 5. Changes adopted on 2026-11-27, and open items

Adopted, as a code change based on the design-C run branch, recorded in the
2026-11-27 note to [ADR-0007](../decisions/ADR-0007-bulk-trained-shift-ladder.md):

1. **Snapshot burn-in.** `uq_compare --snapshot-burn-in-cycles N` trains N extra
   cycles first and keeps the last `--snapshot-count` cycle ends.
   `--snapshot-cycle-epochs` sets the cycle length independently of the
   committee's `--epochs`, so each rung can keep the optimiser steps per cycle
   fixed. The baseline node uses one burn-in cycle: four cycles of 17 epochs,
   members from the last three. That adds one 17-epoch cycle of GPU time, so the
   12 h projection has to be redone.
2. **Member errors and diagnostics.** As in §4.

Open, for the owner to decide:

- **Committee size.** The node runs three members; the shift-ladder plan says
  four. Four costs one more 51-epoch fine-tune per seed set.
- **A more diverse committee variant**, with each member's readout
  re-initialised from its own seed. It would answer the reviewer question "was
  the committee handicapped?", at the cost of one extra committee per rung.
- **Steps per cycle across rungs.** The value of `--snapshot-cycle-epochs` for
  each rung, chosen so the steps per cycle match the 1,000-label node.

## Sources

- Literature review §13 (spread theory) and §14 (committee versus snapshots):
  [literature/REVIEW.md](../literature/REVIEW.md).
- exp-02 numbers: [RESULTS.md](../experiments/exp-02-snapshots/RESULTS.md); the
  per-frame recomputation: `src/mlpdft/utils/exp02_spread_diagnostics.py`.
- Head collapse in the pilot:
  [DIAGNOSIS.md](../experiments/exp-03-mhc/runs/2026-11-05-mhc-pilot/DIAGNOSIS.md).
- MHC-4D: [README.md](../experiments/exp-03-mhc/runs/2026-11-06-mhc4d/README.md).
- The 2026-11-25 review of exp-01 to exp-03 (detrended correlations, restored
  epochs), kept in the project artifacts as `reviews/past-experiments-review.md`.
