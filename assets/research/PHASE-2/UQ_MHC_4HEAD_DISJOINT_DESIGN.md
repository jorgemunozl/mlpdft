# MHC-4D: four-head disjoint committee

**Experiment design and theoretical rationale for the first MHC run**

**Status:** design frozen before training. Supersedes the 8-head,
80%-overlapping configuration in [`UQ_EXPERIMENT_3.md`](UQ_EXPERIMENT_3.md) and
[`EXPERIMENT_3_RUNBOOK.md`](EXPERIMENT_3_RUNBOOK.md) for this experiment.
**Date:** 2026-11-06.
**Method:** Beck et al., *Multi-head committees enable direct uncertainty
prediction for atomistic foundation models*,
[arXiv:2508.09907](https://arxiv.org/abs/2508.09907).
**Dataset:** two-environment interim candidate
(`jorgemunozl/li-f-uq-interim-two-environments`, revision
`c2699e8e52889c18f6d29f657eb996ac3f78524f`): 1,000 training frames
(500 bulk LiF + 500 F-doped Li). The interface is excluded.
**Code:** `src/mlpdft/uq_compare.py`, arm `mhc`.

> **Dataset caveat.** [`UQ_TWO_ENVIRONMENT_CURATION.md`](UQ_TWO_ENVIRONMENT_CURATION.md)
> marks this candidate *development only, not cleared for production UQ
> comparisons*: F-doped burn-in is unverified and the temporal diagnostics hit
> their lag limit. Unless those gates close first, this run is a development
> experiment, not a production measurement. The design below is written for
> 1,000 training frames; the earlier three-environment plan (2,000 frames) would
> double every per-head count.

## 1. Decision

Run the first MHC experiment with **A = 4 heads**, a **disjoint** partition,
randomly initialised readouts, a frozen foundation body, and a force-only
objective. Each head receives **250 frames** (125 bulk + 125 F-doped). The
budget is 120 epochs, about **7,500 optimizer updates** at batch size 16.
Compare against the 3-member fine-tuned committee (primary) and the frozen
single-head probe (secondary).

`A = 4` and `disjoint` are not claims that the paper's configuration is
optimal. They are the cheapest configuration that preserves its mechanism, and
the one that changes the fewest variables relative to the failed pilot. The
design is the **first rung of a pre-registered ladder** (§9), not a final
architecture choice.

## 2. Why begin here

### 2.1 The pilot was already four-head disjoint

The failed pilot (`lif_uq_mhc_002`) used four disjoint heads, `freeze=6`, and
`foundation_filter_elements=True`; its manifest lists `mhc_head_0` through
`mhc_head_3` and no replay heads. It failed for three reasons that are
independent of the head count:

1. every head was warm-started from the same foundation readout;
2. the QE E0 reference was inconsistent with that readout (2.2–2.5 eV/atom);
3. each head saw only about 54 frames.

See [`experiment3_runs/mhc_pilot_2026-11-05/DIAGNOSIS.md`](experiment3_runs/mhc_pilot_2026-11-05/DIAGNOSIS.md).
Changing `A` in the repair run would add a fourth difference and make the
comparison uninterpretable. Keeping `A = 4` isolates the three fixes.

### 2.2 Disjoint is the paper's foundation-model recipe

The paper defines two partitions (Fig. 1b): **disjoint**, an even partition of
the training set with no overlap, and **overlapping**, an independent random
subset per head. For the foundation model it uses disjoint: 8,000 QbC-selected
structures divided across eight heads, 1,000 per head, 800,000 optimization
steps. Disjoint is therefore the partition we should test first. Head count
and per-head count are the budget knobs; the split is not.

### 2.3 Disjoint is the compute-efficient split

MACE concatenates the per-head training files, so the cost of one epoch is the
**sum of the head sizes**, not `A` times the dataset. Under disjoint the sum is
`N` for any `A`; under overlapping it is `A` times the per-head fraction. For
`N = 1,000`, batch 16, 120 epochs:

| Partition | Heads | Frames/head | Samples/epoch | Updates at 120 epochs | Relative cost |
|---|---:|---:|---:|---:|---:|
| Disjoint | 4 | 250 | 1,000 | ~7,500 | **1.0×** |
| Disjoint | 8 | 125 | 1,000 | ~7,500 | 1.0× |
| Overlap 80% | 4 | 800 | 3,200 | ~24,000 | 3.2× |
| Overlap 80% | 8 | 800 | 6,400 | ~48,000 | 6.4× |

The 6.4× figure is why the earlier 2,000-frame, 8-head overlapping plan was
projected at 86 GPU-hours; the same recipe on 1,000 frames would be about half
that. Disjoint cuts the dominant term. Four heads does **not** cut it further:
4 × 250 and 8 × 125 are the same 1,000 samples per epoch.

### 2.4 Under disjoint, four heads dominate eight at equal cost

Both configurations present 1,000 samples per epoch. Four heads give 250
frames per head; eight give 125. The paper attributes the disjoint accuracy
penalty to per-head sparsity: "especially for small training datasets, the
subsets for each head are extremely sparse, negatively affecting the prediction
accuracy" (Results, 3BPA). At fixed compute, `A = 4` buys twice the per-head
data. The price is a noisier uncertainty estimate (§2.5), not a higher bill.

### 2.5 Four is the smallest defensible committee

For `A` exchangeable Gaussian heads, the sample standard deviation has relative
standard error approximately `1 / sqrt(2(A - 1))`. The heads share a body and are
correlated, so the true error is larger; the table is a lower bound.

| Heads | Frames/head | Relative SE of the head std |
|---:|---:|---:|
| 3 | 333 | 50% |
| **4** | **250** | **41%** |
| 5 | 200 | 35% |
| 8 | 125 | 27% |

`A = 3` is rejected. A 50% noise floor is comparable to the between-frame
signal the experiment is trying to rank, and a single divergent head can
dominate the variance. `A = 4` is the smallest committee whose spread is
usable. Frame-level uncertainty aggregates over atoms, which averages down part
of this noise; per-atom correlations will be noisier than frame-level ones.

### 2.6 Four is closer to the three-member baseline

The fine-tuned committee has three members. Comparing eight MHC heads with
three committee models confounds the method with ensemble size. Four versus
three narrows that gap; it does not eliminate it. A perfectly matched `A = 3`
is rejected on estimator-noise grounds. If compute allows, adding a **fourth
committee member** is a better use of roughly 9,000 updates than adding four
more MHC heads, because it removes the remaining member-count confound from
the primary comparison.

## 3. What the paper actually found

The paper does not report one split as universally better:

- **Correlation:** both partitions give a similar level of correlation between
  disagreement and error.
- **Calibration:** the disjoint model shows "preferable scaling of the
  uncertainty"; its fitted `α` is consistently lower than the overlapping
  model's across systems. After `α` scaling, the models behave similarly.
- **Accuracy:** the overlapping model has "a slight advantage in terms of
  accuracy" on 3BPA, because each head sees 80% of the data instead of 12.5%.
  The gap shrinks as the total dataset grows.
- **Foundation model:** disjoint only, eight heads, 1,000 frames each.

So "disjoint is better" is true for diversity and raw calibration;
"overlapping is better" is true for accuracy when per-head data is scarce. At
250 frames per head, accuracy is the binding risk. That is why the primary gate
is accuracy (§6, H1) and why the fallback is **overlap**, not more heads (§9).

## 4. Formal setup

Let `N = 1,000` training frames, `A = 4` heads, and `n_a = 250` frames assigned
to head `a`. The partition is stratified by environment, so each head receives
125 bulk and 125 F-doped frames. Every frame is assigned to exactly one head.

For atom `i` and force component `α`, let `F_{i,a}^{(α)}` be head `a`'s
prediction. The committee prediction and its per-atom uncertainty are

\[
\bar{F}_i^{(\alpha)} = \frac{1}{A}\sum_{a=1}^{A} F_{i,a}^{(\alpha)},
\qquad
\sigma_i = \frac{1}{3}\sum_{\alpha=1}^{3}
\left[\frac{1}{A}\sum_{a=1}^{A}
\left(F_{i,a}^{(\alpha)} - \bar{F}_i^{(\alpha)}\right)^2\right]^{1/2}.
\]

The scalar calibration factor is fitted on the 200-frame calibration split,

\[
\alpha^2 = \frac{1}{N_{\mathrm{cal}}}
\sum_{i \in \mathrm{cal}} \frac{(\Delta y_i)^2}{\sigma_i^2},
\]

and the reported uncertainty is `α σ`. For forces, the mean runs over every
component of every calibration frame; for energies, over frames per atom. The
ranking metrics are invariant to `α`; only calibration and coverage depend on
it.

Construction details:

- **Frozen body.** `freeze=6` plus the radial/Bessel/distance-transform tensors.
  The runner refuses a frozen run with `foundation_filter_elements=True`
  (`src/mlpdft/uq_compare.py`, frozen-body guard), so the new heads start from
  random weights.
- **No original head.** The paper keeps the pretrained head frozen and adds
  eight new heads; the committee uses only the new heads. This implementation
  does not retain the original head. This is a deviation in bookkeeping, not in
  the committee construction.
- **Force-only loss.** `energy_weight=0`. The foundation readout and the QE
  isolated-atom energies use different E0 references; a warm-started energy
  objective would fit that offset instead of the physics. Energy predictions
  are a by-product and are not part of the gate.
- **Optimization.** Adam, learning rate `1e-3`, batch 16, 120 epochs, patience
  50, early stopping on validation force RMSE. One training run; the heads are
  not independent seeds.

## 5. Alternatives considered

| Option | Frames/head | Relative compute | Verdict |
|---|---:|---:|---|
| **4 disjoint** | **250** | **1.0×** | **Chosen.** Paper-faithful split; most per-head data among disjoint options; closest to the 3-member baseline. |
| 8 disjoint | 125 | 1.0× | Dominated: same cost, half the per-head data. |
| 3 disjoint | 333 | 1.0× | Rejected: 50% estimator noise; one head can dominate. |
| 4 overlap 80% | 800 | 3.2× | Fallback if accuracy fails: more per-head data, less diversity, higher `α`. |
| 8 overlap 80% | 800 | 6.4× | Rejected: most expensive, least diverse per unit compute. |
| MH-FT (Exp. 2) | 2,000 (replay) | — | Different method: the body is trained. Not an MHC configuration. |

The ranking is conditional on the goal. If the only objective were force
accuracy, `4 overlap` would be the better first run at 3.2× the cost. The
objective here is a **paper-faithful MHC test under a compute cap**, with
accuracy as a gate rather than the sole target.

## 6. Pre-registered hypotheses and gates

Fixed before seeing the test results.

1. **H1 — accuracy.** MHC preserves accuracy only if its within-group force
   RMSE is at most **1.25×** the 3-member committee's in **both** bulk LiF and
   F-doped Li. If either group fails, report the degradation and its size; do
   not write "preserves accuracy".
2. **H2 — ranking.** Compare within-group frame-level Spearman `ρ` between
   `σ` and force RMSE, with block-bootstrap confidence intervals. "Comparable
   to the committee" requires overlapping intervals in **both** groups. When
   the autocorrelation estimate hits its lag limit, treat the comparison as
   descriptive rather than a significance test.
3. **H3 — calibration.** Report `α`, 1σ/2σ empirical coverage and RLL after
   `α` scaling. `α` is a scale diagnostic, not a pass/fail: a large `α` with
   nominal coverage means the raw spread is small but the rescaling works. The
   run is unusable if 1σ coverage after `α` is below 0.50 or RLL is negative
   (worse than predicting the constant RMSE everywhere).
4. **H4 — cost.** Report MHC wall clock as a fraction of the 3-member
   committee's total. The efficiency claim requires `< 100%`; `≤ 50%` is
   strong efficiency.
5. **H5 — diversity.** Report pairwise head force RMS differences and the
   ratio of between-head spread to mean head RMSE. A near-zero spread (below
   5% of mean head RMSE) invalidates the arm as a committee; a large but
   error-independent spread invalidates it as UQ.

**Smoke gate first.** A 20-epoch smoke run must show four head files with
125/125 frames each, random initialisation, the distance transform copied from
the foundation, finite losses, and non-zero pairwise head spread. The full run
does not start until these pass.

## 7. Run configuration

```bash
SPLIT_DIR=dataset/xyz_files/uq_interim_two_env

uv run --locked python -m mlpdft.uq_compare \
  --run-id lif_uq_mhc_4d_001 \
  --arms mhc,frozen-probe \
  --train-file      "$SPLIT_DIR/uq_train.extxyz" \
  --valid-file      "$SPLIT_DIR/uq_valid.extxyz" \
  --calibration-file "$SPLIT_DIR/uq_calibration.extxyz" \
  --test-file       "$SPLIT_DIR/uq_test.extxyz" \
  --mhc-heads 4 --mhc-split disjoint \
  --mhc-epochs 120 --probe-epochs 600 \
  --energy-weight 0 --learning-rate 1e-3 --patience 50
```

Notes:

- `SPLIT_DIR` must be the directory that produced the uploaded revision.
  The runner's default paths still point at the 214-frame pilot split; pass all
  four paths explicitly.
- Do not pass `--overwrite` on the first run.
- The manifest must record: four heads × 250 frames, per-head `by_group` counts
  of 125/125, `foundation_filter_elements=false`, `freeze=6`, no
  `multiheads_finetuning`, and the calibration file hash.

## 8. Analysis plan

- **Primary:** within-group frame-level Spearman `ρ` between `ασ` and force
  RMSE, reported separately for bulk and F-doped. Do not pool the two
  environments into one headline `ρ`; the between-environment contrast can
  dominate it.
- **Secondary:** Pearson `r`, `α`, 1σ/2σ coverage, RLL, and per-atom `ρ`.
  Report `α` once, fitted on calibration, not per group.
- **Uncertainty:** block bootstrap over time blocks, with block length at least
  the estimated autocorrelation time. Report effective sample size. Do not
  compute i.i.d. frame-level intervals; the frames are temporally correlated.
  If the autocorrelation estimate hits the lag limit — as it does for the
  F-doped windows — the block bootstrap is not trustworthy either: report the
  interval as a lower bound and do not use it to claim superiority.
- **Committee decomposition:** report per-head RMSE, ensemble RMSE, and the gap
  between the ensemble and the best member. A negligible gap is expected for a
  shared-body committee; it is not a failure.

## 9. Failure modes and escalation ladder

| Observation | Most likely diagnosis | Next action |
|---|---|---|
| Pairwise head RMS ≈ 0, `α > 20`, ensemble ≈ average member | Head collapse; initialisation or freeze bug | Fix the recipe and rerun. Not a method result. |
| Accuracy fails H1, but `σ` still ranks | 250 frames/head is too sparse for disjoint | Escalate to **4-head overlap 80%** (3.2× compute), not 8 heads. |
| Accuracy passes, `σ` is flat | Disjoint at 250/head carries no UQ signal | Negative result for this rung; try overlap or more labels. |
| `α` large but ranking and coverage acceptable | Scale, not ranking | Report; `α` is fitted and does not affect `ρ`. |
| Coverage far below nominal after `α` | Gaussian/calibration assumption fails | Report; check heavy tails and per-group coverage. |
| Dataset gates not closed | Curation, not model | Label the result development-only. |

The ladder is deliberately one-dimensional: if disjoint fails on accuracy, the
next run changes the **split**, because that is the variable the paper says
controls the accuracy/calibration trade-off. Changing head count and split
together would repeat the pilot's mistake.

## 10. Claim boundaries

This experiment can claim:

- a test of the paper's MHC **construction** — frozen AEDs, random heads,
  disjoint partition, disagreement as uncertainty, `α` scaling — on a 1,000-frame
  QE Li–F testbed;
- whether that construction preserves force accuracy and ranking at 250
  frames/head, under a stated compute budget.

It cannot claim:

- a reproduction of the paper's setup or numbers: 4 heads vs 8, 250 vs 1,000
  frames/head, ~7,500 vs 800,000 updates, force-only vs energy+force, 1,000 vs
  8,000 structures, two QE environments vs MPtrj;
- that four heads are optimal, or that a negative result rules out MHC in
  general;
- independent labels: the frames are correlated time blocks, and the test is a
  tail block of trajectories represented in training, not an OOD benchmark;
- production-grade evidence while the two-environment candidate remains
  uncleared.

## 11. Compute accounting

Projected from the plan's 86 GPU-hours for 96,000 frozen-head updates
(≈3.2 s/update), unmeasured on this dataset:

| Arm | Updates | Projected time |
|---|---:|---:|
| MHC-4D (this run) | ~7,500 | ~7 h |
| Frozen single-head probe | ~37,500 | ~33 h |
| 3-member committee | ~28,000 | ≥25 h (full-body updates cost more) |

The MHC arm is roughly one fifth of the probe. Cutting heads does not reduce
its cost — disjoint training presents 1,000 samples per epoch whether there are
four heads or eight. If compute must be cut, cut probe epochs or committee
size, not the MHC arm.

## 12. Decision log

- **2026-11-06 — adopt 4-head disjoint as the first MHC run.** Reasons:
  continuity with the pilot, paper-faithful split, 6.4× cheaper than the
  overlapping plan, maximal per-head data among disjoint options, and the
  smallest defensible committee size.
- **Revisit if** the dataset returns to the 2,000-frame three-environment
  version (per-head count doubles to 500; the design is unchanged), or if H1
  fails (escalate to 4-head overlap), or if H5 shows collapse (recipe bug, not
  a split question).

## References

1. H. Beck, P. Simko, L. L. Schaaf, O. Marsalek, C. Schran, *Multi-head
   committees enable direct uncertainty prediction for atomistic foundation
   models*, [arXiv:2508.09907](https://arxiv.org/abs/2508.09907).
2. [`DIAGNOSIS.md`](experiment3_runs/mhc_pilot_2026-11-05/DIAGNOSIS.md) — why
   the four-head pilot failed.
3. [`UQ_TWO_ENVIRONMENT_CURATION.md`](UQ_TWO_ENVIRONMENT_CURATION.md) — current
   dataset composition and unresolved gates.
4. [`UQ_EXPERIMENT_3.md`](UQ_EXPERIMENT_3.md) — broader Experiment 3 protocol.
