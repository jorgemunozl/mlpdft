---
status: active
date: 2026-10-29
revised: 2026-11-27
owner: jorge
---

> **Correction 2026-11-27** (review of 2026-11-25, project artifacts `reviews/past-experiments-review.md`).
> Every number below traces to the saved outputs, and the labels are 40/320 Ry
> throughout. What holds: on these 178 correlated frames and one seed, snapshot
> σ ranks force errors within bulk and F-doped Li at least as well as the
> committee's, at about a third of the training cost; this survives removing a
> linear time trend. What does not hold:
> - "best calibrator": no α, RLL or coverage was computed, so §3 measures
>   ranking (discrimination), not calibration;
> - "CIs excluding zero in every group" and the interface intervals in §3: the
>   interface has 18 frames, fewer than one block, and those intervals cannot
>   be traced;
> - p = 0.002 (§8, §9) as evidence that the arms separate: it needs blocks of
>   20 frames or more, which drop the interface and leave four blocks per group;
>   5- and 10-frame blocks give p = 0.11 and 0.18;
> - "OOD separation is now clean" and "multi-head collapses at the interface":
>   the interface test is the last 18 frames of a trajectory that is in
>   training, and after a linear detrend multi-head's interface ρ is 0.70;
> - the H1–H3 implications in §7.
>
> Two confounds the text does not state. The splits are contiguous with no
> buffer, so fine-tuned test errors rise ×1.15–1.62 along each bulk and F-doped
> test block while zero-shot error stays flat (temporal leakage). Committee
> members and multi-head were restored at their best-validation epochs
> (144/149/148 and 108), the snapshots at fixed epochs 49/99/149.

# UQ Experiment 2 — Results

**Date:** 2026-10-29
**Status:** completed; analyzed 2026-10-29, corrected 2026-11-27.
**Source:** cluster run `lif_uq_001`; local copy `outputs/outputs_v3/` in the main
checkout (git-ignored). Evaluation outputs, manifest and logs are archived in
project artifacts `runs-archive/2026-10-27-exp02-lif_uq_001/`.
**Predecessor:** [`exp-01-committee/DESIGN.md`](../exp-01-committee/DESIGN.md).
**Feeds:** [`exp-04-active-learning/PROTOCOL.md`](../exp-04-active-learning/PROTOCOL.md) (the
AL leg; ceiling RMSE fixed here).

---

## 1. Configuration (confirmed from `experiment_manifest.json`)

- Model: **MACE-OMAT medium** (`mace_omat_medium`), float64, CUDA.
- Budget: **150 epochs**, `snapshot_count = 3`.
- Committee: 3 seeds (123/124/125), **bootstrap enabled**
  (`committee_bootstrapped: true`, block size 5), single cosine decay
  (`scheduler_t0 = 150`).
- Warm snapshots: one run, warm restarts `scheduler_t0 = 50`, snapshots collected
  at **cycle-ends 49/99/149** (one epoch before each restart) — the fix from
  experiment 1.
- Multi-head: 3 bootstrap heads + **Materials Project** replay, requested as
  `pt_train_file = "omat"`. In mace-torch 0.3.15 every `mp`/`omat`/`matpes_*`
  value is routed to `mp_traj_combined.xyz`, so this is MP replay, not OMAT
  (`--pt-samples 1000`).
- Test: 178 frames, NVT only — LIF64 (80) + LIWITHF (80) + LIFINTERFACE (18).

## 2. Force and energy accuracy (atom-weighted)

| method | force RMSE (eV/Å) | energy RMSE (meV/atom) |
|---|---|---|
| zero-shot foundation | 0.0160 | 385 996 (raw) |
| independent committee | **0.0087** | **0.099** |
| multi-head bootstrap | 0.0089 | 0.512 |
| warm snapshots | 0.0101 | 0.116 |

- Fine-tuning buys **1.6–1.8×** force accuracy over zero-shot.
- **Reference-energy fix confirmed:** fine-tuned energy is now ~0.1 meV/atom
  (was 1.5–13 meV/atom in experiment 1). Zero-shot remains raw cohesive-vs-total
  (386 eV/atom) because its labels are unaligned.

## 3. Calibration (uncertainty vs. force error)

| method | pooled Spearman | within-group mean Spearman |
|---|---|---|
| warm snapshots | **0.981** | **0.834** |
| independent committee | 0.943 | 0.656 |
| multi-head bootstrap | 0.254 | 0.580 |

Per-group Spearman with moving-block bootstrap 95% CIs (block ≈ 20, 2000 resamples):

| group | committee | snapshot | multi-head |
|---|---|---|---|
| LIF64 (bulk) | +0.74 [0.29, 0.90] | **+0.93** [0.74, 0.95] | +0.67 [0.07, 0.77] |
| LIWITHF (Li-rich) | +0.63 [0.25, 0.83] | **+0.86** [0.67, 0.95] | +0.95 [0.87, 0.97] |
| LIFINTERFACE (OOD) | +0.59 [0.21, 0.84] | **+0.71** [0.27, 0.77] | +0.12 [−0.51, 0.71] |

### Risk-coverage (force RMSE of X% lowest-uncertainty frames)

| method | 10% | 25% | 50% | 75% | 100% | monotonic? |
|---|---|---|---|---|---|---|
| independent committee | 0.0024 | 0.0025 | 0.0031 | 0.0050 | 0.0087 | yes |
| warm snapshots | 0.0025 | 0.0027 | 0.0035 | 0.0057 | 0.0101 | yes |
| multi-head bootstrap | 0.0044 | 0.0061 | 0.0072 | 0.0073 | 0.0089 | yes (flat tail) |

All three are now monotonic (in experiment 1 only snapshots were), but the
snapshot and committee low-error tails are much steeper than multi-head's
(0.0024–0.0025 vs 0.0044 at the 10% quantile).

> **Regenerated 2026-11-27** from the saved `*_predictions.csv` with the
> atom-weighted formula of `uq_compare.py` (sum of squared errors over the
> number of components), so the 100% column equals the overall force RMSE. The
> earlier rows averaged per-frame RMSEs (committee 100% point 0.0076). The
> ordering did not change. The curve is pooled over environments; §9 shows that
> within an environment it is much flatter.

## 4. Cost (from the manifest, seconds)

| method | cost |
|---|---|
| warm snapshots | **4 996** (one fine-tune) |
| independent committee | 14 975 (3 × ~5 000) |
| multi-head bootstrap | 43 148 (incl. MP replay) |

Snapshot is ~⅓ the committee and ~⅛ the multi-head, while being the best
calibrator.

## 5. Key findings

1. **The cycle-end snapshot fix was decisive.** Snapshots went from the *worst*
   within-group calibrator in experiment 1 (0.271, under-trained at 33/67/99) to
   the *best* here (0.834, at 49/99/149). Snapshot member energies now converge
   (0.188 / 0.099 / 0.066 meV/atom) instead of 26.9 / 16.1 / 2.67.
2. **Snapshots are now the best calibrator** on both pooled (0.981) and
   within-group (0.834) metrics, with CIs excluding zero in every group, and at
   ~⅓ committee cost.
3. **OOD separation is now clean.** Snapshot (0.71) and committee (0.59) stay
   calibrated at the interface; multi-head collapses there (0.12, CI spans zero).
   This is the Kurniawan "uncertainty breaks down OOD" signature, and it now
   singles out multi-head — not snapshots.
4. **Committee remains the most force-accurate** (0.0087) and never inverts (all
   CIs positive); it is now bootstrapped, partially addressing the experiment-1
   committee-collapse confound.
5. **A snapshot tradeoff to name:** the snapshot *ensemble* force RMSE (0.0101)
   is the worst of the three because the ensemble mean folds in the early-cycle
   members; its *final* member alone (epoch 149) is 0.0082, better than the
   committee mean. Snapshot pays ~16% force accuracy for its (better) diversity.

## 6. What this run establishes vs. experiment 1

| item | experiment 1 | experiment 2 |
|---|---|---|
| snapshot collection | evenly spaced 33/67/99 (bug) | cycle-ends 49/99/149 |
| snapshot within-group Spearman | 0.271 | **0.834** |
| within-group ranking | committee > snapshot > multi-head | **snapshot > committee > multi-head** |
| OOD inversion | snapshot & multi-head invert | only multi-head inverts |
| fine-tuned energy RMSE | 1.5–13 meV/atom | 0.1–0.5 meV/atom |
| committee | seed-only (collapsed) | bootstrapped |

## 7. Implication for the active-learning leg

The calibration leg lands exactly where the pre-registered decision rules in
[`exp-04-active-learning/PROTOCOL.md`](../exp-04-active-learning/PROTOCOL.md) anticipate:

- **H1 (calibration → data efficiency):** snapshot leads (0.834 vs 0.656), so it
  should acquire the most informative configs per label.
- **H2 (snapshot cost-efficiency):** snapshot is already at ⅓ committee cost, so
  `compute-to-target ≤ committee/2` holds unless it needs far more labels.
- **H3 (OOD awareness):** snapshot and committee recognize the interface as
  high-error; multi-head does not.

**Ceiling force RMSE** (full-data model) ≈ **0.0087–0.0090 eV/Å**, so the AL
"target reached" threshold is 1.05× ≈ **0.0091–0.0095 eV/Å**.

**Caveat:** the interface test group is only n = 18, so interface CIs are wide;
the OOD conclusion is qualitative there, as already flagged in the protocol.

---

## 8. Expected results — where they are answered

This section maps each outcome anticipated in
[`DESIGN.md`](DESIGN.md) and the pre-registered protocol to
the evidence in this report.

| Expected result | Answered in | Status |
|---|---|---|
| Cycle-end snapshots remove the experiment-1 under-training confound | §5.1, §6 | **Answered.** Within-group Spearman 0.271 → 0.834 and member energies converge. The magnitude is confounded by the test-set change (480 → 178 frames). |
| The snapshot ensemble is the cost-effective calibrator | §3, §4, §5.2, §9 | **Answered with a paired test (pilot).** 0.834 vs 0.656 (committee) at ⅓ the cost; Δ(committee − snapshot) = −0.21, 95% CI [−0.45, −0.07], p = 0.002 on the two resamplable environments. The interface (n = 18) is excluded from the interval. |
| The committee remains the force-accuracy reference | §2, §5.4 | **Answered, but it is a tie.** 0.0087 vs 0.0089 (multi-head), with no CI on the difference. |
| Multi-head is competitive in-domain and collapses at the interface | §3, §5.3 | **Partly answered.** In-domain yes. At the interface n = 18, the CI includes zero, and the interface test is a held-out block of the same trajectory, not an independent OOD set. |
| Reference-energy alignment fixes the energy metric | §2, §5 | **Answered.** 1.5–13 → 0.1–0.5 meV/atom after alignment. |
| Full-data ceiling for the active-learning target | §7 | **Partly answered.** 0.0087–0.0090 is the 214-frame committee, not a model trained on all 214 frames, and it predates the 8,313-frame frozen split. |
| Per-method training cost for compute-to-target | §4, §5.2 | **Answered.** Snapshot 4,996 s; committee 14,975 s; multi-head 43,148 s including the MP replay. |
| Leak-free, frozen splits | §1 | **Configuration recorded.** The split construction is in `build_uq_splits.py`; it is not re-audited here. |

**Not answered by this run:**

- **H1 (calibration → data efficiency)** and **H3 (acquisition behavior)** need
  the active-learning loop; this report stops at the calibration leg.
- **H2 (snapshot cost-efficiency)** is answered on cost, not on labels; the
  labels-to-target side is deferred to the AL loop.
- **OOD claims** need the independent `uq_ood.extxyz` split. The interface test
  used here is a held-out time block of the same trajectory.
- **Statistical separation of the arms** is now reported in §9 with a paired
  block bootstrap; the marginal per-group CIs in this report are not a test of
  the ranking.

**Where the numbers come from:** the run's `evaluation/uq_summary.json` and its
four `*_predictions.csv` files, archived in project artifacts
`runs-archive/2026-10-27-exp02-lif_uq_001/`. The headline force RMSE is
atom-weighted, and since 2026-11-27 so is the risk–coverage table: its 100%
endpoint equals the headline.

---

## 9. Environment-resolved re-analysis (pilot)

The per-group CIs above are marginal intervals, not a test of method
differences, and the risk–coverage table is pooled over environments. Both gaps
are addressed by `src/mlpdft/utils/uq_analysis.py`, which resamples contiguous
20-frame blocks within each environment and uses the *same* blocks for both
methods in a paired comparison. Artifacts and reproduction commands are in
[`analysis/`](analysis/README.md).

Findings on the same 178 test frames:

- **Within-environment risk–coverage is much flatter than the pooled curve.**
  The pooled "steep tail" in §5.2 is partly a group-composition effect: at 10%
  coverage a method can look good by selecting bulk frames rather than by
  ranking well inside an environment.
- **Snapshot > committee on within-environment rank correlation**, on the two
  environments large enough to resample: Δ(committee − snapshot) = −0.21,
  95% CI [−0.45, −0.07], p = 0.002. The interface has 18 frames, fewer than one
  block, so it has no interval and is excluded from the paired test.
- **Committee ≈ multi-head**: Δ(committee − multi-head) = −0.12, 95% CI
  [−0.36, +0.21], p = 0.60. The point estimates differ; the data do not separate
  them.
- **Every arm is under-scaled.** Median force error / predicted spread is
  2.7–6.4× for committee and snapshots, and 13.8–66.5× for multi-head. The
  disagreement ranks errors better than it sizes them; a fitted scale would be
  needed before quoting calibrated error bars.

These are pilot numbers: 178 test frames, one foundation checkpoint, one seed
set, and a 214-frame training pool. They will be regenerated on the frozen
post-generation splits with the same commands.
