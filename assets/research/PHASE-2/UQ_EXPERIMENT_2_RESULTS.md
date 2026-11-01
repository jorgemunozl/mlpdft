# UQ Experiment 2 — Results

**Date:** 2026-10-29
**Status:** ✅ completed — results analyzed.
**Source:** `outputs/outputs_v3/` (local copy of the cluster run
`lif_uq_001`; raw outputs under `src/mlpdft/outputs/uq_diffusion_comparison/`).
**Predecessor:** [`../UQ_EXPERIMENT_1.md`](../UQ_EXPERIMENT_1.md).
**Feeds:** [`../ACTIVE_LEARNING_PROTOCOL.md`](../ACTIVE_LEARNING_PROTOCOL.md) (the
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
| independent committee | 0.0024 | 0.0025 | 0.0032 | 0.0052 | 0.0076 | ✓ |
| warm snapshots | 0.0025 | 0.0027 | 0.0036 | 0.0059 | 0.0089 | ✓ |
| multi-head bootstrap | 0.0045 | 0.0062 | 0.0074 | 0.0075 | 0.0081 | ✓ (flat tail) |

All three are now monotonic (in experiment 1 only snapshots were), but the
snapshot and committee low-error tails are much steeper than multi-head's
(0.0024–0.0025 vs 0.0045 at the 10% quantile).

> **Correction (code fix, not yet applied to this table).** These rows were
> computed by averaging per-frame RMSEs, which up-weights frames with few atoms.
> That is why the 100% row does not reproduce the overall force RMSE
> (committee 0.0076 vs 0.0087). `uq_compare.py` now uses the atom-weighted
> formula (`sum of squared errors / number of components`), which makes the
> 100% row identical to the overall RMSE. Regenerate this table from the saved
> `*_predictions.csv` files before quoting the 10–75% values; the qualitative
> ordering (steep committee/snapshot tails, flat multi-head tail) is expected to
> hold but the numbers will move.

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
[`../ACTIVE_LEARNING_PROTOCOL.md`](../ACTIVE_LEARNING_PROTOCOL.md) anticipate:

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
[`UQ_EXPERIMENT_2.md`](UQ_EXPERIMENT_2.md) and the pre-registered protocol to
the evidence in this report.

| Expected result | Answered in | Status |
|---|---|---|
| Cycle-end snapshots remove the experiment-1 under-training confound | §5.1, §6 | **Answered.** Within-group Spearman 0.271 → 0.834 and member energies converge. The magnitude is confounded by the test-set change (480 → 178 frames). |
| The snapshot ensemble is the cost-effective calibrator | §3, §4, §5.2 | **Answered on point estimates.** 0.834 vs 0.656 (committee) at ⅓ the cost. No paired test of the difference; per-group CIs overlap. |
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
- **Statistical separation of the three arms** needs a paired bootstrap of the
  method differences; the per-group CIs reported here are not a test of the
  ranking.

**Where the numbers come from:** `outputs/outputs_v3/evaluation/uq_summary.json`
and the four `*_predictions.csv` files. The headline force RMSE is
atom-weighted. The risk–coverage table above predates the atom-weighting fix and
is frame-RMS, so its 100% endpoint is 0.0076 (committee) rather than the 0.0087
headline; the runner now uses the atom-weighted formula, whose 100% endpoint
equals the headline. Regenerate that table before quoting it.
