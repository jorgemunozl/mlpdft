# Active-Learning Protocol — Retrospective UQ Comparison

**Date:** 2026-10-29
**Status:** pre-registration v2 — fixes loop design, acquisition, budgets, metrics,
and decision rules *before* the run.
**Implementation:** `src/mlpdft/uq_active_learning.py` (loop mechanics) with the
frozen splits from `PHASE-1/audit/`.
**Scope:** UQ + AL on the Li–F testbed. No transport, no new DFT by default.

---

## 1. Objective and hypotheses

**Objective.** Quantify, for each of the three uncertainty strategies and for a
random control, how many labeled configurations — and how much wall-clock
compute — are needed to reduce held-out force error, starting from a small seed
drawn from the frozen `train_pool`.

The loop is **retrospective**: every "acquisition" reveals the existing DFT label
of a pool frame. No new DFT is generated. This isolates the acquisition rule
from label-generation cost and makes the comparison affordable.

**H1 (calibration → data efficiency).** A method whose uncertainty is better
calibrated against true force error acquires more informative configurations per
label, i.e. reaches a given held-out error with fewer labels.

**H2 (snapshot cost-efficiency).** The warm-restart snapshot ensemble reaches a
comparable error at a small multiple of the committee's label count but at a
fraction (≤ 1/3) of its wall-clock cost, because each retrain costs one
fine-tune instead of K.

**H3 (OOD awareness).** A method with intact out-of-distribution uncertainty
acquires interface frames earlier than a method whose uncertainty collapses on
the interface environment.

### Pre-registered decision rules

| outcome | conclusion |
|---|---|
| method's data efficiency ≤ random (within noise) | its uncertainty carries **no useful AL signal** |
| snapshot `labels-to-target` ≤ 1.5× committee **and** `compute-to-target` ≤ committee/2 | snapshot **replaces** committee for practical AL |
| snapshot ≈ committee on labels but much cheaper | snapshot is the **cost-effective** choice |
| committee (with bootstrap) dominates on both | committee remains the **gold standard**; snapshot is "cheaper but less efficient" |
| all three ≈ random | uncertainty is not the bottleneck; reframe the AL claim around calibration diagnostics |

These are fixed before looking at results and will be reported verbatim.

---

## 2. Data and pool definition

The splits are frozen in [`AUDIT.md`](../PHASE-1/AUDIT.md) and materialized as
extxyz by `build_phase1_splits.py`:

| split | file | frames | bulk LiF | F-doped Li | interface |
|---|---|---:|---:|---:|---:|
| `train_pool` | `uq_train.extxyz` | 8,313 | 5,131 | 1,952 | 1,230 |
| `calibration` | `uq_valid.extxyz` | 1,781 | 1,099 | 418 | 264 |
| `test` | `uq_test.extxyz` | 1,781 | 1,099 | 418 | 264 |
| `ood_test` | `uq_ood.extxyz` | 1,696 | 934 | 333 | 429 |

- **Pool** = `train_pool`. All acquisitions are drawn from it.
- **Validation** = `calibration`, used only for early stopping and calibration
  fitting, never for acquisition.
- **Test** = `test`, permanently locked.
- **OOD test** = `ood_test`, permanently locked and reported separately.

The splits are group-level and time-blocked. The interface OOD set comes from a
trajectory with a 0.54 SCF warning rate and is treated as a stress test; see
`AUDIT.md` §4.

**Leak guarantees:**
1. The test and OOD sets never enter the pool.
2. The calibration set never enters the pool.
3. Acquisition ranks pool frames by model uncertainty **only** — no label of a
   pool frame is used to score it or its neighbours.
4. Every acquisition is a reveal of an existing DFT label; no new DFT is
   generated.

---

## 3. Models and acquisition functions

**Ensemble size K = 3** for every method. Each retrain uses the same recipe
(150 epochs, seeds 123/124/125, SGDR with `T₀ = 50` for the snapshot,
bootstrap-default committee).

The three methods differ only in where the ensemble spread comes from:

| method | spread source |
|---|---|
| Independent committee | std across 3 separately fine-tuned models (bootstrap per member) |
| Multi-head | std across 3 readout heads on one shared body |
| Warm-restart snapshots | std across 3 cycle-end checkpoints of one SGDR run |

**Per-atom uncertainty** is the RMS component standard deviation across the M
members, `σ_i = √( (1/M) Σ_m |F_i^m − F̄_i|² )`.

**Primary acquisition score** (per configuration): the maximum relative force
error over its atoms,

  `score(x) = max_i  σ_i / (|F̄_i| + 0.2 eV/Å)`.

**Secondary acquisition score** (sensitivity check): the mean per-atom force
std, `score'(x) = mean_i σ_i`, to confirm the ranking is not an artifact of the
`0.2` regularizer or the max aggregation. Both are run; the primary score is
used for the headline result.

**Random baseline:** stratified uniform sampling from the pool (seeded), using
the same per-environment quota as the acquisition arms.

---

## 4. Loop mechanics

- **Seed:** 30 frames, 10 per environment, chosen deterministically as
  evenly-spaced indices in trajectory order — not the first 10.
- **Batch size:** 10 frames per round, stratified **4 bulk LiF / 3 F-doped Li /
  3 interface**. The interface is over-sampled relative to its pool share
  because it is the OOD leg and the smallest environment.
- **Acquisition:** rank the eligible pool frames by the current ensemble's
  score, take the top candidates within each environment's quota. **Diversity
  guard:** no two selected frames may be closer than `gap = 500` time-index units
  (~0.24 ps) within the same source trajectory.
- **Retrain:** reveal the 10 labels, retrain the ensemble from seed + acquired,
  evaluate on the calibration set.
- **Budget / stopping:** 15 rounds (150 acquired labels, 180 total), or stop
  earlier if calibration force RMSE has changed by < 1% over three consecutive
  rounds.
- **Locked evaluation:** the test and OOD sets are scored once, after the loop
  and after the arm configurations are frozen.

**Reference points:** the "full-pool" model is trained on all 8,313 `train_pool`
frames and sets the accuracy ceiling. A "target reached" event is test force
RMSE ≤ 1.05 × the full-pool RMSE. If no arm reaches the target within 15
rounds, report the curve and the residual gap rather than forcing a target.

---

## 5. Metrics

Report per method and for the random baseline:

1. **Data-efficiency curve** — calibration force RMSE (and energy RMSE) vs.
   cumulative labeled frames, with per-round points; final locked-test values
   for each arm.
2. **Labels-to-target (LTT)** — acquired labels to first reach the target.
3. **Compute-to-target (CTT)** — cumulative wall-clock across all retrains to
   first reach the target. Per-round cost is measured per method (committee ≈ 3×
   a fine-tune, multi-head ≈ head-training, snapshot ≈ 1× a fine-tune).
4. **Acquired-environment composition** — fraction of acquired frames from each
   environment over rounds (H3 diagnostic).
5. **Within-environment error** — bulk, F-doped, and interface reported
   separately; pooled numbers are secondary.

Both LTT and CTT are reported; the headline is CTT, because the paper's thesis
is about cost.

---

## 6. OOD handling

- **Stratified AL** (primary): acquisition is stratified across bulk LiF,
  F-doped Li, and the interface; evaluation is reported per environment.
- **OOD diagnostic** (secondary): report whether each method acquires interface
  frames earlier than the random control.

The interface pool is now 1,230 train frames, 264 test frames, and 429 OOD
frames, so the OOD leg is quantitative rather than a case study. However, the
interface OOD trajectory has a 0.54 SCF warning rate and forces up to 7.5 eV/Å.
It is reported as a stress test, separately from the clean in-domain result.

---

## 7. Reproducibility

- All random numbers (seed selection, random baseline, committee seeds
  123/124/125) are fixed and recorded.
- The pool ordering, seed indices, and every acquired index are logged per round
  in `al_run_manifest.json`.
- The test and OOD sets are byte-for-byte the same across all methods and rounds.
- The split files are generated from `frames.csv` by
  `build_phase1_splits.py`; their SHA-256 hashes are in `split_manifest.json`.

---

## 8. Outputs

1. Per-method data-efficiency and compute-efficiency curves.
2. LTT and CTT table vs. the random baseline.
3. Acquired-environment-composition plots (H3).
4. `al_run_manifest.json` (seed, pool, per-round acquisitions, costs).
5. A one-line pre-registered verdict per the decision rules in §1.
