# Active-Learning Protocol — Retrospective UQ Comparison

**Date:** 2026-11-04
**Status:** pre-registration v3 — fixes arms, loop design, acquisition, budgets,
metrics, and decision rules *before* the final run. Supersedes v2 (2026-10-29).
**Implementation:** `src/mlpdft/uq_active_learning.py` (loop mechanics),
`src/mlpdft/utils/uq_analysis.py` (statistics), `make_uq_figures.py` (figures).
**Scope:** UQ + AL on the Li–F testbed. No transport, no new DFT by default.

---

## 1. Objective and hypotheses

**Objective.** Quantify, for each uncertainty source and for a random control,
how many labeled configurations — and how much wall-clock compute — are needed
to reduce held-out force error, starting from a small seed drawn from the frozen
`train_pool`.

The loop is **retrospective**: every "acquisition" reveals the existing DFT label
of a pool frame. No new DFT is generated. This isolates the acquisition rule
from label-generation cost and makes the comparison affordable.

The three construction families are **independent committee**, **warm-restart
snapshots**, and **multi-head**. The first two run inside the loop. The
frozen-body multi-head committee (MHC) is evaluated at fixed label budgets
outside the loop (§6): with four heads it needs roughly four times the labels of
the other arms to give each head a comparable sample, and forcing it into a
180-label loop would measure data starvation rather than the method. The
fine-tuned multi-head arm from Experiment 2 (MH-FT: trainable body plus replay)
is not rerun here; it is reported as a pilot and is not the same method as MHC.

**H1 (calibration → data efficiency).** A method whose uncertainty ranks true
force error better *within* an environment acquires more informative
configurations per label, i.e. reaches a given held-out error with fewer labels.

**H2 (snapshot cost-efficiency).** The warm-restart snapshot ensemble reaches a
comparable error at a small multiple of the committee's label count but at a
fraction (≤ 1/3) of its wall-clock cost, because each retrain costs one
fine-tune instead of K.

**H3 (OOD awareness).** A method with intact out-of-distribution uncertainty
acquires interface frames earlier than a method whose uncertainty collapses on
the interface environment.

**H4 (frozen-body MHC).** At a matched label budget, a four-head committee on a
frozen foundation body reaches committee-quality force error and calibration at
a small fraction of its training cost, or the frozen representation is the
bottleneck at the interface.

### Pre-registered decision rules

| outcome | conclusion |
|---|---|
| method's data efficiency ≤ random (paired bootstrap CI includes zero) | its uncertainty carries **no useful AL signal** |
| snapshot `error@budget` ≤ 1.1× committee **and** `compute-to-target` ≤ committee/2 | snapshot **replaces** committee for practical AL |
| snapshot ≈ committee on labels but much cheaper | snapshot is the **cost-effective** choice |
| committee (with bootstrap) dominates on both | committee remains the **gold standard**; snapshot is "cheaper but less efficient" |
| all arms ≈ random | uncertainty is not the bottleneck; reframe the AL claim around calibration diagnostics |

For the multi-head arm:

| outcome at 2,000 labels | conclusion |
|---|---|
| force RMSE ≤ 1.25× committee **and** within-group Spearman ≥ committee − 0.10 | frozen-body MHC is a **paper arm**, not an appendix |
| force RMSE > 1.25× committee, probe ≈ committee | data splitting across heads is the bottleneck |
| force RMSE > 1.25× committee, probe also worse | the frozen representation is the bottleneck; report as a negative foundation-model result |

These rules are fixed before looking at final results and will be reported
verbatim.

---

## 2. Data and pool definition

The pool is rebuilt after the NPT curation and the interface-generation
campaign. Counts below are the measured clean frames; the generated interface
trajectories add 3,000 production frames, and are curated with the same
per-frame SCF and equilibration filter before entering the pool.

| environment | current clean pool | after interface generation (target) |
|---|---:|---:|
| bulk LiF | 5,131 | 5,131 |
| F-doped Li | 4,555 | 4,555 |
| interface | 1,620 | 4,620 |
| **total** | **11,306** | **14,306** |

The F-doped total includes 2,603 recovered dense-NPT frames from
`LIWITHF_NPT_FINAL` (post-transition, SCF-clean); they stay labelled
`F_doped_dense` and are not silently pooled with the NVT family. The interface
total includes 390 recovered NPT frames (`interface_npt`), which are a labelled
sub-environment and not mixed into the new NVT trajectories.

Splits are group-level and time-blocked, rebuilt by `build_phase1_splits.py`
after the pool is frozen:

- **Pool** = `train_pool`: all acquisitions are drawn from it.
- **Validation** = `calibration`: early stopping and calibration fitting only,
  never acquisition.
- **Test** = `test`: locked, scored once.
- **OOD test** = `ood_test`: locked, reported separately.

Target composition is ≥ 500 interface frames in each of test and OOD, and a
pool in which no environment is below 20% of the total. If the generated
interface data fall short of 3,000 clean frames, the interface claim is scoped
as a case study rather than silently reweighted.

**Leak guarantees:**

1. The test and OOD sets never enter the pool.
2. The calibration set never enters the pool.
3. Acquisition ranks pool frames by model uncertainty **only** — no label of a
   pool frame is used to score it or its neighbours.
4. Every acquisition is a reveal of an existing DFT label; no new DFT is
   generated.

---

## 3. Models and acquisition functions

**Ensemble size K = 3** for the loop arms. Each retrain uses the same recipe as
the final fine-tuning run: 150 epochs, seeds 123/124/125, cosine schedule; the
snapshot arm uses SGDR with `T₀ = 50` so that three cycle minima fall inside the
budget.

| method | spread source | body training |
|---|---|---|
| Independent committee | std across 3 separately fine-tuned models (bootstrap per member) | 3 fine-tunes |
| Warm-restart snapshots | std across 3 cycle-end checkpoints of one SGDR run | 1 fine-tune |
| Shared-body multi-head (MHC) | std across K readout heads on one frozen body | 0 (heads only) |

**Per-atom uncertainty** is the RMS component standard deviation across the M
members, `σ_i = √( (1/M) Σ_m |F_i^m − F̄_i|² )`.

**Primary acquisition score** (per configuration): the maximum relative force
standard deviation over its atoms,

  `score(x) = max_i  σ_i / (|F̄_i| + 0.2 eV/Å)`.

**Secondary acquisition score** (sensitivity check): the mean per-atom force
standard deviation, `score'(x) = mean_i σ_i`, to confirm the ranking is not an
artifact of the `0.2` regularizer or the max aggregation. Both are run; the
primary score is used for the headline result.

**Random baseline:** stratified uniform sampling from the pool (seeded), using
the same per-environment quota as the acquisition arms.

The `multihead` strategy that currently exists in `uq_active_learning.py` is the
MH-FT path (trainable body, replay head), not the frozen-body MHC described
here. It must not be reported under the MHC label; the frozen arm needs a
separate head-only training path, and its results are keyed `mhc` in the
analysis.

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
- **Budget / stopping:** 15 rounds (150 acquired labels, 180 total). The loop
  stops earlier if calibration force RMSE has changed by < 1% over three
  consecutive rounds. If the curve is still descending at round 15, the run is
  extended in 5-round blocks up to 30 rounds; the extension rule is applied
  identically to every arm.
- **Locked evaluation:** the test and OOD sets are scored once, after the loop
  and after the arm configurations are frozen.

**Reference points:** the "full-pool" model is trained on the entire `train_pool`
and sets the accuracy ceiling. Because a 180-label model will not reach
`1.05 ×` a 14,000-frame ceiling, the primary AL metric is **error at matched
budget**, not labels-to-target; LTT is reported only if the target is actually
reached.

---

## 5. Metrics

Report per method and for the random baseline:

1. **Error at matched budgets** — locked-test force RMSE at 60, 100, 140, 180
   labels (and at any extension budgets), with a paired block bootstrap over
   trajectories. This is the primary AL comparison.
2. **Data-efficiency curve** — calibration force RMSE (and energy RMSE) vs.
   cumulative labeled frames, with per-round points.
3. **Labels-to-target (LTT)** — acquired labels to first reach `1.05 ×` the
   full-pool RMSE, reported only if reached.
4. **Compute-to-target (CTT)** — cumulative wall-clock to first reach the
   target, or cumulative wall-clock at the largest common budget if never
   reached. Per-round cost is measured per method.
5. **Acquired-environment composition** — fraction of acquired frames from each
   environment over rounds (H3 diagnostic).
6. **Within-environment error** — bulk, F-doped, and interface reported
   separately; pooled numbers are secondary.

Both LTT and CTT are reported; the headline is error at matched budget, because
the paper's thesis is about cost at a fixed label budget.

---

## 6. Frozen-body multi-head arm (outside the loop)

The multi-head committee is evaluated at fixed budgets of 500, 1,000, and 2,000
labels, drawn by the committee arm's acquisition order so that all arms see the
same frames. Four heads share one frozen body and receive disjoint label
partitions (125, 250, and 500 frames per head). A **frozen probe** — one head,
all labels — is run at the same budgets as the attribution control: the
`probe − committee` gap measures the cost of freezing the body, and the
`MHC − probe` gap measures the cost of splitting the labels across heads.

The gate in §1 is evaluated at 2,000 labels. If it passes, the multi-head arm is
promoted into the main text as a cost-ladder point; if it fails, it is reported
as an ablation with the probe attribution, and the AL loop remains the
committee/snapshot/random comparison.

---

## 7. OOD handling

- **Stratified AL** (primary): acquisition is stratified across bulk LiF,
  F-doped Li, and the interface; evaluation is reported per environment.
- **OOD diagnostic** (secondary): report whether each method acquires interface
  frames earlier than the random control.

The clean interface pool is 1,620 frames before generation and about 4,620
after, so the OOD leg is quantitative rather than a case study. The recovered
interface NPT frames (`interface_npt`) are a stress test with a higher SCF
warning rate and are reported separately from the clean in-domain result.

---

## 8. Reproducibility

- All random numbers (seed selection, random baseline, committee seeds
  123/124/125) are fixed and recorded.
- The pool ordering, seed indices, and every acquired index are logged per round
  in `al_manifest.json`.
- The test and OOD sets are byte-for-byte the same across all methods and rounds.
- The split files are generated from `frames.csv` by
  `build_phase1_splits.py`; their SHA-256 hashes are in `split_manifest.json`.
- Statistical intervals use `uq_analysis.py` with a moving-block bootstrap
  (block = 20 frames) and paired resampling for method differences.

---

## 9. Outputs

1. Per-method data-efficiency and compute-efficiency curves.
2. Error-at-matched-budget and CTT tables vs. the random baseline.
3. Acquired-environment-composition plots (H3).
4. `al_manifest.json` (seed, pool, per-round acquisitions, costs).
5. Multi-head fixed-budget table with the frozen-probe attribution.
6. A one-line pre-registered verdict per the decision rules in §1.
