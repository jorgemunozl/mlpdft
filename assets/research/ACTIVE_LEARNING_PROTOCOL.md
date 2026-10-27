# Active-Learning Experimental Protocol — Committee vs. Multi-Head vs. Snapshot UQ

**Date:** 2026-10-27
**Status:** pre-registration — fixes loop design, acquisition, budgets, metrics,
and decision rules *before* the run.
**Implementation:** `src/mlpdft/uq_active_learning.py` (committed with this
protocol). Resolves audit items **C1, C2, C3** (loop parameters, acquisition
function, label leak) and touches **A4** (temporal correlation), **B6** (OOD
regime), **B5** (regularizer), and **E2** (cost).

---

## 1. Objective and hypotheses

**Objective.** Quantify, for each of the three uncertainty strategies, how many
additional labeled configurations — and how much wall-clock compute — are needed
to reach a target force accuracy on the locked test set, starting from a small
seed.

**H1 (calibration → data efficiency).** A method whose uncertainty is better
calibrated against true force error (from the run-2 calibration leg) acquires
more informative configurations per label, i.e. reaches the target with fewer
labels.

**H2 (snapshot cost-efficiency).** The warm-restart snapshot ensemble reaches
the target at a small multiple of the committee's label count but at a fraction
(≤ 1/3) of its wall-clock cost, because each retrain costs one fine-tune instead
of K.

**H3 (OOD awareness).** A method with intact out-of-distribution uncertainty
preferentially acquires interface (OOD) configurations early; a method whose
uncertainty collapses OOD does not.

### Pre-registered decision rules

| outcome | conclusion |
|---|---|
| method's data efficiency ≤ random (within noise) | its uncertainty carries **no useful AL signal** |
| snapshot `labels-to-target` ≤ 1.5× committee **and** `compute-to-target` ≤ committee/2 | snapshot **replaces** committee for practical AL |
| snapshot ≈ committee on labels but much cheaper | snapshot is the **cost-effective** choice |
| committee (with bootstrap) dominates on both | committee remains the **gold standard**; snapshot is "cheaper but less efficient" |
| all three ≈ random | uncertainty is not the bottleneck; reframe the paper's AL claim |

These are fixed before looking at results and will be reported verbatim.

---

## 2. Data and pool definition (resolves C3, A4)

- **Pool** = the training split from run 2 (`uq_train.extxyz`, 214 frames:
  LIF64 96, LIWITHF 96, LIFINTERFACE 22).
- **Validation** = `uq_valid.extxyz` (53 frames) — used **only** for early
  stopping inside each retrain, never for acquisition.
- **Test** = `uq_test.extxyz` (178 frames: 80/80/18) — **permanently locked**;
  its labels are never revealed to training, acquisition, or early stopping at
  any iteration.

These three are already disjoint contiguous time blocks (built by
`build_uq_splits.py`), so the train/valid/test split itself is leak-free (A4)
and the pool carries no test-label information.

**Leak guarantees (C3):**
1. The test set never enters the pool.
2. The validation set never enters the pool.
3. Acquisition ranks pool frames by model uncertainty **only** — no label of a
   pool frame is used to score it or its neighbors.
4. Every "acquisition" is a *reveal* of a pool frame's existing DFT label; no
   new DFT is generated.

---

## 3. Models and acquisition functions (resolves B1, B3, B5, C2)

**Ensemble size K = 3** for every method, matching run 2. Each retrain uses the
same recipe (150 epochs, seed(s) 123/124/125, SGDR with `T₀ = 50` for the
snapshot, bootstrap-default committee).

The three methods differ only in where the ensemble spread comes from:

| method | spread source |
|---|---|
| Independent committee | std across 3 separately fine-tuned models (bootstrap per member) |
| Multi-head | std across 3 readout heads on one shared body |
| Warm-restart snapshots | std across 3 cycle-end checkpoints of one SGDR run |

**Per-atom uncertainty** is the RMS component standard deviation across the M
members, `σ_i = √( (1/M) Σ_m |F_i^m − F̄_i|² )`, matching the calibration leg.

**Primary acquisition score** (per configuration): the maximum relative force
error over its atoms,

  `score(x) = max_i  σ_i / (|F̄_i| + 0.2 eV/Å)`.

**Secondary acquisition score** (sensitivity check, C2/B5): the mean per-atom
force std, `score'(x) = mean_i σ_i`, to confirm the ranking is not an artifact
of the `0.2` regularizer or the max aggregation. Both are run; the primary score
is used for the headline result.

**Random baseline:** uniform sampling from the pool (seeded), as the null that
any UQ signal must beat.

---

## 4. Loop mechanics (resolves C1)

- **Seed**: 30 frames, balanced per group (10 LIF64, 10 LIWITHF, 10
  LIFINTERFACE), chosen deterministically as evenly-spaced indices in trajectory
  order — *not* the first 10, to avoid a temporally clustered seed (A4).
- **Batch size b = 10** frames per iteration.
- **Acquisition** (each iteration): rank all unlabeled pool frames by the
  current ensemble's acquisition score; take the top-b. **Diversity guard**: no
  two selected frames may be closer than `gap = 5` pool positions within the
  same group (prevents acquiring temporally adjacent, redundant neighbors).
- **Retrain**: reveal the b labels, retrain the ensemble from seed + acquired,
  evaluate on the locked test set.
- **Budget / stopping**: stop when the pool is exhausted (~18 iterations), or
  when test force RMSE has changed by < 1% over three consecutive iterations
  (whichever first).

**Reference points:** the "full-data" model = run-2 calibration model trained on
all 214 frames; its test RMSE is the accuracy ceiling the loop approaches. A
"target reached" event is test force RMSE ≤ 1.05 × the full-data RMSE.

---

## 5. Metrics (resolves E2)

Report per method and for the random baseline:

1. **Data-efficiency curve** — test force RMSE (and energy RMSE) vs. cumulative
   labeled fraction, with per-iteration points.
2. **Labels-to-target (LTT)** — number of acquired labels to first reach the
   target.
3. **Compute-to-target (CTT)** — cumulative wall-clock (seconds) across all
   retrains to first reach the target. Per-iteration cost is measured and
   recorded per method (committee ≈ 3× a fine-tune, multi-head ≈ head-training,
   snapshot ≈ 1× a fine-tune).
4. **Acquired-group composition** — the fraction of acquired frames from each
   environment (bulk / Li-rich / interface) over time (H3 diagnostic).

Both LTT and CTT are reported; the headline is CTT, because the paper's thesis
is about cost.

---

## 6. OOD handling (resolves B6)

- **In-domain AL** (primary): acquisition and evaluation within bulk LiF +
  Li-rich Li–F.
- **OOD acquisition** (secondary): the pool includes interface frames; report
  whether each method *acquires* interface frames early (uncertainty recognizes
  them as high-error) or ignores them (uncertainty collapsed OOD).

The interface pool is small (22 train / 18 test), so the OOD leg is reported
**qualitatively** (sign and rank of interface acquisitions), not as a precise
label count — pending the interface-DFT augmentation already planned.

---

## 7. Reproducibility

- All random numbers (seed selection, random baseline, committee seeds 123/124/125)
  are fixed and recorded.
- The pool ordering, seed indices, and every acquired index are logged per
  iteration in `al_run_manifest.json`.
- The run-2 test set is byte-for-byte the same across all methods and iterations.

---

## 8. Deliverables

1. Per-method data-efficiency and compute-efficiency curves (figure).
2. LTT and CTT table vs. random baseline (Table for the paper).
3. Acquired-group-composition plots (H3).
4. `al_run_manifest.json` (seed, pool, per-iteration acquisitions, costs).
5. A one-line pre-registered verdict per the decision rules in §1.
