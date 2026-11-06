# Experiment 3 — why the frozen-body MHC pilot fails

**Status:** diagnosis of existing runs only; no new training.
**Runs:** `lif_uq_mhc_002` (400 epochs, 4 heads) and `mhc150_s123` +
`probe150` (150 epochs). Data: the frozen 214 / 53 / 178 `li_f_uq_splits_v2`
split.
**Companion reports:** [`GATE.md`](GATE.md) (400 epochs) and
[`ANALYSIS.md`](ANALYSIS.md) (150 epochs, probe attribution).

## Verdict

The pilot is **not a test of the paper's MHC**. It is a warm-started,
energy-misaligned, data-starved frozen-readout ensemble. That recipe fails the
pre-registered gate, but the failure cannot be attributed to the frozen-body
multi-head committee as published.

The paper's construction — eight **randomly initialised** heads, frozen AEDs and
original head, 8,000 structures (1,000 per head), 800,000 optimisation steps —
is different from what was trained here. See Beck et al.,
[arXiv:2508.09907](https://arxiv.org/abs/2508.09907), Methods and Fig. 3.

## What the runs show

| Quantity | 400 ep | 150 ep | Committee |
|---|---:|---:|---:|
| Force RMSE (eV/Å) | 0.010237 | 0.010563 | 0.008659 |
| Overall ratio | 1.182 | 1.220 | 1.00 |
| Bulk LiF ratio | **1.641** | **1.360** | 1.00 |
| F-doped ratio | 1.187 | 1.188 | 1.00 |
| Interface ratio (n=18) | 1.143 | 1.226 | 1.00 |
| Within-group mean Spearman | 0.457 | 0.302 | 0.656 |
| Energy RMSE (meV/atom) | 2205 | 2469 | 0.099 |
| Calibration α (force) | — | 51.6 | — |
| Coverage at 1σ (after α) | — | 0.845 | — |
| RLL (after α) | — | −0.138 | — |

The bulk-force failure is robust across both budgets. The rank failure is not:
the interface point estimate (n = 18) flips sign between the two runs and
dominates the three-group mean. Per-head force RMSE is 10.5 / 10.38 / 10.67 /
10.75 meV/Å (150 ep) and 10.15 / 10.16 / 10.42 / 10.29 meV/Å (400 ep); the
ensemble is only 0.01 meV/Å better than the average member and worse than the
best member.

## Causes, ranked

### 1. All four heads were warm-started from the same foundation readout

The 400-epoch [`training/run_config.json`](training/run_config.json) has
`"foundation_filter_elements": true`. In mace-torch 0.3.15 that flag makes
`load_foundations_elements` copy the foundation readout into **every** head
(the weights are repeated across the head axis). All four heads therefore start
identical; the 150-epoch log starts from the same 2.66 eV/atom offset.

The paper explicitly uses **random initialisation** for the new heads. That is
the primary source of head diversity; the disjoint data slices are the second.
With identical initialisation plus a frozen body, the heads converge to nearly
the same function:

- per-head force RMSE differs by < 0.4 meV/Å;
- α = 51.6, i.e. the raw disagreement is ~50× too small for the error;
- the ensemble barely improves on the average member (10.24 vs 10.25 meV/Å at
  400 epochs) and is worse than the best member (10.15 meV/Å): essentially no
  committee variance reduction;
- RLL = −0.14, worse than predicting the constant RMSE.

This is head collapse, not a committee.

**Fix:** random-init heads (`foundation_filter_elements=False`, no
`--warm-start-heads`). Do not use the warm-start option for the paper's MHC; it
is a different method.

### 2. The E0 reference is inconsistent with the warm-started readout

The run uses QE isolated-atom energies (log: Li −205.7175 eV, F −795.4213 eV).
The foundation readout was trained against MACE-MP-0's own E0s. Warm-starting
the readout while swapping E0s leaves a species-dependent residual: the energy
error is 2.2–2.5 eV/atom for the whole run, four orders of magnitude above the
committee's 0.1 meV/atom.

MACE's `UniversalLoss` is a Huber loss with δ = 0.01, so an error of 2.5 eV/atom
produces a **clipped, nearly constant** gradient. The offset is learned very
slowly, dominates the validation loss, and selects a loss-best checkpoint rather
than a force-best one (per-head force optimum 9.33–9.55 meV/Å vs 10.15–10.42 in
the saved model).

**Fix, in order of preference:**

1. Random heads with the dataset's QE E0s (the paper's setup; no inherited
   reference mismatch).
2. If warm-starting, keep the foundation E0s and fit the QE offset with an
   explicit trainable per-head bias.
3. For a force-focused pilot, set `energy_weight = 0` and select the checkpoint
   on validation force RMSE.

The GATE report's "energy term is 375× the force term" overstates the gradient
effect — Huber loss clips large errors — but the E0 offset is still a real
confound for checkpoint selection and energy reporting.

### 3. The frozen body may not be the foundation body

`load_foundations_elements` in mace-torch 0.3.15 copies the node embedding,
interactions, products, Bessel weights, readout and scale/shift, but **not**
`radial_embedding.distance_transform` (an `AgnesiTransform` with its own `q`,
`p`, `a`, `alpha`). Commit `2681ec9` added that copy inside
`_checked_configure_model`. Verify the run log contains no
“distance transform not fully copied” warning; if the copy failed, the frozen
body is not the foundation body and the paper's premise (“leave the AEDs
untouched”) is violated.

**Fix:** add a hard assertion that every descriptor tensor in the frozen model
equals the foundation checkpoint at initialisation (embedding, interactions,
products, radial/Bessel and distance transform).

### 4. The per-head data and step budget are far below the paper

| | this pilot | paper |
|---|---:|---:|
| Heads | 4 | 8 |
| Frames per head | 54 / 54 / 53 / 53 | 1,000 |
| Gradient updates | 2,006 (150 ep), 5,350 (400 ep) | 800,000 |
| Trainable parameters | 1.35% (MHC), 0.33% (probe) | < 0.2% |

The paper's Fig. 3 explicitly warns that disjoint heads lose accuracy when the
per-head subset is small. At 54 frames/head this is the expected regime. An
overlapping block-bootstrap split (each head sees all 214 frames, different
resamples) is the appropriate test at this data scale; the paper supports
overlapping heads and reports similar correlation with worse scaling.

### 5. The target is out of distribution for the foundation model

The committee adapts the body and reaches 2.7 meV/Å in bulk; the frozen probe
(all 214 labels) reaches 4.1 meV/Å (+53%). That is consistent with a genuine
frozen-representation ceiling for QE Li–F, which is not the MPtrj/OMAT regime
where the paper reports only marginal degradation. But the probe is itself
warm-started and E0-mismatched, so this attribution is **not clean** yet.

### 6. The gate's rank leg is unstable

The three-group mean includes the interface with n = 18, less than one
20-frame bootstrap block. It flips from +0.08 (400 ep) to −0.20 (150 ep). The
bulk-force failure is the robust part; the rank leg should be reported on bulk
and F-doped only, with the interface as a case study.

## What is and is not established

**Established:** this warm-started, E0-misaligned, 54-frame/head recipe fails
the gate; bulk force is 1.36–1.64× the committee.

**Not established:** that the frozen-body MHC is the cause. The probe was
supposed to separate freezing from splitting, but it shares causes 1–3. The
`C − D` gap (+4.4% overall, −11.2% bulk) cannot be read as “splitting is cheap”
until both arms use random initialisation and a consistent E0 reference.

## Minimum fair test

1. Random-init four heads, `foundation_filter_elements=False`.
2. QE E0s from the data; no warm-start; or explicit residual-bias correction.
3. Overlapping block-bootstrap heads, all 214 frames per head, or accept that
   disjoint 54-frame heads are not viable at this scale.
4. Force-focused run (`energy_weight=0`), 1,000+ epochs, readout LR ~1e-3,
   early stopping on validation force RMSE, evaluate the force-best checkpoint.
5. Clean random-init frozen probe with the same recipe.
6. Gate on bulk + F-doped force and rank; interface n = 18 reported separately.

## Decision rule

If the clean random-init, overlapping-head run still has bulk force > 1.25× the
committee and within-group ρ below the committee, then the negative result is
real for this testbed: a frozen MACE-MP-0 representation is insufficient for
QE Li–F, and body adaptation is required. That is a defensible claim about
domain shift, not a failure of the published MHC.

If the clean run closes the gap, the pilot failure is a recipe artefact and must
not be reported as evidence against MHC.

## Files

- [`GATE.md`](GATE.md) — 400-epoch gate arithmetic.
- [`ANALYSIS.md`](ANALYSIS.md) — 150-epoch run and probe attribution.
- [`training/run_config.json`](training/run_config.json) — `foundation_filter_elements: true`, weights.
- [`mhc150_s123/train.log`](mhc150_s123/train.log) — per-head progress, E0s, updates.
- [`probe150/train.log`](probe150/train.log) — probe warm start and frozen-body check.
- [`mhc150_s123/evaluation/uq_summary.json`](mhc150_s123/evaluation/uq_summary.json) — α, coverage, RLL.
