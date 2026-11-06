# UQ Experiment 3 — Frozen-Body Multi-Head Committee (MHC)

**Status:** 🟡 **ready to run** — code path implemented, smoke gate not yet passed.
**Predecessors:** [`UQ_EXPERIMENT_2.md`](UQ_EXPERIMENT_2.md),
[`UQ_EXPERIMENT_PROTOCOL.md`](UQ_EXPERIMENT_PROTOCOL.md).
**Method:** Beck et al., *Multi-head committees enable direct uncertainty
prediction for atomistic foundation models*, arXiv:2508.09907.

> **First MHC run superseded.** The first run uses the **4-head, disjoint**
> design in [`UQ_MHC_4HEAD_DISJOINT_DESIGN.md`](UQ_MHC_4HEAD_DISJOINT_DESIGN.md),
> on the 1,000-frame two-environment interim candidate (250 frames/head). The
> 8-head, 80%-overlapping figures and commands below describe the earlier plan
> and are retained for the broader protocol, not for this run.

## Purpose

Experiment 2's `multihead_bootstrap` arm is **not** the paper's MHC. It is the
released `mace-torch 0.3.15` multi-head fine-tuning path: a trainable shared
body, heads warm-started from the foundation readout, and an MP replay head
(requested as `pt_train_file='omat'`, which mace-torch 0.3.15 routes to MP) that
dominates the batch mixture. The paper instead **freezes the atomic
environment descriptors (AEDs)**, attaches **randomly initialised** heads, and
trains **only the readouts** on a partition of the training data (disjoint for
the foundation model; see the design note above).

Because this project is about uncertainty for *foundation* models, the frozen-body
MHC is the arm that actually tests the claim. Experiment 3 adds it, plus the
control needed to attribute any accuracy change.

Experiment 2 is not rerun or edited. Its `multihead_bootstrap` result is retained
and must be labelled **"multi-head fine-tuning with replay"** in the paper, never
"MHC".

## Design

| Arm | Body | Heads | Data | Role |
|---|---|---|---|---|
| A. Independent committee | fine-tuned | 3, independent seeds | bootstrap of 214 | reused from Exp 2 |
| B. Warm snapshots | fine-tuned | 3 cycle-end checkpoints | 214 | reused from Exp 2 |
| **C. Frozen MHC** | **frozen AEDs** | **8, random init** | **overlapping 80% (~172/head)** | new |
| **D. Frozen probe** | **frozen AEDs** | **1, random init** | **all 214** | new — control |

Arm D separates the two things C changes at once: *freezing the body* (D vs A)
and *splitting the data across heads* (C vs D). Without D, a poor C cannot be
attributed.

**Heads:** 8, as in the paper, with the **overlapping** split: each head gets an
independent 80% of the frames (~172 of 214), so every head is trainable. A
disjoint split at this scale would give 27 frames/head; the paper itself warns
that disjoint heads degrade when the per-head subset is small (Fig. 3
discussion). The exact frame indices are written to
`mhc_head_data/mhc_split.json`. `--mhc-split disjoint` remains available for the
reduced-scale comparison, but it is not the default.

**Frozen set:** `node_embedding`, `interactions`, `products`, the radial Bessel
weights, and the (parameter-free) spherical harmonics. Trainable: the readout
blocks only — the per-head output MLPs. No `pt_head`, no replay data,
`multiheads_finetuning=False`.

The runner enforces this at model-build time: it aborts if `freeze != 6`, if
`foundation_filter_elements` is true (which would copy the foundation readout
into every head and make them start identical), or if any descriptor tensor is
still trainable when the optimizer is created.

## Configuration

- Foundation checkpoint, dtype, split files, valid/test sets: identical to Exp 2.
- MHC: `--mhc-heads 8`, `--mhc-split overlapping --mhc-overlap 0.8`,
  `--mhc-epochs 120` (~10k updates over the concatenated heads), batch 16,
  lr 1e-3, `--energy-weight 0`, `--patience 50`, one cosine decay.
- Probe: `--probe-epochs 600`, same force-only loss and lr, random-init head.
- Split seed `20251027`, recorded in `mhc_head_data/mhc_split.json` with the
  exact frame indices per head.
- Evaluation is on the multi-head model directly with `head=<name>` — **not** via
  `remove_pt_head` export, which applies a $1/\sqrt{M}$ rescaling.
- Early stopping (patience 50) selects the best validation **force** checkpoint,
  because the frozen arms are trained with `energy_weight=0`; the α fitted on the
  53 validation frames is therefore on frames not used to fit the readout.
- Checkpoints are not retained: warm-restart runs write only the three cycle-end
  states and delete them after exporting `snapshot_models/`; every other run's
  checkpoint directory is removed once training finishes.
- Isolated-atom references are restricted to the species present in the splits
  (Li, F). `ENERGY_OFFSET` also contains B, and passing it unfiltered fails with
  a 2-vs-3 shape mismatch.
- Committee, snapshot, and zero-shot numbers are merged from the Exp 2
  `uq_summary.json` via `--include-summary`; they are not recomputed.

## Pre-registered decision rules

Fixed before seeing Arm C/D test results.

1. **Accuracy (H1).** MHC "preserves accuracy" only if its within-group force
   RMSE is ≤ 1.25× the committee's in **every** group. Otherwise report the
   degradation and its size; do not call it preserved.
2. **Ranking (H2).** Compare within-group frame-level Spearman with
   block-bootstrap CIs. "Comparable to the committee" requires overlapping CIs
   in LIF64 and LIWITHF. LIFINTERFACE (n=18) is exploratory only.
3. **Cost (H3).** Report MHC wall clock as a fraction of the 3-member committee
   total from Exp 2. The efficiency claim requires < 100% (cheaper than retraining
   three models); ≤ 50% is reported as strong efficiency. At ≥ 100%, drop the
   claim.
4. **Attribution (H4).** Report D vs A (cost of freezing) and C vs D (cost of
   splitting the labels across heads) separately. Whichever gap dominates is the
   explanation to state.
5. **Diversity (H5).** Report pairwise head force RMS differences and the ratio
   of interface to in-distribution mean head σ. A near-zero spread invalidates
   the arm as a committee; a large but error-independent spread invalidates it
   as UQ.

## Smoke gate (must pass before the full run)

```bash
uv run --locked python -m mlpdft.uq_compare \
  --output-dir src/mlpdft/outputs/uq_diffusion_comparison \
  --run-id lif_uq_mhc_smoke \
  --arms mhc,frozen-probe --smoke-test --overwrite
```

Pass criteria, all printed or written to `evaluation/smoke_check.json`:

- `[uq_compare] frozen-body check passed: …` with a trainable fraction consistent
  with readout-only training (a few percent of the model, not the descriptor
  stack);
- no `trainable descriptor tensors` error;
- `smoke check passed: heads differ` — minimum pairwise head force RMS
  difference > 0;
- 8 heads listed as available and selected by name;
- finite predictions, no NaN in forces.
- `learning gate` passes: validation force RMSE falls (or starts ≤0.05 eV/Å and
  stays within 5% of its best).

If any check fails, stop. Do not run the full experiment against a broken arm.

## Full run

The runner's no-flag default **is** this experiment:

```bash
uv run --locked python -m mlpdft.uq_compare
```

It expands to `--arms mhc,frozen-probe --mhc-heads 8 --mhc-split overlapping
--mhc-overlap 0.8 --mhc-epochs 120 --probe-epochs 600 --energy-weight 0
--learning-rate 1e-3 --patience 50 --run-id lif_uq_mhc_003`, and it automatically
merges `lif_uq_001/evaluation/uq_summary.json` when that directory exists. The
explicit equivalent, for a different output directory or run id:

```bash
uv run --locked python -m mlpdft.uq_compare \
  --output-dir src/mlpdft/outputs/uq_diffusion_comparison \
  --run-id lif_uq_mhc_003 \
  --arms mhc,frozen-probe \
  --mhc-heads 8 --mhc-split overlapping --mhc-overlap 0.8 \
  --mhc-epochs 120 --probe-epochs 600 \
  --energy-weight 0 --learning-rate 1e-3 --patience 50 \
  --include-summary src/mlpdft/outputs/uq_diffusion_comparison/lif_uq_001/evaluation/uq_summary.json
```

Expected cost: one MHC run plus one probe run on 214 frames. Note that force
training still backpropagates through the frozen body to the atomic positions,
so the saving is smaller than "readout-only" suggests — do not assume this beats
the committee until the wall clock says so. Record the measured time and do
**not** quote the paper's 800k-step cost.

Hardware: Exp 2 was measured on an A40, not an A100 (17.7 h for the full run).
Run Exp 3 on the same GPU class, or report normalised GPU-hours; raw seconds
from A40 and A100 are not comparable.

## Analysis outputs

Emitted by the runner into `evaluation/`:

- `uq_summary.json` — all methods, per-group and overall: force/energy MAE and
  RMSE; frame-level and per-atom **Spearman** and **Pearson** correlations;
  risk–coverage at 10/25/50/75/100%; calibration α on the 53 validation frames
  (Eq. 5), **RLL** and **1σ/2σ empirical coverage** on the locked test; member
  accuracy.
- `<method>_predictions.csv` — per-frame errors and σ.
- `<method>_atom_level.npz` — per-atom σ and |error|, for per-atom coverage and
  calibration curves.
- `smoke_check.json`, `experiment_manifest.json`, `mhc_head_data/mhc_split.json`.

**Energy caveat:** the frozen arms use `energy_weight=0` because the QE and
foundation E0 references differ. Their energy MAE/RMSE in the summary is a
by-product, not a calibrated prediction; report force accuracy and UQ, and use
the committee's energy as the reference.

Added in analysis, not by the runner: block-bootstrap CIs and the figures. RLL
and coverage are already per group, so they can be read directly from the
summary. Pooled Spearman is secondary and must be labelled as such; per-group
values are the headline.

**Imported arms:** the committee, snapshot and multi-head rows merged from
`lif_uq_001/evaluation/uq_summary.json` were evaluated before Pearson, RLL and
coverage existed, so they carry only the old fields. Re-evaluate their saved
models with the current runner before comparing calibration across arms; the
ranking comparison is unaffected.

## Deliverables

1. Four-arm table (A/B reused, C/D new) with per-group n, force/energy accuracy,
   within-group Spearman + Pearson + CI, α, RLL, 1σ/2σ coverage, risk–coverage.
2. Cost table: training seconds for C and D, versus the Exp 2 committee and
   snapshot runs.
3. One paragraph answering: does a frozen foundation body plus 8 cheap heads give
   committee-quality uncertainty at this data scale?

## Negative-result branch

If C's σ is flat (risk–coverage curve as flat as Exp 2's multi-head arm) or its
accuracy exceeds the 1.25× bound, the paper reports: *at ~172 frames/head
(80% overlapping), a frozen-body MHC does not reproduce committee-quality UQ on
this Li–F testbed*, with the D-vs-A and C-vs-D gaps as the explanation. That is
a valid result and does not require another architecture.

## Limitations to state

- 8 heads at ~172 frames/head (80% overlapping), not the paper's 8 × 1,000
  disjoint frames.
- 10k gradient updates, not 800k.
- One MHC training run (heads are not independent seeds); head spread is
  readout-init plus data-split variance only.
- LIFINTERFACE has 18 test frames; its correlations are case-study evidence.
- Frozen-body accuracy is bounded by the foundation checkpoint's representation
  of Li–F; a poor arm D is a property of the checkpoint, not of MHC.
