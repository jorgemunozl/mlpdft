# Experiment 3 — frozen-body MHC runbook

**Scope:** the frozen-body multi-head committee (MHC) and the single-head
frozen probe, on the 214 / 53 / 178 Li–F split used by Experiment 2.
**Status:** pilot recipe corrected after `DIAGNOSIS.md`; no production run yet.
**Code:** current `main` (`uq_compare.py` with random-init heads, overlapping
split, force-focused frozen arms).

> **Superseded for the first MHC run.** This runbook describes the 214-frame
> pilot split and an 8-head, 80%-overlapping recipe. The first full MHC run
> uses **4 heads, disjoint**, on the 1,000-frame two-environment interim
> candidate: see
> [`UQ_MHC_4HEAD_DISJOINT_DESIGN.md`](UQ_MHC_4HEAD_DISJOINT_DESIGN.md) for the
> rationale, exact command and gates. Use that document, not this runbook, to
> launch the MHC arm.

---

## 1. What this run is testing

| Arm | Body | Heads | Training data |
|---|---|---|---|
| Independent committee (Exp 2) | 3 fine-tunes | 1 per model | all 214, bootstrapped |
| Warm snapshots (Exp 2) | 1 fine-tune | 3 cycle-end checkpoints | all 214 |
| MH-FT (Exp 2) | fine-tuned + replay | 3 bootstrap heads | all 214 |
| **Frozen MHC (this run)** | **frozen** | **8 heads** | **overlapping 80% (~172/head)** |
| **Frozen probe (this run)** | **frozen** | **1 head** | **all 214** |

`C − D` (MHC − probe) measures the cost of splitting labels across heads.
`D − A` (probe − committee) measures the cost of freezing the body.

The heads are **randomly initialised**, as in Beck et al. The earlier pilot
warm-started all heads from the same foundation readout; that collapsed the
committee (α ≈ 52) and is removed from the paper path.

## 2. Data — the 214 split, not the 8,313 pool

Experiment 3 uses `li_f_uq_splits_v2`: **214 train / 53 valid / 178 test**.
The final paper pool is 8,313 frames; that is a different experiment. Uploading
the wrong split makes one epoch take ~20 minutes instead of ~1 minute.

```bash
# on the pod
for f in uq_train uq_valid uq_test; do grep -c '^Lattice' dataset/xyz_files/uq_splits/$f.extxyz; done
# expected: 214 / 53 / 178
```

## 3. Commands

### 3.1 Smoke gate (~1 h)

```bash
uv run --locked python -m mlpdft.uq_compare \
  --smoke-test --output-dir /workspace/runs --run-id smoke1 \
  --device cuda --dtype float64 --overwrite
```

The smoke run uses 20 epochs, 8 heads, overlapping data, `energy_weight=0`,
`lr=1e-3`. It must report:

- `frozen-body check passed` with no trainable descriptor tensors;
- `distance transform copied from the foundation`;
- `smoke check passed: heads differ` over all 8 heads;
- the learning gate passes (validation force RMSE falls, or starts ≤0.05 eV/Å
  and stays within 5% of its best).

If any check fails, stop.

### 3.2 Full run

```bash
uv run --locked python -m mlpdft.uq_compare \
  --output-dir /workspace/runs --run-id lif_uq_mhc_003 \
  --device cuda --dtype float64 \
  --mhc-heads 8 --mhc-split overlapping --mhc-overlap 0.8 \
  --mhc-epochs 120 --probe-epochs 600 \
  --energy-weight 0 --learning-rate 1e-3 --patience 50
```

`--mhc-epochs 120` is one pass over 8 × 80% of 214 frames per epoch
(≈10k gradient updates). The probe sees all 214 frames per epoch and uses 600
epochs. Expect roughly 7–10 h per arm on an A40.

The no-flag default is the same recipe with `--output-dir
src/mlpdft/outputs/uq_diffusion_comparison` and run id `lif_uq_mhc_003`; use the
explicit command above for the corrected run.

### 3.3 Gate and attribution report

```bash
uv run --locked python src/mlpdft/utils/experiment3_report.py \
  --summary /workspace/runs/lif_uq_mhc_003/evaluation/uq_summary.json \
  --outdir  /workspace/runs/lif_uq_mhc_003/analysis
```

Writes `experiment3_report.md` and `.json`: the per-arm table, the per-group
MHC/committee ratio, the gate verdict, and the C−D / D−A deltas. The rank gate
uses only groups with n ≥ 40 (bulk and F-doped); the 18-frame interface is
reported separately.

## 4. Why random init and overlapping heads

- **Random init is the paper's recipe.** Beck et al. attach new heads with
  random weights; the frozen AEDs provide the representation and the heads
  provide diversity. Warm-starting every head from the same readout makes them
  identical and collapses the disagreement.
- **The E0 reference differs between MACE-MP-0 and the QE Li–F data** (Li
  −205.72 eV, F −795.42 eV). A warm-started readout inherits the foundation
  reference and produces a multi-eV/atom offset. With random heads and
  `energy_weight=0`, the arms are trained on forces only; energy predictions are
  not part of this gate.
- **80% overlap is necessary at 214 frames.** Disjoint 8-way gives 27
  frames/head; the paper's Fig. 3 warns that disjoint heads degrade when the
  per-head subset is small. Overlapping gives ~172 frames/head.
- **Distance transform.** `load_foundations_elements` does not copy the Agnesi
  distance transform; `_checked_configure_model` copies and asserts it, and the
  log must show `distance transform copied from the foundation`.

`--warm-start-readout` remains only as a single-head ablation and requires
`--energy-weight 0`. `--warm-start-heads` has been removed.

## 5. Gotchas

- **Energy metrics are not meaningful for the frozen arms** with
  `energy_weight=0`; compare forces and rank/calibration only. The committee's
  energy remains the reference.
- **GPU memory:** batch 16 / float64 on the 8,313-frame pool used ~41.5 GB.
  A 24 GB RTX 4090 will OOM there; an A40 48 GB works. On the 214 split memory
  is lower (~29–36 GB) but still above 24 GB at batch 16.
- **`patience`** defaults to 50 for frozen arms; with `energy_weight=0` the
  validation loss is force-only, so the saved checkpoint is the force-best one.
- **Group names:** the evaluation CSV groups are exactly
  `LIF64_KJPAW_FINAL`, `LIWITHF_KJPAW_FINAL`, `LIFINTERFACE_KJPAW_FINAL`.
- **Analysis alignment:** `uq_analysis.py` asserts identical `(group,
  frame_index)` sets across all prediction CSVs. Use the same 178-frame test
  split for the Exp 2 and Exp 3 CSVs.

## 6. Tracking caveat

The runs above are launched over raw SSH, so they do **not** appear in the
project's Experiments tab, which only lists `orx create-experiment` nodes. To
make a run visible there, create a node with the same run command and launch it
from a host with a writable orx store:

```bash
orx exp run <experimentId> --backend ssh --host <ssh-config-alias>
```

Until then, the durable record is the pod's `/workspace/runs/` plus the copies
pulled into `assets/research/PHASE-2/experiment3_runs/`.
