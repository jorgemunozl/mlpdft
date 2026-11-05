# Experiment 3 — frozen-body MHC runbook

**Scope:** the frozen-body multi-head committee (MHC) and the single-head
frozen probe, on the 214 / 53 / 178 Li–F split used by Experiment 2.
**Status:** pilot recipe frozen; the full `lif_uq_mhc_002` run is executing.
**Code:** `src/mlpdft/uq_compare.py` at `962a5db` or later.

---

## 1. What this run is testing

| Arm | Body | Heads | Training data |
|---|---|---|---|
| Independent committee (Exp 2) | 3 fine-tunes | 1 per model | all 214, bootstrapped |
| Warm snapshots (Exp 2) | 1 fine-tune | 3 cycle-end checkpoints | all 214 |
| MH-FT (Exp 2) | fine-tuned + replay | 3 bootstrap heads | all 214 |
| **Frozen MHC (this run)** | **frozen** | **4 heads** | **disjoint 54/54/53/53** |
| **Frozen probe (this run)** | **frozen** | **1 head** | **all 214** |

`C − D` (MHC − probe) measures the cost of splitting labels across heads.
`D − A` (probe − committee) measures the cost of freezing the body.

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

### 3.1 Warm-start check (2 min)

```bash
uv run --locked python -m mlpdft.uq_compare \
  --arms frozen-probe --output-dir /workspace/runs --run-id warmcheck \
  --device cuda --dtype float64 --probe-epochs 1 --patience 1 \
  --warm-start-readout --skip-evaluation --overwrite
```

The **first** validation line must be ≈15 meV/Å. If it is ≈190 meV/Å, the
warm start did not copy the full readout; do not launch the full run.

### 3.2 Smoke gate (50 min)

```bash
uv run --locked python -m mlpdft.uq_compare \
  --smoke-test --output-dir /workspace/runs --run-id smoke1 \
  --device cuda --dtype float64 --warm-start-readout --warm-start-heads --overwrite
```

The learning gate aborts unless validation force RMSE improves by 10% (or the
arm starts at ≤0.05 eV/Å and does not drift >5% above its best), and the final
force RMSE is ≤0.05 eV/Å (probe) or ≤0.10 eV/Å (MHC).

### 3.3 Full run, detached

```bash
tmux new-session -d -s exp3 "bash /workspace/run_exp3.sh"
```

`run_exp3.sh` trains `lif_uq_mhc_002` and then runs `uq_analysis.py` over all
six methods, writing `/workspace/runs/lif_uq_mhc_002/analysis/`.

```bash
# status
cat /workspace/runs/exp3.status
tmux ls
tail -f /workspace/runs/full.log
```

### 3.4 Gate and attribution report

```bash
uv run --locked python src/mlpdft/utils/experiment3_report.py \
  --summary /workspace/runs/lif_uq_mhc_002/evaluation/uq_summary.json \
  --outdir  /workspace/runs/lif_uq_mhc_002/analysis
```

Writes `experiment3_report.md` and `.json`: the per-arm table, the per-group
MHC/committee ratio, the gate verdict, and the C−D / D−A deltas.

## 4. Why the warm start

`ScaleShiftMACE` has a **two-stage readout** per head (`readouts[0]` for the
atomic energy, `readouts[1]` for the interaction/force path). Copying only
`readouts[0]` leaves the force path random and the probe sits at ~190 meV/Å.
MACE's `load_foundations_elements` also does not copy the **Agnesi distance
transform**, which changes the descriptors even with identical body weights.

`--warm-start-readout` copies both readout stages and the distance transform for
the single-head probe. `--warm-start-heads` uses MACE's readout transfer for all
four MHC heads, so every head starts at foundation accuracy and diversity comes
from the disjoint data slices.

**Random-head MHC fails at this scale.** At 53 frames/head the random readout
must absorb a ~5 eV/atom cohesive-energy offset; even with 100:1 force weighting,
50 epochs leaves it at ~150–190 meV/Å. That is data starvation, not a
representation result.

## 5. Gotchas

- **GPU memory:** batch 16 / float64 on the 8,313-frame pool used ~41.5 GB.
  A 24 GB RTX 4090 will OOM there; an A40 48 GB works. On the 214 split memory
  is lower (~29–36 GB) but still above 24 GB at batch 16.
- **`patience`** defaults to 50 for frozen arms; the best validation checkpoint
  is the one MACE saves and reloads, so early stopping gives the best-loss model.
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
