# UQ Experiment 2 — Corrected Calibration Run

**Status:** 🔵 **running** on the private external cluster (as of 2026-10-27; results arrive as
pasted logs — not tracked in `orx`).
**Predecessor:** [`UQ_EXPERIMENT_1.md`](UQ_EXPERIMENT_1.md) — the run whose confounds this one fixes.

## Purpose

Rerun the three-way comparison with the experiment-1 confounds fixed, and freeze the
**calibration leg** and the **full-data ceiling model** that the active-learning loop
([`ACTIVE_LEARNING_PROTOCOL.md`](ACTIVE_LEARNING_PROTOCOL.md)) will approach.

## Changes from experiment 1 (the fixes)

| item | experiment 1 | experiment 2 |
|---|---|---|
| epoch budget | 100 | **150** |
| snapshot collection | evenly spaced 33/67/99 (bug) | **cycle-ends 49/99/149** (one epoch before each restart, T₀ = 50) |
| primary calibration metric | pooled Spearman | **within-group Spearman + moving-block bootstrap** |
| energy metric | raw | **reference-energy aligned** |
| force-RMSE aggregate | mixed | **atom-weighted global RMS** |

## Configuration

Same foundation checkpoint (MACE-OMAT medium), same three methods (independent committee,
multi-head, warm snapshots), K = 3, seeds 123/124/125. Frozen, leak-free splits (from
`build_uq_splits.py`): train **214**, validation **53**, locked test **178** frames — the same
split set `ACTIVE_LEARNING_PROTOCOL.md` uses as pool/valid/test, so the ceiling RMSE is directly
comparable to the AL loop.

## Deliverables → feeds the active-learning loop

1. Per-method **within-group Spearman + block-bootstrap CIs** and **risk-coverage** curves
   (the calibration leg).
2. **Ceiling force RMSE** = the full-data model trained on all 214 train frames; the AL loop's
   "target reached" event is test RMSE ≤ 1.05 × this ceiling.
3. Per-method training cost, for the compute-to-target accounting.

These numbers parameterize `src/mlpdft/uq_active_learning.py` (seed 30, batch 10, gap 5, ceiling
RMSE). Once they land, the AL leg starts.
