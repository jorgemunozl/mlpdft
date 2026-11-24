---
status: active
date: 2026-11-23
phase: 2
owner: jorge
supersedes: ../archive/2026-11-21-experiments-plan-interim.md
superseded_by: —
---

# UQ experiments on the 40 Ry shift-ladder dataset

**Design:** [ADR-0007](../decisions/ADR-0007-bulk-trained-shift-ladder.md)
(train on bulk LiF, evaluate along a graded shift) and
[ADR-0008](../decisions/ADR-0008-40ry-labels-and-md-protocol.md) (40 Ry labels).
**Data:** [`datasets/shift-ladder-40ry/CURATION.md`](../datasets/shift-ladder-40ry/CURATION.md).
**Code:**
- training: `src/mlpdft/uq_compare.py`;
- evaluation: `src/mlpdft/uq_evaluate.py`;
- metrics: `src/mlpdft/uq_metrics.py`.

## Arms

All arms start from MACE-OMAT-medium.

| Arm | Training | Uncertainty |
|---|---|---|
| `committee` | 3 independent full fine-tunes (different seeds; the runner's maximum) | spread of members |
| `snapshots` | one full fine-tune with warm restarts, 3 snapshots | spread of snapshots |
| `mhc` | frozen body, 4 random heads on disjoint quarters of the pool ([ADR-0001](../decisions/ADR-0001-four-head-disjoint.md)) | spread of heads |
| `shallow-nll` | frozen body, 4 heads on all frames, trained jointly with a Gaussian NLL | spread of heads |
| `frozen-probe` | frozen body, 1 head on all frames | none; accuracy control |
| `llpr` | last-layer prediction rigidity on the probe | analytic σ |
| `zero-shot` | none | accuracy reference |

`shallow-nll` and `llpr` are the baselines reviewers expect (method note:
`methods/baselines.md`, written with the implementation).

## Training

- **Pool:** `train_pool`, 1,536 frames.
- **Label rungs:** 250, 500, 1,000 and 1,536 frames, evenly spaced through
  the pool.
- **Recipes** (revised 2026-11-24 for cost; `configs/shift_ladder/node.env`):
  - float32 for every arm;
  - fixed budgets and the last epoch is the model, with no per-epoch
    validation (ADR-0004, revised 2026-11-24);
  - full fine-tunes: 51 epochs, lr 5e-4. These are the committee members and
    the snapshot run, which is 3 cycles of 17 epochs;
  - frozen-body arms: force-only, lr 1e-3, batch 16. At 1,000 labels each
    head gets about 940 optimizer steps: MHC 60 epochs (250 frames per head),
    shallow-nll and probe 15 epochs (all frames). MHC and shallow-nll
    therefore get the same compute;
  - other rungs keep the number of optimizer steps, not the number of epochs,
    so small rungs are not under-trained (set before those runs).
- **Seeds:** three seed sets per arm and rung.

## Evaluation

`uq_evaluate.py` evaluates every arm on every set:
- in distribution: calibration and test replicas, once they exist;
- `test_timeblock`;
- the four shift sets;
- the two boron sets, for the appendix.

| Question (charter) | Measure |
|---|---|
| Ranking within each set | Spearman of error vs σ, per frame and per atom |
| Calibration in distribution and under shift | NLL, miscalibration area, 1σ/2σ coverage, raw and after α fitted on the calibration replicas; split-conformal 90 % coverage |
| Does σ flag the shift? | AUROC of per-frame σ, in-distribution test vs each shift set |
| Leakage from single-trajectory splits | the same metrics on `test_timeblock` vs the test replicas |
| Cost | GPU-hours per arm, rung and seed set |

Ensembles differ in size (3 members, 3 snapshots, 4 heads), which the
results report next to every comparison.

Intervals come from a block bootstrap:
- in distribution: replicas are the blocks;
- single trajectories: blocks of 1,000 SCF steps.

Arms are compared pairwise on the same frames.

## Order of runs

1. **Baseline:** all arms at 1,000 labels, seed set 1. This checks the whole
   pipeline on the GPU. Until the replicas exist, the evaluation is raw (no
   α), with `test_timeblock` as the in-distribution reference.
2. **Rungs** 250, 500 and 1,536 as siblings of the baseline, seed set 1.
3. **Seed sets 2 and 3** for every rung.
4. **Evaluation-only rerun** when the replicas arrive: calibration, the
   in-distribution test set, and the leakage comparison.
5. **Retrospective active learning** (charter Q3).
   - Start from 100 bulk frames.
   - The pool holds the rest of `train_pool` plus half of each shift set; the
     other halves are kept for testing.
   - Compare committee- and snapshot-driven acquisition against random
     ([ADR-0002](../decisions/ADR-0002-mhc-outside-al-loop.md)).
   - The protocol will be revised in `exp-04-active-learning/`.

## Compute

- One A40-class GPU per run.
- **Baseline:** on a rented A40 within 12 h. A 30-minute smoke run
  (`configs/shift_ladder/smoke.env`) checks every arm and projects the
  baseline time from measured timings. The baseline itself takes about 8 h and
  stops at a 10.5 h deadline. Each arm group is uploaded as soon as it
  finishes. Runbook: `assets/runs/run_shift_ladder_runpod.md`.
- **The other 11 runs** (rungs and seed sets) are scheduled from the
  baseline's measured cost, on the cluster or later rentals.
- Nothing heavy runs on the local machine.
- The GitHub repository is public, so the datasets are not committed. They
  reach the GPU through the transport chosen with the backend.
