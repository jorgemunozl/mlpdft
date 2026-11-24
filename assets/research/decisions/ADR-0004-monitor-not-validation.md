---
status: accepted
date: 2026-11-20
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

# ADR-0004: No validation split for model selection; a monitor slice only

## Context

The training helper sets `patience = max_epochs + 1` by default, so early
stopping never fires and the saved model is the final epoch. Validation
therefore does **not** select the model; it only provides per-epoch logging and
the smoke gate. Meanwhile, a 100-frame validation split is 30% of the 330
independent bulk frames — a large price for monitoring.

## Decision

Drop validation as a scientific split:

- the training budget is fixed a priori (epochs and schedule frozen per arm);
- a **monitor slice** of at most 10% of the training pool exists only to
  satisfy the trainer and run the smoke gate;
- the monitor slice is **never reported** and never used to choose a model,
  checkpoint, hyper-parameter or stopping point;
- freed frames go to calibration and test, not to training.

## Evidence

`src/mlpdft/uq_compare.py:879-882` defaults patience above the epoch budget.
The 4D run logged `patience=50 is not below the epoch budget 90; early stopping
will not trigger`.

## Consequences

- The paper must state the fixed-epoch protocol explicitly; there is no
  best-checkpoint selection to describe.
- The trainer still requires a valid file, so the monitor slice cannot be zero.
- If a future arm needs early stopping, this ADR must be superseded, not
  silently ignored.

## Revision 2026-11-24: the decision was not implemented; now it is

The evidence above was incomplete. MACE writes a checkpoint only on validation
epochs and, unless `save_all_checkpoints` is set, only when the validation loss
improves, so the saved model was the best-validation epoch, not the last one.
The frozen-body arms also defaulted to patience 50, and `shallow-nll` kept its
lowest-validation-NLL epoch.

`uq_compare --validation end`, used by every shift-ladder run, implements the
decision:

- fixed budgets, and the last epoch is the model;
- MACE validates only on the epochs whose checkpoint is kept: the last one, or
  every epoch of the warm-restart run, whose cycle ends are not evenly spaced
  for `eval_interval`;
- `shallow-nll` validates once, after its last epoch, for the record.

This also removes the per-epoch validation cost (a third of the 2026-11-06 run's
time). From the same date every arm trains in float32.

**Interim exception:** LLPR's two calibration constants (ς², α²) are fitted on
the monitor slice until the calibration replicas exist. The evaluation-only
rerun with the replicas refits them there.
