---
status: accepted
date: 2026-11-20
revised: 2026-11-27
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

> **Revised 2026-11-21** with the owner's approval. The heads now train on bulk
> LiF only ([ADR-0007](ADR-0007-bulk-trained-shift-ladder.md)). The evidence
> below comes from the retired interim dataset, which used 20–30 Ry labels, and
> is kept as history. Original text: commit `402298a`.
>
> **Corrected 2026-11-27.** The evidence run used 30 Ry labels in both
> environments (bulk from `LIF64_KJPAW_NPT_V3` only), fails the design's H3 rule
> (RLL < 0) and had no comparator. It is a development record, not evidence for
> this decision, which rests on the design reasoning in
> [`exp-03-mhc/DESIGN.md`](../experiments/exp-03-mhc/DESIGN.md) §2.

# ADR-0001: Four disjoint heads on a frozen foundation body

## Context

The published foundation experiment uses eight heads over 8,000 structures —
1,000 per head — with frozen descriptors and randomly initialised readouts.
Our budget is one A40 and a 1,000–2,000-frame training pool, so an eight-head
disjoint split would give 125–250 frames per head, below the sparse regime the
paper itself warns about.

## Decision

Use **four randomly initialised readout heads on a frozen MACE-OMAT-medium
body**, with a **disjoint split of the bulk LiF training pool**: each head gets
250 bulk LiF frames at the 1,000-label rung. Force-only loss
(`energy_weight=0`), 90 epochs, learning rate 1e-3, batch 16.

Until 2026-11-21 each head got 125 bulk LiF + 125 F-doped Li frames. F-doped Li
is now an evaluation-only shift set.

This is the paper's *disjoint* mode at one eighth of its per-head data, not a
reproduction of its foundation recipe.

## Evidence

Run `mhc4d_a40_001`, 1,000 labels, on the retired interim two-environment
dataset (bulk from the 30 Ry `LIF64_KJPAW_NPT_V3` run, F-doped Li at 30 Ry):

- test force RMSE 33.25 meV/Å (members 32.72–34.49); the ensemble is no better
  than its best member;
- within-group Spearman 0.589 (bulk) / 0.106 (F-doped);
- α = 12.84, 1σ coverage 0.782, RLL −0.394.

Record: [`runs/2026-11-06-mhc4d/README.md`](../experiments/exp-03-mhc/runs/2026-11-06-mhc4d/README.md).
These numbers do not carry over to the 40 Ry bulk pool; the arm is rerun there.

## Consequences

- At 500 labels per head the arm is only meaningful from ~2,000 total labels.
  Below that it measures data starvation, which is why it is not an
  acquisition model.
- The result is a fixed-budget rung, not a verdict on the published method:
  the per-head data, head count and optimization budget all differ.
- If the frozen probe (one head, all labels) is much better than the
  committee, the split is the bottleneck; if both are similar, the frozen body
  is.
- With a shared frozen body and separate heads, the arm is structurally a
  shallow ensemble. It is therefore compared with an NLL-trained shallow
  ensemble and with LLPR ([ADR-0007](ADR-0007-bulk-trained-shift-ladder.md)).
