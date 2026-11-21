---
status: accepted
date: 2026-11-20
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

# ADR-0001: Four disjoint heads on a frozen foundation body

## Context

The published foundation experiment uses eight heads over 8,000 structures —
1,000 per head — with frozen descriptors and randomly initialised readouts.
Our budget is one A40 and a 1,000–2,000-frame training pool, so an eight-head
disjoint split would give 125–250 frames per head, below the sparse regime the
paper itself warns about.

## Decision

Use **four randomly initialised readout heads on a frozen MACE-OMAT-medium
body**, with a **disjoint, environment-stratified** split: each head gets
125 bulk LiF + 125 F-doped Li frames at the 1,000-label rung. Force-only loss
(`energy_weight=0`), 90 epochs, learning rate 1e-3, batch 16.

This is the paper's *disjoint* mode at one eighth of its per-head data, not a
reproduction of its foundation recipe.

## Evidence

Run `mhc4d_a40_001`, 1,000 labels:

- test force RMSE 33.25 meV/Å (members 32.72–34.49; the ensemble is not better
  than its best member);
- within-group Spearman 0.589 (bulk) / 0.106 (F-doped);
- α = 12.84, 1σ coverage 0.782, RLL −0.394.

Record: [`runs/2026-11-06-mhc4d/README.md`](../experiments/exp-03-mhc/runs/2026-11-06-mhc4d/README.md).

## Consequences

- At 500 labels/head the arm is only meaningful from ~2,000 total labels; below
  that it measures data starvation, which is why it is not an acquisition model.
- The result is a fixed-budget rung, not a verdict on the published method: the
  per-head data, head count and optimization budget all differ.
- If the frozen probe (one head, all labels) is much better than the committee,
  the split is the bottleneck; if both are similar, the frozen body is.
