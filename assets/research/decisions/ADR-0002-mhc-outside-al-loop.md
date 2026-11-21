---
status: accepted
date: 2026-11-20
phase: 3
owner: jorge
supersedes: —
superseded_by: —
---

# ADR-0002: MHC is a fixed-budget arm, not an acquisition model

## Context

The active-learning loop retrains at small budgets: a 30-frame seed and
10 frames per round, 15 rounds, so 180 labels at most. With four disjoint
heads, each head would see about 45 frames — the loop would measure starvation,
not the method. The committee and the snapshot ensemble have no such split.

## Decision

- The AL loop compares **committee, warm-restart snapshot and random** only.
- The frozen-body MHC is evaluated at **fixed label budgets** (500 / 1,000 /
  2,000), with disjoint head partitions and a frozen probe at the same budgets.
- If an MHC-in-the-loop claim is wanted later, it is a **separate high-budget
  loop** starting at 500 labels/head, with the same frozen-body trainer on both
  arms (σ-acquisition vs random), never a mixed-trainer comparison.

## Evidence

Protocol section 6: [`experiments/exp-04-active-learning/PROTOCOL.md`](../experiments/exp-04-active-learning/PROTOCOL.md).
Head-only retrains take minutes, so a high-budget loop is affordable; the
constraint is labels per head, not compute.

## Consequences

- The paper's AL claim is about committee vs snapshot; it cannot be "MHC enables
  cheap active learning" without the separate loop.
- MHC still contributes the cost ladder, calibration quality and the gate, and
  the probe attribution is required either way.
- A random control must use the same trainer as the arm it controls, otherwise
  architecture and acquisition are confounded.
