# UQ Experiment 1 — First Fine-Tuning Run

**Status:** ✅ **completed & reviewed** (analysis 2026-10-20).
**Next:** [`UQ_EXPERIMENT_2.md`](UQ_EXPERIMENT_2.md) — the corrected rerun, now running.

## Goal

First full three-way comparison of uncertainty strategies for a fine-tuned MACE foundation
model on the Li/F core: does ensemble disagreement track held-out DFT force error, and at what
cost? Scope per [`DESCRIPTION.md`](DESCRIPTION.md) and [`ROUTE.md`](ROUTE.md).

## Configuration (as actually run)

- Model: **MACE-OMAT medium** (`mace_omat_medium`).
- Budget: **100 epochs** (the later 150-epoch bump in the shared protocol came *after* this run).
- Ensemble size K = 3; committee seeds 123/124/125.
- Warm snapshots collected at **epochs 33/67/99** (evenly spaced) — *this is the bug fixed in
  experiment 2*.
- Multi-head: 3 block-bootstrap heads + OMAT replay head (1 000 pretrain samples).
- Test: 480 frames = 6 groups × 80 (LIF64, LIF64-NPT, LIWITHF, LIWITHF-NPT, LIFINTERFACE,
  LIFINTERFACE-NPT).

## Results — corrected headline

| method | force RMSE (eV/Å) | energy RMSE (meV/atom) | pooled Spearman | within-group mean Spearman | cost (s) |
|---|---|---|---|---|---|
| zero-shot foundation | 0.0251 | 1.64 (aligned) | — | — | 0 |
| independent committee | 0.0150 | 1.50 | 0.391 | **0.608** | 8 632 |
| multi-head bootstrap | **0.0144** | **1.42** | 0.616 | 0.124 | 11 392* |
| warm snapshots | 0.0164 | 13.11 | **0.836** | 0.271 | **2 853** |

\* multi-head cost includes the OMAT pretraining step, so it is not directly comparable to a
single fine-tune.

## Three corrections that change the reading

1. **The zero-shot "energy failure" is a reference-energy artifact.** Zero-shot energy RMSE of
   384 482 meV/atom collapses to **1.64 meV/atom** once per-family offsets are removed (MACE
   predicts cohesive energy; labels are raw totals). Fine-tuning's real payoff is **force**
   accuracy (0.0251 → 0.0144–0.0164), not energy.
2. **"Snapshots win calibration" (pooled 0.836) is a cross-group artifact.** Pooled Spearman is
   driven by between-group scale (snapshot places ~10× the uncertainty on interface vs. bulk),
   not by per-frame error tracking. Within-group the ranking **reverses**: committee **+0.61** >
   snapshot +0.27 > multi-head +0.12. The committee is the only method that never anti-correlates;
   multi-head and snapshot invert in interface/NPT (OOD) groups — the Kurniawan "uncertainty
   breaks down OOD" signature.
3. **Temporal correlation cuts the effective sample ~10×.** Lag-1 force-error autocorrelation
   0.88–0.98, integrated τ = 4.7–33 frames → effective N ≈ 44 (vs. 480 nominal). A moving-block
   bootstrap (block 20, 2000 resamples) keeps the committee's bulk advantage significant and the
   OOD inversions non-significant — the calibration conclusion survives the correlation critique.

## Risk-coverage

Only the snapshot ensemble is monotonic, with the steepest low-error tail (10% of
lowest-uncertainty frames → 0.0038 eV/Å). Committee and multi-head curves dip.

## Confounds found (fixed in experiment 2)

- **Snapshot under-training (audit B2) — root-caused.** Snapshots at 33/67/99 land on the
  warm-restart boundaries (T₀ = 33 = epochs//count), so two of three members are the just-kicked,
  loss-spiked model (energy RMSE 26.9 / 16.1 / 2.67 meV/atom). Fix: collect at **cycle-ends**
  (one epoch before each restart), not evenly spaced.
- **Committee collapse.** Members near-identical (force RMSE 0.0151 / 0.0150 / 0.0150; ensemble
  force σ 0.0006 vs. snapshot 0.0053) — shared initialization removes diversity. A finding, not a
  bug, but needs a stronger baseline.
- **Cost accounting (audit E2).** Committee 8 632 s = 3 × ~2 880 s; snapshot 2 853 s (⅓);
  multi-head 11 392 s *including* OMAT pretrain — report head-only cost separately.

## What this run established

Fine-tuning buys force accuracy; no method dominates (committee = weak-but-reliable, never OOD
inverts; snapshot/multi-head = stronger in-domain, invert OOD); the snapshot's monotonic
risk-coverage at ⅓ cost is the strongest case for the paper's thesis *provided* the
under-training confound is removed.
