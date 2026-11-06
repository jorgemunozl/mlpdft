# Experiment 3 — 150-epoch rerun and probe attribution (pilot)

**Runs:** `mhc150_s123` (MHC, 150 epochs, standard loss), `probe150` (frozen
probe, 150 epochs, standard loss), on the 214 / 53 / 178 split.
**Date:** 2026-11-06.
**Companion:** `GATE.md` is the 400-epoch run; this file is the shorter-budget
rerun and the C/D/A attribution.

## TL;DR

- The 150-epoch gate **fails again**: overall force 1.22× committee (inside the
  1.25 limit), but within-group ρ **0.30 vs 0.66** and bulk force **1.36×**.
- Cutting 400 → 150 epochs **did not rescue the gate**. It improved bulk force
  (1.64× → 1.36×) but made rank worse (0.46 → 0.30); the interface rank inverted
  from +0.08 to −0.20.
- The probe answers the attribution: **freezing the body is the bottleneck**,
  not splitting labels across heads. Freezing costs +17% overall and +53% in
  bulk; splitting costs +4% overall and actually helps in bulk.
- Both frozen arms are still trained under an energy-dominated loss
  (2.4–2.5 eV/atom); the force-focused ablation is running to test that
  confound.

## 150 vs 400 epochs

| Metric | Committee (A) | MHC 150 | MHC 400 | 150 vs 400 |
|---|---:|---:|---:|---|
| Force RMSE (eV/Å) | 0.008659 | 0.010563 | 0.010237 | 150 is 3% worse |
| Overall ratio vs A | 1.00 | **1.220** | 1.182 | both inside 1.25 |
| Within-group mean ρ | 0.656 | **0.302** | 0.457 | 150 is much worse |
| Bulk force ratio | 1.00 | **1.360** | 1.641 | 150 better |
| Bulk ρ | 0.741 | 0.501 | 0.573 | — |
| F-doped ρ | 0.634 | 0.610 | 0.713 | — |
| Interface ρ (n=18) | 0.593 | **−0.201** | 0.084 | sign flip |
| Pooled ρ | 0.943 | **0.917** | 0.858 | pooled hides it |
| Energy RMSE (meV/atom) | 0.099 | 2468.5 | 2205.1 | both broken |
| Median error / spread | 4.45× | 17.4× | 16.1× | under-scaled |

The gate failure is **robust to the epoch budget**, but the two runs fail for
different reasons: 400 fails rank and bulk force; 150 improves bulk force but
degrades rank. The rank leg is unstable at this pilot size — the interface
point estimate swings across zero (n = 18, no interval) and the mean follows it.

## Per-group detail (150 epochs)

| Group | n | MHC force | Committee | Ratio | MHC ρ | Committee ρ | MHC e/σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bulk LiF | 80 | 0.003640 | 0.002678 | **1.360** | 0.501 | 0.741 | 15.1× |
| F-doped Li | 80 | 0.009468 | 0.007973 | 1.188 | 0.610 | 0.634 | 27.2× |
| Interface | 18 | 0.019601 | 0.015985 | 1.226 | **−0.201** | 0.593 | 17.0× |

## Probe attribution: C − D and D − A

`C` = MHC (4 heads, disjoint 54/54/53/53), `D` = frozen probe (1 head, all 214
labels), `A` = independent committee. All three are 150-epoch standard-loss
runs on the same 178-frame test set.

| Comparison | Overall | Bulk | F-doped | Interface |
|---|---:|---:|---:|---:|
| A committee force (eV/Å) | 0.008659 | 0.002678 | 0.007973 | 0.015985 |
| D probe force | 0.010118 | 0.004099 | 0.009489 | 0.018059 |
| C MHC force | 0.010563 | 0.003640 | 0.009468 | 0.019601 |
| **D − A** (freezing) | **+1.459 meV (+16.8%)** | **+1.421 (+53.1%)** | +1.516 (+19.0%) | +2.074 (+13.0%) |
| **C − D** (label splitting) | +0.445 meV (+4.4%) | −0.459 (−11.2%) | −0.021 (−0.2%) | +1.542 (+8.5%) |

Reading:

- **Freezing the body is expensive in every environment.** The single-head
  probe, which sees all labels, is 1.17× the committee overall and 1.53× in
  bulk. That is the dominant gap.
- **Splitting labels across four heads is close to neutral overall.** The MHC is
  only 4.4% worse than the probe overall; in bulk the 4-head average is 11%
  *better* than the single head, which is the expected variance-reduction
  effect. Splitting hurts at the interface (+8.5%), the smallest group.
- This is the protocol's **“probe also worse → frozen representation is the
  bottleneck”** branch, with the caveat that the overall ratios sit just under
  the 1.25 line; the bulk ratios (1.53×, 1.36×) are the clean evidence.

## What this means

- The frozen-body multi-head committee does not clear the pilot gate at either
  150 or 400 epochs. The failure is not an artifact of over-training; the
  150-epoch model is worse on rank and better on bulk force.
- The attribution points at the **frozen representation**, not at the label
  split. A four-head committee cannot recover the committee's bulk accuracy
  because the shared body is the limiting factor.
- The pooled/within-group inversion is the cleanest demonstration yet of the
  paper's methodological point: 150-epoch MHC pooled ρ = 0.917 (nearly the
  committee's 0.943) while within-group ρ = 0.302.
- The energy offset is a real confound. Both C and D spend their loss on a
  2.4–2.5 eV/atom offset; the force-focused ablation (`mhc150_ef0`,
  `energy_weight=0`) and its matched probe are running to see whether the
  freezing conclusion survives a force-focused loss.

## Corrected overnight batch (running)

The first overnight script died immediately because `parse_seeds` requires
exactly three seeds: `--seeds 124` raised `ValueError`, so `mhc150_ef0`,
`mhc150_s124`, `mhc150_s125`, `mhc150_s126` all failed before training. The
probe had already finished successfully.

The corrected batch (`overnight2`) is running:

| Run | Seeds (first used) | Loss | Purpose | ETA |
|---|---|---|---|---|
| `mhc150_ef0` | 123 | energy 0 | does the energy confound explain the gate? | 01:58 UTC |
| `probe150_ef0` | 123 | energy 0 | matched C/D attribution under force-focused loss | 03:14 UTC |
| `mhc150_s124` | 124 | energy 1 | standard-recipe seed robustness | 05:02 UTC |
| `mhc150_s125` | 125 | energy 1 | standard-recipe seed robustness | 06:50 UTC |

## Files

| Path | Contents |
|---|---|
| `mhc150_s123/analysis/uq_analysis.json` | 150-epoch within-environment analysis |
| `mhc150_s123/gate.txt` | 150-epoch gate arithmetic |
| `mhc150_s123/evaluation/frozen_mhc_predictions.csv` | per-frame predictions |
| `mhc150_s123/train.log` | training log |
| `probe150/evaluation/uq_summary.json` | probe force, overall and per group |
| `probe150/evaluation/frozen_probe_predictions.csv` | per-frame probe predictions |
| `probe150/train.log` | probe training log |

## Caveats

- Pilot split: 214 train / 53 valid / 178 test; interface n = 18.
- The gate's rank leg includes the interface point estimate, which has no
  interval and is smaller than one 20-frame bootstrap block. The 150/400 sign
  flip shows this is not a stable quantity.
- Both standard-loss arms have unaligned energy; force comparisons are the only
  meaningful ones.
- The probe is a single head and has no rank correlation by construction; the
  C/D/A attribution is force-only.
