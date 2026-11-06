# Experiment 3 — frozen MHC gate report (pilot)

**Run:** `lif_uq_mhc_002` on the 214 / 53 / 178 split (`li_f_uq_splits_v2`).
**Executed:** 2026-11-05 15:27:26 UTC. MHC trained its full 400 epochs and
finished at 20:16 UTC (4 h 49 m). The probe was stopped at epoch 19 so the MHC
arm could be evaluated immediately; **no probe model exists for this report**.
**Evaluation:** `training/mhc_saved.model` (copy of the MACE checkpoint) run on
the 178-frame test set with `training/eval_mhc_saved.py`, then the standard
within-environment analysis in `analysis/uq_analysis.json`.

## Verdict: FAIL

| Gate criterion | MHC | Committee | Result |
|---|---:|---:|---|
| Force RMSE, atom-weighted (eV/Å) | 0.010237 | 0.008659 | **ratio 1.182** — passes the overall 1.25× limit |
| Within-group mean Spearman (3 envs) | **0.457** | **0.656** | **Δ −0.200** — fails the ≥ −0.10 limit |
| Per-group force ratio (report script) | — | — | **bulk 1.641** — fails the per-group 1.25× limit |

The gate fails on two independent grounds:

1. **Rank.** All-environment mean ρ is 0.457 vs the committee's 0.656. The drop
   is almost entirely the interface: 0.084 vs 0.593 (n = 18). On the two
   resamplable environments the paired difference is **−0.044 [−0.344, +0.213],
   p = 0.856** — not separable. The rank leg is therefore weak evidence; the
   interface is smaller than one bootstrap block.
2. **Bulk force.** In the in-domain environment MHC is **1.64×** worse than the
   committee (0.004394 vs 0.002678 eV/Å), above the 1.25× per-environment
   condition used by `experiment3_report.py`. This failure is independent of the
   n = 18 interface and is the stronger of the two.

Note the pooled/within-group inversion: MHC's **pooled** ρ is 0.858 (committee
0.943), which looks close; its **within-environment** mean is 0.457 (committee
0.656). This is the slide-7 effect reappearing in the new arm.

## Per environment

| Group | n | MHC force (eV/Å) | Committee | Ratio | MHC ρ | Committee ρ | MHC e/σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bulk LiF | 80 | 0.004394 | 0.002678 | **1.641** | 0.573 | 0.741 | 11.6× |
| F-doped Li | 80 | 0.009463 | 0.007973 | 1.187 | 0.713 | 0.634 | 23.9× |
| Interface | 18 | 0.018277 | 0.015985 | 1.143 | 0.084 | 0.593 | 16.1× |

Overall: force 0.010237 eV/Å, energy **2205 meV/atom**, median error / spread
**16.1×** (committee 4.45×). All three groups have positive rank correlation
(3/3).

## Caveats

- **Energy offset confound.** The warm-started heads begin at RMSE_E ≈ 2.66
  eV/atom (epoch 0) and end at 2.21 eV/atom. The loss is dominated by this
  unaligned energy offset (~375× the weighted force term), so the 400 epochs are
  largely fitting an energy offset, not forces. Per-head force plateaued around
  epochs 150–210 (best 9.33–9.55 meV/Å) and drifted slightly to 9.47–9.75 by
  epoch 387. The gate arithmetic above is still the pre-registered arithmetic,
  but the recipe is not a clean force-focused test. The saved `.model` is the
  loss-best (late) checkpoint, not the force-best epoch — so the force numbers
  here are pessimistic relative to the per-head validation best of 9.33–9.55
  meV/Å. That is another reason to rerun force-focused before treating this as a
  clean test of the frozen representation.
- **No attribution.** The probe (`D`) was not trained, so `C − D` (label
  splitting) and `D − A` (freezing the body) cannot be computed. The bulk
  failure is *consistent with* head data starvation (54 frames/head) but that is
  a hypothesis, not a result.
- **Interface.** n = 18, no interval, excluded from paired tests. Any claim that
  "rank collapses at the interface" is descriptive only.
- **Gate implementation.** The arithmetic follows `experiment3_report.py` at
  commit `b416665`: overall ratio, max per-group ratio, and the ρ mean over all
  groups. The deck's slide-12 wording only names the overall force and ρ
  conditions; under that wording the ρ condition still fails, so the verdict
  does not depend on the extra per-group check.

## What this does and does not establish

- It establishes that **this frozen-MHC recipe** (4 heads, 54/54/53/53 frames,
  warm-started readouts, forces 100:1 over energy, 400 epochs) does not clear
  the pre-registered gate on the pilot split.
- It does **not** establish that the frozen representation is the bottleneck.
  That needs the probe.
- It does **not** establish that MHC is worse everywhere: F-doped and interface
  force ratios are 1.19× and 1.14×; the bulk environment carries the failure.

## Files

| File | Contents |
|---|---|
| `training/mhc_saved.model` | MACE checkpoint after 400 MHC epochs (12 MB) |
| `training/mhc_saved_train.log` | MACE training log, 4 heads × 400 epochs |
| `training/full.log` | Full `uq_compare` log including the probe start |
| `training/run_config.json` | Training hyperparameters |
| `training/eval_mhc_saved.py` | Script that re-evaluates the saved checkpoint |
| `evaluation/frozen_mhc_summary.json` | Per-method summary from `_evaluate_method` |
| `evaluation/frozen_mhc_predictions.csv` | Per-frame predictions on the 178 test frames |
| `analysis/uq_analysis.json` | Within-environment analysis vs committee/snapshot/MH-FT |
| `analysis/uq_analysis.md` | Human-readable version of the same |
| `analysis/gate.py`, `analysis/gate.txt` | Gate arithmetic and its output |

## Reproduction

On the pod, from `/workspace/mlpdft`:

```sh
uv run --locked python /workspace/eval_mhc_saved.py
uv run --locked python src/mlpdft/utils/uq_analysis.py \
  --prediction mhc=/workspace/runs/lif_uq_mhc_002/evaluation/frozen_mhc_predictions.csv \
  --prediction committee=/workspace/exp2_csv/independent_committee_predictions.csv \
  --prediction snapshot=/workspace/exp2_csv/warm_snapshots_predictions.csv \
  --prediction multihead=/workspace/exp2_csv/multihead_bootstrap_predictions.csv \
  --output-dir /workspace/runs/lif_uq_mhc_002/analysis_mhc --block 20 --n-boot 2000
python3 /workspace/gate.py
```

## Next options

1. **Run the probe** (400 epochs, ~3.2 h at the observed 30.5 s/epoch) to get
   `C − D` / `D − A` and attribute the failure.
2. **Rerun MHC force-focused** with the energy offset removed (energy weight ≈ 0
   or a per-head bias initialisation) before drawing any conclusion about the
   frozen representation.
3. **Report the negative result as-is**, with the energy confound and the
   missing probe stated explicitly.
