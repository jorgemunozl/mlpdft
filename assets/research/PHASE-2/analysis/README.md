# Experiment 2 — environment-resolved re-analysis

Pilot analysis of the 214-frame run in
`/home/jorge/project/mlpdft/outputs/outputs_v3/` (178 locked test frames:
80 bulk LiF, 80 F-doped Li, 18 interface). It exists to close the statistical
gaps in `UQ_EXPERIMENT_2_RESULTS.md`: within-environment risk--coverage, a
paired test of method differences, and a scale (coverage) diagnostic.

**These are pilot numbers.** The final numbers are regenerated on the frozen
post-generation splits with the same commands.

## Files

| file | contents |
|---|---|
| `uq_analysis.json` | full machine-readable report |
| `uq_analysis.md` | per-method tables and paired differences |
| `figures/fig_risk_coverage.pdf` | risk--coverage inside each environment |
| `figures/fig_error_over_sigma.pdf` | median error / predicted spread per arm |

Regenerate with:

```bash
D=/home/jorge/project/mlpdft/outputs/outputs_v3/evaluation
python src/mlpdft/utils/uq_analysis.py \
  --prediction committee=$D/independent_committee_predictions.csv \
  --prediction snapshot=$D/warm_snapshots_predictions.csv \
  --prediction multihead=$D/multihead_bootstrap_predictions.csv \
  --prediction zero_shot=$D/zero_shot_predictions.csv \
  --output-dir assets/research/PHASE-2/analysis --block 20 --n-boot 2000

python src/mlpdft/utils/make_uq_figures.py \
  --analysis assets/research/PHASE-2/analysis/uq_analysis.json \
  --outdir assets/research/PHASE-2/analysis/figures
```

## Method

- **Unit of resampling:** contiguous 20-frame blocks within each group, not
  individual frames. Frames from one AIMD trajectory are autocorrelated; an
  i.i.d. bootstrap would understate every interval.
- **Within-group mean Spearman:** equal weight per environment (three groups),
  computed from the per-frame uncertainty and per-frame force RMSE.
- **Paired differences:** the same resampled blocks are used for both methods,
  so the interval is on the difference rather than on two marginal intervals.
- **Scale diagnostic:** median of `force RMSE / force uncertainty`. A value
  above 1 means the ensemble spread is too small to be an error bar.

## Reading the pilot result

- All three arms rank error positively within every environment; none inverts.
- Snapshots have the highest within-group mean Spearman (+0.83), committee
  +0.66, multi-head +0.58. The paired interval puts snapshots above committee
  (−0.21 for committee − snapshot, 95% CI [−0.45, −0.07], p = 0.002) in the two
  environments large enough to resample.
- The **interface group has 18 frames**, fewer than one 20-frame block, so it
  has no interval and is excluded from the paired test. Any interface claim
  from this run is descriptive only.
- Every arm is **under-scaled**: median error / spread is 2.7–6.4× for committee
  and snapshots, and 13.8–66.5× for multi-head. The disagreement ranks errors
  better than it sizes them; calibrated error bars would need a fitted scale.
- Risk--coverage within an environment is much flatter than the pooled curve in
  the draft. The pooled "steep tail" was partly a group-composition effect.

## Limits

- 178 test frames, 18 of them interface; the paired interval rests on two
  groups of 80.
- One foundation checkpoint, one seed set, one 214-frame training pool.
- The `e/σ` coverage diagnostic is a Gaussian reference on a per-frame RMS
  error, not a formal predictive-interval test.
