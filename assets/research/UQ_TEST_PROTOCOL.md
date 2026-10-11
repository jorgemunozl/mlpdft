# Test Plan: UQ Methods for Foundation MACE

## Goal

Test whether a warm-restart snapshot ensemble from a fine-tuned MACE foundation
model gives useful, compute-efficient uncertainty estimates for Li-containing
SEI-like materials. The main claim is not that snapshot ensembles are new. The
test is whether snapshot disagreement predicts errors and physical unreliability
as well as an independent-model committee on held-out DFT configurations and
Li-transport simulations.

## First Step

Build and freeze the core dataset manifest and splits before training. Use the
six final Li/F groups: `LIF64_KJPAW_FINAL`, `LIF64_KJPAW_NPT_FINAL`,
`LIWITHF_KJPAW_FINAL`, `LIWITHF_NPT_FINAL`, `LIFINTERFACE_KJPAW_FINAL`, and
`LIFINTERFACE_KJPAW_NPT_FINAL`. After checking equilibration, select about 200
time-spaced frames per group and make a contiguous-block 60/40 train/locked-test
split. Keep a separate validation block from the training portion for model
selection; do not tune on the locked test. Record each source frame and its
DFT/ensemble metadata in the manifest.

## Questions and Hypotheses

1. Does force disagreement across ensemble members rank held-out DFT force errors?
2. At comparable prediction accuracy, does a warm-snapshot committee provide
   useful calibration and risk-versus-coverage compared with independently
   fine-tuned models?
3. Does high committee uncertainty flag configurations or trajectories where
   predicted Li transport differs substantially across models?

The hypothesis is not that snapshots must outperform independent models. A
result where snapshots are cheaper but less calibrated is still informative.

## Scope and Data

The primary test is Li/F diffusion without boron chemistry. Use these final,
non-isolated groups from [`DATASET.md`](DATASET.md):

- `LIF64_KJPAW_FINAL` and `LIF64_KJPAW_NPT_FINAL` for bulk LiF.
- `LIWITHF_KJPAW_FINAL` and `LIWITHF_NPT_FINAL` for Li-rich Li-F.
- `LIFINTERFACE_KJPAW_FINAL` and `LIFINTERFACE_KJPAW_NPT_FINAL` for the Li|LiF
  interface.

Keep NVT and NPT labels in the manifest and report their results separately.
Do not silently merge groups with different DFT settings. Before training,
verify pseudopotentials, functional, k-points, smearing, cutoffs, SCF
convergence, restart overlap, and whether logs contain concatenated runs. Keep
isolated structures for a separate reference-energy experiment, not transport.

Treat `LIWITHBF`, `LIBF4`, `BLI`, `BLI3`, and their interface groups as an
**optional boron-chemistry extension**, not required for the primary Li/F
diffusion claim. Add them as a separately reported ablation only if testing
transfer into B-containing environments is a research objective. `BCC_54` is
pure BCC Li, not boron chemistry; it is an optional metal-side control.

For the initial comparison, target about **200 time-spaced configurations
per group** after equilibration, when the group has enough usable frames. This is
a sampling target, not a claim that 200 structures are statistically
independent. Use equal group contributions for the initial comparison and report
per-group results so the larger or longer trajectories do not dominate. Treat
`LIWITHBF` and `LIBF4` as a separate salt/additive ablation if the primary paper
is narrowed to LiF/Li-rich/interfacial transport.

### Equilibration and Sampling

- Determine the burn-in separately for each trajectory from temperature,
  potential energy, cell/volume for NPT, and relevant structural observables.
  Do not assume that the first 100 saved frames are always enough.
- After burn-in, thin with a stride of 10 only if diagnostics show it reduces
  short-time correlation. Select the target structures across the full usable
  trajectory, not just the first 200 strided frames.
- Exclude frames without valid DFT energy and forces, and investigate any
  non-converged SCF or restart/concatenation boundary.
- Store a manifest with group, source file, original frame index, ensemble,
  composition, cell, DFT settings, and split assignment.

## Splits

Use the requested **60% training / 40% held-out evaluation** at the trajectory
block level, never a random neighboring-frame split. With 200 selected frames
per group, this is approximately 120 training and 80 held-out frames per group.
Keep held-out frames locked until model choices and uncertainty calibration are
fixed.

For a stronger transfer test, add a separate **leave-one-family-out** result:
hold out an entire environment family (for example, one interface family) from
all training and tune only on other families. Report this separately from the
within-trajectory 60/40 test. A 60/40 split within a trajectory measures
blocked temporal generalization; it does not, by itself, prove transfer to a new
chemistry.

Use the new `src/mlpdft/uq_compare.py` runner with explicit, time-blocked
`--valid-file` and locked `--test-file` inputs. The older `multi-train.py` uses
an internal `valid_frac`; do not use its random split as the final test. If the
test protocol changes, preserve the locked test set and record the split
manifest.

## Model Comparisons

All methods must start from the **same MACE-OMAT foundation checkpoint**, use
the same training and validation structures, architecture, energy reference,
loss weights, precision, and total epoch budget. Keep the test set out of all
training, scheduler selection, and calibration decisions.

| Method | Construction | Purpose |
|---|---|---|
| Zero-shot foundation model | No fine-tuning | Reference for the value of adaptation |
| Single fine-tune | Report each independently fine-tuned committee member individually | Accuracy reference; not a separate extra run |
| Independent committee | Three fine-tunes from the same foundation checkpoint, seeds 123/124/125, single cosine decay | Standard committee-UQ baseline |
| Warm-snapshot committee | One fine-tune with cosine warm restarts; save three cycle-minimum checkpoints | Candidate low-cost UQ method |
| Multi-head committee | One shared-backbone run with three bootstrap-resampled target heads plus an OMAT replay head | Compare lower-cost head disagreement; report shared-model correlation and replay-data cost |

The `uq_compare.py` defaults use 100 epochs and three warm-restart snapshots
near epochs 33, 66, and 99. Confirm the exported checkpoints correspond to
cycle minima and have converged comparably. Report both equal-member and
training-cost comparisons (including OMAT replay data for multi-head fine-
tuning). Do not interpret a larger snapshot or head spread as better UQ unless
it predicts larger actual errors.

## Evaluation

### Held-Out DFT Configurations

Report results overall and by group, ensemble, and environment:

- Energy MAE/RMSE in meV/atom and force MAE/RMSE in eV/angstrom.
- Ensemble force uncertainty from member disagreement, with the aggregation
  rule fixed in advance (for example, RMS component standard deviation across
  atoms, plus maximum per-atom uncertainty for detecting local failures).
- Rank correlation between predicted uncertainty and actual held-out force
  error; error by uncertainty quantile; risk-versus-coverage curves.
- Calibration on validation data and empirical interval coverage on the locked
  test set. Calibrate scales only on validation data.
- Energy uncertainty as a secondary metric. Check energy-reference alignment;
  a shared systematic energy-zero error can affect every committee member and
  remain invisible to disagreement.

### Physical Validation: Li Transport

The primary physical endpoints are Li diffusivity and effective activation
energy. Use long MACE MD, not the short DFT-AIMD trajectories, for transport
estimates. Keep the training/evaluation trajectories distinct from the MACE
production trajectories. Start with:

1. Bulk LiF from `LIF64_KJPAW_FINAL` as the bulk reference.
2. Li-rich Li-F from `LIWITHF_KJPAW_FINAL` as a defect/disorder comparison.
3. The LiF interface groups for interface-local transport, after checking
  periodicity, slab construction, and interface normal.

The B-containing interface groups are optional follow-up tests, not part of the
primary diffusion result. The BCC_54 pure-Li group can be used as an optional
metal-side control if the Li|LiF model requires a validated Li-metal region.

For bulk and each interface environment, run each committee member at the same
fixed cell and temperature with multiple independent velocity seeds. Use NPT
only for cell/volume equilibration if needed, then measure production MSD in
fixed-cell NVT. For interfaces, resolve Li displacement into in-plane and
normal components; only report a normal diffusion coefficient when an MSD
linear regime exists. NPT trajectories require cell-aware unwrapping.

Estimate Li diffusivity from the long-time MSD slope. Estimate effective
activation energy only from a multi-temperature series with adequate sampling
and independent replicas; the existing DFT data at roughly 300 K do not alone
provide an Arrhenius slope. Report bulk LiF, Li-rich Li-F, and Li|LiF interface
results separately. Track Li hops, coordination changes, and trajectory
stability alongside diffusivity. Compare member-to-member spread in these
observables with uncertainty on the structures visited during the trajectories.

Do not claim that committee agreement is proof of physical accuracy. Where
possible, validate representative high-uncertainty hops or barriers with DFT
single points or NEB. The multi-nanosecond sampling used in published LiF MACE
diffusion work is a reminder that short runs can give noisy transport estimates.

## Decision Criteria

Before running the final test, record the model-selection and UQ-calibration
rules. The warm-snapshot approach is useful only if it offers a favorable
accuracy/cost tradeoff and its disagreement tracks held-out error or physical
failure. Compare against the independent committee at matched ensemble size,
and separately report matched training cost. If snapshot disagreement is larger
but uncorrelated with error, conclude it is not useful UQ for this setting.

## Required Outputs

- Versioned group/frame manifest and block-based train/validation/test files.
- Per-model training configuration, seed, checkpoint epoch, compute time, and
  energy reference.
- Per-group DFT energy/force accuracy and UQ calibration/ranking plots.
- NVT production-MD inputs, random seeds, temperatures, MSD/hop analyses, and
  uncertainty across model members and trajectory replicas.
- A limitations statement for short DFT trajectories, group overlap, missing
  original QE inputs, and any DFT-setting mismatch.
