# UQ Experiment: Warm-Snapshot vs. Independent Committee for MACE

> **Status (2026-10-29):** the transport endpoints in this planning document are
> superseded by the UQ + AL framing in `../DESCRIPTION.md` and `../ROUTE.md`.
> The frozen splits are in `../PHASE-1/AUDIT.md`; the retrospective AL protocol
> is `../PHASE-3/ACTIVE_LEARNING_PROTOCOL.md`. This file is retained as a
> planning record.



## Goal

Test whether a warm-restart snapshot ensemble from a fine-tuned MACE foundation
model gives useful, compute-efficient uncertainty estimates (UQ) for
Li-containing SEI-like materials. The claim is not that snapshot ensembles are
new, but that their disagreement can predict held-out force errors and physical
unreliability as well as an independent-model committee — or not.

## Data & Scope

Primary test: Li/F diffusion, no boron chemistry. Use the final, non-isolated
**NVT** groups from [`DATASET.md`](../PHASE-1/DATASET.md):

- Bulk LiF: `LIF64_KJPAW_FINAL`
- Li-rich Li–F: `LIWITHF_KJPAW_FINAL`
- Li|LiF interface: `LIFINTERFACE_KJPAW_FINAL`

The NPT variants are excluded from the primary scope: the downstream diffusion
validation runs in fixed-cell NVT, and NPT carries a density/barostat confound
(non-uniform settings). They may be reintroduced later as a separate
ensemble-transfer OOD probe.

Rules:

- Before training, verify pseudopotentials, functional, k-points, smearing,
  cutoffs, SCF convergence, restart overlap, and concatenated runs.
- Keep isolated structures for a separate reference-energy experiment, not
  transport.
- Treat `LIWITHBF`, `LIBF4`, `BLI`, `BLI3` and their interfaces as an
  **optional boron ablation**, reported separately. `BCC_54` (pure BCC Li) is an
  optional metal-side control.

Sampling: ~200 time-spaced frames per group after equilibration, with equal
group contributions; report per-group results so long trajectories do not
dominate.

### Equilibration & Sampling

- Determine burn-in per trajectory (temperature, energy, cell/volume for NPT,
  structural observables) — do not assume the first 100 frames suffice.
- After burn-in, thin with stride 10 only if it reduces short-time correlation;
  select across the full usable trajectory, not just the first 200 strided frames.
- Exclude frames without valid DFT energy/forces; investigate non-converged SCF
  or restart boundaries.
- Store a manifest: group, source file, original frame index, ensemble,
  composition, cell, DFT settings, split.

## First Step

Build and freeze the dataset manifest and splits before any training.

## Splits

- **60/40 block-based** train/held-out split at the trajectory level (never a
  random neighboring-frame split) — ~120 train / 80 held-out per group.
- Keep the held-out set locked until model choices and calibration are fixed.
- Add a separate **leave-one-family-out** result (hold out an entire environment
  family) to test transfer; report it separately from the within-trajectory
  60/40 test.
- Use `src/mlpdft/uq_compare.py` with explicit time-blocked `--valid-file` and
  locked `--test-file`. Do not use `multi-train.py`'s internal random split as
  the final test.

## Questions & Hypotheses

1. Does force disagreement rank held-out DFT force errors?
2. At comparable accuracy, does a warm-snapshot committee provide useful
   calibration and risk-vs-coverage versus independently fine-tuned models?
3. Does high uncertainty flag configurations/trajectories where predicted Li
   transport differs across models?

The hypothesis is not that snapshots must win; "cheaper but less calibrated" is
still informative.

## Model Comparisons

All methods start from the **same MACE-OMAT checkpoint** with the same data,
architecture, energy reference, loss weights, precision, and epoch budget. The
test set is excluded from all training, scheduler selection, and calibration.

| Method | Construction | Purpose |
|---|---|---|
| Zero-shot | No fine-tuning | Value of adaptation |
| Single fine-tune | Each committee member reported individually | Accuracy reference |
| Independent committee | 3 fine-tunes, seeds 123/124/125, single cosine decay | Standard committee-UQ baseline |
| Warm-snapshot committee | 1 fine-tune, cosine warm restarts; 3 cycle-minimum checkpoints | Candidate low-cost UQ |
| Multi-head committee | Shared backbone, 3 bootstrap heads + OMAT replay head | Lower-cost head disagreement; report correlation + replay cost |

`uq_compare.py` defaults: 150 epochs, snapshots near epochs 50/100/150 (cycle
ends 49/99/149 under T₀ = epochs//count = 50). Confirm checkpoints are cycle
minima and converged. Report equal-member and matched-cost comparisons. A larger
spread is not "better UQ" unless it predicts larger errors.

## Evaluation

### Held-Out DFT Configurations

Report overall and by group/ensemble/environment:

- Energy MAE/RMSE (meV/atom); force MAE/RMSE (eV/Å).
- Ensemble force uncertainty from member disagreement, with the aggregation rule
  fixed in advance (RMS component std across atoms + max per-atom uncertainty).
- Uncertainty-vs-error rank correlation; error by uncertainty quantile;
  risk-vs-coverage curves.
- Calibration on validation; empirical coverage on the locked test. Calibrate
  scales on validation only.
- Energy uncertainty as secondary; check energy-reference alignment (shared
  offsets are invisible to disagreement).

### Physical Validation: Li Transport

Endpoints: Li diffusivity and effective activation energy, from long MACE MD
(not short DFT-AIMD). Keep training/evaluation trajectories distinct from the
MACE production trajectories.

Start with: (1) bulk LiF `LIF64_KJPAW_FINAL`, (2) Li-rich `LIWITHF_KJPAW_FINAL`,
(3) the LiF interface groups (check periodicity, slab construction, and
interface normal).

- Per environment: run each member at fixed cell and temperature with multiple
  velocity seeds; equilibrate in NPT if needed, then measure production MSD in
  fixed-cell NVT.
- Interfaces: resolve Li displacement into in-plane and normal components;
  report a normal diffusion coefficient only with a linear MSD regime. Unwrap
  NPT cells.
- Estimate D from the long-time MSD slope; estimate E_a only from a
  multi-temperature series with independent replicas (the ~300 K DFT data alone
  give no Arrhenius slope).
- Report bulk, Li-rich, and interface results separately; track Li hops,
  coordination changes, and stability.
- Do not treat agreement as proof of accuracy; validate representative
  high-uncertainty hops/barriers with DFT single points or NEB where possible.

## Decision Criteria

Record model-selection and calibration rules before the final test. The
warm-snapshot approach is useful only if it has a favorable accuracy/cost
tradeoff *and* its disagreement tracks held-out error or physical failure.
Compare at matched ensemble size and matched training cost. If disagreement is
larger but uncorrelated with error, conclude it is not useful UQ here.

## Required Outputs

- Versioned group/frame manifest and block-based train/valid/test files.
- Per-model config, seed, checkpoint epoch, compute time, energy reference.
- Per-group DFT accuracy and UQ calibration/ranking plots.
- NVT production-MD inputs, seeds, temperatures, MSD/hop analyses, and
  uncertainty across members and replicas.
- Limitations: short DFT trajectories, group overlap, missing original QE inputs,
  and DFT-setting mismatch.
