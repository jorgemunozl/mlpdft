# Project Route: UQ-Guided Active Learning for Li Diffusion with MACE

## Research Goal

Determine whether uncertainty from fine-tuned MACE ensembles can select DFT
configurations that improve predictions of lithium diffusion more efficiently
than random selection. Compare independent-seed fine-tuning, warm-restart
snapshots, and multi-head fine-tuning. The central result is the tradeoff
between DFT labels acquired and reliable Li-transport predictions, not a broad
benchmark across all Li-F-B battery chemistry.

The FitSNAP comparison and broad cathode/electrolyte/SEI survey are out of
scope. Boron-containing groups are optional transfer extensions.

## Scope

Use the final non-isolated Li/F groups:

- `LIF64_KJPAW_FINAL` and `LIF64_KJPAW_NPT_FINAL`: bulk LiF.
- `LIWITHF_KJPAW_FINAL` and `LIWITHF_NPT_FINAL`: Li-rich Li-F.
- `LIFINTERFACE_KJPAW_FINAL` and `LIFINTERFACE_KJPAW_NPT_FINAL`: Li|LiF
  interface.

Keep NVT and NPT labels. Use NPT to equilibrate cell/volume where needed, then
use fixed-cell NVT production trajectories for MSD analysis. Include `BCC_54`
only as an optional pure-Li metal-side control; it is not a boron-containing
group. `LIWITHBF`, `LIBF4`, `BLI`, `BLI3`, and their interfaces are optional and
do not enter the primary Li/F diffusion claim.

## Phase 1 — Dataset Audit and Frozen Splits

- Verify composition, periodic cell, trajectory boundaries, timestamps,
  functional, pseudopotentials, cutoffs, k-points, smearing, and SCF convergence
  for every selected group.
- Record group, trajectory/cell ID, source file, original frame index,
  ensemble, composition, cell, and DFT setup for every structure.
- Determine the equilibration cutoff independently for each trajectory using
  temperature, energy, cell/volume for NPT, and structural observables.
- Split by contiguous trajectory blocks or independent runs, never by random
  neighboring frames. Freeze training, validation/calibration, locked test, and
  candidate-pool membership before model comparisons.
- Keep the locked test disjoint from every acquisition candidate and never
  query it with DFT during active learning.

## Phase 2 — Initial Training Set and Baselines

Build the seed training set from time-spaced equilibrated configurations in the
six core groups. Use equal per-group contributions initially; treat a target of
about 200 structures per group as a sampling target, not 200 independent
observations. Use the same seed set, validation data, foundation checkpoint,
energy reference, and training budget for all methods.

Evaluate the zero-shot MACE foundation model and train:

1. Three independent fine-tunes from the same foundation checkpoint (reference
   committee).
2. One warm-restart fine-tune; export checkpoints at planned cycle minima.
3. One multi-head fine-tune with three block-bootstrap target heads and the
   required OMAT replay head. Report replay data and cost separately.

## Phase 3 — UQ Validation Before Acquisition

Use the validation/calibration data to check energy/force MAE and RMSE,
force-error ranking by ensemble disagreement, uncertainty calibration, and
risk-versus-coverage. Fit any calibration scale using calibration data only.
Report these diagnostics by Li/F environment and NVT/NPT provenance. Use them to
freeze acquisition scores and settings before querying new DFT structures.
Do not inspect or report locked-test results during this phase or between
active-learning rounds.

Use a predeclared score, such as maximum per-atom force standard deviation for
each frame. Include a single-model point-estimate baseline. Do not treat greater
ensemble spread as better UQ unless it predicts greater error.

## Phase 4 — Active Learning Loop (Required)

Generate an unlabeled candidate pool from MACE MD in the target Li/F diffusion
environments: bulk LiF, Li-rich Li-F, and Li|LiF interface. Use temperatures
relevant to the transport study, filter unphysical structures, and preserve
candidate trajectory IDs. Keep this pool separate from calibration and locked
test data.

For each acquisition round:

1. Score the same candidate pool with independent, snapshot, and multi-head
   ensembles using the same predeclared uncertainty score.
2. Select the same number of candidates per method, stratified across the three
   environments so one system cannot consume the entire query budget.
3. Select an equal-size random batch from the same pool as the control. Dedup
   overlapping selections and do not query locked test frames.
4. Calculate DFT energy and forces for selected candidates with consistent
   settings; record convergence and discard or separately flag failures.
5. Add each method's acquired labels to its own training set and retrain with
   the same validation/calibration and locked-test sets.

Run multiple rounds if compute permits. Plot held-out force error and transport
QoI error against cumulative new DFT labels. Compare UQ selection with random
selection at equal label budgets, and report GPU-hours/inference cost as well.
The active-learning claim requires new DFT labels; uncertainty ranking alone
does not constitute an active-learning result.

## Phase 5 — Physical Validation: Li Transport

The primary quantities of interest are Li diffusivity and effective diffusion
activation energy.

- For every method and AL round, run long MACE MD with identical cells,
  temperatures, and multiple independent velocity seeds.
- Use NPT only for cell equilibration when needed, then measure production MSD
  in fixed-cell NVT. Use cell-aware unwrapping for any variable-cell analysis.
- Estimate diffusivity from a supported long-time MSD linear regime. At the
  interface, resolve in-plane and normal components only when geometry and
  sampling support them.
- Obtain effective activation energy from a multi-temperature Arrhenius fit
  with independent replicas and uncertainty estimates. One approximately
  300 K AIMD dataset is not a temperature series.
- Track Li hops and coordination changes. Check convergence of hop statistics,
  MSD windows, replica-to-replica diffusivity, and activation energy.
- Validate representative high-uncertainty hops with DFT single points or NEB
  where feasible. Ensemble agreement alone does not establish physical
  accuracy.

## Phase 6 — Analysis and Success Criteria

The primary success criterion is that an uncertainty-driven acquisition method
reaches predeclared force-error and Li-transport accuracy targets with fewer new
DFT labels than random selection. Before opening the locked test, predeclare
these target tolerances, acquisition batch size, calibration method, and stopping
rule. Use validation/calibration data during rounds; evaluate the locked test
once after acquisition and model choices are complete.

Report:

- Final locked-test force/energy errors, uncertainty-error ranking, calibration,
  and risk-versus-coverage by group and ensemble.
- Li diffusivity and activation energy with model-member and replica spread.
- Active-learning curves versus cumulative DFT labels for each method and the
  random control.
- DFT-label count, GPU-hours, training time, and inference cost.

If UQ fails to rank errors or does not improve label efficiency, report the
negative result. Do not broaden the primary claim to boron, electrolyte, or
other battery chemistries based on the Li/F study.

## Deliverables

- Frozen manifest and disjoint train, calibration/validation, locked-test, and
  candidate-pool IDs.
- Per-round DFT query lists, input/output files, convergence status, and added
  label counts.
- Reproducible configs and checkpoints for all three UQ methods.
- Figures for uncertainty versus error, calibration, active-learning label
  efficiency, Li diffusivity, and activation-energy convergence.
- A limitations report covering trajectory correlation/length, group overlap,
  DFT-setting consistency, and transport sampling uncertainty.
