# Project Route: UQ and Active Learning for Fine-Tuned MACE on a Li–F Testbed

## Research goal

At equal label budget, with measured compute cost, determine which of three
ensemble-based uncertainty-quantification strategies for a fine-tuned MACE
foundation potential best ranks held-out DFT error within chemically distinct
Li–F environments, stays calibrated, and selects informative DFT configurations
more efficiently than random sampling.

The Li–F data is a controlled testbed. The paper does not claim a transport,
SEI-formation, or electrochemical result.

## Phase 0 — Freeze scope and claims

Record before training:

- the claims and non-claims from `DESCRIPTION.md`;
- the environment definitions (bulk LiF, F-doped Li, interface film, Li metal);
- the primary metrics and the pre-registered success thresholds;
- the negative-result branch.

No result may broaden the claim to transport, SEI formation, or other
chemistries.

## Phase 1 — Dataset audit, provenance, and frozen splits (current focus)

### Provenance and de-duplication

- Build a per-file table: group, filename, stride, frame count, frame-0 hash,
  composition, cell, DFT settings, and SCF status.
- Collapse files that are strides or subsets of the same trajectory. Known
  hazard: `LIFINTERFACE` V1/V2/NPT/NPT_V2 share one frame-0 hash, and some
  `*_NPT_FINAL` files duplicate `LIF64_KJPAW_V2` / `LIWITHF_V3` frames.
- Report unique trajectories and unique frames, not file counts.
- Confirm composition and cell for every group. Measured: `LIWITHF` is
  53 Li + 1 F (F-doped Li metal); `LIFINTERFACE` is 86 Li + 36 F (a two-layer
  LiF film on Li, ~4 Å of LiF, with a ~2 Å gap to the periodic image).
- Verify functional, pseudopotentials, cutoffs, k-points, smearing, timestep,
  thermostat, and SCF convergence. Treat the 110/440 Ry literature values as
  external references, not pool data.

### Environment labels

Label every frame by environment from its structure, not only its directory:

- bulk LiF: Li and F coordination consistent with rocksalt;
- F-doped Li: Li-rich metal with isolated F;
- interface film: LiF layers in contact with Li metal;
- Li metal: pure Li.

Record per-frame Li coordination, distance to the LiF/Li boundary, and local
composition. These labels drive the stratified metrics in Phase 3.

### Frozen splits

- Split by contiguous trajectory blocks and by independent trajectory, never by
  random neighbouring frames.
- Hold out at least one independent trajectory per environment where more than
  one exists; otherwise hold out contiguous blocks and state the limitation. Do
  not let versioned copies of the same trajectory cross the split.
- Keep the locked test disjoint from calibration and from every acquisition
  candidate.
- Target a usable per-environment test size. If the interface trajectory cannot
  supply it, report the interface result as a case study and say so before
  looking at outcomes.
- Freeze the manifest (file, frame index, environment, split) before any model
  comparison.

### Phase 1 outputs

The frozen audit is [`PHASE-1/AUDIT.md`](PHASE-1/AUDIT.md) plus `PHASE-1/audit/provenance.csv`,
`PHASE-1/audit/frames.csv` (per-frame environment and split), and `PHASE-1/audit/manifest.json`.
Split counts, equilibration cutoffs, and excluded sets are recorded there.

## Phase 2 — Models, baselines, and cost accounting

Use one foundation checkpoint, one seed set, one energy reference, and one
training budget for all arms:

1. three independent fine-tunes (reference committee);
2. one warm-restart fine-tune with cycle-end checkpoints;
3. one multi-head fine-tune with block-bootstrap heads and the replay head.

Baselines: zero-shot MACE, a single fine-tune point estimate, and random
acquisition. Record wall-clock, GPU-hours, and inference cost per arm and per
retrain.

## Phase 3 — UQ validation before acquisition

On calibration data only, report:

- force and energy MAE/RMSE, overall and **within each environment**;
- calibration: ECE, NLL, calibration curves;
- discrimination: Spearman ρ(σ, error), AULC;
- risk–coverage curves.

Pooled numbers are secondary and must be labelled as such: the environments
differ in scale, so a pooled correlation can be driven entirely by
between-group contrast. Fit any calibration scale on calibration data only.
Freeze the acquisition score before any label is revealed.

Include a single-model point-estimate baseline. Do not treat larger spread as
better UQ unless it predicts larger error.

## Phase 4 — Active-learning evaluation

Primary evidence is a **retrospective** AL simulation on the existing labeled
pool: hide labels from the acquisition rule, acquire in rounds, retrain, and
measure held-out error against cumulative labels. This compares acquisition
rules at equal label budget without new DFT.

If budget allows, confirm with one real-DFT round on a small batch; report it
separately and do not pool it with the retrospective curves.

For each round:

1. score the same pool with all three arms using the frozen score;
2. select equal-size batches stratified by environment;
3. select an equal-size random batch from the same pool as the control;
4. deduplicate selections; never acquire locked-test frames;
5. retrain each arm on its own acquired labels;
6. evaluate on calibration data only until the loop closes.

Report held-out error versus cumulative labels, acquisition composition per
round, and GPU-hours per arm. An uncertainty-ranking result alone is not an AL
result.

## Phase 5 — Tier-0 physical anchor

One static check, no new DFT by default:

- build Li vacancy and Li interstitial cells in bulk LiF (3×3×3, cross-checked
  at 2×2×2);
- evaluate the fine-tuned MACE potential;
- compare formation energies with the published values (≈0.76 eV vacancy,
  ≈1.73 eV interstitial).

Accept agreement within ~0.2 eV, accounting for the 110/440 Ry literature
settings versus 40/320 Ry here. If the disagreement exceeds that, run a small
matched-settings DFT check before making any physical claim. The anchor is a
sanity check on the potential, not a second research thrust.

## Phase 6 — Analysis and success criteria

Pre-register before Phase 3:

- the primary metric (within-environment ranking and calibration);
- the AL comparison (area under the label-efficiency curve versus random);
- the minimum effect size worth claiming;
- the stopping rule and the negative-result branch.

Open the locked test once, after acquisition and model choices are complete.
Report:

- locked-test force/energy errors and calibration by environment;
- within-environment Spearman/AULC and risk–coverage by arm;
- AL curves versus cumulative labels for each arm and random;
- cost per arm;
- tier-0 anchor result;
- B/salt transfer results, if included, reported separately and never pooled
  with the primary claim.

If the arms do not separate, report the null result. A well-measured "cheap
snapshots match the committee" is a result, not a failure.

## Phase 7 — Limitations and reporting

State explicitly:

- small dataset, one foundation checkpoint, one seed set;
- environments are compositionally distinct, so pooled metrics are misleading;
- the interface cell is a thin LiF/Li film, not a buried interface;
- no transport, SEI-formation, or electrochemical validation;
- DFT settings are not uniform across the pool and the literature;
- trajectory correlation limits the effective sample size.
