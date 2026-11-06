# Interim Li–F UQ dataset — curation audit and required actions

**Status:** curation audit of the planned `li-f-uq-interim-2900` dataset.
No dataset rebuild or training run was performed for this report.
**Scope:** the 2,000 / 300 / 300 / 300 interim split used by
[`UQ_INTERIM_DATASET_EXPERIMENTS.md`](UQ_INTERIM_DATASET_EXPERIMENTS.md).
**Builder:** [`build_interim_uq_dataset.py`](../../../src/mlpdft/utils/build_interim_uq_dataset.py),
which selects from the Phase 1 audit table
[`frames.csv`](../PHASE-1/audit/frames.csv) and materialises the extxyz files.

This report distinguishes what the builder already enforces from what must be
verified or fixed before the dataset can support the planned UQ claims.

## 1. Dataset identity

- Hugging Face dataset: `jorgemunozl/li-f-uq-interim-2900`.
- Pinned revision: `0014d10c5f6355e84c596fbcf25a6eadad6c81fc` (from the
  experiment plan).
- Files: `uq_train.extxyz`, `uq_valid.extxyz`, `uq_calibration.extxyz`,
  `uq_test.extxyz`, `frame_manifest.csv`, `dataset_manifest.json`, and the
  dataset card / checksums.
- Labels: `REF_energy` (eV) and `REF_forces` (eV/Å).
- The runner now accepts `--calibration-file`
  ([`uq_calibration.py`](../../../src/mlpdft/uq_calibration.py)); pass all four
  paths explicitly. Omitting it falls back to validation with a warning, which
  does **not** satisfy this protocol.

## 2. Current selection recipe

- Environment quotas: bulk 800, F-doped 800, interface 400 training frames.
- Validation / calibration / test: 100 frames per environment each.
- Selection: deterministic evenly spaced indices within eligible time blocks;
  output cadence is irregular, not a fixed stride.
- Time guard at every split boundary: `GUARD = 16` time indices, where
  `time_index = source_frame_index × source_stride` (MD steps).
- Thermalisation cutoffs from the audit: bulk 3, F-doped 0, interface 586 time
  steps. The interface cutoff removes a ≈0.7 eV/atom initial relaxation; bulk
  and F-doped are treated as equilibrated from the start.
- Validation and calibration are the first and second halves of the audit's
  calibration block, separated by the guard; test is the audit's test block.

## 3. Reconstructed composition

Reconstructed from `frames.csv` with the builder's own selection logic.
**Confirm against the shipped `frame_manifest.csv` before relying on it.**

### Bulk LiF (NPT) — three source runs, two strides

| split | LIF64_KJPAW_NPT (stride 3) | LIF64_KJPAW_NPT_V2 (stride 15) | LIF64_KJPAW_NPT_V3 (stride 3) |
|---|---:|---:|---:|
| train | 363 | 71 | 366 |
| validation | 46 | 10 | 44 |
| calibration | 47 | 9 | 44 |
| test | 50 | **0** | 50 |

The test set contains **no `LIF64_KJPAW_NPT_V2` frames**, although that run is
9% of training and present in validation/calibration. The pooled time sort with
tie-breaking places the stride-15 frames after the stride-3 frames at equal
times, and the evenly spaced selection never lands on them. This is a selection
artifact, not a physical exclusion.

### F-doped Li — one run, two strides

| split | LIWITHF_V3 (stride 3) | LIWITHF_V3 (stride 5) |
|---|---:|---:|
| train | 570 | 230 |
| validation | 72 | 28 |
| calibration | 71 | 29 |
| test | 70 | 30 |

### Interface — one run, one stride

`LIFINTERFACE_KJPAW_V2`, stride 1: 400 / 100 / 100 / 100 frames.

## 4. QC the builder already enforces

- Exact expected counts (2,000 train, 300 per held-out split).
- No duplicate rounded structures across splits (`species`, positions, cell
  diagonal rounded to 0.001 Å).
- Finite energy, forces, cell and positions in every written frame.
- Per-frame SCF-to-energy alignment check (`verify_scf`): no warned frames and
  energy-offset spread below 0.05 eV.
- Frame count > 0 with maximum force reported; frames with force above
  5 eV/Å are counted.
- Prior pilot-test hash overlap is reported when `--previous-test` is supplied.

These are necessary but not sufficient for UQ: they do not test temporal
independence or split representativeness.

## 5. Empirical dependence (critical)

Lag-1 autocorrelation of energy per atom on the selected frames, and a naive
effective sample size `n(1−r)/(1+r)` (triage only; compute the full
autocorrelation function before final analysis):

| Environment | split | median gap (time steps) | lag-1 energy | naive ESS |
|---|---|---:|---:|---:|
| Bulk LiF | train (mixed runs) | 9 | 0.10 | not meaningful (mixture) |
| Bulk LiF | test | 15 | 0.86 | ~7 / 100 |
| F-doped Li | train | 5 | 0.993 | ~3 / 800 |
| F-doped Li | test | 9 | 0.89 | ~6 / 100 |
| Interface | train | 3 | 1.000 | ~0 / 400 |
| Interface | test | 2 | 0.99 | ~1–3 / 100 |

The 16-step guard is a **leakage guard, not a decorrelation buffer**. The
correlation time is tens to ~100 frames; the selected frames are a small number
of quasi-independent blocks, not 100 (or 800) independent samples. Energy is a
slow global proxy; force autocorrelation must be measured separately, but the
order of magnitude is clear.

**Consequence:** 100 test frames per environment cannot support narrow
frame-level CIs, and 2,000 labels are not 2,000 independent structures. Do not
call this a blind or OOD benchmark; the interface is a time-block of a
trajectory represented in training.

## 6. Required curation actions before training

### D1 — Fix the bulk test composition (blocking)

Either:

- **stratify selection by `(group, stride)`** within every environment, with
  quotas proportional to eligible frames (bulk test should then contain ~9%
  `LIF64_KJPAW_NPT_V2`), or
- **restrict bulk to one NPT run** (for example `LIF64_KJPAW_NPT_V3`) in all
  four splits and state that bulk is one trajectory, not a three-run mixture.

Do not keep the current mixture in training while the test is a different
mixture. Rebuild and re-upload if this changes the dataset revision.

### D2 — Quantify and use temporal dependence (blocking for CIs)

- Compute autocorrelation of energy per atom and force components per
  environment and split; record the integrated autocorrelation time and the
  resulting block length.
- Use block bootstrap or segment-level resampling with blocks at least as long
  as the measured correlation time. Do not compute frame-level confidence
  intervals as if frames were independent.
- Choose explicitly between: (a) keep 100 frames per split and analyse at block
  level (≈5–10 blocks), or (b) coarsen to ~20–30 separated frames per split and
  report honest CIs. The trajectories cannot supply 100 independent frames with
  these correlation times.

### D3 — Verify burn-in for bulk and F-doped

The audit's block test is coarse; a cutoff of 3 or 0 means no burn-in. Inspect
the first 50–100 steps of each source run (energy per atom, force distribution,
cell) against later blocks. If a transient is present, discard a fixed burn-in
and record it.

### D4 — Make the split guard physical

A 16-step guard is shorter than the correlation time. Increase it to at least
the measured correlation length (environment-specific, in MD steps), or state
that splits are adjacent time blocks with residual dependence. The current
33–39-step boundary gaps are not decorrelation.

### D5 — Report the bulk mixture

If the three NPT runs are kept, the dataset card and every result table must say
so; report run identity as a random effect or stratify metrics by run. The
"bulk" number otherwise mixes thermostats/settings that the audit could not
verify from the extxyz.

### D6 — Frame-level split disjointness

The builder checks rounded-structure duplicates. Also check that no
`(source_file, frame_index)` pair appears in two splits using
`frame_manifest.csv`. File-hash checks in `resolve_calibration_file` cannot
detect partial overlap.

### D7 — Keep the prior pilot out of this comparison

The experiment plan reports 3 / 4 / 8 rounded-structure overlaps between the
earlier pilot test and the new validation / calibration / test splits. Treat the
new test as previously observed; do not tune on it, and make final claims on a
later independent dataset. Fresh output directories must be used so the runner
does not auto-import `lif_uq_001`.

## 7. Acceptance gates

Before a production run, all of the following must hold and be recorded in the
run manifest:

- [ ] HF revision and SHA256 checksums verified.
- [ ] 2,000 / 300 / 300 / 300 frames; per environment 800 / 100 / 100 / 100
      and 400 / 100 / 100 / 100.
- [ ] No duplicated rounded structure across splits; no shared
      `(source_file, frame_index)` across splits.
- [ ] Bulk test group/stride composition matches training within a stated
      tolerance, or one-run bulk is declared.
- [ ] SCF warning rate 0 in selected frames; energy-offset spread < 0.05 eV.
- [ ] No NaN/Inf; maximum force reported.
- [ ] Burn-in verified for bulk and F-doped; interface cutoff 586 recorded.
- [ ] Autocorrelation time and block length recorded per environment/split;
      ESS reported alongside every frame count.
- [ ] Calibration split physically separate from validation and test; α fitted
      only there; calibration path recorded by the runner.
- [ ] `frame_manifest.csv` and `dataset_manifest.json` archived with the run.

## 8. Reproduction

```bash
uv run --locked python src/mlpdft/utils/build_interim_uq_dataset.py \
  --data-root /path/to/qe_outputs \
  --frames assets/research/PHASE-1/audit/frames.csv \
  --output-dir dataset/xyz_files/uq_interim \
  --previous-test dataset/xyz_files/uq_splits/uq_test.extxyz
```

The builder writes `frame_manifest.csv` and `dataset_manifest.json`; verify the
composition table in §3 and the QC entries before uploading a new revision.

## 9. Claims this dataset cannot support

- "2,000 independent DFT labels" — the frames are correlated time blocks.
- "A blind/OOD test set" — the test is a tail block of trajectories represented
  in training, and overlaps the prior pilot.
- "Data starvation is solved" — the label count is higher, but effective
  diversity is much lower than the count; report ESS.
- "This reproduces the paper's MHC setup" — the dataset, split and budget are a
  new adaptation.
