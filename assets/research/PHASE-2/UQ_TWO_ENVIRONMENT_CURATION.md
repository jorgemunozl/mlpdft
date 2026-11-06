# Two-environment interim UQ dataset: curation result

**Status:** development candidate; **not cleared for production UQ
comparisons**. The interface is excluded from all files, as requested. The
candidate follows the applicable checks in
[`UQ_INTERIM_DATASET_CURATION.md`](UQ_INTERIM_DATASET_CURATION.md), with the
three-environment frame-count gate replaced by the counts below.

Private dataset:
[`jorgemunozl/li-f-uq-interim-two-environments`](https://huggingface.co/datasets/jorgemunozl/li-f-uq-interim-two-environments),
revision `c2699e8e52889c18f6d29f657eb996ac3f78524f`. All four split files,
manifests, diagnostics and README were downloaded after upload and their hashes
matched local files. The previous dataset and its revision remain unchanged.

| Environment | Train | Validation | Calibration | Test |
|---|---:|---:|---:|---:|
| Bulk LiF, NPT | 500 | 100 | 100 | 100 |
| F-doped Li, NVT | 500 | 100 | 100 | 100 |
| **Total** | **1,000** | **200** | **200** | **200** |

## Curation decisions

- Use exactly one source per environment, `LIF64_KJPAW_NPT_V3` and
  `LIWITHF_V3`, each at source stride 3. This resolves the old bulk mixture's
  different train/test source composition. It does not establish that the two
  sources have equivalent QE settings.
- The earlier audit's F-doped zero-step cutoff is not defensible as proof of
  equilibration: the mean force and energy change appreciably through the
  early trajectory. Exclude the first **2,400 inferred steps** for F-doped and
  **100** for bulk as provisional burn-ins, not as verified equilibrium
  times. The F-doped choice leaves fewer than 800 eligible training frames;
  use 500 per environment rather than filling the deficit with early frames.
- Maintain the original audit's time blocks, split its calibration block into
  validation and calibration, and exclude 60 steps on each side of each
  boundary. Time spacing in the final selection is irregular. Remaining
  segments are from the same trajectories and retain temporal dependence.

## Checks and unresolved gates

- 1,600 configurations; no repeated `(source_group, source_file, frame_index)`
  and no repeated rounded geometry hash in or across the splits. No selected
  frame has an eigenvalue warning in its mapped SCF block; energy offset spread
  is < 0.05 eV for each source. Finite labels, positive cells, periodicity and
  force-array sizes were checked; the largest force is < 5 eV/Å.
- No exact rounded-geometry matches to the old pilot test were found. This does
  not rule out similar or temporally adjacent configurations; the test is not
  an independent-trajectory or OOD benchmark.
- Energy-per-atom and pooled atom-wise force-component autocorrelations were
  computed per environment/split on the regular stride-3 source windows. The
  manifest includes IAT, approximate source-window ESS, proposed block lengths,
  and a maximum-lag flag. Several F-doped estimates reach the lag limit, so
  neither the reported ESS nor suggested block lengths establish reliable
  confidence intervals. The train energy estimate is approximately 3.5 source
  observations; the test force estimate is approximately 3.9; both hit the lag
  limit. These are triage diagnostics, not counts of independent selected
  frames.
- Burn-in is **not verified**: F-doped statistics continue to drift, while
  early bulk volume also varies. Inspect the 300-step source windows and secure
  independent trajectories or justify stationarity before interpreting results
  as production evidence. The 60-step guard does not establish independence.
- Pass all four split paths, including `--calibration-file`, in a future run.
  Do not import old pilot summaries. Check actual per-head data and training
  budgets again because the pool now has 1,000 rather than 2,000 frames.

The published dataset card, `dataset_manifest.json`, `frame_manifest.csv`,
`temporal_diagnostics.json` and `SHA256SUMS` contain the full details. The
builder is [`build_interim_two_environment.py`](../../../src/mlpdft/utils/build_interim_two_environment.py);
it reuses the existing split materializer and per-source SCF checker.
