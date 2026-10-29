# Phase 1 Audit — UQ + AL Li–F Testbed

**Date:** 2026-10-29
**Scope:** Phase 1 only — provenance, de-duplication, environment labels, and
frozen splits. No model training, no active-learning run, no transport work.
**Script:** [`src/mlpdft/utils/phase1_audit.py`](../../../src/mlpdft/utils/phase1_audit.py)
**Data root:** `/home/jorge/project/mlpdft/dataset` (read-only)
**Artifacts:** `provenance.csv`, `frames.csv`, `manifest.json` in this directory.

Reproduce with:

```bash
python3 src/mlpdft/utils/phase1_audit.py \
  --data-root /home/jorge/project/mlpdft/dataset \
  --output-dir assets/research/PHASE-1/audit
```

---

## 1. Provenance

| Quantity | Value |
|---|---:|
| Groups with extxyz files | 17 |
| Files parsed | 63 |
| Raw frames | 38,759 |
| Unique frames | 27,960 |
| Duplicate frames removed | 10,799 (27.9%) |
| Files contributing zero unique frames | 37 |
| Isolated reference frames (excluded) | 222 |

`BCC_54_KJPAW_FINAL` and `BCC_54_KJPAW_NPT_FINAL` contain only `.out` logs and
no `xyz_files/`, so there is **no pure-Li control in the pool**.

The duplicate rate is not noise. Multiple files are strides or copies of the
same trajectory:

- `LIFINTERFACE` V1, V2, NPT, and NPT_V2 share one frame-0 hash.
- `LIF64_KJPAW_NPT_FINAL_15_666` duplicates `LIF64_KJPAW_V2` frames.
- The long `LIWITHF_NPT_FINAL` files duplicate `LIWITHF_V3` frames.
- 37 of 63 files add no new frame at all.

Consequence: file count and frame count are not sample counts. All splits below
are assigned by trajectory family, after position-hash de-duplication.

## 2. Measured compositions

Group names are not reliable descriptors. The measured compositions are:

| Family | Composition | Atoms | Cell (Å) | What it is |
|---|---:|---:|---|---|
| `LIF64` | 32 Li + 32 F | 64 | 8.16 cubic | 2×2×2 rocksalt LiF |
| `LIWITHF` | 53 Li + 1 F | 54 | 10.53 cubic | 3×3×3 BCC Li with one F substitution |
| `LIFINTERFACE` | 86 Li + 36 F | 122 | 12.24 × 12.24 × 11.75 | two-layer LiF film on Li metal |
| Isolated `LIF64` | 32 Li + 32 F | 64 | 28.16 | isolated reference |
| Isolated `LIWITHF` | 53 Li + 1 F | 54 | 30.53 | isolated reference |

`LIWITHF` is **F-doped Li metal**, not a Li-rich Li–F compound. `LIFINTERFACE`
is a thin-film model, not a buried interface.

Structural descriptors confirm the labels:

| Environment | Li–F coordination (mean) | Li–Li nearest-neighbour (median) |
|---|---:|---:|
| Bulk LiF | 6.00 | 2.73 Å |
| F-doped Li | 0.07 | 2.71 Å |
| Interface | 2.26 | 2.60 Å |

The Li–Li values are below the ideal rocksalt value (2.88 Å) because of thermal
and local distortion; they are a sanity check, not a lattice-constant
measurement.

## 3. Interface geometry

- Exactly **36 F atoms in two planes**, at z ≈ 0 and z ≈ 2.05 Å, 18 F per plane.
  The 2.04 Å spacing is the LiF (001) interplanar distance, so the film is
  **two LiF layers, ≈4 Å thick**.
- Li metal occupies z ≈ 2.3–9.5 Å.
- The top of the Li slab sits only ≈2.1 Å below the periodic image of the LiF
  plane — too close to be vacuum, close enough to bond.
- The in-plane cell is 12.24 Å = 3 × 4.08 Å (LiF), but 12.24 / 3.51 = 3.49 for
  BCC Li. The metal is strained or incommensurate with the LiF mesh.
- Post-equilibration frames still contain short Li–Li contacts (minimum
  2.19 Å) and forces up to 7.5 eV/Å in the NPT interface trajectory.

This cell is admissible only as an **OOD structural environment**. It is not a
physical buried Li|LiF interface, and it is not used for transport.

## 4. Electronic-structure quality

SCF warning rate = `eigenvalues not converged` / `convergence has been
achieved`, from each group's `.out` log.

| Group | Warning rate |
|---|---:|
| `LIFINTERFACE_KJPAW_V1` | 9.34 |
| `LIFINTERFACE_KJPAW_NPT_FINAL` | 0.82 |
| `LIFINTERFACE_KJPAW_NPT_V2` | 0.54 |
| `LIWITHF_NPT_FINAL` | 0.31 |
| `LIFINTERFACE_KJPAW_NPT` | 0.30 |
| `LIFINTERFACE_KJPAW_V2` | 0.001 |
| All `LIF64` groups, `LIWITHF_V3`, `LIWITHF_KJPAW_FINAL`, `LIFINTERFACE_KJPAW_FINAL` | 0.00 |

Frames from groups with rate **> 0.1** are excluded from the primary blocks:
**348 frames** (`excluded_scf`), almost all from `LIFINTERFACE_KJPAW_V1`.

The interface primary trajectory is clean (`LIFINTERFACE_KJPAW_V2`, 0.001). The
interface **OOD** trajectory (`LIFINTERFACE_KJPAW_NPT`, 0.54) is not clean; there
is no independent interface trajectory with both zero warnings and more than
200 frames. It is kept as a flagged stress test, not as a clean benchmark.

## 5. Equilibration

Energy per atom and volume per atom are compared in 16 blocks against the last
four blocks; earlier blocks outside ±0.02 eV/atom or ±0.5% volume are marked
`excluded_equilibration`.

| Environment | Cutoff (time index) | Frames excluded | Reason |
|---|---:|---:|---|
| Bulk LiF | 3 | 0 | stable from the start |
| F-doped Li (clean V3) | 0 | 0 | NVT, stable density |
| Interface | 586 | 1,870 | ≈0.7 eV/atom relaxation in the first block |

The F-doped NPT run (`LIWITHF_NPT_FINAL`) has a density transition at time
≈6,069 and a 0.31 SCF warning rate. It is excluded from the primary split
(`excluded_extra`, 7,550 frames), not used as training or test data.

## 6. Frozen splits

Rule:

1. Within each environment, families are ranked by SCF cleanliness, then size.
2. Frames with SCF warning rate > 0.1 are `excluded_scf`.
3. The primary family is ordered by timestep and split **contiguously**:
   first 70% `train_pool`, next 15% `calibration`, last 15% `test`.
4. The largest non-primary family becomes `ood_test`, evenly subsampled to at
   most 1,000 frames.
5. Remaining families are `excluded_extra`; isolated cells are `excluded`.

| Environment | Unique | Train | Calibration | Test | OOD test | Excl. SCF | Excl. equil. | Excl. extra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bulk LiF | 12,462 | 5,131 | 1,099 | 1,099 | 934 | 0 | 0 | 4,199 |
| F-doped Li | 10,671 | 1,952 | 418 | 418 | 333 | 0 | 0 | 7,550 |
| Interface | 4,605 | 1,230 | 264 | 264 | 429 | 348 | 1,870 | 200 |
| Isolated | 222 | — | — | — | — | — | — | 222 |
| **Total** | **27,960** | **8,313** | **1,781** | **1,781** | **1,696** | **348** | **1,870** | **11,949** |

What is frozen:

- `train_pool` is the active-learning pool; its labels are hidden from the
  acquisition rule during the retrospective simulation.
- `calibration` is used for calibration fitting and early stopping only.
- `test` is the in-domain locked test: a contiguous tail of the same primary
  trajectory, separated from training by the calibration block.
- `ood_test` is an independent trajectory, capped at 1,000 frames.

The in-domain test and the OOD test are both locked. Neither is queried during
active learning.

## 7. Unresolved and blocking

1. **Thermostat and timestep are not in the extxyz.** The `.out` logs contain no
   input namelist, so temperature and thermostat cannot be verified for most
   groups. Only `LIFINTERFACE_KJPAW_V2` states 300 K with an Andersen
   thermostat. Input decks must be recovered before claiming a settings-matched
   pool.
2. **No pure-Li control.** `BCC_54_*` has not been converted to extxyz.
3. **No clean interface OOD trajectory.** The available NPT interface run has a
   0.54 SCF warning rate and forces up to 7.5 eV/Å. Treat its result as a
   stress test and report it separately.
4. **The clean F-doped leg is small.** Excluding the NPT run leaves 2,788 V3
   frames; after the 70/15/15 split, training has 1,952 frames. This is enough
   for a controlled comparison but limits per-environment calibration curves.
5. **The manifest is not the extxyz split.** `frames.csv` records file and
   frame indices; a builder is still needed to materialize
   `uq_train.extxyz` / `uq_valid.extxyz` / `uq_test.extxyz`.
6. **Li–Li distances are not lattice constants.** Thermal and local distortion
   dominate the nearest-neighbour statistic.

## 8. Next actions

- Convert `BCC_54_KJPAW_FINAL` and `BCC_54_KJPAW_NPT_FINAL` to extxyz and audit
  them.
- Recover the QE input decks for temperature, thermostat, timestep, and SCF
  thresholds.
- Build the extxyz splits from `frames.csv`, then verify frame counts and label
  keys before training.
- Decide whether the excluded 7,550-frame F-doped NPT run is worth keeping as
  an auxiliary high-density OOD set.
- Update `ACTIVE_LEARNING_PROTOCOL.md`, `DIFFUSION_CELL.md`, and
  `LITERATURE_REVIEW.md` to the UQ + AL framing.
