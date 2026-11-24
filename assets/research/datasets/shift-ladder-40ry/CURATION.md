---
status: active
date: 2026-11-23
phase: 2
owner: jorge
supersedes: ../final-3env/CURATION.md
superseded_by: —
---

# Shift-ladder dataset (40 Ry): build record

**Builder:** `src/mlpdft/utils/build_shift_ladder_dataset.py`, recipe in its
`SETS` table. It uses `src/mlpdft/utils/extract_qe_trajectory.py`.
**Output:** `outputs/shift_ladder_40ry/` (gitignored, about 26 MB), with
`manifest.json` holding the source run, steps, stride, cutoffs, pseudopotential
checksums and SHA-256 of every file.
**Plan and roles:** [PLAN.md](PLAN.md).

## Sets

All sets are 40/320 Ry and read the same pseudopotential files as the old runs
(QE MD5 Li `d2f1e9ef…`, F `48ea186a…`). No SCF step appears in two sets.

| Set | Role | Source run | SCF steps (stride) | Frames | Atoms | Force rms (eV/Å) |
|---|---|---|---|---:|---:|---:|
| `train_pool` | training | `LIF64_KJPAW_FINAL` (NVT 300 K) | 500–15,850 (10) | 1,536 | 64 | 0.403 |
| `monitor` | MACE's checkpoint validation and the interim LLPR calibration; never reported (ADR-0004) | same | 505–15,805 (150), offset from the pool | 103 | 64 | 0.403 |
| `test_timeblock` | time-block holdout | same | 15,980–18,970 (10), after a 120-step guard | 300 | 64 | 0.409 |
| `shift_bulk_npt` | state shift | `LIF64_KJPAW_NPT_FINAL` (variable cell, no thermostat) | 500–9,975 (25) | 380 | 64 | 0.445 |
| `shift_fdoped` | chemistry shift | `LIWITHF_KJPAW_FINAL` | 500–8,700 (20) | 411 | 54 | 0.231 |
| `shift_li_metal` | chemistry shift | `BCC_54_KJPAW_FINAL` | 500–9,975 (25) | 380 | 54 | 0.214 |
| `shift_interface` | structure shift | `LIFINTERFACE_KJPAW_V2` | 586–2,341 (5) | 352 | 122 | 0.392 |
| `far_libf4` | far shift (appendix) | `LIBF4_FINAL` | 500–19,950 (50) | 390 | 18 | 0.751 |
| `far_bli3` | far shift (appendix) | `BLI3_KJPAW_FINAL` (log `BLi3.out`) | 500–9,975 (25) | 380 | 32 | 0.555 |

Calibration and in-distribution test sets will come from the bulk replicas of
the 2026-11 QE campaign. They are not built yet.

## Checks

- **Pairing of positions with labels.** SCF block k takes the positions
  printed in MD iteration k, the same rule as ASE's `read_espresso_out`.
  - Against ASE on a fixed-cell log (`LIF64_ISOLATED`, 283 frames) and a
    variable-cell log (`LIFINTERFACE_KJPAW_NPT`, 1,292 frames), positions and
    cells are identical, energies agree within 9 × 10⁻⁶ eV and forces within
    2 × 10⁻⁸ eV/Å.
  - Shifting the pairing by one block changes the forces by 0.05 and 4.5
    eV/Å, so an off-by-one would be caught.
- **Units:** QE's own CODATA 2006 constants, as ASE uses. A CODATA 2018
  conversion would differ by a factor of 8.7 × 10⁻⁸ (−2.8 meV on a −32,500 eV
  total).
- **The variable-cell set is not simply colder.** Its kinetic temperature is
  about 160 K (QE's printed T agrees) and it is 2 % denser, but its forces are
  larger than in the 300 K run (rms 0.445 vs 0.403 eV/Å). It is labelled a
  state shift, not a temperature shift.
- **`far_bli3` is not stationary:** its energy per atom varies by 125 meV
  across the run. It is an appendix set only.

## Reproduce

```bash
PYTHONPATH=src python3 src/mlpdft/utils/build_shift_ladder_dataset.py \
  --dataset-root /home/jorge/project/mlpdft/dataset --output-dir outputs/shift_ladder_40ry
```
