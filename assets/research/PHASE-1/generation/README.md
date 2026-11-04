# Data Generation — Interface Closure

**Date:** 2026-11-04
**Scope:** fill the interface gap for the frozen UQ + AL dataset. F-doped is
recovered from the existing NPT run; only the interface needs new DFT.

---

## 1. What was recovered before generating

Per-frame SCF curation (`src/mlpdft/utils/curate_npt_scf.py`) maps each extxyz
frame to its SCF block in the QE `.out` log (`frame i` ↔ block `i × stride`) and
keeps only frames with zero `eigenvalues not converged` warnings after the
equilibration cutoff.

| Source | Frames | SCF-clean | Post-equil. | Selected | Regime |
|---|---:|---:|---:|---:|---|
| `LIWITHF_NPT_FINAL_3_5394` | 5,394 | 4,009 | 3,372 | **2,603** | dense (t ≥ 6,180) |
| `LIFINTERFACE_KJPAW_NPT_3_431` | 431 | 344 | 297 | **233** | NPT |
| `LIFINTERFACE_KJPAW_NPT_V2_3_795` | 795 | 482 | 299 | **157** | NPT |

The two interface files have **zero frame overlap**: 390 unique interface
frames. Outputs are in `dataset/xyz_files/curated_npt/` (git-ignored).

Resulting clean pool before generation:

| Environment | Clean pool | Target | Status |
|---|---:|---:|---|
| Bulk LiF | 5,131 | 5,000–6,000 | enough |
| F-doped Li | 1,952 + 2,603 = **4,555** | 4,000–5,000 | closed |
| Interface | 1,230 + 390 = **1,620** | 3,000–4,000 | short |

No new F-doped DFT is needed.

## 2. What to generate

Two independent NVT interface trajectories, same 122-atom cell as the existing
data:

| ID | Start frame | Frames | Role |
|---|---|---:|---|
| `LIFINTERFACE_AL_V1` | `LIFINTERFACE_KJPAW_V2` frame 1008 | 1,500 production | pool |
| `LIFINTERFACE_AL_V2` | `LIFINTERFACE_KJPAW_V2` frame 1500 | 1,500 production | test/OOD |

Both start from equilibrated frames (the audit cutoff is 586) and randomise the
velocities, so they are independent trajectories of the same cell, not new
geometry.

Input decks:

- [`LIFINTERFACE_AL_V1.in`](LIFINTERFACE_AL_V1.in)
- [`LIFINTERFACE_AL_V2.in`](LIFINTERFACE_AL_V2.in)

## 3. Settings

| Parameter | Value |
|---|---|
| Functional | PBE |
| Pseudopotentials | `Li.pbe-sl-kjpaw_psl.1.0.0.UPF`, `F.pbe-n-kjpaw_psl.1.0.0.UPF` |
| `ecutwfc` / `ecutrho` | 40 / 320 Ry |
| `conv_thr` | 1e-8 Ry |
| k-points | 3×3×3 automatic (14 irreducible, matching V2) |
| Smearing | Methfessel–Paxton, 0.02 Ry |
| Ensemble | NVT (`cell_dynamics = 'none'`) |
| Thermostat | Andersen, 300 K, `nraise = 1` |
| Timestep | 10 a.u. (0.4838 fs) |
| Steps | 1,800 (300 equilibration + 1,500 production) |

The settings match `LIFINTERFACE_KJPAW_V2`; do not switch to Gamma-only or a
different cutoff, or the new frames will not be pool-compatible.

## 4. Run

```bash
mkdir -p pseudo tmp
# copy the two UPF files into pseudo/
pw.x -in LIFINTERFACE_AL_V1.in > LIFINTERFACE_AL_V1.out
pw.x -in LIFINTERFACE_AL_V2.in > LIFINTERFACE_AL_V2.out
```

Convert the QE output to extxyz with the project converter, using `frame_stride=1`
and `max_frames=None`. Name the groups `LIFINTERFACE_KJPAW_AL_V1` and
`LIFINTERFACE_KJPAW_AL_V2`.

## 5. Post-run QC

1. Discard the first 300 frames (equilibration).
2. Run `curate_npt_scf.py` on each new trajectory to mark per-frame SCF quality.
3. Check energy drift and volume stability over the production block.
4. Flag frames with `max|F| > 5 eV/Å` or Li–Li contacts below 2.0 Å.
5. Split the surviving frames into contiguous pool / calibration / test blocks;
   keep `AL_V2` as the independent test/OOD trajectory.

## 6. Cost

- 122 atoms, 14 k-points, 1,800 steps: roughly 2–4 days per trajectory on one
  node, depending on the GPU/CPU partition.
- Two trajectories: about one week.
- F-doped: zero new DFT.
