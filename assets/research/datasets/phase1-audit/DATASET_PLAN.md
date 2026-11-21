# Dataset plan — closing the interface gap

**One page for collaborators and slides.** The detailed provenance and generation
decks live in [`audit/`](AUDIT.md) and [`generation/`](generation/README.md);
this file is the plan and the status.

> **Status (2026-11-05): plan only.** No DFT has been submitted. The two
> interface decks are ready, but the 3,000 frames do not exist. Every
> "after generation" count below is a target conditional on approval.

---

## 1. The issue

The frozen UQ + AL dataset has three environments. Bulk LiF and F-doped Li are
large enough for the final comparison; the **interface is not**.

| Environment | Clean frames now | Target | Gap |
|---|---:|---:|---:|
| Bulk LiF | 5,131 | 5,000–6,000 | none |
| F-doped Li | 4,555 | 4,000–5,000 | none |
| Interface | **1,620** | 3,000–4,000 | **−1,400 to −2,400** |
| **Total** | **11,306** | 12,000–14,000 | |

The interface is also the OOD leg: the locked test and OOD sets need
**≥500 interface frames each**, and the current clean interface pool cannot
supply both without starving the training pool.

## 2. What is already fixed

**F-doped is closed without new DFT.** Per-frame SCF curation of the excluded
NPT run recovered **2,603** post-transition dense frames (1,952 → 4,555). The
pre-transition expanded frames are excluded; the recovered frames stay labelled
`F_doped_dense`.

**Interface recovered +390** unique frames from the two NPT runs, labelled
`interface_npt` (1,230 → 1,620). They are a labelled sub-environment, not mixed
into the new NVT trajectories.

## 3. What is planned (not started)

Two independent NVT interface trajectories are planned, using the same 122-atom
cell as the existing data and starting from equilibrated frames:

| ID | Start | Production frames | Role |
|---|---|---:|---|
| `LIFINTERFACE_AL_V1` | V2 frame 1008 | 1,500 | pool |
| `LIFINTERFACE_AL_V2` | V2 frame 1500 | 1,500 | test / OOD |

Settings match `LIFINTERFACE_KJPAW_V2`: PBE + KJPAW, 40/320 Ry, 3×3×3 k-points
(14 irreducible), MP 0.02 Ry, NVT Andersen 300 K, `dt = 10 a.u.`, 1,800 steps
(300 equilibration + 1,500 production).

Decks: [`LIFINTERFACE_AL_V1.in`](generation/LIFINTERFACE_AL_V1.in),
[`LIFINTERFACE_AL_V2.in`](generation/LIFINTERFACE_AL_V2.in).

**Cost if approved:** 2–4 days per trajectory on one CPU node, about one week
total.

**If approved and completed:** interface 1,620 + 3,000 = **4,620**; total pool
**14,306**.

## 4. Post-generation QC

1. Convert QE output to extxyz, `frame_stride=1`, groups
   `LIFINTERFACE_KJPAW_AL_V1/V2`.
2. Drop the first 300 frames of each trajectory.
3. Run `curate_npt_scf.py` per frame; keep zero-warning frames only.
4. Check energy drift and volume stability over the production block.
5. Flag frames with `max|F| > 5 eV/Å` or Li–Li contacts below 2.0 Å.
6. Split into contiguous pool / calibration / test blocks; keep `AL_V2` as the
   independent test/OOD trajectory.

## 5. Split rebuild

`build_phase1_splits.py` is re-run after the new frames are curated. The
manifest must carry the sub-environment labels explicitly:

- `bulk_LiF`
- `F_doped_Li` and `F_doped_dense` (recovered NPT)
- `interface` and `interface_npt` (recovered NPT)
- `interface` from `AL_V1` for the pool, `AL_V2` for test/OOD

Rules that stay fixed:

- group-level and time-blocked splits, never random frames;
- no frame overlap between pool, calibration, test, and OOD;
- test and OOD are locked before any model selection;
- the OOD set is an independent trajectory family.

The target split sizes are pool ~12,000–14,000, test 2,000–3,000, OOD
1,500–2,000, with ≥500 interface frames in each of test and OOD.

## 6. Dependency order

```
generate AL_V1/V2  →  per-frame QC  →  extend manifest  →  rebuild + freeze splits
                                                              ↓
                                             final MHC/probe at 2,000 labels
                                                              ↓
                                                        AL loop
```

The frozen-MHC pilot on the 214-frame split is independent of this and is
run separately; it locks the recipe, not the dataset.

## 7. Fallback

If generation is not approved, or slips, the interface stays at ~1,620 clean
frames and is reported as a **reduced-size environment / case study**, not
silently reweighted. That weakens the balanced three-environment claim, so it is
the last resort.

## 8. Status

- F-doped recovery: **done** (2,603 + 390 interface frames) — the only part
  already realized.
- Interface generation: **not started; plan only**. Decks are ready, nothing
  has been submitted, and no compute has been allocated for it.
- Split rebuild: **pending**, and conditional on generation.
- The Experiment 3 pilot uses the 214-frame split and is independent of this
  plan.
