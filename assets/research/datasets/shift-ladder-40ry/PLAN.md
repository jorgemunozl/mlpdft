---
status: active
date: 2026-11-21
phase: 2
owner: jorge
supersedes: —
superseded_by: —
---

# Shift-ladder dataset at 40 Ry: DFT campaign and build plan

The paper trains on bulk LiF, calibrates and tests in distribution on
independent bulk replicas, and evaluates the arms along a graded distribution
shift ([ADR-0007](../../decisions/ADR-0007-bulk-trained-shift-ladder.md)). All
labels use the 40/320 Ry settings
([ADR-0008](../../decisions/ADR-0008-40ry-labels-and-md-protocol.md)).

Job folders: [`assets/quantum_expresso/campaign-2026-11/`](../../../quantum_expresso/campaign-2026-11/README.md).

## 1. Role of each data source

| Role | Source (all 40/320 Ry) | Exists? |
|---|---|---|
| Training pool | `LIF64_KJPAW_FINAL`: bulk LiF, NVT 300 K, 18,980 steps | yes; re-extract from the log |
| Calibration, in distribution | 3 new bulk LiF replicas (300 K) | stages 1 and 3 |
| Test, in distribution | 3 more new bulk LiF replicas (300 K) | stages 1 and 3 |
| Shift 1, temperature and density | `LIF64_KJPAW_NPT_FINAL`: bulk LiF, no thermostat, ≈ 162 K, V ≈ 531 Å³ | yes |
| Shift 2, chemistry | `LIWITHF_KJPAW_FINAL` (F-doped Li), `BCC_54_KJPAW_FINAL` (Li metal); optional 2 F-doped replicas | yes / stage 4 |
| Shift 3, structure | `LIFINTERFACE_KJPAW_V2` after step 586 (LiF/Li film; drifting stress test) | yes |
| Shift 4, appendix only | `LIBF4_FINAL`, `BLI3_*`, `LIWITHBF_*` (boron) | yes |

Excluded:
- every 30 Ry and 20 Ry run, including `LIF64_KJPAW_NPT_V3` and `LIWITHF_V3` from
  the provisional build;
- `LIFINTERFACE_KJPAW_FINAL`, a copy of steps 908–2,343 of V2.

The training run's frames are correlated (force τ ≈ 720 steps), and that is
acceptable for training. Calibration and test come from replicas that share no
history with it.

## 2. DFT campaign

| Stage | Jobs | Node-hours (est.) | Wall time, run in parallel | Needs |
|---|---|---:|---:|---|
| 0 Preflight | 1 bulk single point that must reproduce `LIF64_KJPAW_FINAL` at SCF step 501 | < 0.1 | ~5 min | — |
| 1 Thermostat pilot | 2 bulk MD runs × 3,000 steps: SVR and Andersen, `nraise = 100` | ~37 | ~18 h | stage 0 passes |
| 2 Cutoff and k-point check | 4 jobs, 77 single points on existing frames | ~7 | ≤ 9 h | — |
| 3 Bulk replicas | 5 bulk MD runs × 3,000 steps with the winning thermostat | ~92 | ~18 h | stage 1 decision |
| 4 F-doped replicas (optional) | 2 MD runs × 3,000 steps | ~110 | ~55 h | stage 1 decision |

Stages 0–2 are submitted together with `./submit.sh all`; the pilots wait for
the preflight. The total is about 5.6 node-days without
stage 4 and 10 with it. Costs come from the measured 22 s (LiF), 66 s
(F-doped), 57 s (Li metal) and 246 s (interface) per MD step on one 48-core
node.

### Settings (identical to the existing 40 Ry runs)

- PBE; PSLibrary 1.0.0 PAW `Li.pbe-sl-kjpaw`, `F.pbe-n-kjpaw`, read from the
  repository (`assets/quantum_expresso/pseudo`) and pinned by checksum.
  - They are differently formatted copies of the files the old runs read.
  - So a preflight single point must first reproduce the old 40 Ry bulk data:
    force RMSE ≤ 1 meV/Å, largest deviation ≤ 3 meV/Å, |ΔE| ≤ 1e-4 Ry.
  - The MD pilots start only if it passes.
- `ecutwfc = 40`, `ecutrho = 320`, 3×3×3 Monkhorst–Pack grid, `nosym`.
- `conv_thr = 1e-8`. `mixing_beta` is 0.7 (LiF), 0.6 (F-doped, interface) or
  0.4 (Li metal).
- Methfessel–Paxton smearing, 0.02 Ry, for the metallic cells.
- `dt = 10` a.u. (0.4838 fs), second-order extrapolation, `tstress`.
- QE 7.4, 48 MPI ranks, `-nk 2`.

### What changes

- **Thermostat.** The old inputs set `ion_temperature = 'andersen'` without
  `nraise`. QE's default `nraise = 1` re-thermalises every atom at every step.
  The new runs use `nraise = 100`: for Andersen that is a collision per atom
  every ~48 fs; for SVR it is a 48 fs coupling time.
- **Start.** The ideal lattice plus independent Gaussian displacements
  (σ = 0.06 Å, one seed per run), so no replica shares history with the
  training run.

### Decision after stage 1

Analyse both pilots after a 1,000-step burn-in:

```bash
PYTHONPATH=src python3 src/mlpdft/utils/estimate_qe_correlation.py --out <pw.out> --burn-in 1000 --forces --json
PYTHONPATH=src python3 src/mlpdft/utils/diagnose_qe_drift.py --out <pw.out> --burn-in 1000
```

A thermostat is acceptable if:
- the mean temperature is within 300 ± 10 K;
- there is no drift warning;
- the short-lag MSD exponent is ≥ 1.5, i.e. the motion is inertial.

Among acceptable thermostats, choose the one whose force |ρ| falls below 0.1
sooner, using smaller τ_F as the tie-break. Its pilot becomes replica 1. The
other pilot is kept as a spare in-distribution set. If neither passes, rerun
the pilot with Andersen `nraise = 50` before spending stage 3.

The production burn-in is the longer of 1,000 steps and the point where the
pilot energy reaches its plateau.

### Acceptance of stage 2

```bash
python3 src/mlpdft/utils/analyze_cutoff_check.py --stage-dir <stage2_cutoff> \
  --dataset-root /home/jorge/project/mlpdft/dataset --json cutoff.json
```

- **Settings check first:** each frame is a configuration from an existing
  40 Ry run, labelled with its SCF step. The new 40 Ry 3×3×3 single points must
  reproduce that run's forces at the same step within 1 meV/Å. If they don't,
  the new jobs differ from the old data in some setting and stage 3 must wait.
- This check matters: both pseudopotential files suggest more than 40 Ry
  (Li 50 Ry, F 49 Ry).
- 40 Ry is accepted if the force RMSE against 80 Ry is ≤ 10 meV/Å in every
  environment. The models' force error is about 30 meV/Å.
- The same tolerance applies to the 4×4×4 k-point check.
- If an environment fails, report the difference as a label-noise floor and
  discuss it. Do not recompute the data.
- The 30 Ry rows document why the 30 Ry runs were dropped.

## 3. Dataset build (no DFT; can start now)

1. **Re-extract full trajectories** from the 40 Ry logs: positions, cell, `!`
   energy, forces, stress, run and step. The existing 40 Ry extxyz files hold
   only 200–666 frames each, out of time order.
2. **Training pool:** `LIF64_KJPAW_FINAL` after a 500-step burn-in, at a fixed
   stride of 10 steps (~1,850 frames). Its τ and ESS are reported, not used to
   choose the stride ([ADR-0003](../../decisions/ADR-0003-qe-decorrelation-stride.md)).
3. **Calibration and test:** each replica after burn-in at stride 10 (200
   frames per replica). Split by replica, 3 + 3.
4. **Shift sets:** each run after burn-in at a fixed stride giving 200–500
   frames, with ESS and failed checks reported.
5. **Manifest:** source run, step, cutoff and pseudopotential checksum for
   every frame. No duplicate geometries. Every frame at 40 Ry.

## 4. Models and evaluation (GPU, after the dataset)

- **Arms:**
  - independent fine-tuning committee (4 members);
  - snapshot ensemble;
  - frozen-body multi-head committee (4 heads);
  - frozen probe (one head).
- **Baselines reviewers will expect:**
  - LLPR (last-layer prediction rigidity);
  - a shallow ensemble trained with a Gaussian NLL loss, DPOSE-style.
- **Training:** bulk LiF only, at label rungs 250 / 500 / 1,000 / ~1,850. At
  least 3 seeds per arm.
- **Metrics,** on the in-distribution test set and on each shift set:
  - ranking of force error by uncertainty (Spearman, per atom and per frame);
  - calibration: miscalibration area, NLL, and coverage after conformal
    recalibration on the calibration replicas;
  - shift detection: AUROC of the in-distribution test set against each
    shift set;
  - risk–coverage;
  - GPU-hours.
- **Statistics:**
  - in distribution: bootstrap over replicas;
  - single-trajectory shift sets: time-block bootstrap with blocks at least as
    long as the force envelope lag;
  - arms compared pairwise on the same frames.
- **Active learning:** retrospective. Start from a small bulk seed set, draw
  from a pool of bulk training frames plus held-out frames of the shift sets,
  and compare uncertainty-driven against random acquisition at equal label
  budget, on the in-distribution test set and the shift sets.

## 5. Risks

- **Random seeds.** pw.x does not expose a seed for initial velocities or
  thermostat noise. Replicas differ through their starting displacements, and
  chaotic divergence separates them within a few hundred steps. Check this by
  correlating forces across replicas.
- **Wrong pseudopotentials.**
  - Other files are caught by the checksum guard within ~10 minutes of a job's
    start.
  - Numerically different copies are caught by the preflight before any MD
    runs. If it fails, use the cluster's original files (QE MD5 Li
    `d2f1e9ef…`, F `48ea186a…`) and regenerate the jobs with those checksums.
- **Time limits.** `max_seconds` stops pw.x cleanly one hour before the SLURM
  limit, and the run can be restarted.
- **A coherent oscillation survives in the bulk crystal under both
  thermostats.** The Andersen `nraise = 50` fallback above covers it. The
  envelope criterion in `estimate_qe_correlation.py` detects it.
