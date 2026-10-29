# Dataset Families for MACE UQ

## Purpose

This dataset was originally assembled for training a FitSNAP potential and is
being reused to study uncertainty quantification (UQ) in a fine-tuned MACE
foundation model. The paper compares independent committees, warm-restart
snapshots, and multi-head fine-tuning, asking whether ensemble disagreement
predicts held-out DFT error, stays calibrated across environments, and improves
active-learning label efficiency over random sampling. The Li-F groups are a
testbed with three chemically distinct environments; transport and SEI formation
are out of scope. Boron-containing families are optional extensions for testing
transfer to B-modified chemistry. These component systems do not by themselves
simulate electrochemical SEI formation.

## Core Li/F Environments

| Family | Why it belongs in the UQ study |
|---|---|
| **LIWITHF** | 53 Li + 1 F: F-doped Li metal, a compositionally shifted anode-side environment distinct from bulk LiF. |
| **LIFINTERFACE** | 86 Li + 36 F: a thin two-layer LiF film on Li metal, used as the OOD interface environment. The cell is a thin-film model, not a buried interface. |
| **LIF64** | Stoichiometric bulk LiF is the ionic-crystal control for Li coordination and the bulk limit. Isolated LiF is a reference environment, not condensed-phase data. |

## Optional Li-Metal Control

| Family | Why it may be useful |
|---|---|
| **BCC54** | Pure body-centered-cubic Li, not a boron-containing cell. It provides a pure-metal reference environment for UQ transfer, but is optional. |

## Optional Boron-Containing Extensions

| Family | Why it may be useful |
|---|---|
| **LIWITHBF** | Mixed Li-B-F environments test transfer into B/F-modified Li-rich chemistry, distinct from stoichiometric LiBF4 salt. |
| **LIBF4** | LiBF4 provides B-F salt chemistry and a covalent bonding regime alongside ionic Li interactions; include it for salt/additive questions. |
| **BLI3** | B-Li bulk environments provide an alloy-side reference for BLI3 interfaces. Confirm exact stoichiometry from structures rather than inferring it from the group name. |
| **BLI3_INTERFACE** | B-Li-containing interfaces test UQ transfer at an alloy/interphase boundary; identify adjoining phases and interface orientation from the structure. |
| **BLI** | B-Li bulk environments provide metallic coordination references for BLI interfaces. |
| **BLI_INTERFACE** | Li-B contact interfaces test committee UQ at a metal/compound contact in boron-containing chemistry. |

## MD Ensembles

The tables list groups by QE ensemble only; they do not imply identical DFT
settings. Isolated reference groups are omitted because they are not
condensed-phase trajectories.

### NVT (Fixed Cell)

| Groups | `ecutwfc` / `ecutrho` (Ry) | `conv_thr` (Ry) | Requested `nstep` |
|---|---:|---:|---:|
| Core: `LIWITHF_KJPAW_FINAL`, `LIFINTERFACE_KJPAW_FINAL`, `LIF64_KJPAW_FINAL` | 40 / 320 | `1e-8` | 10,000 |
| Optional B: `LIWITHBF_FINAL`, `BLI3_KJPAW_FINAL`, `BLI3_INTERFACE_FINAL`, `BLI_INTERFACE_FINAL` | 40 / 320 | `1e-8` | 10,000 |
| Optional Li-metal control: `BCC_54_KJPAW_FINAL` | 40 / 320 | `1e-8` | 10,000 |
| Optional B/salt: `LIBF4_FINAL` | 40 / 320 | `1e-7` | 20,000 |

### NPT (Variable Cell)

| Groups | `ecutwfc` / `ecutrho` (Ry) | `conv_thr` (Ry) | Requested `nstep` |
|---|---:|---:|---:|
| Core: `LIWITHF_NPT_FINAL`, `LIF64_KJPAW_NPT_FINAL`, `LIFINTERFACE_KJPAW_NPT_FINAL` | 40 / 320 | `1e-8` | 10,000 |
| Optional B: `LIWITHBF_NPT_FINAL`, `BLI_NPT`, `BLI3_NPT_FINAL`, `BLI_INTERFACE_NPT_FINAL`, `BLI3_INTERFACE_NPT_FINAL` | 40 / 320 | `1e-8` | 10,000 |
| Optional Li-metal control: `BCC_54_KJPAW_NPT_FINAL` | 40 / 320 | `1e-8` | 10,000 |
| Optional B/salt: `LIBF4_NPT_FINAL` | 40 / 320 | `1e-8` | 20,000 |

Most listed groups therefore share one settings profile per ensemble, but the
exceptions above matter: cutoff and convergence settings are not uniform. Some
groups have output logs but not their original input decks, so these values are
the recorded settings available locally and should be checked against the
original input before pooling. Local examples report `dt=10` a.u. (0.4838 fs)
and a 300 K Andersen thermostat, but these values are not verified for every
group. Some logs contain multiple MD sections; check restarts and timestamps
rather than inferring duration from `nstep` or SCF-marker counts. Preserve
ensemble metadata; NPT cell fluctuations require cell-aware displacement
analysis.

## Sampling and UQ Splits

- Determine thermalization cutoffs per trajectory from temperature, energy,
  volume/cell, and structural observables; a fixed number of initial frames is
  not a universal equilibration time.
- Frames from an AIMD trajectory are temporally correlated. Sample across the
  equilibrated trajectory and split validation/test by contiguous time blocks
  or by independent runs, not by a random frame-level split.
- Check whether versioned/final groups reuse or resample the same trajectory
  before treating one as an independent held-out test.
- Report parsed frame counts as configurations, not as independent statistical
  samples. The effective sample size is set by the number of independent
  trajectories and contiguous blocks, not by the number of frames.
- Use measured compositions, not group names, to define environments:
  `LIWITHF` is 53 Li + 1 F and `LIFINTERFACE` is 86 Li + 36 F.

## Missing Inputs to Resolve

The ten numbered families are now represented by local directories. The EMIM
systems and `LIBF4_RELAX` remain absent. Several newly added directories contain output logs but no original
input deck; recover those inputs before treating the final groups as fully
settings-matched. Confirm composition, cell geometry, interface normal,
pseudopotentials, k-point sampling, smearing, cutoffs, convergence thresholds,
and overlap between versioned/final trajectories before constructing the final
training manifest. `scripts/count.txt` predates these additions and does not
provide counts for them.