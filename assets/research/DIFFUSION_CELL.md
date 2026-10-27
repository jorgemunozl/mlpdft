# Diffusion-Validation Cell Decision — Interstitial Li in Bulk LiF

**Date:** 2026-10-27
**Question:** which DFT cell is the right one to validate Li⁺ diffusivity and
activation energy for the active-learning claim, and is the existing Li|LiF
interface dataset the best basis?

**Decision:** the canonical Li⁺-diffusion cell is an **interstitial Li in a bulk
LiF supercell**, *not* the Li|LiF interface. The existing dataset does **not**
contain this configuration, so a small interstitial-Li DFT dataset should be
generated with the project's own settings before the AL leg.

---

## 1. What the literature says about the diffusion mechanism

- The dominant low-temperature Li⁺ conduction mechanism in LiF is an
  **interstitial knock-off** (an interstitial Li displaces a lattice Li), not a
  vacancy hop. This is the mechanism used in the direct neighbor:
  Alghamdi, De Angelis, Asinari, Chiavazzo, *Comparing fine-tuning strategies
  for machine learning force fields in lithium-ion diffusion*, J. Phys. Energy,
  arXiv:2510.05020 (2025) / 10.1088/2515-7655/ae6c5a.
- Reference activation energies in that paper: MACE-MPA-0 **zero-shot Ea =
  0.22 eV**, fine-tuned (300 points) **0.20 eV**, against a DeePMD reference of
  **0.24 eV** (trained on 40,000 active-learned points).
- LiF point-defect energetics (vacancy vs. interstitial formation enthalpies
  ≈ 0.76 vs. 1.73 eV) and a collective ring-diffusion channel are quantified in
  *Exploring lithium diffusion in LiF with machine learning potentials*, npj
  Comput. Mater. (2026), 10.1038/s41524-026-02132-8.

So the physical quantity we intend to validate (Ea ≈ 0.20–0.27 eV) is tied to
the **interstitial** configuration. The interface is a different, higher-energy
region and does not isolate the bulk Li⁺ hop we want to benchmark against the
literature.

## 2. Why the existing dataset is insufficient for this validation

The project dataset ([`DATASET.md`](DATASET.md)) contains bulk LiF (`LIF64_*`),
Li-rich Li–F (`LIWITHF_*`), and Li|LiF interface (`LIFINTERFACE_*`) groups, plus
B-/salt groups. It **lacks a bulk LiF supercell with an interstitial Li** — the
one configuration the diffusion mechanism actually probes. Relying on the
interface for the activation-energy claim would (a) benchmark a different
mechanism and (b) give the active learner no interstitial frames to recognize as
high-value, which is exactly the OOD signal the AL leg is supposed to reward.

## 3. Kinetic-energy-cutoff comparison (our DFT vs. Alghamdi's)

Both datasets are PBE + PAW, but at different cutoffs:

| | this project | Alghamdi et al. |
|---|---|---|
| pseudopotential | KJPAW (PAW) | PAW |
| ecutwfc / ecutrho | **40 / 320 Ry** | **110 / 440 Ry** |
| conv_thr | 1e-8 | — |
| MD dt | 10 a.u. (0.4838 fs) | — |
| thermostat | Andersen, 300 K | — |
| license / availability | private | **public, CC-BY 4.0** (github.com/AlghamdiNada/Li_diffusion_with_MACE) |

Consequence: Alghamdi's public data **does** include the interstitial-Li
configurations (the DeePMD/MACE fine-tuning splits are 216-atom cells, and the
MACE-MPA-0 active-learning steps `a_step_1/2`, `b_step_1` exist), but at a
cutoff (110/440 Ry) that is **not** consistent with this project's 40/320 Ry
KJPAW convention. Mixing the two would introduce a DFT-setting mismatch that
[`DATASET.md`](DATASET.md) already flags as a hazard ("cutoff and convergence
settings are not uniform").

## 4. Recommendation

Generate the interstitial-Li dataset with the project's **own** DFT settings,
not Alghamdi's, so it is self-consistent with the rest of the pool:

- **Cell:** 3×3×3 bulk LiF supercell (~216 atoms, cubic).
- **Defect:** one Li interstitial (knock-off geometry; relax then run MD).
- **Settings:** PBE + KJPAW, ecutwfc/ecutrho = 40/320 Ry, conv_thr = 1e-8,
  dt = 10 a.u., Andersen thermostat.
- **Temperature range:** 300–600 K (at least two temperatures) so an Arrhenius
  slope is possible; ~60–100 frames each after equilibration.
- **Output:** extended XYZ with `REF_energy` / `REF_forces`, added to the pool
  and (sparingly) to the seed so the active learner has interstitial frames to
  acquire.

This gives a self-consistent, citable activation-energy target and closes the
gap identified in [`DATASET.md`](DATASET.md), while keeping the interface data
for the separate OOD/transfer leg.
