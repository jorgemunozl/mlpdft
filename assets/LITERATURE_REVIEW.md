# Literature Review — Reference-energy (E₀) mismatch in foundation-model MLIPs for SEI

**Date:** 2026-09-13
**Scope:** foundation-model interatomic potentials, zero-shot vs fine-tuned transfer, the per-element
reference-energy (E₀) problem, and MLIPs for the solid-electrolyte interphase (SEI) / Li-ion systems.
**Purpose:** map the territory around the thesis and locate the exact gap.

---

## 1. Foundation-model MLIPs — the landscape

The field has moved from per-system "train-from-scratch" potentials to **pretrained, general-purpose
foundation models** that can be applied out-of-the-box and then fine-tuned on a handful of data points.

- **MACE-MP-0** ([Batatia et al., *A foundation model for atomistic materials chemistry*, arXiv:2401.00096](https://arxiv.org/abs/2401.00096), 2024) is the dominant model here. Built on the MACE equivariant-message-passing architecture ([Batatia et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html)), trained on the ~1.6M-configuration MPtrj dataset, 89 elements, PBE/PBE+U reference.
- Contemporaries: **CHGNet**, **M3GNet**, **SevenNet**, **Orb**, **MatterSim**, **DPA/OpenLAM** — same paradigm, different architectures/training sets.
- The **MACE-MPA-0** variant is the one used in the SEI/LiF fine-tuning literature (see §4).

## 2. Zero-shot vs fine-tuned transfer (energy & force)

The general picture is consistent across systems:

- Zero-shot foundation models give **reasonable but imperfect** predictions that degrade out-of-domain.
  On gold surfaces, MACE zero-shot gave 14.8 meV/atom energy / 74.7 meV/Å force; fine-tuning on **40**
  configurations cut this to 4.3 meV/atom / 37.4 meV/Å ([*Fine-tuning bulk-oriented universal interatomic
  potentials for surfaces*, arXiv:2509.25807](https://arxiv.org/abs/2509.25807)).
- Fine-tuning beats zero-shot **and** from-scratch by roughly 7–10× (energy) and 10–20× (forces)
  (Mo-based dilute alloys; *Materials* 2026).
- Cross-geometry transfer is fragile: geometry-specific fine-tuning improves in-domain accuracy but can
  **negatively transfer** to other structural classes ([*Cross-Geometry Transferability Assessment of
  Universal MLIPs*, arXiv:2608.06662](https://arxiv.org/abs/2608.06662)).
- Foundation potentials systematically **soften** high-energy regions and **underestimate** surface,
  defect, and formation energies (PES-softening benchmark; MACE surface-energy MAE ≈ 0.032 eV/Å²,
  ion-migration barriers underpredicted).

## 3. The reference-energy (E₀) problem — the crux

This is the heart of the thesis, and it is **well-known but often treated as a preprocessing detail
rather than a phenomenon**. The convention: a model's energy is defined as

```
E_model = E_total − Σ_i n_i · E₀^i
```

where `E₀^i` is a per-element reference energy (usually an isolated-atom energy) for element `i`.
Forces — gradients — are **unaffected** by any constant per-species shift, but absolute energies are not.

### 3.1 It is a real, general, and quantified issue

The closest and most important prior work is:

- **Tompa, Varga-Umbrich, Batatia, Elena, Bernstein, Csányi, *Fine-tuning MLIP foundation models:
  strategies for accuracy and transferability*, arXiv:2606.12704 (2026).** This is the state of the art
  on E₀ handling. It contrasts three E₀ conventions and finds **E₀ choice outweighs the fine-tuning
  method**:
  - *Averaged* (least-squares per-element fit) E₀s gave force RMSEs **2–3× worse** than *re-estimated* E₀s.
  - *Model-aware E₀ re-estimation* (solve for per-element corrections aligning the pretrained energy zero
    with the labels) cut mean absolute E₀ error from **3.13 eV → 0.33 eV** on SPICE.
  - Fine-tuned ice models were **MD-unstable within 50 ps** with averaged E₀s, but stable over 250 ps and
    a 50→800 K ramp with re-estimated E₀s.
  - Advice: supply true isolated-atom training points; **never** use averaged E₀s.

- **[*Cross-functional transferability in foundation machine learning interatomic potentials*, npj
  Computational Materials, 10.1038/s41524-025-01796-y (2025).](https://www.nature.com/articles/s41524-025-01796-y)** Its "Method 4"
  — **shift the reference energy first, then train the GNN** — was the best strategy, reducing
  decomposition-energy MAE to **23.66 meV/atom** (vs 37–41 meV/atom for the alternatives) and formation-energy
  MAE to 29.38 meV/atom. A **trainable** reference-energy term (their Method 2) actively *hurt*.

### 3.2 Practitioner-level confirmation

- MACE discussion [#1119 "Reference energies (E0s) and MACE fine-tuning"](https://github.com/ACEsuit/mace/discussions/1119):
  E₀s must match the fine-tuning DFT setup exactly (functional, smearing, spin, free-energy vs energy).
- DeepMD-kit discussion [#3311 "Finetune with large energy shift"](https://github.com/deepmodeling/deepmd-kit/discussions/3311):
  the dataset's elemental-energy bias may not equal the isolated-atom energy (the NN contributes its own bias).
- *Cross Learning between Electronic Structure Theories*, arXiv:2510.25380 — the explicit
  `E_atm = E_tot − Σ_i E₀^i` convention across levels of theory.

### 3.3 Where the gap still is

The prior work treats E₀ as a **fine-tuning prerequisite** — "get it right so your fine-tuned model
works." What is *not* explicitly framed is the **zero-shot consequence**:

> Zero-shot foundation-model transfer is **property-dependent**: force-derived quantities (diffusion,
> activation energy, vibrational/thermal properties) transfer cleanly, while energy-derived quantities
> (interface adhesion, defect/doping formation, mixing, phase stability) **silently fail** — not because
> of any representational limit, but purely because of the E₀ mismatch.

That is the specific, citable gap.

## 4. MLIPs for SEI / Li-ion — the application

- **[*Comparing fine-tuning strategies for machine learning force fields in lithium-ion diffusion*,
  Alghamdi, De Angelis, Asinari, Chiavazzo, J. Phys. Energy, 10.1088/2515-7655/ae6c5a, arXiv:2510.05020 (2025).](https://beta.iopscience.iop.org/article/10.1088/2515-7655/ae6c5a/meta)**
  The direct neighbor. MACE-MPA-0 **zero-shot** predicts LiF interstitial-diffusion Ea = **0.22 eV** and
  fine-tuned (300 points) **0.20 eV**, both ≈ the DeePMD reference (0.24 eV) trained on 40,000
  active-learned points. **Crucially, its headline quantities — diffusivity and Ea — are force-derived
  and thus immune to the E₀ mismatch.** It never tests an energy-derived property.
- **[*Exploring lithium diffusion in LiF with machine learning potentials: from point defects to
  collective ring diffusion*, npj Comput. Mater., 10.1038/s41524-026-02132-8 (2026).](https://www.nature.com/articles/s41524-026-02132-8)**
  LiF point defects (vacancy vs interstitial formation enthalpies ≈ 0.76 vs 1.73 eV), plus a collective
  ring-diffusion mechanism. Relevant because the project dataset contains F-doped Li (Li53F1) and
  interfaces — but this paper is LiF-only, crystalline.
- Thermal conductivity of LiF/Li₂O via MLIP (Comput. Mater. Sci. 2026); amorphous fluorophosphate fast-ion
  conductors ([arXiv:2510.22912](https://arxiv.org/abs/2510.22912)); SEI structure via ML (Aalto thesis) —
  context, not competitors.

## 5. Ensemble UQ — tangential (the abandoned thread)

- **[*Comparative study of ensemble-based uncertainty quantification methods for neural network
  interatomic potentials*, Kurniawan, Wen, Tadmor, Transtrum, arXiv:2508.06456 (2025).](https://arxiv.org/abs/2508.06456)**
  Benchmarks bootstrap / dropout / random-init / **snapshot** ensembles, on carbon allotropes; finds UQ
  can be **counterintuitive in OOD** (uncertainty plateaus/decreases as error grows). This closes the
  "snapshot committee as novelty" route.
- Wimer et al., *Benchmarking UQ methods for NNPs*, J. Cheminformatics (2026) — single-shot aleatoric
  methods competitive with ensembles in data-rich regions.

---

## 6. Synthesis — the gap and our position

| Thread | Status |
|---|---|
| Foundation models transfer zero-shot | Established, but degrades OOD (§2) |
| E₀ mismatch is real and fixable | Established as a *fine-tuning prerequisite* (§3) |
| Zero-shot transfer is property-dependent (force vs energy) | **Under-framed — this is our gap** |
| LiF SEI is a "zero-shot works" success story | Established, but force-derived only (§4) |
| Multi-phase SEI (anode/LiF/interfaces/doping) + energy-derived properties | **Open** |

**Our claim, restated against the literature:** zero-shot MACE-MPA-0 transfers for *force-derived* SEI
properties (diffusion, Ea — confirming Alghamdi et al. 2025) but *silently fails* for *energy-derived*
properties (Li/LiF and BLi/LiF interface adhesion; F-, B-, and F+B-doping formation energies; LiBF₄/BLi
formation) because of the per-element E₀ mismatch — and a minimal isolated-atom reference correction
(or light fine-tuning, per Tompa et al. 2026 and npj 2025) recovers these at near-zero data cost.

**Honest risk to keep in view:** Tompa et al. (2026) already covers E₀ re-estimation comprehensively.
Our novelty is *not* "E₀ mismatch exists," but the **property-dependent framing of zero-shot transfer**
and the **demonstration on a realistic multi-phase SEI system with energy-derived quantities that the
"zero-shot is good enough" narrative never tested.** Position against §3.1/§4 explicitly, or a reviewer
will do it for us.

---

## 7. Candidate citation list (primary)

1. Batatia et al., *A foundation model for atomistic materials chemistry*, arXiv:2401.00096 (2024).
2. Batatia et al., *MACE: higher order equivariant message passing…*, NeurIPS (2022).
3. Alghamdi, De Angelis, Asinari, Chiavazzo, *Comparing fine-tuning strategies for MLFFs in lithium-ion
   diffusion*, J. Phys. Energy (2025), arXiv:2510.05020.
4. Tompa, Varga-Umbrich, Batatia, Elena, Bernstein, Csányi, *Fine-tuning MLIP foundation models:
   strategies for accuracy and transferability*, arXiv:2606.12704 (2026).
5. *Cross-functional transferability in foundation machine learning interatomic potentials*, npj Comput.
   Mater. (2025), 10.1038/s41524-025-01796-y.
6. *Exploring lithium diffusion in LiF with MLPs: from point defects to collective ring diffusion*, npj
   Comput. Mater. (2026), 10.1038/s41524-026-02132-8.
7. Kurniawan, Wen, Tadmor, Transtrum, *Comparative study of ensemble-based UQ methods for NNP*,
   arXiv:2508.06456 (2025).
8. *Cross-Geometry Transferability Assessment of Universal MLIPs*, arXiv:2608.06662 (2026).
9. *Fine-tuning bulk-oriented universal interatomic potentials for surfaces*, arXiv:2509.25807 (2025).
10. *Machine-Learning-Guided Insights into SEI Conductivity*, arXiv:2510.22912 / ACS Energy Lett. (2025).
