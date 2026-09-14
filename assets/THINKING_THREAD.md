# Thinking Thread — from "train an MLIP on Li-F-B" to a precise paper

**Date:** 2026-09-13
**Status:** converged on a precise, defensible thesis.

## 1. Where this started

I began with a vague goal: train an MLIP on a Li-F-B dataset to get better energy/force
RMSE than the professor's existing FitSnap model. The professor built the DFT dataset,
trained FitSnap, and wrote a paper (currently **in revision**). The original framing was
"expand his work with his data."

## 2. The winding path (every idea I tried, and why it changed)

1. **FitSnap** (descriptor-based MLP). Goal: better RMSE on Li-F-B.
   → Rejected: FitSnap didn't feel like a good architecture.
2. **MACE** (equivariant MPNN, foundation model), post-trained.
   → Rejected as a standalone paper: "just post-training" isn't enough.
3. **+ active learning.**
   → Hit a wall: active learning needs a committee.
4. **Cheap committee via warm restarts / cosine annealing (SGDR snapshots).**
   Get N checkpoints from one run, hoping they land in different minima.
   → Ran it: compared snapshot committee (1 run) vs full committee (3 runs).
5. **Novelty check.** Realized snapshot ensembling is not new
   (Huang et al. 2017; Loshchilov & Hutter 2017).
   → Needed to understand the domain better to publish.
6. **Learned the physics.** SEI, diffusion coefficient, Arrhenius activation energy.
7. **"Final" idea (doubted): FitSnap vs MACE head-to-head.**
   FitSnap trained normally + committee + active learning; MACE with the snapshot committee.
   → Doubted: are they even comparable? FitSnap takes far more time/RAM.

## 3. The biases I caught

- **Kitchen-sink / sunk cost.** Each "final" idea tried to keep every earlier idea alive
  (FitSnap *and* MACE *and* active learning *and* snapshots *and* SEI physics). The tell
  between a genuine synthesis and a bias-driven Frankenstein is the **direction of reasoning**:
  - Genuine: "my question is X; ideas 1 and 2 are necessary to answer X."
  - Biased: "I have ideas 1 and 2; let me find a question that uses them."

## 4. Facts that set the constraints

- Professor's FitSnap paper is **in revision** (not published yet).
- I **have** the trained FitSnap model + its reported numbers (a baseline, no re-training needed).
- I **can run new DFT** (real active learning is possible).
- I have **free will** — any topic.

## 5. Literature reality check

- **"Comparative study of ensemble-based uncertainty quantification methods for neural
  network interatomic potentials."** Covers snapshot ensembles among UQ methods, with a
  negative OOD result on carbon allotropes.
  → Confirms the method-novelty route (snapshot committee) is crowded. Killed the method paper.
- **"Comparing fine-tuning strategies for machine learning force fields in lithium-ion
  diffusion."** Fine-tunes MACE on **LiF**, computes interstitial Li diffusivity + Ea, finds
  zero-shot / fine-tuned MACE ≈ DeePMD (40k points) with only ~300 points.
  → Covers the naive "MACE beats a descriptor baseline for Li diffusion" paper. Killed Option 2
  as originally framed.

## 6. Understanding the dataset properly

Not "LiF + a bit of B." It's a **designed SEI/anode system** (see data catalog, Tables 1 & 2):

- **Li metal anode** (Li54), **LiF crystal** (LiF64)
- **Li/LiF interphase**, **BLi/LiF interphase**
- **F-doping** (Li53F1) and **B- and F+B-doping** (Li52F1B1) in Li metal — SEI formation chemistry
- **LiBF₄** salt, **BLi** alloy
- isolated Li₂ / F₂ / B₂ for reference energies

## 7. The turning point — the E₀ observation

From my own zero-shot run:

> **Forces transfer zero-shot; energies don't** — there's a per-species reference-energy (E₀)
> mismatch. I fixed it by running DFT on isolated Li/F/B to align references.

**Mechanism:** MACE-MPA-0 carries its own isolated-atom energy references from its training
DFT setup. Applied to my DFT setup, energies are off by a constant Σ nᵢ ΔE₀ⁱ per species.
Forces are gradients → immune to this shift.

**The link that makes it a paper:** paper 2's headline (zero-shot ≈ fine-tuned) was on
*diffusivity + Ea* — force-derived / energy-difference properties, **immune to E₀**. My dataset
is built for *energy-derived* properties (interface adhesion, defect/doping formation energies),
exactly where E₀ bites. So I'm not duplicating paper 2 — I'm mapping the boundary it never crossed.

## 8. The precise paper

**Thesis:** Zero-shot foundation MACE transfers for force-derived SEI properties (diffusion, Ea —
confirming paper 2) but silently fails for energy-derived ones (interface adhesion, defect/doping
formation) because of the reference-energy mismatch; quantified on a realistic
Li-anode / LiF / LiBF₄ / BLi dataset, where a minimal isolated-atom reference correction
(or light fine-tuning) recovers it at near-zero data cost.

**Core result = one table, four columns** (zero-shot / zero-shot+E₀-corrected / fine-tuned /
FitSnap), rows = energy-sensitive quantities:

- Li/LiF interface adhesion energy
- BLi/LiF interface energy
- F-in-Li, B-in-Li, F+B-in-Li defect formation energies
- LiBF₄ and BLi formation/cohesive energies

(all vs DFT reference)

**FitSnap's role:** a from-scratch descriptor potential fit directly to this dataset — it never
has the E₀ problem, so it's the right baseline for "what does the foundation model actually save
you, and what does it cost in energy accuracy?"

## 9. What remains to verify

- **Lit check:** is "reference-energy / E₀ alignment for foundation-model MLIPs" already fully
  characterized, or is the *SEI-specific, energy-property-quantified* version still open? (Known
  issue in general; the specific quantification is likely open.)
- **Decisive experiment:** the four-column property table above. Zero-shot is already run
  (forces good, E₀ mismatch observed); the correction is already done (isolated-atom DFT).
  Fine-tuned + FitSnap remain.

## 10. Meta-lesson

The oscillation through 7+ ideas wasn't wasted — but it wasn't a straight line either. Two biases
did the steering (kitchen-sink + anchoring on the professor's work). What finally produced a
*precise* paper was an **empirical observation I made myself** (the E₀ mismatch), not an assembly
of inherited ingredients. The one idea that was genuinely mine all along was the seed.
