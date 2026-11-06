# Four-head disjoint MHC: A40 development run

**Status:** prepared; no A40 run or scientific result yet. This document applies
the design in [`UQ_MHC_4HEAD_DISJOINT_DESIGN.md`](UQ_MHC_4HEAD_DISJOINT_DESIGN.md)
to the user's approximately eight-hour compute budget.

## Objective and scope

Test whether four randomly initialized, disjoint readout heads trained on the
frozen MACE-OMAT-medium body learn useful force predictions and uncertainty on
the existing two-environment Li–F candidate. This is a **development** test:
F-doped equilibration and dependence remain unresolved, and the test frames are
later segments of training trajectories. Do not claim independent/OOD or
production-grade performance from this run.

The foundation-model experiment in Beck et al. used eight new random heads on
disjoint subsets. This run uses four heads, fewer labels, a force-only loss and a
different foundation checkpoint; it tests the construction, not a numerical
reproduction. [Paper](https://www.alphaxiv.org/pdf/2508.09907)

## Inputs and configuration

- Private dataset `jorgemunozl/li-f-uq-interim-two-environments`, revision
  `c2699e8e52889c18f6d29f657eb996ac3f78524f`, with 1,000 train, 200 validation,
  200 calibration, 200 test frames. SHA-256 values are checked in preflight.
- Frozen foundation `jorgemunozl/mace_omat_medium`, revision
  `d64b6faf564bd682ba1e9ca337408f6925aa908d`; expected model SHA-256
  `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`.
  Private Hugging Face access is required from the execution host.
- Four disjoint, group-stratified heads, each with 125 bulk and 125 F-doped
  frames, seed 20251027; 1,000 unique frames across all heads. Training seed
  123. No replay or foundation readout warm start.
- One A40, float64, batch 16, force weight 100, energy weight 0, learning rate
  0.001, patience 50. Energy metrics are not scientifically interpretable as
  energy predictions from a force-only run.
- A 20-epoch smoke uses validation frames for its head-diversity check. Only
  after it passes, the full MHC runs for **at most 90 epochs**. This replaces
  the design's 120-epoch proposal to leave time for smoke and evaluation
  within the eight-hour budget. Convergence is not guaranteed. No 600-epoch
  frozen probe or newly trained full-body committee is included.

## Checks and budget

The committed runner installs locked dependencies when `uv` is available,
downloads pinned inputs, verifies file hashes, species/group counts, finite
labels and periodic cells, checks that the head files form the exact disjoint
125/125 partition, and checks that CUDA identifies an A40. It copies and checks
the foundation distance transform, verifies element-remapped embedding and key
shape-stable foundation parameters, checks initial linear head weights differ,
and fails if descriptor parameters are trainable. This set of tensor checks is
not proof of equality of every transformed descriptor tensor.

The smoke stage checks head diversity and a validation learning gate. It reads
no final test frames. The final evaluation fits uncertainty scaling and the
constant RMSE reference for relative log-likelihood on the **separate
calibration split**; it then evaluates the test split. The output manifest and
summary must identify paths, head membership, actual runtime and results.
The smoke's validation-only check cannot guarantee full-run convergence.
It also evaluates the archived 400-epoch pilot MHC on these same calibration
and test frames; its SHA-256 is
`dd6c542a4fc10d7a9e4a724cb06ba07b6a3fda22dc735f343bb8c1baa4b67a11`.
The locally available old pilot train, validation and test splits have no exact
rounded-geometry matches to the new calibration or test frames. This does not
establish trajectory independence or exclude similar configurations.

An eight-hour wall-clock timeout applies to setup, smoke and the full command.
The full stage is also stopped with 15 minutes remaining. A timed-out or
failed run is incomplete; preserve logs and checkpoints. This bounds the
experiment process, **not the rental billing**: the rented machine must be
stopped separately when no longer needed.

## Interpretation

Report per-environment force RMSE, within-environment uncertainty ranking,
calibration factor, coverage and RLL, plus actual A40 time and per-head errors.
Interpret temporal-confidence intervals as exploratory; F-doped autocorrelation
estimates hit their lag limit. One run does not measure seed variability.

The existing pilot's originally reported metrics were evaluated on a different
dataset. The new same-frame pilot comparison is descriptive: the training data,
initialization and loss all changed, so it cannot isolate their individual
effects. A 3-member committee or frozen probe comparison additionally needs
those models evaluated on the same frames; this run alone cannot establish
superiority over them.

## Launch contract

[`scripts/run_mhc_4d.sh`](../../../scripts/run_mhc_4d.sh) is the fixed run
recipe. It expects an authenticated Hugging Face environment, or private inputs
staged at the pinned locations before preflight. It produces `preflight.json`,
head manifests, smoke and full run outputs, and prints the final UQ summary.
Use a fresh output directory. Capture outputs before the remote run directory
is cleaned up. Host setup, GPU visibility, authentication, time and storage
must be checked after the user provides the SSH entry. No compute was launched
while preparing this document.
