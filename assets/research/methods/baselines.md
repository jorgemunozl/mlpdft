---
status: active
date: 2026-11-23
phase: 2
owner: jorge
---

# Single-model UQ baselines: shallow NLL ensemble and LLPR

**Date:** 2026-11-23.
**Scope:** two baselines that share the MHC arm's frozen MACE-OMAT-medium body
and cost about one model, added to `src/mlpdft/uq_compare.py` as
`--arms shallow-nll` and `--arms llpr`.
**Code:** `src/mlpdft/shallow_ensemble.py`, `src/mlpdft/llpr.py`,
`src/mlpdft/mace_batches.py`, tests in `tests/test_uq_baselines.py`.
**Status:** implemented and unit-tested on CPU. Nothing has been trained with the
real model. The training path runs first in the GPU smoke procedure (§5).

## 1. Shallow NLL ensemble (`shallow-nll`)

Reference: Schäfer, Kellner, Kästner, Ceriotti, *How to Train a Shallow
Ensemble*, <https://www.alphaxiv.org/abs/2602.15747> (DPOSE-style).

**Model.** Same as the MHC arm: the foundation body is frozen
(`FROZEN_BODY_LAYERS = 6`, `foundation_filter_elements=False`), and there are K
randomly initialised readout heads. The starting model comes from MACE's
`run_train` with the same arguments as the MHC arm, but every head's file is the
full training set and the optimisation loop is skipped (`_skip_mace_optimisation`).
The existing frozen-body checks therefore run on it unchanged: element-mapped
body weights, distinct initial head readouts, and no trainable descriptor tensor.

**Loss.** All heads see every training frame and are trained jointly in a
custom loop. For each force component $c$ (atom × direction) in a batch:

$$\mu_c = \frac1K\sum_k F_{k,c},\qquad s^2_c = \frac1K\sum_k (F_{k,c}-\mu_c)^2 + \varepsilon,$$

$$\mathcal L_\text{NLL} = \operatorname{mean}_c\ \tfrac12\Big(\log s^2_c + \frac{(y_c-\mu_c)^2}{s^2_c}\Big).$$

The $2\pi$ constant is omitted. $\varepsilon = 10^{-6}$ eV²/Å², so σ ≥ 10⁻³ eV/Å.
During the first `warmup_fraction` of the epochs the loss is the per-head force
MSE, $\operatorname{mean}_{k,c}(F_{k,c}-y_c)^2$. Because the heads have disjoint
weights, this is K independent fits. Energies are not used (energy weight 0).

**Forces.** Each step calls the model once per head with `training=True`, which
computes forces with `create_graph`. The loss reaches the readout weights through
$F = -\partial E/\partial R$. Body parameters have `requires_grad=False`.

**Selection.** The saved model has the lowest validation NLL among the NLL-phase
epochs. If there are no NLL epochs, the last epoch is kept. Training stops early
after `patience` NLL-phase epochs without improvement. Every epoch is logged to
`shallow_nll/shallow_nll_train.jsonl` with: phase, train loss, lr, validation
NLL, RMSE of the ensemble mean, RMS σ and per-head RMSE.

**Output.** `shallow_nll/models/shallow_nll.model` is a normal MACE multi-head
model. `_evaluate_method` evaluates it head by head, exactly like `frozen_mhc`
(heads `shallow_head_0…K-1`).

| Setting | Default | Flag |
|---|---|---|
| Heads K | 4 | `--shallow-heads` |
| Epochs (1 epoch = one pass over the full train set, all heads) | 120 (as `--mhc-epochs`) | `--shallow-epochs` (`--smoke-epochs` in smoke mode) |
| MSE warm-up | 20% of epochs (round) | `--shallow-warmup-fraction` |
| Patience (NLL phase, validation NLL) | 50 | `--shallow-patience` |
| Variance floor ε | 1e-6 eV²/Å² | `NLL_VARIANCE_EPS` |
| Optimiser | Adam, amsgrad, lr 1e-3, weight decay 0, grad-norm clip 10 | `--learning-rate` |
| Schedule | cosine annealing over all epochs to 1e-6, single cycle | — |
| Batch size | 16 | `--batch-size` |
| Seed | first `--seeds` value | — |

## 2. Last-layer prediction rigidity (`llpr`)

References: Bigi, Chong, Ceriotti, Grasselli, *A prediction rigidity formalism
for low-cost uncertainties in trained neural networks*,
<https://www.alphaxiv.org/abs/2403.02251>; applied to foundation-model
fine-tuning in <https://www.alphaxiv.org/abs/2607.14486>.

LLPR is applied post hoc to the trained `frozen-probe` model, or to the model
given with `--llpr-model` and `--llpr-head`.

**Last-layer features.** For head $k$, MACE computes

$$E = \sum_i E^0_{z_i} + \sum_i \big[s_k(p_i + \textstyle\sum_l o_{l,i}) + b_k\big],$$

where $p_i$ is the ZBL pair term and $o_{l,i}$ is readout $l$. Readout 0 is
linear on the layer-0 scalar (0e) features; the vector channels get zero weight,
and the code checks this. The last readout is Linear → SiLU → Linear, and its
hidden block for head $k$ is the channel slice that `mask_head` keeps. The
features are the inputs of these final linear maps, summed over atoms:

$$f = \sum_i\big[h^{(0)}_{i,0e};\ \phi(W_1 h^{(1)}_i)\big],\qquad E = w^\top f + E_\text{rest},$$

with $w = s_k[W_0[:,k];\,W_2[\text{block}_k,k]]$ and
$E_\text{rest} = \sum E^0 + s_k\sum p + N b_k$. $E_\text{rest}$ contains no
trainable readout weight; it depends on positions only through the ZBL term.
For MACE-OMAT-medium, $d = 128 + 16 = 144$.

**Force features.** $J = -\partial f/\partial R$ (d × 3N), computed with one
backward pass per feature (`JACOBIAN_CHUNK = 1`). Then
$F = w^\top J + F_\text{rest}$, where $F_\text{rest} = -s_k\,\partial(\sum p)/\partial R$.

**Covariance and variance.**

$$C = \sum_{\text{train frames}} J J^\top\ \big(+\,\lambda_E \textstyle\sum f f^\top\big),\qquad \sigma^2_c = \alpha^2\, j_c^\top (C+\varsigma^2 I)^{-1} j_c .$$

$C$ is diagonalised once, $C = V\Lambda V^\top$, which gives
$u_c = \sum_m (V^\top j_c)_m^2/(\lambda_m+\varsigma^2)$.

**Calibration** on the calibration file (`--calibration-file`). For each
$\varsigma^2$ on a grid of $\lambda_\max\cdot10^{[-12,2]}$ (85 log-spaced
points), the NLL-optimal scale has the closed form $\alpha^2 = \operatorname{mean}(e_c^2/u_c)$,
with errors $e = F_\text{model}-F_\text{ref}$. The pair with the lowest Gaussian
NLL is kept, and a warning is printed if it lies on the grid edge.

An energy σ, $\alpha^2 f^\top(C+\varsigma^2I)^{-1}f$, is also reported so the
summary format is complete. It is calibrated on forces only, and the evaluator
rescales it with `alpha_energy` as it does for the ensembles.

**Output.** `llpr/llpr_covariance.npz` holds C, its eigen-decomposition, w,
varsigma2, alpha2, the NLL grid, the training-frame indices used, and the
maximum reconstruction errors. `llpr/llpr_record.json` holds the fit summary.
`mlpdft.llpr.predict_force_sigma(model_path, head, cov_npz, frames)` returns one
`(n_atoms, 3)` σ array per frame; it refuses a covariance built for different
readout weights.

**Evaluation.** The prediction is the probe's. The σ come from LLPR, not from
member disagreement. The new `_summarize_predictions` writes `llpr_predictions.csv`,
`llpr_atom_level.npz` and the summary entry in the same format as the other
arms. Because α is fitted on the calibration set, the evaluator's
recalibration factor `alpha_force` on that same set is 1 by construction. RLL
and coverage on the test set are the informative numbers.

| Setting | Default | Flag |
|---|---|---|
| Training frames for C | all if ≤ 300; otherwise a random subset of 300 (seed `--split-seed`) | `--llpr-max-train-frames` |
| Energy term in C, λ_E | 0 (off) | `--llpr-energy-weight` |
| ς² grid | λ_max·10^[-12, 2], 85 points | `fit_calibration(grid=…)` |
| Model | the `frozen-probe` of the same run | `--llpr-model`, `--llpr-head` |

## 3. Run manifest and evaluator refactor

- `arms_manifest.json` (run directory) is written before evaluation. For
  committee, snapshots, multihead-bootstrap, mhc, frozen-probe and shallow-nll
  it lists members `{name, model_path, head}`. For llpr it lists `model_path`,
  `head` and `covariance`. Paths are relative to the run directory, and
  `head: null` means the model's only head. The foundation path is recorded
  as absolute.
- `_evaluate_method` now collects member predictions and passes them to
  `_summarize_predictions`, which the llpr arm also uses. On a randomised
  regression check against commit 42d4f77 (fake calculators, 7 test and 5
  calibration frames, 3-member and 1-member cases), the summary JSON,
  predictions CSV and atom-level NPZ were byte-identical.
- `--arms all` now includes `shallow-nll` and `llpr`. The no-flag default is
  unchanged (`mhc,frozen-probe`).

## 4. Verification (CPU, 2026-11-23)

`uv run --locked python -m unittest discover -s tests`: **23 tests OK, ~9 s**
(13 new tests, 10 existing).

| Check | Result |
|---|---|
| LLPR = Bayesian linear regression posterior predictive variance (α² = σ_n², ς² = σ_n²/τ²) | equal to rtol 1e-10 |
| Closed-form α² recovers the true scale (12,000 synthetic components); ±20–25% perturbations raise the NLL; ς² recovered within 0.5 decade, not at the grid edge | pass |
| `J` rows vs analytic gradient (chunk 1 and vmap chunk 3) | equal |
| Toy 3-head ScaleShiftMACE with ZBL (d = 6 + 4): max\|F − (wᵀJ + F_rest)\|, \|E − (w·f + E_rest)\| for every head | ≤ 2.6e-16 eV/Å, 0 eV |
| **MACE-OMAT-medium**, one forward and one vector-Jacobian product $-\partial(w\cdot f)/\partial R$ on an 8-atom rattled rock-salt LiF cell (812 edges, float64) | d = 144; max\|ΔF\| = 2.0e-15 eV/Å (max\|F\| = 0.665); \|ΔE\| = 7.1e-15 eV; ~1.2 s, 0.94 GB |
| LLPR fit/predict on the toy model: σ finite and > 0, shape (8, 3); probe forces equal `MACECalculator` to 1e-10; unknown head rejected | pass |
| NLL loss: matches the formula; finite when all heads agree (ε floor); ∂L/∂μ > 0 when μ > y; spread grows when residual² ≫ s² and shrinks when ≪; fixed variance ⇒ ½·MSE/s² + const with identical gradient | pass |
| Shallow trainer on the toy model (2 epochs, 4 steps): phases mse → nll, only `readouts.*` change, the saved model evaluates through `MACECalculator` for every head with distinct finite forces equal to the trainer's | pass |
| `arms_manifest.json` relative paths; llpr summary/CSV/NPZ keys identical to an ensemble's | pass |

**Not verified here.** The explicit 144-row J on the foundation model and the
real `run_train` initialisation path were not run on this machine; both are in
§5. A float64 forward of the foundation model with its graph kept, on one
122-atom frame, was killed for memory on this 7 GB CPU host.

## 5. GPU smoke procedure (to run before any production launch)

```bash
SPLIT_DIR=dataset/xyz_files/uq_interim_two_env   # the split the production run will use
uv run --locked python -m mlpdft.uq_compare \
  --smoke-test --output-dir /workspace/runs --run-id lif_uq_baselines_smoke --overwrite \
  --arms frozen-probe,shallow-nll,llpr \
  --train-file "$SPLIT_DIR/uq_train.extxyz" --valid-file "$SPLIT_DIR/uq_valid.extxyz" \
  --calibration-file "$SPLIT_DIR/uq_calibration.extxyz" --test-file "$SPLIT_DIR/uq_test.extxyz" \
  --energy-weight 0 --learning-rate 1e-3
```

Smoke mode gives 20 epochs per arm (`--smoke-epochs`): shallow-nll then has 4
MSE epochs and 16 NLL epochs. LLPR uses the first 8 training frames and 8
calibration frames (`--smoke-frames`). The probe learning gate runs first and
aborts the smoke run if it fails.

What to check:

1. **Init.** The log for `shallow_nll_init` shows `distinct initial head readouts
   verified`, `frozen body: 4 head(s) ['shallow_head_0', …]` and
   `frozen-body check passed`, and no per-epoch training lines. The file
   `shallow_nll/init/models/shallow_nll_init.model` exists.
2. **`shallow_nll/shallow_nll_train.jsonl`.** Phases are `mse` ×4, then `nll`.
   `valid_rmse_mean_force` falls during warm-up. `valid_nll` is finite and falls
   in the NLL phase. `valid_rms_sigma` neither collapses to the 1e-3 eV/Å floor
   nor grows without bound, and should end on the order of `valid_rmse_mean_force`.
   The four `rmse_per_head` values are finite and distinct. Record seconds per
   epoch for costing (`shallow_nll_record.json: seconds`).
3. **`llpr/llpr_record.json`.** `feature_dim` = 144.
   `max_abs_force_reconstruction_error` < 1e-6 eV/Å; this is the explicit-J
   check on real frames. `max_abs_energy_reconstruction_error` < 1e-6 eV.
   `varsigma2_at_grid_edge` may be true with only 8 frames; on a production run
   it must be false.
4. **`evaluation/uq_summary.json`.** It contains `shallow_nll` (`member_count`
   4, calibration block present) and `llpr` with `calibration.alpha_force` =
   1.000 ± 1e-6, since it is the same set α was fitted on.
   `evaluation/{shallow_nll,llpr}_{predictions.csv,atom_level.npz}` exist.
5. **Independent reload from the manifest**:

   ```bash
   uv run --locked python - <<'EOF'
   import json, pathlib
   from mace.calculators.mace import MACECalculator
   from mlpdft.llpr import predict_force_sigma
   from ase.io import read
   run = pathlib.Path("/workspace/runs/lif_uq_baselines_smoke")
   m = json.loads((run / "arms_manifest.json").read_text())["arms"]
   atoms = read("dataset/xyz_files/uq_interim_two_env/uq_test.extxyz", 0)
   for member in m["shallow-nll"]["members"]:
       calc = MACECalculator(model_paths=[str(run / member["model_path"])], head=member["head"], device="cuda", default_dtype="float64")
       assert calc.head == member["head"], calc.head   # MACECalculator silently falls back to the last head otherwise
       a = atoms.copy(); a.calc = calc; print(member["head"], abs(a.get_forces()).max())
   l = m["llpr"]
   s = predict_force_sigma(run / l["model_path"], l["head"], run / l["covariance"], [atoms], device="cuda")
   print("llpr sigma", s[0].min(), s[0].max())
   EOF
   ```

## 6. Expected GPU cost (projection, unmeasured)

The figures use the Exp 3 design's projection of ≈3.2 s per frozen-head update
at batch 16 ([`exp-03-mhc/DESIGN.md`](../experiments/exp-03-mhc/DESIGN.md) §11),
on 1,000 training frames. Replace it with the smoke timing.

- **shallow-nll.** One step runs K forward passes plus K force double-backward
  passes, so it costs about K frozen-head updates. At 120 epochs:
  63 steps/epoch × 120 × 4 ≈ 30,000 head-updates ≈ **27 h**. That is between
  MHC-4D (~7 h) and the probe (~33 h), plus a per-epoch validation pass and a
  few minutes for the `run_train` init. Early stopping can only shorten it.
  Reusing one forward for the K backward passes would save roughly a third.
- **llpr.** There are d = 144 backward passes per frame, without
  `create_graph`. The frames processed are the 300 covariance frames, the
  calibration frames twice (fit, then evaluation) and the test frames. With
  about 800 frames that is ≈115k single-frame backward passes, an estimated
  **0.5–2 h** on an A40. Memory holds one frame's graph at a time. Raising
  `llpr.JACOBIAN_CHUNK` vectorises rows at proportionally more memory.
- **Peak memory.** shallow-nll keeps K double-backward graphs per batch of 16.
  Expect about the MHC arm's footprint (29–41 GB at batch 16, float64), so use
  a 48 GB card or lower `--batch-size`.

## 7. Deviations from the brief and open choices

- The CPU training smoke test was replaced by §5, on the project owner's
  instruction that this machine runs no training. The trainer runs here only
  for 4 steps on a toy model (126 trainable readout parameters).
- The warm-up uses plain MSE on forces, not MACE's universal (Huber) loss.
- The foundation check uses the vector-Jacobian product $w^\top J$ (one
  backward pass), not the full 144-row J, to stay within the one-pass limit
  here. The full J is checked on the toy model and in §5 item 3.
- The ZBL pair term is not a readout. It is carried in $E_\text{rest}$ and
  $F_\text{rest}$ rather than in $f$, and is excluded from the LLPR features.
- The default `--shallow-epochs` is 120 (as the MHC arm), chosen for cost (§6);
  it is not tuned.
