---
status: active
date: 2026-11-27
owner: jorge
---

# Literature Review — Uncertainty Quantification and Active Learning for Foundation MACE on Li–F

- **Date:** 2026-11-27
- **Update 2026-11-27:** §14 explains why the warm-restart snapshots ranked errors at least as well as the fine-tuned committee in exp-02: a committee fine-tuned from one checkpoint is a local ensemble, and part of the snapshot spread is a convergence trend. §10.7 adds its two references. The consequences for each design-C arm are in [methods/ensemble-diversity.md](../methods/ensemble-diversity.md).
- **Update 2026-11-23:** the study moved to design C (train on bulk LiF, evaluate along a graded shift; [ADR-0007](../decisions/ADR-0007-bulk-trained-shift-ladder.md), [ADR-0008](../decisions/ADR-0008-40ry-labels-and-md-protocol.md)). §10.5 adds seven references and §12 explains what they change; the rest of the review is unchanged and predates the redesign.
- **Theory addendum 2026-11-07:** §13 adds the ensemble-spread theory (spread identity, correlation parameter, common-mode blindness, frozen-body and snapshot consequences, online versus retrospective AL) and §10.6 its references. The eight papers whose numbers are quoted there were read in full text. This addendum is independent of the design-C update in §12.
- **Status:** reframed on 2026-10-29 for the UQ + AL + tier-0 anchor scope; supersedes the 2026-09-13 draft (E₀ and zero-shot property dependence) and the 2026-10-27 transport framing.
- **Scope:** uncertainty quantification (UQ) for machine-learning interatomic potentials (MLIPs), with emphasis on cheap versus expensive ensemble constructions and uncertainty-guided active learning on a Li–F testbed. Transport is context, not a claimed observable.
- **Project cross-reference:** [`DESCRIPTION.md`](../CHARTER.md), [`ROUTE.md`](../ROADMAP.md), [`UQ_EXPERIMENT_0.md`](../archive/2026-10-29-uq-protocol-v0.md), [`ACTIVE_LEARNING_PROTOCOL.md`](../experiments/exp-04-active-learning/PROTOCOL.md).

**Search.** arXiv, alphaXiv, and Crossref; queries combined MLIP/MLFF, uncertainty quantification, ensemble, snapshot, multi-head, active learning, and Li-ion/LiF diffusion; cutoff 2026-10-27. The ranked retrieval set supplied for this revision was checked against canonical records. **Inclusion:** UQ methods or benchmarks for MLIPs, active learning, and Li–F/Li-ion context. **Exclusion:** E₀-only or zero-shot-transfer papers, except where they affect energy metrics.

**Evidence basis.** Bibliographic metadata for every reference below was checked against the arXiv API or Crossref on 2026-10-27. Claims are taken from titles and abstracts; no claim-level full-text audit has been performed. This is a bounded review, not a systematic review or a proof that no other relevant paper exists.

**Framing note (2026-10-29).** The Li-ion/LiF transport literature in §7 is retained as testbed justification and as a model-form-uncertainty caveat. It is no longer a validation target: the paper does not report \(D\) or \(E_a\). Passages that still mention a "transport QoI" are read as historical context; the current claims are in `DESCRIPTION.md`.

**Theory addendum (2026-11-07).** §13 adds the ensemble-theory literature: a spread identity that makes committee uncertainty invariant to shared error, a correlation parameter that makes \(\rho\) rather than \(K\) the governing quantity, and the mode-connectivity and common-bias evidence that explains why fine-tuned committees collapse and why out-of-distribution uncertainty inverts. It also bounds the AL claim to retrospective acquisition efficiency.

**Bottom line.** The literature establishes three things: ensemble disagreement is the practical default for MLIP UQ but is expensive and unreliable out of distribution; weight-sharing and single-model alternatives can cut that cost substantially; and better uncertainty does not automatically produce better active learning. It does **not** contain a matched comparison of independent fine-tuning committees, warm-restart snapshot ensembles, and the two multi-head variants (MH-FT and MHC) for a foundation MACE model on a multi-environment Li–F dataset, nor a test of whether those uncertainty signals improve active learning for held-out force error over random acquisition. That is the paper's opening.

---

## 1. What this paper is, and what the literature must support

The paper compares four ensemble arms around the same foundation MACE checkpoint. The first three are the core comparison; the fourth separates the published method from the fine-tuning implementation that is often confused with it:

1. **Independent committee** — three full fine-tunes from the same checkpoint, with different seeds and/or bootstrap resamples.
2. **Warm-restart snapshot ensemble** — one fine-tune with a cyclic (SGDR-style) learning-rate schedule; the three members are checkpoints taken at cycle minima.
3. **Multi-head fine-tuning with replay (MH-FT)** — one *trainable* backbone with three output heads trained on different bootstrap targets, plus a Materials Project replay head. This is the released `mace-torch` multi-head fine-tuning path and the arm measured in Experiment 2.
4. **Frozen-body multi-head committee (MHC)** — Beck et al.'s construction: the atomic-environment descriptors and the original head are frozen, and four randomly initialised heads are trained on disjoint slices with no replay. This is the arm added in Experiment 3.

Arms 3 and 4 share the multi-head mechanism but not what is trained. They have different accuracy, cost and diversity behaviour, so they are reported separately and are not interchangeable.

The project checkpoint is **MACE-OMAT medium** ([Batatia et al., 2023](https://arxiv.org/abs/2401.00096)). The papers below often use MACE-MP-0 (Beck et al.) or MACE-MPA-0 (Alghamdi et al.); their results are method evidence, not same-checkpoint benchmarks.

The comparison is not merely "which method has the lowest force error." It asks:

- Does ensemble disagreement **rank** held-out DFT force errors within each Li/F environment?
- Is the uncertainty **calibrated** in force-error units, and does it produce useful risk–coverage curves?
- Does the spread across members stay calibrated within each environment, and does that calibration survive the shift from bulk LiF to F-doped Li and the interface film?
- At matched ensemble size and honest compute accounting, which construction gives the best accuracy/UQ per unit cost?
- Does using that uncertainty to acquire DFT configurations beat random acquisition at equal label budget?

These are five different questions. A method can win one and lose another. The review is organized around that distinction.

The previous draft's E₀/property-dependent zero-shot story is **out of scope**. Reference-energy alignment remains a preprocessing check for energy metrics, but it is not the paper's hypothesis and should not organize the related-work section. Energy metrics are secondary here; force accuracy, uncertainty ranking, within-environment calibration, and label/compute efficiency are primary.

### 1.1 Terminology used below

| Term | Meaning in this paper |
|---|---|
| **Independent committee** | \(K\) separately fine-tuned models; spread comes from initialization, data order, and bootstrap resampling. |
| **Warm-restart snapshot ensemble** | One fine-tune with cyclic learning-rate restarts; members are cycle-end checkpoints. "Warm restart" here means an optimization restart, **not** warm-starting from another task. |
| **Multi-head committee** | One shared body with \(K\) output heads; spread comes from head-specific parameters and training targets. In Beck et al. the body is **frozen** and the heads are new. |
| **Multi-head fine-tuning (MH-FT)** | The released MACE multi-head path: the body is trained jointly, heads are warm-started, and a replay head limits forgetting. Not the same method as MHC, even though both use multiple heads. |
| **Shallow ensemble** | A weight-sharing ensemble in which most of the model is shared and only a shallow part is replicated; trained with a probabilistic objective. Related to, but not identical with, a multi-head committee. |
| **Discrimination / ranking** | Whether uncertainty is larger for more erroneous predictions (e.g. Spearman correlation, error-by-uncertainty quantile). |
| **Calibration** | Whether predicted uncertainty has the stated coverage or likelihood; a ranking can be good while calibration is poor, and vice versa. |
| **Risk–coverage** | Error among retained predictions as the lowest-uncertainty fraction is varied. |
| **Environment** | Bulk LiF, F-doped Li metal, or the thin LiF/Li interface film; splits and metrics are reported per environment. |
| **Labels-to-target / compute-to-target** | New DFT labels, or wall-clock compute, needed to reach a pre-declared accuracy target. |

---

## 2. The UQ problem in foundation MLIPs

### 2.1 Why ensemble UQ is the default

Deep ensembles are the standard baseline ([Lakshminarayanan et al., 2017](https://arxiv.org/abs/1612.01474)) because they are simple, parallelizable, and usually better than a single point estimate at detecting distribution shift. The idea is straightforward: train several models, treat their disagreement as epistemic uncertainty, and use it for error analysis or acquisition. In MLIPs this usually means several independently trained networks, or several members differing by seed, bootstrap, dropout, or snapshot.

The cost is equally straightforward. An independent \(K\)-member committee costs roughly \(K\) fine-tunes, and \(K\) forward passes per candidate during active learning. For a foundation model that is already expensive to fine-tune, this is the bottleneck. Bi et al. state the problem directly: standard model-committee uncertainty is impractical for foundation MLFFs because each member requires a separate fine-tuning run ([Bi et al., 2026](https://arxiv.org/abs/2607.14486)). Beck et al. make the same point from the architecture side: a committee of independent foundation models is costly, so they attach multiple output heads to one shared MACE body ([Beck et al., 2025](https://doi.org/10.1063/5.0302097)).

This cost structure is why "cheap ensembles" is a real research question, not just an engineering optimization. The relevant comparison is not only training cost. In an active-learning loop, scoring a large candidate pool is repeated every round, so **inference cost** matters too:

- Independent committee: \(K\times\) training, \(K\times\) inference.
- Warm-restart snapshots: \(\approx 1\times\) training, \(K\times\) inference (one checkpoint per member).
- Multi-head: \(\approx 1\times\) backbone training plus head training, \(\approx 1\times\) inference.
- Single-model alternatives: \(1\times\) training, \(1\times\) inference, but a different uncertainty model.

A paper that reports only training time will miss the inference side; one that reports only inference will miss the fine-tuning side. Both should be reported, and both should be tied to the active-learning budget.

### 2.2 Why ensemble UQ is not automatically trustworthy

The more important literature is the one that limits the claim. Several recent papers show that ensemble disagreement is a useful but fragile signal:

- **Disagreement can correlate weakly with error at foundation scale.** Mehdi et al. argue that prevailing MLIP UQ methods rely on independently trained backbones, scale unfavorably with foundation models, and that member disagreement correlates weakly with per-molecule prediction error. Their PROBE method instead trains a classifier on frozen per-atom representations and reports a per-prediction reliability probability ([Mehdi et al., 2026](https://arxiv.org/abs/2605.00640)).
- **Uncertainty can plateau or decrease as error grows out of distribution.** Kurniawan et al. benchmark bootstrap, dropout, random-initialization, and snapshot ensembles on carbon allotropes and find exactly this counterintuitive behavior in OOD tests ([Kurniawan et al., 2026](https://doi.org/10.1088/2632-2153/ae9fb4)).
- **Data heterogeneity degrades calibration.** Shuang et al. find that ensemble and D-optimality uncertainties are reasonably calibrated on homogeneous training sets but underpredict errors and lose novelty sensitivity on heterogeneous datasets; clustering the configuration space improves novelty detection ([Shuang et al., 2026](https://doi.org/10.1088/2632-2153/ae3d80)).
- **Independently trained graph networks can collapse to similar predictions.** Vieira et al. benchmark deep ensembles across eight graph datasets and find only marginal gains over a single model; an aleatoric–epistemic decomposition shows "epistemic collapse," where members converge to overly similar predictions ([Vieira et al., 2026](https://arxiv.org/abs/2605.22593)). This is graph-level evidence, not an MLIP result, but it is a direct warning for committees whose members share architecture, initialization, and training data.
- **Calibration does not guarantee active-learning gains.** Dale et al. evaluate ALIGNN ensembles, XGBoost, random forests, and neural networks on solubility, bandgap, and formation-energy tasks. Calibration methods generalize unevenly from in-domain to OOD data, and calibrated uncertainties were generally **unsuccessful** in reducing the data needed to improve over random sampling on OOD data ([Dale et al., 2025](https://arxiv.org/abs/2511.17760)).

The practical implication is that "ensemble spread" should never be treated as a synonym for "reliable uncertainty." The paper should report discrimination, calibration, and coverage separately, and should be prepared to find that a method with larger spread is worse.

### 2.3 What ensemble spread does and does not capture

A fine-tuning ensemble around one foundation checkpoint captures **parameter and optimization uncertainty** within that model family. It does not, by default, capture:

- uncertainty in the exchange–correlation functional;
- uncertainty from the foundation model's pretraining distribution;
- uncertainty from the architecture choice;
- uncertainty from the DFT labels themselves;
- uncertainty from finite MD sampling and replica-to-replica variability.

This matters for Li-ion transport. Li, Gong et al. show that changing the XC functional and changing the neural-network architecture can produce differences in predicted Li⁺ diffusivity of the same order of magnitude in Li₆PS₅Cl ([Li et al., 2025](https://arxiv.org/abs/2512.11650)). Guo et al. benchmark several universal MLIPs on alkali-ion battery kinetics and find that architecture helps static barriers, while training-data coverage dominates dynamic transport accuracy ([Guo et al., 2026](https://arxiv.org/abs/2601.10938)). An ensemble of three fine-tunes of the same MACE checkpoint will not see either of those model-form uncertainties. The paper should say so explicitly, rather than presenting its spread as total predictive uncertainty.

---

## 3. The three ensemble constructions

### 3.1 Independent committees: the reference, not the novelty

The independent committee is the right reference because it is the most studied and the easiest to interpret: each member is a legitimate model, and disagreement is not artificially constrained by shared weights. It is also the most expensive.

Kurniawan et al. include random-initialization and bootstrap ensembles in their comparison, alongside dropout and snapshot ensembles. Their key result is not that one construction wins everywhere, but that all of them can fail OOD: uncertainty may plateau or decrease while error grows. That makes the independent committee a necessary baseline but a weak novelty claim.

The project's own first run is consistent with the broader warning: committee members can be highly correlated when they share a foundation checkpoint and training recipe, so "three seeds" does not guarantee three independent hypotheses ([`UQ_EXPERIMENT_1.md`](../experiments/exp-01-committee/DESIGN.md)). The literature on GNN epistemic collapse offers a plausible mechanism, though it is not an MLIP-specific result ([Vieira et al., 2026](https://arxiv.org/abs/2605.22593)). The practical lesson is to report member-to-member diversity, not just ensemble size.

Note 2026-11-27: every committee in this project is fine-tuned from the same MACE-OMAT-0 checkpoint, so it is what Sadrtdinov et al. call a local deep ensemble, not the from-scratch deep ensemble the "gold standard" label refers to. §14 gives the evidence and what it means for the comparison with snapshots.

**Role in this paper:** reference construction for cost and accuracy; the method that must be beaten or matched on discrimination, calibration, and label efficiency.

### 3.2 Warm-restart snapshots: cheap, plausible, and under-evidenced

The warm-restart snapshot ensemble is the paper's most distinctive arm. The construction is simple: fine-tune once with a cyclic learning-rate schedule, save the model at each cycle minimum, and treat those checkpoints as ensemble members. Training cost is approximately one fine-tune; inference cost is still \(K\) forward passes.

The idea has a clear generic deep-learning lineage. Huang et al. proposed snapshot ensembles to obtain several local minima along one optimization path at no additional training cost, using cyclic learning rates ([Huang et al., 2017](https://arxiv.org/abs/1704.00109)). Kurniawan et al. include snapshot ensembles in their MLIP benchmark, so the construction is not unknown in the MLIP literature ([Kurniawan et al., 2026](https://doi.org/10.1088/2632-2153/ae9fb4)).

But the specific claim this paper needs — that a warm-restart snapshot ensemble from a **fine-tuned foundation MACE model** is a cost-effective UQ method for **Li–F transport**, competitive with an independent committee and the two multi-head variants (MH-FT and MHC) — is not established by those papers. The retrieval found no dedicated MLIP paper that uses "warm snapshots" as its central method. The closest conceptual neighbors are:

- **snapshot ensembles** as one of several generic ensemble constructions (Kurniawan et al.);
- **shallow ensembles** as a weight-sharing way to reduce ensemble training cost (Schäfer et al.);
- **multi-head methods** that extract several predictions from one shared model (Beck et al.).

These are not the same thing. A shallow ensemble shares most weights across members and is trained with a probabilistic objective; a warm-restart snapshot ensemble shares one optimization trajectory and varies the checkpoint. The paper should define its arm precisely and should not cite shallow ensembles as if they were direct evidence for warm snapshots.

There is also an operational trap. A checkpoint taken at an arbitrary epoch is not an ensemble member; it may be a loss-spiked model just after a restart. The members should be taken at **cycle minima**, and the selection rule should be fixed before looking at test performance. The project's first run collected snapshots at evenly spaced epochs that coincided with restart boundaries, and two of the three members were loss-spiked just after a restart; the corrected run uses cycle-end checkpoints ([`UQ_EXPERIMENT_1.md`](../experiments/exp-01-committee/DESIGN.md), [`UQ_EXPERIMENT_2.md`](../experiments/exp-02-snapshots/DESIGN.md)). This is a concrete design lesson, not a literature result: the snapshot recipe is only as good as its checkpoint-selection rule.

**Role in this paper:** the low-training-cost candidate. The hypothesis is not that snapshots are intrinsically better, but that they may reach comparable discrimination and transport-QoI uncertainty at a fraction of the fine-tuning cost. The risk is that their members are too correlated, or that the cycle-end checkpoints are not diverse enough to represent epistemic uncertainty.

Note 2026-11-27: exp-02 showed the opposite risk. The snapshot spread was 1.2–2.8 times the committee's, and part of it is a convergence trend between cycles rather than diversity between minima (§14.4).

### 3.3 Multi-head committees: the strongest cost-reduction evidence

Beck et al. implement a committee by attaching multiple output modules to the same MACE atomic-environment descriptors. The standard deviation across heads is used as uncertainty. On custom models, force uncertainty correlates well with true errors. On MACE-MP-0, they train only the newly attached heads while keeping the rest of the model fixed, and use the resulting committee in an active-learning workflow that condenses the training set to about 5% of its original size without substantial accuracy loss ([Beck et al., 2025](https://doi.org/10.1063/5.0302097)).

This is the most directly relevant cost-reduction result for the paper. It shows that a shared backbone with multiple heads can produce useful force uncertainty and support active learning, at a fraction of the cost of independent fine-tunes. It also makes the multi-head arm a serious competitor, not a straw man.

But there are caveats:

- The heads share the same representation, so their errors may be correlated; head disagreement is not the same as independent-model disagreement.
- Beck et al. train only the new heads on a frozen foundation model. If the backbone is fine-tuned instead, both the cost and the uncertainty behaviour change. The project keeps these as two separate arms (MH-FT and MHC) for exactly that reason.
- MH-FT includes a replay head trained on Materials Project data (requested as `omat`) to limit forgetting. Replay adds data and compute; it is reported separately, not hidden in the head-training cost. MHC has no replay head.
- Few heads mean a coarse spread: three bootstrap heads in MH-FT, four disjoint heads in MHC. The spread may be dominated by head initialisation, especially when each head sees a small slice.

The related **shallow ensemble** literature is also relevant, but it is a different design. Schäfer et al. systematically study training strategies for shallow ensembles and find that explicit negative-log-likelihood (NLL) optimization improves calibration relative to random-initialization ensembles or a last-layer Laplace approximation. Models trained only on energy objectives give miscalibrated force uncertainties; explicitly modeling force uncertainty via an NLL objective is essential, though expensive. They then show that full-model fine-tuning of a shallow ensemble originally trained with a probabilistic energy loss — or sampled from the Laplace posterior — retains calibration while reducing training time by up to 96% ([Schäfer et al., 2026](https://arxiv.org/abs/2602.15747)).

That is the closest cost–accuracy evidence in the retrieval, but it is **not** a warm-restart snapshot study. It supports the general proposition that weight sharing plus a probabilistic objective can make ensembles much cheaper; it does not tell us whether cycle-end checkpoints from one SGDR run will be calibrated on Li–F forces.

**Role in this paper:** the low-inference-cost, shared-backbone competitors. MH-FT tests whether the released fine-tuning recipe is cheaper than an independent committee and whether its replay cost erodes that advantage. MHC tests whether freezing the descriptors preserves the foundation model's accuracy while giving committee-quality uncertainty at head-only cost.

### 3.4 Comparison of the arms

| Property | Independent committee | Warm-restart snapshots | Multi-head fine-tuning (MH-FT) | Frozen-body MHC |
|---|---|---|---|---|
| Source of spread | Separate fine-tunes (seed/bootstrap) | Checkpoints along one optimization trajectory | Head parameters, bootstrap targets, jointly trained body | Random head initialisation on disjoint data slices, frozen body |
| Training cost | \(K\times\) fine-tune | \(\approx 1\times\) fine-tune | \(\approx 1\times\) fine-tune + MP replay | Forward-heavy body, readouts only; no replay |
| Inference cost | \(K\times\) | \(K\times\) | \(\approx 1\times\) mean, \(K\) passes for force spread | \(\approx 1\times\) mean, \(K\) passes for force spread |
| Diversity mechanism | Initialisation, data order, bootstrap | Optimization path / cycle minima | Head initialisation, bootstrap targets | Disjoint data slices plus random head initialisation |
| Main strength | Most interpretable; closest to gold-standard deep ensemble | Cheapest training; simple to add to an existing fine-tune | Strong published evidence; one model | Preserves the foundation representation; no replay; cheap heads |
| Main risk | Expensive; members can collapse | Members may be highly correlated; checkpoint selection is fragile | Replay confounds cost; the body can drift from the foundation model | Few frames per head; accuracy bounded by the frozen body |
| Direct MLIP evidence | Strong (generic ensembles; benchmarks) | Weak for foundation MLIPs; snapshot ensembles only as one benchmark arm | Strong for MACE fine-tuning | Strong for MACE force UQ and AL (Beck et al.) |
| Role here | Reference | Candidate low-cost method | Experiment 2 arm | Experiment 3 arm; the paper's method |

The table makes the paper's contribution legible: it is not "another ensemble." It is a **matched, cost-aware comparison** of three ensemble families — with the multi-head family split into the fine-tuning implementation and the frozen-body committee — followed by a test of whether their uncertainty is useful for within-environment error ranking and active learning.

---

## 4. Single-model alternatives: the cost floor and the competition

A reviewer will ask why the paper compares three ensemble families (four arms) instead of using a cheaper single-model UQ method. Single-model methods approach the same prediction problem with a different uncertainty model; they set the cost floor and should be discussed as alternatives, not ignored.

### 4.1 Last-layer projection regression (LLPR)

Bi et al. propose an active-learning workflow based on last-layer-projection regression, a forward-pass-cheap per-configuration uncertainty estimator. They argue that standard committee UQ is impractical for foundation MLFFs because each member needs a separate fine-tune. Across molecular, condensed-phase, and electrolyte systems, LLPR selects compact, high-value training sets that recover full-data accuracy with a small fraction of electronic-structure labels. In foundation-model fine-tuning, LLPR-selected configurations approach the practical fine-tuning ceiling with substantially fewer labels than random selection. In iterative electrolyte fine-tuning, it detects unphysical local coordination before new labeling, calibrates uncertainty in force-error units, and supports retrospective convergence monitoring ([Bi et al., 2026](https://arxiv.org/abs/2607.14486)).

This is the most directly relevant active-learning paper in the retrieval. It shows that a cheap, single-model uncertainty estimator can compete with committees for label efficiency, at least in the systems studied. It also raises the bar for the paper's AL claim: if a single-model method can do this, a three-member ensemble needs to justify its extra cost.

**Caveat:** LLPR is not an ensemble and does not provide member-to-member spread. It is a last-layer Bayesian/linear-projection approximation; its uncertainty may be poorly calibrated under strong distribution shift. The paper should cite it as a competitor and a cost benchmark, not as evidence that ensembles are unnecessary.

### 4.2 Neural tangent kernel methods

Varga-Umbrich et al. extend the neural tangent kernel (NTK) to a force-aware setting using mixed parameter-coordinate derivatives, giving force and joint energy–force NTKs. They introduce a chunked feature-space posterior-variance shortlisting method that scales to large candidate pools (screening ~200k structures within hours). On OC20, force-aware acquisition gives the lowest energy and force errors across metrics and distribution splits. Across T1x, PMechDB, and RGD, their force-NTK methods remain competitive with established baselines while being significantly more efficient than committee-based approaches; under a controlled candidate-pool shift on T1x, pretrained-MLIP embedding/NTK acquisition remains robust while committee-based methods exhibit higher variance ([Varga-Umbrich et al., 2026](https://arxiv.org/abs/2605.13788)).

AdaptNTK goes further: it measures uncertainty as a regularized Mahalanobis distance in empirical NTK feature space, updates the acquisition score recursively after each selection without retraining, and reports the highest mean correlation with force errors on held-out rMD17 (Spearman 0.68, Pearson 0.71), matching a three-member ensemble in error retention. In active learning on rMD17 and Transition-1X, it achieves the lowest force errors, with a 2.6-fold speedup per Transition-1X cycle relative to the ensemble ([Ananth & Yue, 2026](https://arxiv.org/abs/2609.00488)).

These results are important because they show that a **single pretrained model plus a feature-space uncertainty score** can be more efficient and sometimes more robust than a committee. They do not make ensembles obsolete: NTK methods require a choice of kernel/feature space, can be sensitive to the training-set geometry, and have not been validated on Li–F transport. But they are a strong reminder that "more models" is not automatically "better UQ."

### 4.3 Evidential and probabilistic single-model methods

- **Equivariant evidential deep learning (e²IP).** Wang et al. represent force uncertainty as a full \(3\times3\) symmetric positive-definite covariance tensor that transforms equivariantly under rotations. They report a stronger accuracy–efficiency–reliability balance than a non-equivariant evidential baseline and a widely used ensemble, with single-model inference efficiency ([Wang et al., 2026](https://arxiv.org/abs/2602.10419)).
- **Learned functional perturbations.** Zaghen et al. turn a deterministic MLIP into a probabilistic one by learning functional perturbations and fine-tuning end-to-end with the Continuous Ranked Probability Score (CRPS). On a charged-particle benchmark, their P-EGNN improves CRPS over the Bayesian MLIP baseline BLIP by 19–32% across training sizes; on silica, fine-tuning Orb-v3 raises the Spearman correlation between predicted uncertainty and actual error from 0.75 to 0.84 ([Zaghen et al., 2026](https://arxiv.org/abs/2605.19939)).
- **Black-box uncertainty from existing multi-head outputs.** Fonea et al. analyze an ensemble-based UQ signal that can be extracted from a multi-headed regression network without changing the architecture, making it suitable for sealed or black-box models. They apply it to Na and Al systems, scale the signal for heteroscedasticity, and show robustness for OOD detection and training convergence ([Fonea et al., 2025](https://arxiv.org/abs/2511.16439)). This is not a new committee construction; it is a way to read uncertainty out of an existing multi-head model.
- **Fast single-network UQ.** Zhu et al. propose a single-network uncertainty estimate that matches deep-ensemble quality with virtually no additional training or inference overhead, and demonstrate active learning at order-of-magnitude lower computational cost than an ensemble strategy ([Zhu et al., 2023](https://doi.org/10.1063/5.0136574)). This is an early and influential statement of the "single model ≈ ensemble at much lower cost" idea.
- **Evidential deep learning for interatomic potentials.** Xu et al. develop an evidential formulation for MLIPs and report consistent improvements across datasets without significant computational overhead, with single-model inference ([Xu et al., 2025](https://doi.org/10.1038/s41467-025-67663-y)). It is a useful reference for the single-pass alternative, though not a drop-in replacement for ensemble spread.

The collective message is that the ensemble is a **design choice**, not a necessity. The paper's contribution is to compare three ensemble-based choices under one protocol, while acknowledging that single-model methods could be cheaper and, in some regimes, better.

### 4.4 Design-based uncertainty

Shuang et al. compare ensemble learning with D-optimality in the atomic cluster expansion framework. D-optimality gives more conservative estimates; on heterogeneous data, both approaches underpredict errors, and clustering-enhanced local D-optimality substantially improves novelty detection ([Shuang et al., 2026](https://doi.org/10.1088/2632-2153/ae3d80)). This is a different uncertainty paradigm — design-based rather than parameter-ensemble — but it reinforces the paper's emphasis on data heterogeneity and on evaluating UQ by environment rather than pooling all frames.

---

## 5. What the evidence actually supports about UQ quality

The literature is often summarized as "ensembles work." A more accurate summary is: **ensembles often rank in-domain errors reasonably well, but their calibration and OOD behavior are fragile, and their advantage over cheaper methods is not universal.** The paper should be built around that more defensible claim.

### 5.1 In-domain ranking is the best-case regime

Beck et al. report good correlation between force uncertainty and true errors on custom datasets and MACE-MP-0. Zhu et al. report single-network uncertainty matching deep ensembles. These are positive existence proofs, not general guarantees; the results above are mostly in-domain ranking results rather than interval-coverage tests.

### 5.2 OOD is where the methods separate — and often fail

Kurniawan et al. find uncertainty plateauing or decreasing as error grows OOD. Dale et al. find calibration methods generalize unevenly and calibrated uncertainty often fails to improve AL over random on OOD data. Shuang et al. find heterogeneous data causes underprediction of errors. These are three different failure modes:

1. **Rank failure:** uncertainty stops increasing with error.
2. **Calibration failure:** intervals are too narrow or too wide.
3. **Utility failure:** even a calibrated signal does not select better training data.

The paper should test all three. A method that has a high pooled Spearman but anti-correlates within the interface group is not a good UQ method for this application.

### 5.3 Pooled correlations can hide environment-level failures

If bulk LiF has low error and low spread while the interface has high error and high spread, a pooled Spearman correlation across all frames can look impressive even if within each environment the correlation is weak or negative. This is a general statistical caution, and it is directly relevant to a dataset with bulk LiF, F-doped Li, and interface groups. The paper should report **within-group** discrimination and calibration as primary, with pooled numbers only as a secondary summary.

### 5.4 Calibration is not the same as ranking

A method can rank errors well but produce badly scaled error bars. Individual papers often emphasize one or the other. For risk–coverage and active learning, the scale matters. The paper should separate:

- **discrimination:** does \(\sigma\) order errors correctly?
- **calibration:** does \(\sigma\) have the right magnitude/coverage?
- **utility:** does \(\sigma\) select useful configurations?

This decomposition is more informative than a single "UQ score."

### 5.5 BNNs versus deep ensembles are not settled

Farris et al. provide a systematic comparison of variational Bayesian neural networks and deep ensembles on 7,815 TiO₂ structures, training on the full dataset and a subset to test sensitivity to data representation. They frame the result as practical guidance on strengths and limitations, rather than a single-method verdict ([Farris et al., 2025](https://arxiv.org/abs/2509.19180)). This is the closest methodological sibling to the paper's "which UQ method" question, but it is not foundation MACE and not Li–F. It supports the need for a head-to-head comparison; it does not supply the answer.

### 5.6 Summary of evidence quality

| Claim | Evidence | Confidence |
|---|---|---|
| Independent committees are expensive | Structural; stated by Bi et al., Beck et al. | High |
| Multi-head committees can give useful force UQ at lower cost | Beck et al. on MACE-MP-0; 5% training set | Moderate–high, single group |
| Shallow ensembles can cut training time substantially | Schäfer et al.; up to 96% reduction | Moderate, preprint |
| Snapshot ensembles are a recognized construction | Kurniawan et al.; generic lineage Huang et al. | High that it exists; low that it is validated for foundation MACE |
| Ensemble disagreement often fails OOD | Kurniawan, Dale, Shuang, PROBE | Moderate–high, multiple settings |
| Single-model UQ can match ensembles | Zhu, LLPR, AdaptNTK, e²IP | Moderate, method-specific |
| Better UQ improves active learning | LLPR, AdaptNTK, Beck; counterevidence Dale | Mixed; not established for Li–F environments |
| UQ stays calibrated across Li–F environments | No direct study found | Low / open |

---

## 6. Active learning: label efficiency is not compute efficiency

The active-learning claim is the most demanding part of the argument, because it requires more than a good uncertainty ranking. It requires that the ranking selects configurations that improve the model **per unit cost** relative to random selection.

### 6.1 What the positive evidence shows

- **LLPR** recovers full-data accuracy with a fraction of the labels in molecular, condensed-phase, and electrolyte systems, and approaches the fine-tuning ceiling with fewer labels than random selection ([Bi et al., 2026](https://arxiv.org/abs/2607.14486)).
- **Force-aware NTK / AdaptNTK** achieve strong force-error correlations and competitive or better active learning than committees, with substantial speedups ([Varga-Umbrich et al., 2026](https://arxiv.org/abs/2605.13788); [Ananth & Yue, 2026](https://arxiv.org/abs/2609.00488)).
- **Multi-head committees** condense the MACE-MP-0 training set to about 5% of its original size while maintaining accuracy and reliable uncertainty ([Beck et al., 2025](https://doi.org/10.1063/5.0302097)).
- **Chem. Rev. review.** Kulichenko et al. review data-generation strategies for MLIPs, including active learning and the role of uncertainty in deciding what to label ([Kulichenko et al., 2024](https://doi.org/10.1021/acs.chemrev.4c00572)). This is a useful authoritative context, not a method comparison.

### 6.2 The negative evidence is just as important

Dale et al. find that calibrated uncertainties were generally unsuccessful in reducing the data required to improve over random sampling on OOD data, across ALIGNN, XGBoost, random forests, and neural networks ([Dale et al., 2025](https://arxiv.org/abs/2511.17760)). This is the strongest caution in the retrieval. It means the paper must include a random-acquisition control and must be willing to report a negative result.

There are several reasons AL can fail even with a good ranking:

- **Redundancy:** top-uncertainty frames may be near-duplicates; diversity guards are needed.
- **Label noise / DFT inconsistency:** acquiring a high-uncertainty frame is useless if its label is unreliable.
- **Distribution mismatch:** the candidate pool may not contain the configurations that would improve the target QoI.
- **Cost asymmetry:** a method may need fewer labels but more compute per label, so "labels-to-target" and "compute-to-target" can disagree.
- **Calibration disconnect:** pooled force RMSE can improve while within-environment calibration does not, especially when the environments differ in scale.

The project's protocol already anticipates these issues: same candidate pool, equal batch size, random control, group stratification, diversity gap, locked test, and both labels-to-target and compute-to-target ([`ACTIVE_LEARNING_PROTOCOL.md`](../experiments/exp-04-active-learning/PROTOCOL.md)). The literature supports that design.

### 6.3 What "matched cost" should mean

The paper should report at least three cost measures:

1. **Training cost per round:** wall-clock or GPU-hours for the full ensemble, including replay for multi-head.
2. **Inference cost per acquisition:** cost to score the candidate pool, times \(K\) for committees/snapshots.
3. **Cumulative compute-to-target:** sum of training and inference until the target is reached, not just the final model cost.

A cheap-training method with expensive inference can lose in an AL loop with a large pool. A cheap-inference method with expensive training can lose in the early rounds. The comparison should make this trade-off visible.

---

## 7. Li-ion and LiF application context

The Li-ion literature is not primarily a UQ literature. Its role here is to establish that LiF is a scientifically meaningful testbed, that model choice materially changes LiF behavior, and that ensemble UQ has not been validated across chemically distinct Li–F environments.

### 7.1 LiF interstitial diffusion as physical context

> Transport is not a claimed output of this paper. This section explains why the LiF system is a meaningful testbed and why fine-tuning ensembles cannot capture model-form uncertainty.

Alghamdi et al. compare fine-tuning strategies for MACE on interstitial Li diffusion in LiF. MACE-MPA-0 zero-shot predicts \(E_a = 0.22\) eV; a fine-tuned model with only 300 data points predicts \(E_a = 0.20\) eV; both agree with a DeePMD reference of \(E_a = 0.24\) eV trained on more than 40,000 actively learned points ([Alghamdi et al., 2026](https://doi.org/10.1088/2515-7655/ae6c5a)). This is a strong transfer result and a useful accuracy anchor. Crucially, it is a **force-derived transport** result; it does not test uncertainty calibration, and it does not test whether ensemble disagreement predicts transport error.

De Angelis et al. train an MLIP for large-scale MD of Li diffusion in bulk LiF and identify vacancy, interstitial, and a collective six-ion ring-diffusion mechanism at high interstitial concentration ([De Angelis et al., 2026](https://doi.org/10.1038/s41524-026-02132-8)). This broadens the mechanistic scope: a UQ method that only sees bulk equilibrium frames may miss the high-interstitial configurations that control transport.

### 7.2 Transport is sensitive to model form, not just parameter noise

Li, Gong et al. study Li⁺ conductivity in Li₆PS₅Cl with MLFFs built from semilocal and hybrid DFT data. They average MSD over 300 independent 70 ps trajectories to reduce statistical uncertainty below 1%. Semilocal functionals predict consistently higher diffusion coefficients than hybrid functionals because they underestimate band gaps and migration barriers. Differences between neural-network architectures are of the same order of magnitude as differences between functionals ([Li et al., 2025](https://arxiv.org/abs/2512.11650)).

This is a crucial caveat for the paper: an ensemble of fine-tunes of one MACE checkpoint captures neither XC-functional uncertainty nor architecture uncertainty. If the paper claims "UQ for Li transport," it should specify that it is **within-model epistemic uncertainty**, and it should discuss functional/architecture uncertainty as a limitation. It can also use the Li, Gong et al. result to justify multiple MD replicas: transport averages need many independent trajectories, not just many ensemble members.

### 7.3 Universal MLIPs are accurate enough to be useful, not accurate enough to be trusted blindly

Shaaban Kabakibo et al. benchmark MD approaches for ionic conductivity in 21 solid Li electrolytes, comparing DFT-driven and MACE-driven simulations. They find comparable performance, with MACE more than 350× faster than DFT on one GPU versus a 64-CPU node ([Shaaban Kabakibo et al., 2026](https://arxiv.org/abs/2603.28012)). This supports the practical value of foundation MLIPs for Li transport.

Guo et al. benchmark M3GNet, CHGNet, MACE, SevenNet, GRACE, and Orb against DFT for alkali-ion battery kinetics. Orb-v3 excels at static migration barriers (MAE ≈ 75–111 meV), while GRACE trained on OMat24 reproduces ion diffusivities and structural correlations better. They conclude that high-temperature, non-equilibrium training data is the dominant driver of kinetic accuracy, more than architectural sophistication ([Guo et al., 2026](https://arxiv.org/abs/2601.10938)).

These papers reinforce two points:

- Foundation MLIPs are good enough that UQ is the remaining bottleneck for deployment.
- The dominant error source may be the training distribution, not the ensemble construction. A UQ method that only measures parameter disagreement may be blind to the most important failure mode.

### 7.4 SEI relevance

Zhong and Persson use diffusion-based structure prediction and MLIPs to study amorphous lithium fluorophosphate phases in the SEI, reporting a projected room-temperature conductivity of about \(0.18\) mS cm⁻¹ and \(E_a \approx 0.40\) eV for the amorphous phase ([Zhong & Persson, 2025](https://doi.org/10.1021/acsenergylett.5c03526)). This is application context, not a UQ method paper, but it shows why Li–F–P–O–F disorder and interfaces matter for transport. The project's LiF/F-doped-Li/interface scope is a simplified but defensible slice of that problem.

### 7.5 Why LiF is a good UQ testbed

LiF offers a useful combination:

- a well-defined defect chemistry with a literature reference \(E_a\);
- bulk LiF, F-doped Li, and interface environments that create a natural in-domain/OOD axis;
- a literature anchor (Alghamdi et al.) for accuracy, not just uncertainty;
- a physically interpretable static sanity check (defect formation energies) that does not require transport.

The gap is that no retrieved paper tests whether ensemble uncertainty **ranks** error within each Li–F environment or **improves** held-out accuracy per label. That is the application-level contribution.

---

## 8. Synthesis: the gap and the paper's position

### 8.1 What is established

| Finding | Representative evidence |
|---|---|
| Independent committees are the default but cost \(K\) fine-tunes and \(K\) forward passes | Bi et al.; Beck et al. |
| Multi-head committees can provide useful force UQ at lower cost | Beck et al.; 5% training set on MACE-MP-0 |
| Shallow/weight-sharing ensembles can cut training time substantially, but need a probabilistic force objective | Schäfer et al.; up to 96% reduction |
| Snapshot ensembles are a recognized generic construction | Huang et al.; Kurniawan et al. |
| Ensemble disagreement often fails OOD, and calibration does not guarantee AL gains | Kurniawan; Dale; Shuang; PROBE |
| Single-model alternatives can match ensembles at much lower cost | Zhu; LLPR; AdaptNTK; e²IP |
| LiF interstitial transport is a credible foundation-MLIP application | Alghamdi et al.; De Angelis et al. |
| Transport is sensitive to XC functional and architecture, not just seed | Li, Gong et al.; Guo et al. |

### 8.2 What is not established in the retrieved set

1. **No matched comparison for foundation MACE on Li–F.** The retrieved literature compares some ensemble methods on some systems, but not independent committees vs warm-restart snapshots vs the two multi-head variants (MH-FT and MHC) under one checkpoint, one data budget, one ensemble size, and one cost accounting.
2. **No dedicated evidence for warm-restart snapshots in foundation MLIPs.** Snapshot ensembles appear as one arm in a broader benchmark, and shallow ensembles are a different construction. The specific cost/calibration claim for warm snapshots on MACE is open.
3. **No environment-stratified calibration benchmark.** UQ papers evaluate energies, forces, or generic OOD detection. They rarely report whether calibration and risk–coverage hold separately for bulk, doped-metal, and interface environments.
4. **No Li–F active-learning comparison against random with honest compute accounting.** LLPR and NTK papers are strong precedents, but they are not Li–F environment studies. Beck et al. demonstrate AL for MACE-MP-0, but not across a bulk/doped/interface split.
5. **No separation of parameter uncertainty from model-form uncertainty.** Fine-tuning ensembles cannot see XC-functional or architecture uncertainty, which the Li-ion literature shows can be comparable to ensemble spread.

### 8.3 The gap statement

> In the retrieved literature, no study compares independent fine-tuning committees, warm-restart snapshot ensembles, and the two multi-head variants (fine-tuning with replay and frozen-body committees) under a common foundation MACE checkpoint, matched ensemble size, and matched compute accounting, then tests whether their uncertainty ranks force errors within Li–F environments and improves active learning for held-out force error over random acquisition.

This is a **bounded-search gap claim**, not a claim of absolute novelty. The paper should phrase it as "we found no study that..." and should cite the nearest neighbors explicitly: Beck et al. for multi-head, Schäfer et al. for shallow ensembles, Kurniawan et al. for snapshot/ensemble benchmarks, Bi et al. and Ananth & Yue for cheap AL, and Alghamdi et al. for LiF transport.

### 8.4 Contributions the paper can defend

1. **A matched comparison of three ensemble families (four arms)** at the same foundation checkpoint, data budget, ensemble size, and evaluation protocol. The multi-head family is split into MH-FT (fine-tuned body + replay) and MHC (frozen body + disjoint heads) because they are different methods, not two names for one baseline.
2. **Environment-resolved UQ diagnostics** — within-group discrimination, calibration, and risk–coverage for bulk LiF, F-doped Li, and the thin LiF/Li interface film, rather than pooled correlations.
3. **A distribution-shift test** — whether calibration fitted in bulk LiF survives F-doped Li and the interface film, rather than only pooled force RMSE.
4. **A cost-aware active-learning test** — labels-to-target and compute-to-target against a random control, with training and inference costs separated.
5. **A negative-result-ready claim** — if cheap ensembles do not beat random or do not stay calibrated across environments, the paper reports that rather than forcing a "snapshots win" narrative.

The headline is the label/compute cost of reaching a held-out force-error target, not a leaderboard over every UQ method.

### 8.5 What would falsify the paper's claims

| Outcome | Interpretation |
|---|---|
| Warm snapshots match the committee on within-group discrimination and calibration at substantially lower compute cost (≤1/2, ideally ≤1/3) | Supports the cost-efficiency hypothesis. |
| Warm snapshots are cheaper but anti-correlate or collapse OOD | Cheap but not useful UQ; report as a negative result. |
| A multi-head arm dominates on both discrimination and compute | The snapshot route is not justified; report which variant (MH-FT or MHC) dominates and at what cost. |
| All three are ≈ random in AL | Uncertainty is not the bottleneck; reframe the paper around calibration diagnostics, not AL. |
| UQ ranks pooled error but fails within an environment | Pooled discrimination is misleading; report environment-stratified results. |

---

## 9. Hypotheses and how they map to the literature

| Hypothesis | Literature basis | Test in this paper | Falsified if |
|---|---|---|---|
| **H1. Better calibration implies better label efficiency.** | LLPR, AdaptNTK, Beck et al. show UQ-guided AL can beat random; Dale et al. show it often does not. | Compare labels-to-target across methods and random, with calibration on validation only. | A better-calibrated method needs more labels than random, or calibration does not predict AL gain. |
| **H2. Warm snapshots are the cost-effective choice.** | One fine-tune vs \(K\); snapshot ensembles are a recognized construction; no direct foundation-MLIP evidence. | Compute-to-target and training/inference cost at matched \(K\). | Snapshots need >1.5× the committee's labels or >1/2 its compute, or their UQ is unusable. |
| **H3. Methods with intact OOD uncertainty acquire interface frames earlier.** | Kurniawan et al. show OOD collapse; PROBE shows disagreement can be weak; Shuang et al. show heterogeneity hurts calibration. | Track acquired-group composition over rounds; evaluate interface separately. | A method's uncertainty collapses on interface frames and it does not acquire them preferentially. |
| **H0. No method beats random.** | Dale et al. provide direct counterevidence for AL gains. | Random control at equal label budget. | Any method shows a reproducible label/compute advantage over random. |

The hypotheses are deliberately separable. H1 is about calibration utility, H2 about cost, H3 about OOD awareness, and H0 is the null that protects against over-interpreting a pretty uncertainty plot.

---

## 10. Reference list

References are grouped by role. The first group is the core comparison set most directly relevant to the paper's comparison; the second group covers single-model alternatives and UQ quality; the third covers active learning and application context; the fourth is method lineage. Links are to the canonical arXiv or DOI record.

### 10.1 Core comparison set

1. **Beck, Simko, Schaaf, Marsalek, Schran (2025).** *Multi-head committees enable direct uncertainty prediction for atomistic foundation models.* J. Chem. Phys. 163, 234103. [DOI](https://doi.org/10.1063/5.0302097) · [arXiv:2508.09907](https://arxiv.org/abs/2508.09907). Shared-backbone multi-head committee on MACE-MP-0; force UQ correlates with error; 5% training set in AL. **Direct source of the MHC arm (Experiment 3); MH-FT uses the released MACE fine-tuning path instead.**
2. **Schäfer, Kellner, Kästner, Ceriotti (2026).** *How to Train a Shallow Ensemble.* [arXiv:2602.15747](https://arxiv.org/abs/2602.15747). Weight-sharing shallow ensembles; NLL force loss needed for calibration; full-model fine-tuning of a probabilistic shallow ensemble cuts training time by up to 96%. **Closest cost–accuracy evidence, but not warm snapshots.**
3. **Kurniawan, Wen, Tadmor, Transtrum (2026).** *Comparative study of ensemble-based uncertainty quantification methods for neural network interatomic potentials.* Mach. Learn.: Sci. Technol. 7, 055018. [DOI](https://doi.org/10.1088/2632-2153/ae9fb4) · [arXiv:2508.06456](https://arxiv.org/abs/2508.06456). Benchmarks bootstrap, dropout, random-init, and snapshot ensembles on carbon allotropes; uncertainty can plateau or decrease OOD. **Direct precedent for snapshot ensembles and the OOD warning.**
4. **Bi, Wang, Cheng (2026).** *Full-data accuracy with fewer labels for training and fine-tuning machine-learning force fields.* [arXiv:2607.14486](https://arxiv.org/abs/2607.14486). LLPR, a forward-pass-cheap per-configuration UQ estimator; committee UQ called impractical for foundation MLFFs; electrolyte AL. **Direct competitor for cheap AL.**
5. **Mehdi, Cho, Isayev (2026).** *Knowing when to trust machine-learned interatomic potentials.* [arXiv:2605.00640](https://arxiv.org/abs/2605.00640). PROBE: frozen-representation classifier for per-prediction reliability; ensemble disagreement scales poorly and correlates weakly with error. **Strongest critique of committee UQ at foundation scale.**
6. **Varga-Umbrich, Weller-Davies, Duckworth, Tilly, Peltre, Surana (2026).** *Force-Aware Neural Tangent Kernels for Scalable and Robust Active Learning of MLIPs.* [arXiv:2605.13788](https://arxiv.org/abs/2605.13788). Force/joint energy–force NTKs; scalable shortlisting; competitive with committees at lower cost; committees higher variance under pool shift. **Single-model AL competitor.**
7. **Farris, Telari, Artrith, Neyman, Bruix (2025).** *Bayesian Neural Networks versus deep ensembles for uncertainty quantification in machine learning interatomic potentials.* [arXiv:2509.19180](https://arxiv.org/abs/2509.19180). Variational BNN vs deep ensembles on 7,815 TiO₂ structures, full and subset training. **Closest methodological sibling; not MACE or Li–F.**

### 10.2 Single-model alternatives and UQ quality

8. **Ananth & Yue (2026).** *AdaptNTK: Adaptive Uncertainty Quantification and Active Learning for Neural Network Potentials.* [arXiv:2609.00488](https://arxiv.org/abs/2609.00488). Regularized Mahalanobis distance in empirical NTK space; recursive batch updates; Spearman 0.68 / Pearson 0.71 on rMD17; matches 3-member ensemble; 2.6× speedup per Transition-1X cycle.
9. **Wang, Cui, Zou, Zhang, Yan, Ouyang, Tan, Su (2026).** *Equivariant Evidential Deep Learning for Interatomic Potentials.* [arXiv:2602.10419](https://arxiv.org/abs/2602.10419). e²IP: equivariant full-covariance force uncertainty; single-model inference; stronger accuracy–efficiency–reliability balance than evidential and ensemble baselines.
10. **Zaghen, Zhdanov, Coscia, Wessels, Bekkers (2026).** *Uncertainty-aware Machine Learning Interatomic Potentials via Learned Functional Perturbations.* [arXiv:2605.19939](https://arxiv.org/abs/2605.19939). CRPS fine-tuning; P-EGNN and P-Orb; 19–32% CRPS improvement over BLIP; silica Spearman 0.75 → 0.84.
11. **Fonea, Peles, Niv, Gordon, Natan (2025).** *Black-Box Uncertainty Estimation for Deep Learning Models in Atomistic Simulations.* [arXiv:2511.16439](https://arxiv.org/abs/2511.16439). Extracts an ensemble signal from an existing multi-headed network without architecture change; Na/Al; heteroscedastic scaling; OOD detection.
12. **Zhu, Batzner, Musaelian, Kozinsky (2023).** *Fast uncertainty estimates in deep learning interatomic potentials.* J. Chem. Phys. 158, 164111. [DOI](https://doi.org/10.1063/5.0136574). Single-network UQ matching deep ensembles at order-of-magnitude lower cost; AL demonstration. **Classic cost-floor reference.**
13. **Shuang, Wei, Liu, Gao, Dey (2026).** *Model accuracy and data heterogeneity shape uncertainty quantification in machine learning interatomic potentials.* Mach. Learn.: Sci. Technol. 7, 025002. [DOI](https://doi.org/10.1088/2632-2153/ae3d80). Ensemble vs D-optimality in ACE; calibration degrades on heterogeneous data; clustering-enhanced local D-optimality.
14. **Xu, Cui, Tang, Ma, Zhou, Li, Gao, Gong, Ouyang, Zhang, Su (2025).** *Evidential deep learning for interatomic potentials.* Nat. Commun. 17, 937. [DOI](https://doi.org/10.1038/s41467-025-67663-y). Single-model evidential UQ for MLIPs; consistent improvements across datasets without significant overhead; single-pass inference.

### 10.3 Active learning, critiques, and application context

15. **Kulichenko, Nebgen, Lubbers, Smith, Barros, Allen, Habib, Shinkle, Fedik, Li, Messerly, Tretiak (2024).** *Data Generation for Machine Learning Interatomic Potentials and Beyond.* Chem. Rev. 124, 13681–13714. [DOI](https://doi.org/10.1021/acs.chemrev.4c00572). Review of data-generation strategies, including active learning and UQ for MLIP training.
16. **Dale, Li, DeCost, Wan, Han, Fehlis, Hattrick-Simpers (2025).** *When Active Learning Fails, Uncalibrated Out of Distribution Uncertainty Quantification Might Be the Problem.* [arXiv:2511.17760](https://arxiv.org/abs/2511.17760). Calibration methods generalize unevenly; calibrated UQ often fails to beat random AL on OOD data across ALIGNN, XGBoost, RF, and NN. **Essential negative control.**
17. **Vieira, Ribeiro, Borovitskiy (2026).** *Do Deep Ensembles Actually Capture Uncertainty in Graph Neural Networks?* [arXiv:2605.22593](https://arxiv.org/abs/2605.22593). Eight graph datasets; marginal ensemble gains; epistemic collapse from shared message passing. **Warning about correlated members; not MLIP-specific.**
18. **Alghamdi, de Angelis, Asinari, Chiavazzo (2026).** *Comparing fine-tuning strategies for machine learning force fields in lithium-ion diffusion.* J. Phys. Energy 8, 025013. [DOI](https://doi.org/10.1088/2515-7655/ae6c5a) · [arXiv:2510.05020](https://arxiv.org/abs/2510.05020). MACE-MPA-0 zero-shot \(E_a = 0.22\) eV; 300-point fine-tune \(E_a = 0.20\) eV; DeePMD reference 0.24 eV. **Accuracy anchor for LiF interstitial transport; no UQ.**
19. **De Angelis, Raucci, Mambretti, Fasano, Chiavazzo, Asinari, Parrinello (2026).** *Exploring lithium diffusion in LiF with machine learning potentials: from point defects to collective ring diffusion.* npj Comput. Mater. 12, 213. [DOI](https://doi.org/10.1038/s41524-026-02132-8). Vacancy/interstitial mechanisms plus collective six-ion ring diffusion at high interstitial concentration. **Mechanistic context.**
20. **Shaaban Kabakibo, Therrien, Bengio, Côté, Guo, Shin, Hernandez-Garcia (2026).** *A Comparative Study of Molecular Dynamics Approaches for Simulating Ionic Conductivity in Solid Lithium Electrolytes.* [arXiv:2603.28012](https://arxiv.org/abs/2603.28012). 21 solid electrolytes; DFT vs MACE; comparable performance, >350× faster on one GPU. **Application scale.**
21. **Guo, Gui, Wang (2026).** *Are Universal Potentials Ready for Alkali-Ion Battery Kinetics?* [arXiv:2601.10938](https://arxiv.org/abs/2601.10938). Benchmarks M3GNet, CHGNet, MACE, SevenNet, GRACE, Orb on cathodes/solid electrolytes; Orb-v3 static barriers, GRACE dynamics; training-data coverage dominates. **Model-form context.**
22. **Li, Gong, Xiao, Ren (2025).** *Influence of Exchange-Correlation Functionals and Neural Network Architectures on Li⁺-Ion Conductivity in Solid-State Electrolyte from Molecular Dynamics Simulations with Machine-Learning Force Fields.* [arXiv:2512.11650](https://arxiv.org/abs/2512.11650). Li₆PS₅Cl; semilocal vs hybrid functionals; 300 × 70 ps trajectories; architecture differences same order as functional differences. **Key limitation for ensemble-only UQ.**
23. **Zhong & Persson (2025).** *Machine-Learning-Guided Insights into Solid-Electrolyte Interphase Conductivity: Are Amorphous Lithium Fluorophosphates the Key?* ACS Energy Lett. 11, 806–812. [DOI](https://doi.org/10.1021/acsenergylett.5c03526). Amorphous Li–P–O–F phases; projected \(\sigma \approx 0.18\) mS cm⁻¹ and \(E_a \approx 0.40\) eV. **SEI context.**

### 10.4 Method lineage

24. **Batatia et al. (2023).** *A foundation model for atomistic materials chemistry.* [arXiv:2401.00096](https://arxiv.org/abs/2401.00096). MACE foundation model and MPtrj training set; background for the MACE checkpoint used here.
25. **Huang, Li, Pleiss, Liu, Hopcroft, Weinberger (2017).** *Snapshot Ensembles: Train 1, get M for free.* [arXiv:1704.00109](https://arxiv.org/abs/1704.00109). Cyclic-learning-rate snapshots as a cheap ensemble; generic deep-learning origin of the warm-restart idea.
26. **Lakshminarayanan, Pritzel, Blundell (2017).** *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* [arXiv:1612.01474](https://arxiv.org/abs/1612.01474). Deep ensembles as a practical predictive-uncertainty baseline.

### 10.5 Added 2026-11-23 (design C)

Identifiers were checked on 2026-11-23 against alphaXiv (arXiv papers) and OpenAlex (DOIs). The "Read" note says how much of each paper was read.

27. **Schmid, Setzer, Scheurer (2026).** *Towards Out-of-Distribution Uncertainty Quantification for Machine-Learned Interatomic Potentials via Pre-Training.* [arXiv:2610.41801](https://www.alphaxiv.org/abs/2610.41801). Post-hoc uncertainty estimators (latent-space distance, mean-variance estimation, shallow ensembles) are pre-trained on large materials datasets. They give zero-shot uncertainty across datasets, and fine-tuning yields positive uncertainty–error correlations even on the hardest sets. Read: abstract and introduction. **Evidence that out-of-distribution evaluation is the open gap.**
28. **Bigi, Chong, Ceriotti, Grasselli (2024).** *A prediction rigidity formalism for low-cost uncertainties in trained neural networks.* [arXiv:2403.02251](https://www.alphaxiv.org/abs/2403.02251). The paper defines prediction rigidities from a constrained optimisation, connects them to Bayesian inference, and gives a last-layer approximation that needs no change to the network or its training. Read: abstract. **Method behind the `llpr` arm.**
29. **Andersen (1980).** *Molecular dynamics simulations at constant pressure and/or temperature.* J. Chem. Phys. [DOI:10.1063/1.439486](https://doi.org/10.1063/1.439486). Stochastic-collision thermostat. The existing NVT runs used it with a collision for every atom at every step. **Lineage of the thermostat finding.**
30. **Bussi, Donadio, Parrinello (2007).** *Canonical sampling through velocity rescaling.* J. Chem. Phys. [DOI:10.1063/1.2408420](https://doi.org/10.1063/1.2408420). The stochastic velocity-rescaling (SVR) thermostat, one of the two candidates in the QE pilot.
31. **Madras, Sokal (1988).** *The pivot algorithm: A highly efficient Monte Carlo method for the self-avoiding walk.* J. Stat. Phys. [DOI:10.1007/BF01022990](https://doi.org/10.1007/BF01022990). Standard source for the self-consistent window of the integrated autocorrelation time and its statistical error, both used in `estimate_qe_correlation.py`.
32. **Flyvbjerg, Petersen (1989).** *Error estimates on averages of correlated data.* J. Chem. Phys. [DOI:10.1063/1.457480](https://doi.org/10.1063/1.457480). Blocking analysis, used as a cross-check in the correlation review.
33. **Chodera (2016).** *A simple method for automated equilibration detection in molecular simulations.* J. Chem. Theory Comput. [DOI:10.1021/acs.jctc.5b00784](https://doi.org/10.1021/acs.jctc.5b00784). Picks the burn-in that maximises the number of effective samples; applied to the interface run, where no start point gave a stationary segment.
### 10.6 Diversity, correlation and theory

34. **Krogh & Vedelsby (1995).** *Neural Network Ensembles, Cross Validation, and Active Learning.* Advances in Neural Information Processing Systems 7. Ambiguity decomposition: ensemble error equals mean member error minus diversity. **Source of the identity in §11.1.**
35. **Brown, Wyatt, Harris, Yao (2005).** *Diversity creation methods: a survey and categorisation.* Information Fusion 6(1), 5–20. [DOI](https://doi.org/10.1016/j.inffus.2004.04.004). Taxonomy of data, parameter and architecture manipulations that create ensemble diversity.
36. **Fort, Hu, Lakshminarayanan (2019).** *Deep Ensembles: A Loss Landscape Perspective.* [arXiv:1912.02757](https://arxiv.org/abs/1912.02757). Random initialization explores distinct modes; functions along one trajectory cluster in prediction space; cyclic shared-initialization sampling is less diverse than independent runs. **Mechanism for committee collapse.**
37. **Neyshabur, Sedghi, Zhang (2020).** *What is being transferred in transfer learning?* [arXiv:2008.11687](https://arxiv.org/abs/2008.11687). Models fine-tuned from one checkpoint stay in the same basin, show no loss barrier, and make similar mistakes. **Direct evidence that shared initialization persists.**
38. **Garipov, Izmailov, Podoprikhin, Vetrov, Wilson (2018).** *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs.* [arXiv:1802.10026](https://arxiv.org/abs/1802.10026). Low-loss paths connect distinct minima; fast geometric ensembling traverses them at single-run cost.
39. **Izmailov, Podoprikhin, Garipov, Vetrov, Wilson (2018).** *Averaging Weights Leads to Wider Optima and Better Generalization.* [arXiv:1803.05407](https://arxiv.org/abs/1803.05407). Weight averaging stays inside one basin; a better point estimate, not a committee.
40. **Frankle, Dziugaite, Roy, Carbin (2020).** *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* [arXiv:1912.05671](https://arxiv.org/abs/1912.05671). Fine-tuned solutions are linearly connected; underpins the same-basin argument.
41. **Wortsman et al. (2022).** *Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy without Increasing Inference Time.* [arXiv:2203.05482](https://arxiv.org/abs/2203.05482). Averaging independently fine-tuned models from one initialization improves accuracy without a barrier.
42. **Ainsworth, Hayase, Srinivasa (2023).** *Git Re-Basin: Merging Models modulo Permutation Symmetries.* [arXiv:2209.04836](https://arxiv.org/abs/2209.04836). Permutation symmetries map distinct parameters to identical functions. **Weight-space distance is not functional diversity.**
43. **Ovadia et al. (2019).** *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.* [arXiv:1906.02530](https://arxiv.org/abs/1906.02530). Deep ensembles are the strongest baseline under shift, while all methods degrade.
44. **Guo, Pleiss, Sun, Weinberger (2017).** *On Calibration of Modern Neural Networks.* [arXiv:1706.04599](https://arxiv.org/abs/1706.04599). Calibration is a property separate from accuracy and degrades with capacity and shift.
45. **Wilson & Izmailov (2020).** *Bayesian Deep Learning and a Probabilistic Perspective on Generalization.* [arXiv:2002.08791](https://arxiv.org/abs/2002.08791). Deep ensembles as an approximate Bayesian model average over posterior modes.
46. **Gal & Ghahramani (2016).** *Dropout as a Bayesian Approximation.* [arXiv:1506.02142](https://arxiv.org/abs/1506.02142). The implicit-ensemble analogy Beck et al. invoke for the shared-body committee.
47. **Carrete, Montes-Campos, Wanzenböck, Heid, Madsen (2023).** *Deep ensembles vs committees for uncertainty estimation in neural-network force fields.* J. Chem. Phys. 158, 204801. [DOI](https://doi.org/10.1063/5.0146905) · [arXiv:2302.08805](https://arxiv.org/abs/2302.08805). Committee matches bootstrap as an error proxy (Spearman 0.90 vs 0.91, SrTiO₃) at far lower cost; descriptors computed once, so inference overhead is small.
48. **Lu, Ghiringhelli, Carbogno, Wang, Scheffler (2023).** *On the Uncertainty Estimates of Equivariant-Neural-Network-Ensembles Interatomic Potentials.* [arXiv:2309.00195](https://arxiv.org/abs/2309.00195). Ensemble uncertainty is overconfident and can move opposite to error; constant rescaling cannot repair it. **Equivariant-MLIP evidence for common-mode blindness.**
49. **Kahle & Zipoli (2022).** *Quality of uncertainty estimates from neural network potential ensembles.* Phys. Rev. E 105, 015311. [DOI](https://doi.org/10.1103/PhysRevE.105.015311). Common-bias problem of committee members; cited by Beck et al. as the reason to decorrelate heads.
50. **Tan, Urata, Goldman, Dietschreit, Gómez-Bombarelli (2023).** *Single-model uncertainty quantification in neural network potentials does not consistently outperform model ensembles.* [arXiv:2305.01754](https://arxiv.org/abs/2305.01754). The committee remains the reference; cheap single-model UQ closes the cost gap but not the quality gap.
51. **Zamyatin, Indri, Malhotra, Gärtner (2026).** *Is BatchEnsemble a Single Model? On Calibration and Diversity of Efficient Ensembles.* [arXiv:2601.16936](https://arxiv.org/abs/2601.16936). Shared-backbone efficient ensembles reach a measure-zero subset of parameter space, are near-identical in function space, and have JSD 0.002 vs 0.037 for a deep ensemble.
52. **Jinnouchi, Karsai, Kresse (2019).** *On-the-fly machine learning force field generation: Application to melting points.* Phys. Rev. B 100, 014105. [DOI](https://doi.org/10.1103/PhysRevB.100.014105). Canonical online on-the-fly AL loop; contrast for the retrospective design.
53. **Vandermause et al. (2020).** *On-the-fly active learning of interpretable Bayesian force fields for atomistic rare events.* npj Comput. Mater. 6, 20. [DOI](https://doi.org/10.1038/s41524-020-0283-z). Online Bayesian AL; second online reference for the scope caveat.

### 10.7 Added 2026-11-27 (committee versus snapshot diversity)

Both were found by retrieval on 2026-11-27 and read in full text.

54. **Sadrtdinov, Pozdeev, Vetrov, Lobacheva (2023).** *To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning.* NeurIPS 2023. [arXiv:2303.03374](https://www.alphaxiv.org/abs/2303.03374). Models fine-tuned from one pre-trained checkpoint form a "local" deep ensemble in one basin; a snapshot ensemble with a tuned cyclic schedule matches or beats it but does not reach an ensemble fine-tuned from different pre-trained checkpoints. Image classification (ResNet-50, CLIP), not MLIPs. **Main source for §14.2.**
55. **Ashukha, Lyzhov, Molchanov, Vetrov (2020).** *Pitfalls of In-Domain Uncertainty Estimation and Ensembling in Deep Learning.* ICLR 2020. [arXiv:2002.06470](https://www.alphaxiv.org/abs/2002.06470). Defines the deep ensemble equivalent (DEE): the number of independently trained networks that match a method's calibrated log-likelihood. Trained from scratch, snapshot ensembles come second only to deep ensembles. **The from-scratch baseline that §14.2 contrasts with.**

---

## 11. Verification and scope notes

These notes record where the retrieved set does **not** support a convenient claim, so the paper does not over-cite.

- **"Warm snapshots" is not an established MLIP term.** The closest MLIP evidence is snapshot ensembles as one arm in Kurniawan et al., not a dedicated foundation-MACE method. The paper should define the term and present it as a hypothesis, not as prior art.
- **Shallow ensembles are not warm snapshots.** Schäfer et al. share weights across members and train with a probabilistic objective; the optimization trajectory is not the source of diversity. Cite it for the cost–calibration trade-off, not as direct validation of cycle-end checkpoints.
- **LLPR is electrolyte-adjacent, not LiF.** Bi et al. demonstrate electrolyte fine-tuning and Li–O coordination; they do not study LiF interstitial diffusion. Cite it for the AL workflow and cost argument, not as a LiF result.
- **PROBE is not a periodic-LiF force-UQ study.** Mehdi et al. evaluate per-molecule reliability on molecular benchmarks. It is a strong critique of ensemble disagreement, but it does not establish that a representation classifier would work for LiF force errors.
- **Farris et al. is TiO₂, not MACE.** The BNN-vs-ensemble comparison is a methodological sibling, not a foundation-MACE result.
- **The black-box multi-head paper is not a new committee training method.** Fonea et al. extract a signal from an existing multi-head network; it is relevant to inference cost, not to the training-cost comparison.
- **Li-ion application papers are not UQ papers.** Alghamdi et al., De Angelis et al., Shaaban Kabakibo et al., Guo et al., Li, Gong et al., and Zhong & Persson establish accuracy, mechanisms, and model-form sensitivity. They do not test ensemble calibration or AL label efficiency.
- **Partial full-text audit now done (2026-11-07).** For the theory addendum the full text of Fort et al. (2019), Neyshabur et al. (2020), Huang et al. (2017), Beck et al. (2025), Kurniawan et al. (2026), Carrete et al. (2023), Lu et al. (2023) and Zamyatin et al. (2026) was read, and the numbers quoted in §13 come from those texts. The remaining §10.6 entries are cited at metadata level and are marked as such in the §13.8 ledger.
- **Still missing a claim-level audit for the core cost comparison.** Schäfer et al. and Bi et al. remain abstract-level; reading their full text is the next step if the paper needs exact cost and calibration numbers for the shallow-ensemble and LLPR arms.

---

## 12. Update 2026-11-23: what design C changes

**What changed.**
- The fine-tuned arms are trained on bulk LiF only, then calibrated and tested on independent bulk replicas.
- They are evaluated along a graded shift: variable-cell bulk, then F-doped Li and Li metal, then the LiF/Li film, with the boron systems in an appendix.
- LLPR and an NLL-trained shallow ensemble join the comparison as baselines.
- Labels are 40/320 Ry only ([ADR-0007](../decisions/ADR-0007-bulk-trained-shift-ladder.md), [ADR-0008](../decisions/ADR-0008-40ry-labels-and-md-protocol.md)).

**What the new references change.**

- **The gap moves to distribution shift.** Schmid et al. observe that "for a vast majority of cases, post-hoc UQ models are trained and evaluated on specific (mostly in-domain) datasets, which also often consist of only thousands of training examples (see Figure 1)." ([Schmid et al., p. 3](https://www.alphaxiv.org/pdf/2610.41801?page=3)). The gap statement in §8.3 narrows accordingly: we found no study that compares ensemble and last-layer UQ for a fine-tuned foundation MACE model along a graded distribution shift, with calibration and test sets drawn from independent trajectories.
- **Shallow ensembles and LLPR become arms, not only related work.** §3.3 and §4.1 cite them as neighbours; they are now implemented as `shallow-nll` (ref. 2) and `llpr` (refs. 4 and 28).
  - The frozen-body MHC has the same structure as a shallow ensemble: a frozen body with separate heads.
  - It differs in training each head on a disjoint quarter of the data with MSE, instead of all heads jointly with an NLL. Comparing the two arms isolates those two choices.
  - The caution in §11 still holds: neither is evidence about warm-restart snapshots.
- **Independence is now a requirement reviewers will check.** The data decisions (ADR-0003, ADR-0005, ADR-0008) rest on refs. 29–33:
  - the integrated autocorrelation time and its error (Madras & Sokal);
  - blocking analysis (Flyvbjerg & Petersen);
  - equilibration detection (Chodera);
  - the two thermostats (Andersen; Bussi et al.).

  The project's own analysis is [methods/autocorrelation.md](../methods/autocorrelation.md).

**Found, not yet read.** These are titles only; do not cite them before reading. Each note says why the paper may matter, judged from its title.

- Lysogorskiy et al., *A latent-space extrapolation grade built into graph atomic cluster expansion foundation potentials*, [arXiv:2609.40060](https://www.alphaxiv.org/abs/2609.40060). An out-of-distribution signal built into a foundation potential.
- Lortaraprasert et al., *Does System Size Matter in Uncertainty Quantification for Machine Learning Interatomic Potentials?*, [arXiv:2610.40126](https://www.alphaxiv.org/abs/2610.40126). Our evaluation sets range from 18 to 122 atoms.
- Su et al., *EPIC: Differentiable Energy-Force Consistent Uncertainty Quantification and Propagation for Machine-Learned Interatomic Potentials*, [arXiv:2611.00942](https://www.alphaxiv.org/abs/2611.00942). Another UQ method a reviewer may ask about.
- Fisher et al., *Multivariate conformal uncertainty propagation in multitask atomistic simulation: Successes and pitfalls*, [arXiv:2609.31384](https://www.alphaxiv.org/abs/2609.31384). Conformal calibration, which our evaluation uses.
- Stepurko et al., *Out-of-training properties in frozen interatomic potential embeddings are decodable and ordered by physical scale*, [arXiv:2611.21116](https://www.alphaxiv.org/abs/2611.21116). Frozen-body representations, as in the MHC and probe arms.
- Wong & Yang, *Bias in Universal Machine-Learned Interatomic Potentials and its Effects on Fine-Tuning*, [arXiv:2603.10159](https://www.alphaxiv.org/abs/2603.10159). Fine-tuning from a biased foundation model.

---

## 13. Theory addendum (2026-11): why a spread measures only what members do differently

This section records the theoretical literature added after the main retrieval. Its purpose is narrow: to give the paper a *mechanism* for the correlation, calibration and out-of-distribution failures it reports, rather than treating them as empirical surprises. The central results are two identities and one consequence; everything else is evidence for them.

### 11.1 The spread identity, and invariance to common error

Let \(y(x)\) be the reference and \(f_1,\dots,f_M\) the member predictions, with spread \(s^2 = \frac{1}{M}\sum_m \lVert f_m - \bar f\rVert^2\). Then, exactly, \( \frac{1}{M}\sum_m \lVert f_m - y\rVert^2 = \lVert \bar f - y\rVert^2 + s^2 \). The left side is the mean member error and the first right-hand term is the ensemble error, so \(s^2\) equals the error that averaging removes — the ambiguity decomposition of Krogh and Vedelsby (Krogh & Vedelsby, 1995, NIPS 7). The consequence the paper needs is an invariance: if every member shares an error component \(b\), so that \(e_m = b + u_m\), then \(s^2\) depends only on the \(u_m\) and is **independent of \(b\)**. A shared foundation checkpoint, a frozen body, and a fixed DFT dataset are all shared error channels, and no committee can see them. This is the formal statement behind §2.3 and §5.3.

### 11.2 One correlation parameter governs the committee

For exchangeable members with error variance \(\sigma^2\) and pairwise error correlation \(\rho\), \( \mathbb{E}[s^2] = \sigma^2(1-\rho) \) and \( \mathrm{Var}(\bar f) = \sigma^2(\rho + (1-\rho)/M) \). Both the uncertainty signal and the accuracy advantage of averaging are proportional to \(1-\rho\), and both vanish as \(\rho \to 1\); increasing \(M\) cannot go below the floor \(\rho\sigma^2\). Three consequences for the paper:

- "More seeds" is not a fix. \(\rho\), not \(K\), governs the signal, and \(\rho\) is set by the training protocol.
- A committee whose members collapse is simultaneously a bad ensemble and a blind sensor; the two failures are one failure.
- The quantity to report as the mechanistic covariate is \(\rho\) (or function-space disagreement), not ensemble size. This is the quantitative version of the warning already in §3.1 from Vieira et al.

### 11.3 Where diversity comes from, and why fine-tuning closes the channels

Diversity has a catalogued taxonomy of data, parameter and architecture manipulations ([Brown et al., 2005](https://doi.org/10.1016/j.inffus.2004.04.004)), but for deep networks the dominant channel is random initialization. Fort et al. decouple initialization from minibatch order and find initialization dominant, with shuffling contributing only at high learning rates where gradient noise is large; they also show that functions along one optimization trajectory cluster within a single mode in prediction space, and that cyclic sampling from a shared initialization is significantly less diverse than independent runs ([Fort et al., 2019](https://arxiv.org/abs/1912.02757)).

The fine-tuning protocol used in this project closes the channels:

- **Shared initialization persists.** Models trained from pre-trained weights stay in the same basin, show no loss barrier between runs, and make similar mistakes ([Neyshabur et al., 2020](https://arxiv.org/abs/2008.11687)). Solutions fine-tuned from one checkpoint are linearly connected ([Frankle et al., 2020](https://arxiv.org/abs/1912.05671)), which is why averaging them works without a barrier ([Wortsman et al., 2022](https://arxiv.org/abs/2203.05482)).
- **Annealing removes gradient noise.** SGD-noise temperature scales with learning rate, and every cycle ends at \(\eta_{\min}\).
- **Data order and subsampling are fixed.** No bootstrap, no per-member partition.
- **What remains is permutation symmetry**, which changes parameters without changing the function ([Ainsworth et al., 2023](https://arxiv.org/abs/2209.04836)) — so weight-space distance is not evidence of functional diversity.

Mechanistically, the loss landscape supplies the alternatives: minima are joined by low-loss paths traversable at single-run cost by fast geometric ensembling ([Garipov et al., 2018](https://arxiv.org/abs/1802.10026)), while weight averaging deliberately stays inside one basin and trades diversity for a better point estimate ([Izmailov et al., 2018](https://arxiv.org/abs/1803.05407)).

**What this changes in the review.** §3.1's empirical warning that "three seeds does not guarantee three hypotheses" now has a mechanism and a prediction: a committee differing only in seed, from one checkpoint, under an annealed schedule, should have \(\rho \to 1\). The project's own 0.0151 / 0.0150 / 0.0150 eV Å⁻¹ member RMSE is the predicted outcome, not an unlucky draw.

### 11.4 Common-mode blindness and the OOD inversion

Out of distribution the dominant error is usually shared: every member saw the same frames and inherits the same prior, so \(b\) grows while the members agree more. Then \(s^2\) can fall as the error rises, making the spread not merely uninformative but anti-correlated with total error. This is the mechanism behind the pattern in §5.2, and it is reported directly in the MLIP literature:

- Kurniawan et al. find uncertainty plateauing or decreasing as errors grow across bootstrap, dropout, random-initialization and snapshot ensembles ([Kurniawan et al., 2026](https://doi.org/10.1088/2632-2153/ae9fb4)).
- Lu et al. test an equivariant-network ensemble on out-of-training-domain silicon phases and find the uncertainties "generally overconfident" with "little quantitative predictive power for the actual errors"; they report paths where uncertainty decreases while error increases, and note that this defeats calibration by a constant factor. Their clearest case is a pair of equations of state with the same uncertainty of about 2 meV/atom, one accurate and one qualitatively wrong ([Lu et al., 2023](https://arxiv.org/abs/2309.00195)).
- Kahle and Zipoli document the common-bias problem of committee members, which Beck et al. cite as the reason to decorrelate heads by disjoint data ([Kahle & Zipoli, 2022](https://doi.org/10.1103/PhysRevE.105.015311)).

Two properties must therefore be reported separately, as §5.4 already argues: coverage is a *scale* property repairable by one fitted factor, while rank correlation is a *shape* property that no rescaling repairs. A pooled correlation mixes the between-environment scale (an environment detector) with the within-environment shape that acquisition needs.

### 11.5 The frozen body is maximal common mode

With \(f_k = h_k(\phi(x))\) and \(\phi\) frozen, the spread lives entirely in the readout. If the heads are linear and the loss strictly convex in them, all heads trained on the same features and targets share a unique minimizer, so \(s^2 \to 0\) at convergence: head diversity must come from under-training or from different data partitions. Beck et al. take the second route and state the reason explicitly — the disjoint partition "increases the diversity between the output heads more strongly, mitigating known problems of a common bias between the committee members" ([Beck et al., 2025](https://doi.org/10.1063/5.0302097)).

Their own calibration makes the hidden term numerical. Rescaling the disagreement by a validation-fitted factor to correct "an underestimation of the true uncertainty due to biases inherent to the models" gives \(\alpha = 1.32\) for a naive committee, \(2.00\) for the disjoint multi-head committee and \(6.94\) for the overlapping one on 3BPA: the shared-body spread is two to seven times smaller than the error it is meant to estimate, and \(\alpha\) is fitted in distribution, so it cannot transfer to the extrapolative regime where the bias is largest. Beck et al. also supply the right analogy — the construction is "conceptually similar to adding a dropout layer to the output blocks" ([Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142)) — and the same structural limit is documented outside atomistics: a shared backbone with per-member low-rank perturbations reaches only a measure-zero subset of the parameter space, yields near-identical members, and has an epistemic uncertainty an order of magnitude below a deep ensemble (Jensen–Shannon divergence 0.002 versus 0.037) ([Zamyatin et al., 2026](https://arxiv.org/abs/2601.16936)).

**Prediction for the AL loop.** A frozen body cannot be corrected by acquiring labels, so the MHC arm should saturate at the body's error floor; its acquisitions are those that most excite the readout, not those that most reduce body error.

### 11.6 Snapshots carry a convergence trend, not just basin diversity

Write a snapshot's error as \(e_m = b + u_m + v_m\), where \(u_m\) is basin-to-basin diversity and \(v_m\) is a deterministic function of cycle index measuring distance from convergence. The ensemble mean then carries a bias \(\bar v\) and the spread carries \(\mathrm{Var}(v)\); neither is epistemic uncertainty. Huang et al. are explicit that the construction requires each snapshot to be a converged minimum, and their own diagnostics show early snapshots with much higher error and late snapshots in the same minimum as the final model, with the usable ensemble "comparable with ensembling of 2 independent models" ([Huang et al., 2017](https://arxiv.org/abs/1704.00109)). Kurniawan et al.'s protocol is the practical remedy: discard early cycles as burn-in and snapshot only after the loss plateaus, spaced widely enough to be approximately independent ([Kurniawan et al., 2026](https://doi.org/10.1088/2632-2153/ae9fb4)).

**What this changes in the review.** §3.2's "operational trap" is not merely a checkpoint-selection bug; it is a bias term. The project's 26.9 / 16.1 / 2.67 meV/atom member energy RMSE is \(v_m\), and the 13.11 meV/atom arm-level energy RMSE is \(\bar v\). Any pooled correlation computed on that ensemble is partly a convergence trend aligned with environment difficulty.

### 11.7 Online versus retrospective active learning

The AL literature this review cites is mostly *online*: the model's uncertainty decides which configurations along a running trajectory are labelled ([Jinnouchi et al., 2019](https://doi.org/10.1103/PhysRevB.100.014105); [Vandermause et al., 2020](https://doi.org/10.1038/s41524-020-0283-z)). The project's loop is *retrospective*: labels are revealed from a fixed, pre-labelled pool. That isolates the acquisition function at equal label cost and removes trajectory variance, but it also removes the feedback from acquisition to the distribution the model is later scored on. The paper should therefore claim **retrospective acquisition efficiency**, not closed-loop data efficiency, unless an online check is run.

### 11.8 Claim ledger

| Claim the paper makes | Source | Evidence level |
|---|---|---|
| Ensemble spread equals the reducible part of member error | Krogh & Vedelsby 1995; derived identity | Textbook result; derivation in §13.1 |
| Spread is invariant to shared error | Corollary of the identity | Analytic |
| \(\rho\), not \(K\), controls committee quality | Standard exchangeable-ensemble algebra | Analytic |
| Fine-tuning from one checkpoint stays in one basin | Neyshabur et al. 2020; Frankle et al. 2020 | Full text read (Neyshabur); metadata (Frankle) |
| Random initialization is the dominant diversity channel | Fort et al. 2019 | Full text read |
| Committees match bootstrap ensembles at lower cost | Carrete et al. 2023 (0.90 vs 0.91 Spearman, SrTiO₃) | Full text read |
| Shared-body committees underestimate error by 2–7× | Beck et al. 2025 (\(\alpha\) values) | Full text read |
| Ensembles are overconfident and can invert OOD | Lu et al. 2023; Kurniawan et al. 2026 | Full text read |
| BatchEnsemble-class methods are near-single-model | Zamyatin et al. 2026 | Full text read |
| Snapshot ensembles require converged cycle minima | Huang et al. 2017 | Full text read |
| Single-model UQ does not consistently beat ensembles | Tan et al. 2023 | Metadata only |
| Ensembles are strongest under shift, but degrade | Ovadia et al. 2019; Guo et al. 2017 | Metadata only |

**Caveat.** The "metadata only" rows are used for framing, not for numbers. The three entries marked full text were read for this addendum; the rest are cited at the level their source supports.

---

## 14. Update 2026-11-27: why the snapshots matched the committee

### 14.1 The observation

In exp-02, the warm-restart snapshot ensemble ranked force errors within the bulk and F-doped test blocks at least as well as the bootstrapped three-member committee, at a third of its training cost (within-environment Spearman 0.834 against 0.656; 4,996 s against 14,975 s; [exp-02 RESULTS](../experiments/exp-02-snapshots/RESULTS.md)). The expectation had been the reverse, because independent ensembles are the reference construction (§3.1). The 2026-11-25 review limits what the run supports: one seed, 178 time-correlated frames, about 3–5 effective samples per environment, and a paired test that separates the arms only with 20-frame blocks. "At least as good" survives; "better" is not established.

### 14.2 A committee fine-tuned from one checkpoint is a local ensemble

The expectation holds for networks trained from scratch. There, members start from different random weights and end in different basins, and the cheaper constructions fall short of them: "SSE and cSGLD outperform all other techniques except deep ensembles and enjoy a near-linear scaling of DEE with the number of samples on CIFAR datasets." ([Ashukha et al., p. 8](https://www.alphaxiv.org/pdf/2002.06470?page=8)). SSE is the snapshot ensemble, cSGLD a cyclic Langevin sampler, and DEE the deep ensemble equivalent (ref. 55).

A MACE fine-tune has no random initialisation: every weight comes from the foundation checkpoint. In exp-02 the committee members differed only in data order and bootstrap draw, and all annealed on one cosine schedule to a learning rate of 1e-6. Sadrtdinov et al. study exactly this situation in transfer learning. Models fine-tuned from one checkpoint "end up in the same basin of the loss landscape, which we call the pre-train basin, and thus have limited diversity" ([p. 1](https://www.alphaxiv.org/pdf/2303.03374?page=1)), and they call such a committee a local deep ensemble. A snapshot ensemble whose restarts are tuned to explore that basin can match it: "Overall, SSE with optimally chosen hyperparameters can construct an ensemble of similar, and in some cases even better, quality than Local DE but is not able to close the gap with Global DE (see Table 1)." ([p. 6](https://www.alphaxiv.org/pdf/2303.03374?page=6)). Global DE means members fine-tuned from different pre-trained checkpoints. In their Table 1, at five members, the tuned snapshot ensemble beats the local ensemble clearly in two of six settings, ties in two and loses in two; its advantage is largest at two or three members (their Figure 1).

Two limits. Their evidence is classification accuracy, not ranking of force errors. And their snapshot members get worse with each cycle, because they drift away from the pre-trained checkpoint and overfit the target data ([p. 7](https://www.alphaxiv.org/pdf/2303.03374?page=7)); exp-02's members got better with each cycle (§14.4). No MLIP paper found so far compares snapshots with a committee for a fine-tuned foundation model; §11 already noted the gap.

### 14.3 What exp-02's saved predictions show

Recomputed from the per-frame predictions of exp-02 (`outputs/outputs_v3/evaluation/*_predictions.csv`) with `src/mlpdft/utils/exp02_spread_diagnostics.py`. ρ is the Spearman correlation over frames within one environment; σ is the per-frame force spread.

| | Bulk LiF (80) | F-doped Li (80) | Interface (18) |
|---|---|---|---|
| Median σ, committee / snapshot (meV/Å) | 0.62 / 0.77 | 1.24 / 3.50 | 3.72 / 5.73 |
| Median per-frame ratio σ_snapshot / σ_committee | 1.2 | 2.8 | 1.5 |
| Median error/σ, committee / snapshot | 4.2 / 3.7 | 6.4 / 2.7 | 4.3 / 3.2 |
| ρ(snapshot error, committee error) | 0.98 | 0.99 | 0.91 |
| ρ(committee σ, committee error) | 0.74 | 0.63 | 0.59 |
| ρ(snapshot σ, committee error) | 0.92 | 0.82 | 0.61 |

The two arms err on nearly the same frames, which is what a shared checkpoint and shared data predict (§13.1). They differ in the spread: the snapshot spread is larger, and it ranks the committee's own errors better than the committee's spread does. The per-frame correlations carry the same time trend as the rest of exp-02 (contiguous splits), so they are descriptive.

### 14.4 Part of the snapshot spread is a convergence trend

The snapshot members are not exchangeable. Their energy RMSE falls from 0.188 to 0.099 to 0.066 meV/atom over epochs 49, 99 and 149, and the three-member mean has a worse force RMSE (0.0101 eV/Å) than the last member alone (0.0082 eV/Å). The epoch-49 member is partly a less-trained model, so σ contains the distance between an early and a late model: the term \(v_m\) of §13.6.

Inference, not measured: the frames that the fine-tune was still changing at epoch 49 are plausibly the frames it still gets wrong at epoch 149, which would make this term rank errors well. It is not uncertainty about which model is right, it costs accuracy when the ensemble mean is used, and nothing shows that it survives a distribution shift. The saved outputs cannot separate it from diversity between minima: no run saved per-member predictions, and the exp-02 copy lacks the epoch-49 and epoch-149 members.

### 14.5 Consequences for design C

- The committee arm is a local ensemble by construction. Its weakness relative to snapshots is a property of fine-tuning from one checkpoint, and the paper should present it that way, with the member diversity measured, not assumed.
- The snapshot arm needs converged members, so that its spread is diversity and not a trend. Discarding the first cycle and holding the optimiser steps per cycle fixed across label rungs are the two remedies.
- The per-arm review, the diagnostics and the changes adopted are in [methods/ensemble-diversity.md](../methods/ensemble-diversity.md) and the 2026-11-27 note in [ADR-0007](../decisions/ADR-0007-bulk-trained-shift-ladder.md).

### 14.6 Claim ledger additions

| Claim | Source | Evidence level |
|---|---|---|
| Models fine-tuned from one checkpoint form a low-diversity local ensemble | Sadrtdinov et al. 2023; Neyshabur et al. 2020 | Full text read |
| A tuned snapshot ensemble can match a local ensemble, not a global one | Sadrtdinov et al. 2023 (accuracy, image classification) | Full text read |
| Trained from scratch, snapshot ensembles come second to deep ensembles | Ashukha et al. 2020 | Full text read |
| In exp-02 the snapshot spread ranked errors at least as well as the committee's | Project data, §14.3 | One seed, correlated frames |
| Part of exp-02's snapshot spread is a convergence trend | Member errors and ensemble-mean penalty, §14.4 | Inference |
