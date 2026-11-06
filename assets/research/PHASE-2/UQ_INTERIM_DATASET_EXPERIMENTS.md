# Three UQ experiments on the interim Li–F dataset

**Status:** planning only. No experiment on this dataset has been launched.

## Research question

At the same DFT label budget, how do an independent committee, warm-restart
snapshots, and a frozen-body multi-head committee compare in force accuracy,
uncertainty quality, and computational cost across three Li–F environments?
The independent committee is the reference, snapshots test a cheaper ensemble,
and the multi-head committee tests uncertainty from multiple readouts sharing a
frozen foundation representation. A single frozen readout is a control for the
multi-head experiment.

These are new experiments on a new dataset. Historical results and the 214 /
53 / 178 pilot belong to different splits and must not be merged into this
comparison. See [Experiment 2 results](UQ_EXPERIMENT_2_RESULTS.md) and the
[MHC pilot diagnosis](experiment3_runs/mhc_pilot_2026-11-05/DIAGNOSIS.md) for
their background and limitations.

## Shared dataset

Use the private Hugging Face dataset
[`jorgemunozl/li-f-uq-interim-2900`](https://huggingface.co/datasets/jorgemunozl/li-f-uq-interim-2900),
pinned to revision `0014d10c5f6355e84c596fbcf25a6eadad6c81fc`. Verify the
downloaded files against `SHA256SUMS` and record the revision and file hashes in
every run manifest.

Curation audit, composition checks, temporal-dependence caveats and acceptance
gates: [`UQ_INTERIM_DATASET_CURATION.md`](UQ_INTERIM_DATASET_CURATION.md).

| Environment | Train | Validation | Calibration | Test | Source stride (MD steps) | Audit thermalization cutoff (steps) |
|---|---:|---:|---:|---:|---|---:|
| Bulk LiF, NPT | 800 | 100 | 100 | 100 | 3 or 15 | 3 |
| F-doped Li, NVT | 800 | 100 | 100 | 100 | 3 or 5 | 0 |
| LiF/Li thin-film interface, NVT | 400 | 100 | 100 | 100 | 1 | 586 |
| **Total** | **2,000** | **300** | **300** | **300** | — | — |

Use `uq_train.extxyz` to fit models, `uq_valid.extxyz` for checkpoint and
hyperparameter selection, `uq_calibration.extxyz` to fit uncertainty scaling,
and `uq_test.extxyz` for final evaluation. Each training arm uses only the
2,000 unique training frames; bootstrap and head subsets do not increase the
number of unique DFT labels. `REF_energy` is in eV and `REF_forces` in eV/Å.

Selection is time-blocked, with 16 source-timestep buffers at split boundaries
and irregular output spacing. In particular, the interface validation and
calibration blocks include consecutive frames. The buffers have not been shown
to exceed the correlation time, and the audit's zero cutoff for F-doped Li is
not proof of equilibration. All splits contain related trajectory segments, so
frame counts are not independent sample counts. No split should be called an
independent-trajectory or OOD benchmark. The interface is represented in
training and is not an unseen-environment test.

The earlier pilot test shares 3, 4 and 8 rounded-structure hashes with the new
validation, calibration and test sets respectively. The new test is therefore
**previously observed**, not a blind final benchmark. Avoid using it to select
models or tune settings for these experiments; validate the final paper claims
on a later independent dataset. The dataset card and manifests document the
source logs, SCF checks, frame selection, exact split hashes, and limitations.

## Common protocol

- Start all methods from the same recorded foundation checkpoint and model
  version; `mace_omat_medium` is the proposed default because it was used in the
  earlier comparison. Record its checkpoint hash, dtype, device and software
  environment. The published Beck et al. foundation demonstration used
  MACE-MP-0b, so this is an adaptation, not an exact reproduction of that
  experiment. [Beck et al.](https://www.alphaxiv.org/pdf/2508.09907)
- Primary outcome: force accuracy and force uncertainty. Fix the force
  aggregation, loss weights, checkpoint rules, calibration rule, seeds and
  training budgets before runs. If methods use different losses or budgets,
  report them explicitly and do not attribute the whole difference to the
  architecture. A force-only setting makes the frozen heads unsuitable for
  energy-accuracy claims; if energy is evaluated, first verify reference-energy
  consistency.
- Report per-environment force RMSE, uncertainty–error Spearman and Pearson
  correlations, risk–coverage, calibration scale, RLL, and empirical 1σ / 2σ
  coverage. Fit calibration on the separate calibration split only. Report
  pooled metrics as secondary, and use paired time-block resampling where
  defensible, reporting the block length and uncertainty. Do not equate
  overlapping confidence intervals with proof of equivalence.
- Record actual training and inference cost, hardware, number of optimizer
  updates, head count and wall time. Compare at equal label budget; additionally
  report member-count and compute-budget differences. Three independent models
  and eight MHC heads are not a matched-member comparison.
- Before full training, verify dataset hashes and split usage; run a small smoke
  test for finite forces, loss reduction and working checkpoint selection. Do
  not use the held-out test to adjust a smoke test or training settings.

## Experiment 1 — Independent committee

**Why:** establish the reference for force prediction and disagreement-based
uncertainty using independently trained, fine-tuned models.

**Design:** train three models from the same foundation checkpoint with seeds
123, 124 and 125. Use the same 2,000-frame pool, with recorded time-block
bootstrap samples if bootstrapping is enabled. Obtain the prediction from the
member mean and uncertainty from their force disagreement. Evaluate every
member and the ensemble. Fix optimizer, loss and stopping rules on validation
before training; fit the uncertainty scale only on the calibration split.

**Question:** how well does the reference committee rank and calibrate force
errors within each environment, and what does it cost to train and evaluate?
Its new results provide the baseline for Experiments 2 and 3.

## Experiment 2 — Warm-restart snapshot ensemble

**Why:** test whether checkpoints from one fine-tuning run provide useful UQ at
lower training cost than independently training three models.

**Design:** train one model on the same 2,000-frame pool with a recorded
three-cycle warm-restart schedule. Retain one checkpoint at the end of each
cycle, immediately before the next restart, and form a three-member ensemble.
Verify the scheduler and checkpoint epochs from actual logs; the earlier
boundary-selection problem must not recur. Use the same validation,
calibration and test roles as Experiment 1.

**Question:** relative to the newly trained independent committee, how much
accuracy, ranking, calibration and inference/training cost does the snapshot
ensemble gain or lose? A larger spread alone does not demonstrate better UQ.

## Experiment 3 — Frozen-body multi-head committee

**Why:** test whether diverse readouts on one frozen foundation representation
can provide useful uncertainty at this label budget. The earlier pilot used
identical warm-started heads and a mismatched energy reference, so its negative
result does not answer this question.

**Design:** preserve and verify all foundation descriptor tensors, freeze the
body and original head, and train eight newly **randomly initialized** readout
heads. Start with independently sampled, environment-stratified 80% overlapping
training subsets: 640 bulk + 640 F-doped + 320 interface frames per head,
1,600 per head. Save each head's exact frame indices and seed. The ensemble
uses the eight new heads, not the original foundation head. Test a disjoint
split as a separate ablation if resources permit: 250 frames per head at this
budget. Overlapping subsets are an adaptation to this data; the paper's
foundation-model demonstration used disjoint subsets and a larger budget.

Train a **single random-initialized frozen-head probe** on all 2,000 training
frames, using the same frozen body and a comparable force-focused recipe. The
probe versus the fine-tuned committee and MHC versus probe comparisons are
diagnostic, but change more than one factor, including ensemble averaging and
training subsets. They do not by themselves isolate the causal cost of freezing
or splitting. Additional matched controls are needed for causal attribution.

**Question:** can the frozen multi-head committee achieve useful force accuracy
and uncertainty ranking on this dataset, and how does its cost compare with the
other methods? Check pairwise head force differences and initialization before
full training; use calibration and test outcomes to assess whether diversity is
informative.

## Implementation readiness before the first run

The runner accepts `--calibration-file` for a separate calibration split. Pass
all four paths explicitly. For backward compatibility, omitting this flag uses
validation for calibration and emits a warning; that legacy fallback does not
meet this protocol. The runner checks that the explicit calibration file exists
and is not byte-identical to another split, and records the path and fallback
status in its run manifest. This is not a frame-level overlap check; verify the
dataset manifest separately. The overlapping MHC sampler stratifies by
`config_type` and records selected indices. Confirm the actual head manifest
contains 640 / 640 / 320 frames per head before interpreting a full run.

Use fresh output directories so the runner does not automatically import a
previous pilot summary. Its old default dataset path and dataset download target
refer to the 214-frame pilot, so download the pinned interim dataset separately
and verify all four paths before use. Matched loss choices still need to be
fixed before launch.

## Proposed A40 compute and training budget

Use one NVIDIA A40 per run, CUDA, `float64`, batch size 16 and validation batch
size 1 as the initial configuration. Confirm memory fit in a short benchmark;
do not assume that the pilot's batch fits this dataset. Record GPU model, peak
memory, elapsed time, actual updates, early-stopping epoch, and training and
evaluation times. These are proposed caps, not established convergence:

| Study | Training jobs per repetition | Maximum epochs | Label budget |
|---|---:|---:|---:|
| Independent committee | 3 separate fine-tunes | 150 each | 2,000 unique frames |
| Snapshots | 1 fine-tune, 3 cycles | 150 total, 50 per cycle | 2,000 |
| Frozen MHC | 1 eight-head training job | 120 | 2,000 unique; 1,600 per head |
| Frozen single-head control | 1 training job | 600 | 2,000 |
| MHC split/budget ablation | 5 additional eight-head jobs | 120 each | 500, 1,000, 2,000; two split schemes |

The ablation has six configurations, but reuses the 2,000-frame overlapping
MHC from the main study when all settings and seed match, so five jobs are
additional. One repetition is therefore **11 training jobs**; three repetitions
would be **33 jobs**, excluding smoke tests and evaluation. The MHC and probe
use a proposed patience of 50 validation checks; the snapshot run must complete
its cycles to yield all three checkpoints. Record actual stop points and
updates; the planned maximum epoch count is not a guarantee of convergence.

**Timing is not yet measured for this dataset.** On the earlier 214-frame
pilot, one 150-epoch snapshot fine-tune took 4,996 s, the 400-epoch four-head
disjoint MHC took about 4 h 49 min, and the 150-epoch probe took about 1 h 17
min. A naive linear scaling of those times by training-frame presentations
and epoch count gives roughly 13 GPU-hours per 150-epoch fine-tune, 86 hours
for the 120-epoch eight-head overlapping MHC, and 48 hours for the 600-epoch
probe. This implies about 186 GPU-hours for the first six jobs and roughly
275 GPU-hours for all 11 jobs per repetition, **before** separate evaluation,
startup, failed runs or the longer validation sets. Three repetitions would be
roughly 825 GPU-hours by the same crude arithmetic. These are planning-scale
extrapolations, not quotes or reliable upper bounds: model, head, atom-count,
validation, batch, early-stopping and I/O costs can alter them substantially.
No hourly rental price is assumed; multiply measured GPU-hours by the actual
A40 hourly rate for cost.

Before committing to those budgets, run a short, tracked A40 benchmark for a
fine-tune, MHC and probe using the full dataset and intended validation setup.
Measure startup and steady-state epoch/update time separately, peak memory and
evaluation time; then revise the full and ablation estimates. For the benchmark,
use validation as the runner's test input and skip final-test evaluation: MACE
receives the `--test-file` during training as well as the later evaluation.
References: [pilot gate](experiment3_runs/mhc_pilot_2026-11-05/GATE.md),
[probe training log](experiment3_runs/mhc_pilot_2026-11-05/probe150/train.log),
and [Experiment 2 cost](UQ_EXPERIMENT_2_RESULTS.md).

## Minimum remaining campaign and transition to active learning

The recommended last campaign on this interim dataset is **three main method
experiments plus one targeted MHC ablation**. An experiment here is a research
comparison, not one training run: the committee alone has three members, and
repeat seeds or budgets require additional runs. Treat this as an upper bound on
the planned questions, not a promise that one seed can establish a robust win.

1. **Independent committee:** establish the 2,000-label reference; evaluate the
   zero-shot foundation and individual fine-tunes as accuracy controls.
2. **Snapshots:** compare three cycle-end checkpoints with that reference on
   the same splits and report both accuracy/UQ and actual cost.
3. **Frozen MHC:** compare eight random heads on 80% overlapping subsets with
   the reference, alongside the all-data frozen-head probe and the head
   diversity/initialization checks described above.
4. **MHC data-split and budget ablation:** on the same nested, environment-
   stratified 500 / 1,000 / 2,000-frame training subsets, compare overlapping
   and disjoint heads with the same head count, frozen foundation, loss and
   stopping policy. Use the common validation and calibration splits; keep the
   test untouched until the analysis and settings are frozen. Record subset
   manifests and actual optimizer updates. This asks whether data per head and
   overlap explain performance; it is not an isolated proof of either cause,
   because the schemes also change membership and correlations between heads.

For each comparison repeat independent training/split seeds when affordable,
using at least three recorded repetitions as the target. Report variability
across repetitions and account for correlated trajectory blocks. If compute
limits repetitions, label the study exploratory rather than treating the heads
or adjacent frames as independent experimental replicates. Match the held-out
data and label budgets across methods; report differences in model count,
optimization updates and compute. A negative or inconclusive result is a valid
outcome and does not mandate a new method sweep.

Before the final long train, freeze the method, configuration, dataset
revision, reference energies, calibration rules and evaluation plan based on
the development evidence. Build and audit the separately planned final dataset;
its new interface trajectories are planned, not existing data. A model trained
on all its training pool is a **full-pool reference**, not proof that an
acquisition strategy works or a guaranteed accuracy ceiling.

The later retrospective active-learning study is a distinct experiment:
compare uncertainty acquisition with a random-selection control, paired initial
seeds, equal label budgets, matching environment quotas and diversity rules,
and the same retraining recipe and held-out data. Report error versus labels,
uncertainty discrimination, and cumulative training/inference compute over
multiple acquisition seeds. An uncertainty–error correlation alone does not
establish an AL benefit. If environment quotas are fixed, they cannot support
a claim that one method naturally acquires interface structures earlier;
answer that question with a separately specified unstratified diagnostic.
An actual DFT acquisition round, if later undertaken, is a separate prospective
validation rather than part of the retrospective result.

The older [active-learning protocol](../PHASE-3/ACTIVE_LEARNING_PROTOCOL.md)
contains preliminary assumptions about four disjoint MHC heads, split
attribution and interface/OOD status. Reconcile it with the selected method and
the eventual final dataset before pre-registering the AL study; do not treat its
existing thresholds or proposed frame counts as verified outcomes.
