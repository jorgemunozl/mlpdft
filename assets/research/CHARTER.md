---
status: active
date: 2026-11-21
owner: jorge
---

# Uncertainty Quantification and Active Learning for Fine-Tuned MACE on a Li–F Testbed

## One-sentence description

We compare three ensemble-based uncertainty-quantification strategies for a
fine-tuned MACE foundation potential — an independent fine-tuning committee,
warm-restart snapshots, and a shared-body multi-head committee trained with the
foundation body frozen — against two cheap last-layer baselines (LLPR and an
NLL-trained shallow ensemble), on whether they rank held-out DFT error, stay
calibrated as the data move away from the bulk LiF training distribution, and
select informative DFT configurations more efficiently than random sampling.

## Why this system

Foundation MLIPs are fine-tuned on small DFT datasets, and users must choose an
uncertainty estimate at a fixed compute budget. The model is fine-tuned on bulk
LiF only. It is then evaluated along a graded shift
([ADR-0007](decisions/ADR-0007-bulk-trained-shift-ladder.md)):
1. independent bulk LiF replicas at 300 K (in distribution);
2. bulk LiF at ≈ 162 K;
3. F-doped Li metal and pure Li metal;
4. a thin LiF/Li film.

All labels use 40/320 Ry
([ADR-0008](decisions/ADR-0008-40ry-labels-and-md-protocol.md)). The dataset
is small and fully labeled, which makes it a controlled testbed for comparing
UQ methods rather than a production benchmark.

The physics is deliberately secondary: the paper's claim is about uncertainty
quality and label efficiency, not about a battery property.

## Question

1. Which arm ranks held-out force and energy error best **within** each
   evaluation set (the in-distribution bulk replicas and each shifted
   environment), not only when the sets are pooled?
2. Which arm is best calibrated in distribution, and how does its calibration
   degrade along the shift?
3. Does uncertainty-driven acquisition beat random acquisition at equal label
   budget on held-out DFT error?
4. What is the cost–accuracy tradeoff, in GPU-hours per retrain and per
   acquisition cycle?

## Scope

In scope:

- `LIF64`: stoichiometric bulk LiF. It is the training environment. Its
  calibration and test sets are new independent replicas at 300 K. A colder
  (≈ 162 K) variable-cell run is the first shift.
- `LIWITHF`: 53 Li + 1 F, F-doped Li metal. Chemistry shift, evaluation only.
- `BCC54`: pure Li metal. Chemistry shift, evaluation only.
- `LIFINTERFACE`: 86 Li + 36 F, a thin two-layer LiF film on Li metal.
  Structure-shift stress test, evaluation only.
- `LIWITHBF`, `LIBF4`, `BLI3`: B/salt chemistry. Far shift, appendix only.

Out of scope:

- Li diffusivity, activation energy, and any transport observable;
- rebuilding the Li|LiF interface or generating new diffusion cells;
- SEI formation, electrochemical potential, charge transfer, and voltage;
- carbonate, oxide, hydroxide, and organic interphase chemistry;
- a DFT defect campaign beyond the tier-0 sanity check below.

The interface cell is a thin-film model, not a buried interface. Its geometry is
described as measured, and it is used as a distribution shift, not as a physical
interface claim.

## Tier-0 physical anchor

One static check, no new DFT by default: evaluate the fine-tuned MACE potential
on Li vacancy and Li interstitial cells in bulk LiF and compare the formation
energies with the published values (≈0.76 eV vacancy, ≈1.73 eV interstitial).
Accept agreement within ~0.2 eV, accounting for the published 110/440 Ry
settings versus this project's 40/320 Ry. If the disagreement exceeds that
tolerance, escalate to a small matched-settings DFT check before making any
physical claim.

## Claims

We will claim:

- a measured ranking of the arms on within-environment error
  discrimination, calibration, and risk–coverage;
- a measured label-efficiency comparison against random acquisition;
- a measured cost–accuracy tradeoff that includes the zero-fine-tune
  (frozen-body) operating point;
- a negative result, stated as such, if the arms do not separate.

We will not claim:

- that any arm is universally better across chemistries;
- that the interface cell is a physical buried Li|LiF interface;
- that these results transfer to transport, SEI formation, or other chemistry;
- that ensemble agreement alone establishes physical accuracy.
