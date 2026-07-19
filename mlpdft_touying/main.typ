#import "@preview/cetz:0.4.2"
#import "@preview/fletcher:0.5.8" as fletcher: edge, node
#import "@preview/touying:0.6.1": *
#import "@local/touying-simpl-uni:1.0.0": *
#import "@preview/fontawesome:0.5.0": *


// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#show: ecnu-theme.with(
  // Lang and font configuration
  lang: "en",
  font: ("Libertinus Serif", "Noto Sans"),

  // Basic information
  config-info(
    title: [_mlpdft_: Leveraging Foundational Models for accurate and blazing speed Li-F-B cells],
    short-title: [_mlpdft_: Message passing neural networks for battery cells],
    subtitle: [Message passing neural networks for battery cell simulation],
    author: [Jorge Munoz Laredo],
    date: datetime.today(),
    institution: [National University of Engineering],
  ),
)

#title-slide()

= What we actually have and why we are here

== A battery - ish dataset
I
=== Examples of configuration-group names (catalog)

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  // LEFT: LiF / Li-rich
  figure(
    table(
      columns: (auto, auto),
      stroke: 0.4pt,
      inset: 3pt,

      table.cell(fill: luma(230), colspan: 2)[*LiF / Li-rich*],
      table.cell(fill: luma(230))[*Group*],
      table.cell(fill: luma(230))[*Frames*],

      [`LIFINTERFACE_KJPAW_V1`], [100],
      [`LIFINTERFACE_KJPAW_NPT_V2`], [100],
      [`LIFINTERFACE_KJPAW_NPT`], [100],
      [`LIWITHF_NPT_FINAL`], [100],
      [`LIWITHF_ISOLATED`], [100],
      [`LIF64_KJPAW_V2`], [100],
      [`LIWITHF_V3`], [100],
      [`LIF64_ISOLATED`], [100],
    ),
    caption: [LiF and Li-rich configuration groups],
  ),
  // RIGHT: Salts, B-Li, ionics
  figure(
    table(
      columns: (auto, auto),
      stroke: 0.4pt,
      inset: 3pt,

      table.cell(fill: luma(230), colspan: 2)[*Salts, B--Li, ionics*],
      table.cell(fill: luma(230))[*Group*],
      table.cell(fill: luma(230))[*Frames*],

      [`BLI_V2`], [100],
      [`LIBF4_V4`], [100],
      [`LIBF4`], [100],
    ),
    caption: [Salts, B--Li, and ionic configuration groups],
  ),
)

== The FitSNAP model we actually have

=== Configuration of the FitSNAP model we have

- *FitSNAP stack* in words:
  - *Scraper* — reads quantum-chemistry frames (e.g. JSON) into structures.
  - *Calculator* — builds *SNAP bispectrum* descriptors per atom in LAMMPS (radial cutoff, `twojmax`, chemical channels, etc.).
  - *Solver* — *PyTorch* MLP.
  - *Layers:* PyTorch MLP `num_desc → 256 → 128 → 64 → 64 → 1`, `multi_element_option = 2` (per-element nets).
  - *Training:* learning rate `5e-6`, `70` epochs, batch `50`.

=== Metrics for the FitSNAP model

#figure(
  image("images/Metrics MLPDFT.png", width: 80%),
  caption: [Metrics obtained by two models],
)

= High Ordern ACE model

== What the MACE model is

#slide(
  figure(
    image("images/image.png", width: 80%),
    caption: [I. Batatia et al. 2022],
  ),
)

#slide(
  figure(
    image("images/mpnn.png", width: 80%),
    caption: [Message Passing Neural Network Concept],
  ),
)

#set par(spacing: 0.5em)
#slide(
  grid(
    columns: (1fr, 1fr),
    [
      *MACE Step (1): Atom Embedding*
      A molecule or crystal is represented as a *graph*:
      atoms are nodes, edges connect pairs within a cutoff radius $r_c$.

      Each atom $i$ is assigned an initial feature vector:
      $ bold(h)_i^((0)) = bold(W)_(Z_i) $

      a learned embedding indexed by atomic number $Z_i$ (lookup table).

      *Goal:* produce, for each atom, a scalar energy $E_i$
      that reflects its local chemical environment.
      The total energy is $E = sum_i E_i$.
    ],
    [

    ],
  ),
)


#slide(
  [
    *Depth on MPNN*

    - $t=0$, the atom/node only can see itself.
    - $t=1$, the atom can see its neighborhoods — *TWO BODY*.
    - $t=2$, the atom can see the neighborhoods from its neighborhoods — TRIPLETS — *THREE BODY*

    Three and higher order bodies are computationally expensive.
  ],
)

#slide(
  [
    *MACE Step (2): Interaction Block*

    For each atom $i$, messages from neighbors $j$ are built as:
    $ bold(m)_(i j) = sum_k W_k R_k(r_(i j)) dot Y_l^m(hat(r)_(i j)) ⊗ bold(h)_j^((t)) $

    $W_k$ learned matrix.

    - $Y_l^m(hat(r)_(i j))$: *spherical harmonics* — fixed geometric
      functions encoding the _direction_ to neighbor $j$.
    - $R_k(r_(i j))$: *radial network* — a small MLP acting on the
      scalar distance. _This is where most parameters live._
    - $⊗$: tensor product combining neighbor features with
      edge geometry, preserving equivariance via
      Clebsch--Gordan coefficients.

    Messages are aggregated: $bold(m)_i = sum_(j in cal(N)(i)) bold(m)_(i j)$
  ],
)

#slide(
  [
    MACE Step (2): Interaction Block

    For each atom $i$, messages from neighbors $j$ are built as:
    $ bold(m)_(i j) = sum_k W_k R_k(r_(i j)) dot Y_l^m(hat(r)_(i j)) ⊗ bold(h)_j^((t)) $

    $W_k$ learned matrix.
  ],
)

#slide(
  [
    MACE Step (3): Equivariant Product Block

    After aggregating neighbors, atom $i$ holds a feature vector
    $bold(m)_i$ encoding its local environment.

    *The problem:* one message passing layer only captures
    pairwise $(i,j)$ interactions — that is *2-body*.

    *The MACE idea:* take tensor products of $bold(m)_i$
    with itself $nu$ times:
    $
      bold(m)_i ⊗ bold(m)_i ⊗ dots.h.c
      quad (nu " times")
    $

    This mixes contributions from _different_ neighbors $j, k$
    simultaneously, producing *$(nu+1)$-body correlations*
    without any extra message passing.

    - With $nu = 2$: reaches *3-body* after layer 1,
      compounding across layers.
    - The CG coefficients ensure the result stays
      *equivariant* — no parameters, fixed by symmetry.

  ],
)

#slide(
  [
    After $T$ interaction layers, each atom $i$ holds a feature vector
    $bold(h)_i^((t))$ with components of different equivariance order $L$.

    Each layer contributes a partial atomic energy via its $L=0$ features:
    $ E_i = E_i^((0)) + E_i^((1)) + ... + E_i^((T)) $

    where each $E_i^((t))$ is a linear projection of
    $bold(h)_i^((t))|_(L=0)$,
    except the last layer which uses a small MLP.

    The total energy and forces are then:
    $
      E = sum_i E_i, \
      bold(F)_i = - (partial E) / (partial bold(r)_i)
    $

    *Forces are free* — no extra network needed, just
    automatic differentiation through the entire graph.
  ],
)


=== MACE-MP-0 Parameters Distribution

#figure(
  table(
    columns: (2fr, auto, auto),
    stroke: 0.4pt,
    inset: 5pt,
    align: (left, center, center),

    table.cell(fill: luma(230))[*Block*],
    table.cell(fill: luma(230))[*Parameters*],
    table.cell(fill: luma(230))[*Share*],

    [`interactions` (2 layers)], [3 163 392], [\~82%],
    [`products` (2 layers, corr. / sym. contraction)], [670 720], [\~17%],
    [`node_embedding`], [11 392], [\~0.3%],
    [`readouts`], [2 192], [\~0.1%],
    table.cell(fill: luma(220))[*Total*],
    table.cell(fill: luma(220))[*3 847 696*],
    table.cell(fill: luma(220))[*100%*],
  ),
  caption: [Parameter count by block — MACE-MP-0 Small],
)

#v(0.6em)

#figure(
  table(
    columns: (auto, auto, 3fr),
    stroke: 0.4pt,
    inset: 5pt,
    align: (left, center, left),

    table.cell(fill: luma(230))[*Submodule (layer 0)*],
    table.cell(fill: luma(230))[*Params*],
    table.cell(fill: luma(230))[*Role*],

    [`skip_tp`], [\~1.46 M], [Main equivariant tensor-product / mixing path],
    [`linear`], [\~65 k], [Channel mixing linear on node features],
    [`conv_tp_weights`], [\~42 k], [Radial MLP output → weights for the convolution],
    [`linear_up`], [\~16 k], [Projection up in channels],
  ),
  caption: [Dominant submodules inside one interaction layer],
)

== The differents MACE flavors and how you pick your favorite one

=== How you are going to pick your flavor

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    stroke: 0.4pt,
    inset: 5pt,
    align: (left, center, center, center, center, center),

    table.cell(fill: luma(230))[*Model*],
    table.cell(fill: luma(230))[*Parameters*],
    table.cell(fill: luma(230))[*Training dataset*],
    table.cell(fill: luma(230))[*Structures*],
    table.cell(fill: luma(230))[*DFT level*],
    table.cell(fill: luma(230))[*License*],

    [`MACE-MP-0`], [3 847 696], [MPTrj], [1.58 M], [PBE+U], [MIT],
    [`MACE-MP-0B3`], [9 063 204], [MPTrj], [1.58 M], [PBE+U], [MIT],
    [`MACE-OMAT`], [9 063 204], [OMat24], [\~100 M], [PBE+U VASP], [ASL],
  ),
  caption: "MACE foundation model variants and their training datasets",
)

== Zero-shot evaluation

== Zero-Shot MACE-MP Evaluation on LiF 54

== FitSnap First Metrics Obtained

#figure(
  image("images/fitsnap_metrics_table.pdf", width: 100%),
  caption: "FitSnap first metrics obtained",
)

== MACE OMAT First Metrics Obtained

#figure(
  image("images/mace_metrics_table.pdf", width: 80%),
  caption: "MACE OMAT first metrics obtained",
)

= Leveraging foundational model

== State of post training

Fine tunning is a field well study thanks the NLP success

== Catastrophic Forgetting

What is really happening on the first epochs a knowledge destillation strategy

== Using a freezing strategy

#figure(
  image("images/frozen_title.png", width: 90%),
  caption: "The different freezing strategies used in MACE fine-tuning",
)

== Freezing Strategy

#figure(
  image("images/frozen.png", width: 60%),
  caption: [Only the last layers are unfrozen],
)

== Multi Head Fine Tuning

What the heck is this

== Strategies for Fine Tunning

#figure(
  image("images/strategies.png", width: 100%),
  caption: "",
)

== Fine-Tuning Performance Study

#figure(
  image("images/performance.png", width: 100%),
  caption: "Performance of fine-tuned MACE models on the ",
)


== What we are trying to reach (Goals)

*We are trying to do*

1. Improve metrics from the existing FitSNAP model.
2. Blazing speed inference with message passing neural networks.
3. Obtain a model with a fraction of computional cost.

*Key resources:*

#align(center)[
  #link("https://github.com/jorgemunozl/mlpdft")[#fa-github() MLPDFT Github Repo]

  #link(
    "https://hf.co/collections/jorgemunozl/message-passing-neural-networks-for-lithium-fluoride-cells",
  )[🤗 Hugging Face Collection with Dataset Models and Arxiv Papers] ]

== Challenges Part One


== Challenges of Fine-Tuning MACE on a Specific Dataset

- *Reference energy mismatch ($E_0$ problem)*
  The foundation model and the target DFT dataset use different
  atomic reference energies. A linear regression over the
  training set is required to align them before fine-tuning.

- *DFT functional mismatch*
  MACE-MP was trained with very specific data, if different
  functional (DFT), the model must _unlearn_ systematic
  biases — which can hurt generalization.

- *Out-of-domain chemistry*
  LiBF#sub[4] contains ionic species and Li, which are outside
  the MACE-OFF23 organic training domain. The model has no
  prior knowledge of ionic interactions, charge transfer,
  or metal coordination.

= Training time

== Goals - First Training

+ Get used to the pipeline (set everything around)
+ Know how much RAM and compute time training takes for a small dataset.

== Dataset - First Training

#align(center)[
  #figure(
    table(
      columns: (auto, 2fr, auto),
      stroke: 0.5pt,
      inset: 6pt,
      align: (center, left, center),

      table.cell(fill: luma(230))[*\#*],
      table.cell(fill: luma(230))[*Group*],
      table.cell(fill: luma(230))[*Frame count*],

      [1], [`LIFINTERFACE_KJPAW_V1`], [149],
      [2], [`LIFINTERFACE_KJPAW_NPT_V2`], [477],
      [3], [`LIFINTERFACE_KJPAW_NPT`], [258],
      [4], [`LIWITHF_NPT_FINAL`], [3236],
      [5], [`LIWITHF_ISOLATED`], [38],
      [6], [`LIF64_KJPAW_V2`], [2000],
      [7], [`LIWITHF_V3`], [1195],
      [8], [`LIF64_ISOLATED`], [56],
    ),
    caption: [Total frames: 7409],
  )
]

== Hyperparameter Selection - First Training
#text(size: 0.8em)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 0.5em,

    // left — second part: scheduler, regularization, lora
    align(center, table(
      columns: (auto, auto),
      stroke: 0.4pt,
      inset: 3pt,

      table.cell(fill: luma(230))[*hyperparameter*],
      table.cell(fill: luma(230))[*value*],

      // === model architecture ===
      table.cell(fill: luma(220), colspan: 2)[*model architecture*],

      [cutoff radius ($r_max$)], [$5$],
      [channels], [$128$],
      [max $l$], [$1$],
      [max $ell$], [$3$],
      [interactions], [$2$],
      [correlation], [$3$],
      [radial basis functions], [$8$],
      [cutoff basis functions], [$5$],

      // === training ===
      table.cell(fill: luma(220), colspan: 2)[*training*],

      [batch size], [$4$],
      [max epochs], [$10$],
      [validation fraction], [$0.5$],
      [loss function], [`weighted`],
      [$lambda_(text("forces"))$], [$1.0$],
      [$lambda_(text("energy"))$], [$1.0$],
    )),

    // right — first part: model architecture, training, optimizer
    align(center, table(
      columns: (auto, auto),
      stroke: 0.4pt,
      inset: 3pt,

      table.cell(fill: luma(230))[*hyperparameter*],
      table.cell(fill: luma(230))[*value*],

      // === optimizer ===
      table.cell(fill: luma(220), colspan: 2)[*optimizer*],

      [optimizer], [`adam`],
      [learning rate], [$0.01$],
      [amsgrad], [`true`],
      [weight decay], [$5 times 10^(-7)$],
      [gradient clip], [$10.0$],

      // === scheduler ===
      table.cell(fill: luma(220), colspan: 2)[*scheduler*],

      [type], [`reducelronplateau`],
      [lr factor], [$0.8$],
      [patience], [$50$],

      // === regularization ===
      table.cell(fill: luma(220), colspan: 2)[*regularization*],

      [ema], [`true`],
      [ema decay], [$0.99$],
      [early stopping patience], [$4$],

      // === lora ===
      table.cell(fill: luma(220), colspan: 2)[*lora*],

      [lora enabled], [`true`],
      [lora rank], [$8$],
    )),
  ),
]

== Resources Employed - First training

#grid(
  columns: (1fr, 1fr),
  [
    #align(center)[
      #figure(
        table(
          columns: (auto, auto),
          stroke: 0.4pt,
          inset: 5pt,
          align: (left, center),

          table.cell(fill: luma(230))[*Resource*],
          table.cell(fill: luma(230))[*Value*],

          [GPU], [RTX 4000 ADA],
          [TFLOPS], [10],
          [Training time], [3 hours],
          [RAM used], [19 GB],
          [GPU usage], [\(approx 100\%\)],
        ),
        caption: [Resources employed in the first training],
      )
    ]

  ],
  [#figure(image("images/ada.webp", width: 70%), caption: "RTX 4000 ADA"),],
)

== MACE auto generated results - First Training

#figure(
  image("images/mace_results.png", width: 70%),
  caption: "Mace Training Supervision",
)

== Group Results - First Training

#figure(
  image("images/energy_rmse_per_group.pdf", width: 70%),
  caption: [Energy RMSE per atom (meV/atom) — MACE mock\_2\_test],
)

#v(0.5em)

#figure(
  image("images/force_rmse_per_group.pdf", width: 70%),
  caption: [Force RMSE (meV/Å) — MACE vs FitSNAP per group],
)

== Takeaways - First training

The take aways

== Goals - Second Training

Experiment with Scaling Laws

== Dataset - Second Training

#align(center)[
  #figure(
    table(
      columns: (auto, 2fr, auto),
      stroke: 0.5pt,
      inset: 6pt,
      align: (center, left, center),

      table.cell(fill: luma(230))[*\#*],
      table.cell(fill: luma(230))[*Group*],
      table.cell(fill: luma(230))[*Frame count*],

      [1], [`LIFINTERFACE_KJPAW_V1`], [248],
      [2], [`LIFINTERFACE_KJPAW_V2`], [781],
      [3], [`LIFINTERFACE_KJPAW_NPT_V2`], [795],
      [4], [`LIFINTERFACE_KJPAW_NPT`], [431],
      [5], [`LIWITHF_NPT_FINAL`], [5394],
      [6], [`LIWITHF_ISOLATED`], [63],
      [7], [`LIF64_KJPAW_V2`], [3333],
      [8], [`LIF64_KJPAW_NPT`], [3333],
      [9], [`LIF64_KJPAW_NPT_V3`], [3333],
      [10], [`LIWITHF_V3`], [1992],
      [11], [`LIF64_ISOLATED`], [94],
    ),
    caption: [Total frames: 19,797 (stride 3)],
  )
]


== Hyperparameter Selection - Second Training

#text(size: 0.8em)[
  #grid(
    columns: 1fr,
    gutter: 0.5em,
    figure(
      table(
        columns: (auto, auto),
        stroke: 0.4pt,
        inset: 3pt,

        table.cell(fill: luma(230))[*Hyperparameter*],
        table.cell(fill: luma(230))[*Value*],

        // === Model Architecture ===
        table.cell(fill: luma(220), colspan: 2)[*Model Architecture*],

        [Cutoff radius ($r_max$)], [$8$],

        // === Training ===
        table.cell(fill: luma(220), colspan: 2)[*Training*],

        [Max epochs], [$40$],
        [Validation fraction], [$0.15$],
      ),
      caption: [Hyperparameter values],
    )
  )
  Intended changes, channels 128 max 𝑙 1 max ℓ 3 interactions 2 correlation 3 radial basis functions 8 cutoff basis functions 5
]

== Resources Employed - Second Training

== Mace auto generated results - Second Training

== Group Results - Second Training


== Takeaways - Second Training



== Iterative Learning

#figure(
  image("images/iterative_training.png", width: 40%),
  caption: [Iterative training example],
)

== Active Learning

#figure(
  image("images/active_learning_commitee.png", width: 40%),
  caption: [Active learning example],
)
