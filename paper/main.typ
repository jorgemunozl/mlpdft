#set page(margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 17pt)[
    Foundation Models for MACE Machine Learning Interatomic Potentials
  ] \
  #text(size: 12pt)[{working title}]

  #v(1em)

  #text(size: 12pt)[
    Jorge A. Munñz#super[\*1], {edit your name as you want}
    Diego E. Galvez-Aranda#super[3],
    Jorge M. Seminario#super[3,4,5], and
    Luis A. Selis#super[1,2,\*]
  ]

  #v(0.5em)

  #text(size: 10pt)[
    #super[1]Facultad de Ciencias, Universidad Nacional de Ingeniería, Av. Túpac Amaru 210, Lima, Perú \
    #super[2]Facultad de Ingeniería Eléctrica y Electrónica, Universidad Nacional de Ingeniería, Av. Túpac Amaru 210, Lima, Perú \
    #super[3]Department of Chemical Engineering, Texas A&M University, College Station, TX 77843, USA \
    #super[4]Department of Electrical and Computer Engineering, Texas A&M University, College Station, TX 77843, USA \
    #super[5]Department of Materials Science and Engineering, Texas A&M University, College Station, TX 77843, USA

    #v(0.3em)
    #super[\*]Corresponding author: #link("mailto:luis.selis.v@uni.edu.pe")[luis.selis.v\@uni.edu.pe], Telephone: +51 940 117 469
  ]
]

#v(1em)

#align(center)[*Abstract*]

Obtain a committe of models to make active learning is expensive. I propose a new approach to active learning that uses a committee of models to make predictions.
// ABSTRACT

#v(0.5em)

*Keywords:* Machine learning; MACE; Foundation models; Active learning; Li-ion; Molecular Dynamics; Nanotechnology

#v(1em)

= Introduction

Foundational models are revolutionizing molecular dynamics.

Active learning is actually way too important. Without it, molecular dynamics simulations would crash.
Static datasets are not enough.

Committee active learning requires a diverse set of models to make predictions. And obtain good deviations

Post training for foundation models looses its diversity and collapses into a single model. Becuase the seed that makes the model differs only affects the data shuffling. Lack of diversity in the model can lead to collapse into a single local minima.


https://www.researchgate.net/publication/370344599_Fast_uncertainty_estimates_in_deep_learning_interatomic_potentials

Wait the guys from above do really interesting stuff.

== Related Work

Here NACE OMAT is way too important.

Snapshot Ensembles Train one and get M for free.

Comparative Study of

Cite the 2025 paper on Snapshot Ensembles for MLIPs (Kurniawan et al.).
Cite papers on MLIP Active Learning.



*   **The Theoretical Methodology:**
    *   Write down the math of Cosine Annealing with Warm Restarts (SGDR).
    *   Write the equation for calculating variance (Uncertainty) from a committee of models.

*(Do not write the Results or Discussion yet! Wait for the cluster.)*

### 2. Plan B: What if the PoC "horribly fails"?
Let's define what a "failure" looks like: You plot the graph of *Predicted Uncertainty vs. True Error*, and it looks like a random cloud of dots instead of a nice diagonal line.

If that happens, do not panic. It just means the models didn't get enough "diversity." If it fails, here is your immediate pivot:

*   **Fix 1 (The Easy Fix):** The "Warm Restart" (the spike in the learning rate) wasn't high enough to kick the model out of the foundation model's valley. You simply increase `lr_max` in your scheduler and run it again.
*   **Fix 2 (The Multi-Head Pivot):** If snapshots stubbornly refuse to work, you change a few lines of code to freeze the MACE message-passing layers and initialize 3 different random linear readout layers (Multi-Head). It's incredibly cheap and guarantees diversity at the final step.
*   **Fix 3 (MC Dropout):** You just turn on Dropout during inference. It’s a classic, zero-cost UQ method.

By having a Plan B, you don't have to stress about the PoC. You are acting like a true researcher: testing a hypothesis, and adjusting if nature says no.

### 3. The Coding Alternative (Zero-Waste Effort)
If you don't feel like writing text, there is one piece of code you can write right now that you will 100% need, whether you use Snapshots, Deep Ensembles, or Multi-Head models.

**The Automated Quantum Espresso (QE) Bridge (`qe_oracle.py`).**
You know that eventually, your LAMMPS MD simulation will pause and hand a `.xyz` frame to Python. Python needs to:
1. Read the `.xyz` frame using `ase.io.read`.
2. Convert it into a `pw.scf.in` file (Quantum Espresso input format).
3. Generate a Slurm `.sh` script (e.g., `sbatch run_qe.sh`).
4. Wait for QE to finish computing the forces and energies.
5. Parse the output and append it to your MACE training dataset.

Since you are waiting on the cluster anyway,would you like me to write the Python/ASE code for this automated Quantum Espresso bridge? You can copy-paste it into your repo, and it will be ready to go the second your PoC finishes.

= Methodology

Trian this like that.


== MACE methods

== DFT data

== Model training and hyperparameters

= Results and Discussion

== Accuracy of energies and forces

== Improvement respect to Spectral Neighbor Analysis Potential

== Limitations of the Model

= Conclusions

#heading(numbering: none)[Acknowledgements]

We appreciate the support of computational resources from Texas A&M High Performance Research Computing (TAMU-HPRC).

#heading(numbering: none)[References]
