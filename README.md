# mlpdft, Message Passing Neural Networks for Ionic Molecular Dynamics

![MPNN Architecture](beamer/images/mpnn_illustration.png)

Inference study
Active learning
Experiments around **fine-tuning MACE foundation models**
(MACE-MP, MACE-OFF, MACE-OMAT) on lithium–fluoride DFT (Quantum ESPRESSO)
data, plus comparison with a **FitSNAP** SNAP baseline.

---

**Hugging Face**

| Resource | Link |
|----------|------|
| Dataset | [`jorgemunozl/minimal_li_f_mace_dataset`](https://huggingface.co/datasets/jorgemunozl/minimal_li_f_mace_dataset) |
| Fine-tuned MACE-OMAT model | [`jorgemunozl/mace_omat_medium`](https://huggingface.co/jorgemunozl/mace_omat_medium) |

---

## Dataset

All DFT data lives under `dataset/`. Each group is a Quantum ESPRESSO
`pw.x` run:

| Group | Description |
|-------|-------------|
| `LIF64_KJPAW_V2` | Bulk LiF — NVE / NPT trajectories |
| `LIF64_ISOLATED` | Isolated bulk LiF frames |
| `LIFINTERFACE_KJPAW_V1` | LiF interface (first version) |
| `LIFINTERFACE_KJPAW_NPT` | LiF interface — NPT |
| `LIFINTERFACE_KJPAW_NPT_V2` | LiF interface — NPT (second version) |
| `LIWITHF_V3` | Li + F slabs (third version) |
| `LIWITHF_ISOLATED` | Isolated Li + F frames |
| `LIWITHF_NPT_FINAL` | Li + F — NPT (final) |
| `BLI_V2` | B–Li system |
| `LIBF4_V4` | Li–B–F system |
| `LIBF4` | Li–B–F |

The full merged dataset is published on [Hugging Face](https://huggingface.co/datasets/jorgemunozl/minimal_li_f_mace_dataset).

### Convert QE output → `.extxyz`

```bash
uv run python -c "
from mlpdft.config import MaceConfig
from mlpdft.utils.qe_out_to_extxyz import convert_qe_out_to_extxyz

config = MaceConfig(
    group='LIFINTERFACE_KJPAW_V1',
    frame_stride=5,
    max_frames=None,       # use all frames after striding
)
convert_qe_out_to_extxyz(config)
"
```

Each configuration carries:
- `REF_energy` — total DFT energy (eV) in `atoms.info`
- `REF_forces` — per-atom forces (eV/Å) in `atoms.arrays`
- `stress` — stress tensor (when available)
- `config_type` — group label

### Merge all groups & upload to Hugging Face

```bash
export HF_TOKEN="hf_…"
uv run python src/mlpdft/utils/upload_ds_hf.py
```

The script reads every group's `.extxyz`, merges all frames into a single
file, builds a dataset card from
[`dataset_readme_template.md`](src/mlpdft/utils/dataset_readme_template.md),
and uploads everything to the Hub.

---

## Fine-tuning MACE

[`src/mlpdft/train.py`](src/mlpdft/train.py) fine-tunes a MACE foundation
model with **LoRA** on any of the available groups:

```bash
uv run python src/mlpdft/train.py
```

The script uses `Mace_TrainerConfig` — edit the config at the top of the
file to change group, model key, hyperparameters, etc.:

```python
config = Mace_TrainerConfig(
    model_key="0-small",          # or "0b3-medium", "0-omat-medium"
    group="LIF_KJPAW",            # dataset group
    experiment_name="small_first_second",
    r_max=0.1,
    device="cpu",
    max_num_epochs=2,
    batch_size=1,
    lora=True,
    lora_rank=8,
)
```

Checkpoints, models, logs, and results are written to
`src/mlpdft/outputs/`.

### Shell alternative

A full CLI template with every `mace_run_train` flag is at
[`scripts/run_train.sh`](scripts/run_train.sh).

---

## Evaluation

### MACE on a single group

```python
from mlpdft.config import MaceConfig
from mlpdft.evaluate import eval

config = MaceConfig(
    model_key="0-omat-medium",
    group="LIF_KJPAW",
    frame_stride=10,
    max_frames=20,
)
eval(config)
```

Predictions are written to an `.extxyz` file with `energy`, `forces`, and
optionally `node_energy` prefixed fields.

### Batch MACE evaluation (all groups)

[`src/mlpdft/evaluate_mace_metrics.py`](src/mlpdft/evaluate_mace_metrics.py)
runs the `0-omat-medium` model on **all groups** and prints a summary table
with energy/force MAE, RMSE, and MaxAE:

```bash
uv run python src/mlpdft/evaluate_mace_metrics.py
```

### FitSNAP single-point evaluation

```bash
uv run python scripts/evaluate_fitsnap_singlepoint.py
```

Loads a trained FitTorch model into LAMMPS + mliappy, evaluates on `.extxyz`
frames, and writes metrics `.txt` files to `src/mlpdft/fitsnap/metrics/`.

### Inspect a FitSNAP checkpoint

```bash
uv run python scripts/inspect_fitsnap_pt.py fitsnap_models/LI_F/checkpoints/LiF_Pytorch.pt
```

---

## Molecular Dynamics

[`src/mlpdft/md.py`](src/mlpdft/md.py) runs MD from the first frame of a
dataset using a MACE potential:

```python
from mlpdft.config import MaceConfig
from mlpdft.md import run_md

config = MaceConfig(
    model_key="0-omat-medium",
    group="LIF_KJPAW",
    frame_stride=10,
    max_frames=20,
)
run_md(config, temperature_K=300.0, n_steps=10_000)
```

Supported thermostats: Langevin, Nose–Hoover chain, Velocity Verlet.
Trajectories and logs are saved alongside the model output directory.

---

## FitSNAP metrics

Stored in [`src/mlpdft/fitsnap/metrics/`](src/mlpdft/fitsnap/metrics/).
A Typst table summarises energy and force errors across all datasets:

```bash
typst compile src/mlpdft/fitsnap/metrics/metrics_table.typ
```

---

## Python environment

[`pyproject.toml`](pyproject.toml) installs **`mace-torch`** and
**`huggingface-hub`** with **`torch` from the CPU-only PyTorch index**
(avoids pulling NVIDIA CUDA wheels on Linux).

```bash
cd /path/to/mlpdft
uv sync
# Cache MACE‑MP checkpoints under the repo
export XDG_CACHE_HOME=$PWD/.cache
```

---

## Presentation

[`beamer/main.typst`](beamer/main.typst) — proof-of-concept deck (Typst).

```bash
typst compile beamer/main.typst
```

---

## Prerequisites

- **`uv`** — Python package manager
- **MACE foundation models** — downloaded automatically on first use to
  `~/.cache/mace/` (or `$XDG_CACHE_HOME`)
- **FitSNAP / LAMMPS** — only needed for FitSNAP evaluation scripts
  (`evaluate_fitsnap_singlepoint.py`, `evaluate_dataset.py`)
- **Hugging Face token** — only needed for `upload_ds_hf.py`
  (`export HF_TOKEN="hf_…"` or `huggingface-cli login`)
