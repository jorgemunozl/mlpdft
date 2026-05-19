# mlpdft — MACE vs FitSNAP (LiF proof of concept)

Experiments and helpers for comparing **[MACE](https://github.com/ACEsuit/mace)** to **[FitSNAP](https://github.com/FitSNAP/FitSNAP)** on lithium–fluoride DFT data. This is not a general-purpose library: it holds runnable scripts, evaluation code, trained FitSNAP artifacts, and slides.

Training data and FitSNAP input decks live **outside** the repo (shared project storage). Pass paths explicitly to the scripts below.

## Layout

```text
.
├── beamer/           # Beamer deck (dataset, FitSNAP baseline, MACE, evaluation)
├── model_LiF/        # FitSNAP outputs for the LiF model used in comparisons
├── scripts/          # FitSNAP runners, converters, demos, small MACE utilities
├── src/              # MACE evaluation on held-out frames
├── pyproject.toml    # uv: mace-torch + CPU PyTorch
└── README.md
```

## Trained model (`model_LiF/`)

Checkpoints, SNAP potential files, and FitSNAP metrics for the **LiF** potential (the one used for tests in this project). See [model_LiF/README.md](model_LiF/README.md).

Typical contents: `LiF_Pytorch.pt`, `FitTorch_Pytorch.pt`, `LiF64_NEWJSON_pot.*`, `perconfig.dat`, `LiF-example.in`.

## Python environment

[`pyproject.toml`](pyproject.toml) installs **`mace-torch`** with **`torch` from the PyTorch CPU wheel index** (avoids pulling CUDA wheels on Linux).

```bash
cd /path/to/mlpdft
uv sync
# Optional: keep MACE-MP checkpoints under the repo
export XDG_CACHE_HOME=$PWD/.cache
```

## Scripts (`scripts/`)

| Path | Role |
|------|------|
| `run_fitsnap3_patched.py` | Run FitSNAP 3 from a `.in` file. Paths resolve from the **repo root**. Patches `randint` for **Python 3.14+**. |
| `run_lif64_fitsnap.py` | Convenience wrapper around `run_fitsnap3_patched.py` for a LiF64 NEWJSON deck (you supply the `.in` file and JSON directory). |
| `fitsnap_json_scrape.py` | Minimal scrape: `scrape_groups` → `motion_configs` → `scrape_configs`. |
| `fitsnap_snap_matrix.py` | `snap_design_matrix()`: scrape → `process_configs` → SNAP bispectrum matrix `A`. |
| `snap_bispectrum.py` | CLI: print `A.shape`, optional `-o` saves NumPy `.npy`. Needs LAMMPS with SNAP. |
| `fitsnap_json_to_extxyz.py` | FitSNAP-style JSON → `train.xyz` / `test.xyz` (same 80/20 split as FitSNAP `random_sampling=0`). Requires `--json-dir` and `--out-dir`. |
| `mace_on_qe_out.py` | Last frame from a Quantum ESPRESSO **pw.x** `.out` → **MACE-MP** energy/forces on CPU. |
| `print_mace_model_arch.py` | Load a MACE `.model` and print `r_max`, `num_interactions`, `hidden_irreps`, etc. |
| `random_energy_demo.py` | Pedagogical: JSON frames → toy features → random linear “energy”. |
| `toy_energy.py` | Shared toy-feature helpers. |
| `random_energy_fitsnap.py` | CLI combining scrape + toy energy (no bispectrum). |

Pedagogical demos need a `--glob` (or similar) pointing at your JSON frames; there is no bundled dataset in this repo.

## MACE evaluation (`src/`)

Configure and run via `MaceEvalConfig` (no CLI flags):

```python
from pathlib import Path
from mace_eval_fitsnap_test import MaceEvalConfig, run

cfg = MaceEvalConfig(
    test_extxyz=Path("/path/to/test.xyz"),
    mace_model=Path("/path/to/checkpoints/final.model"),  # omit for MACE-MP zero-shot
    device="cuda",
    out_csv=Path("outputs/mace_eval.csv"),
)
run(cfg)
```

For JSON or `perconfig.dat` splits, set `json_root` and optionally `perconfig`. Per-config energy and force errors are in **eV** and **eV/Å**.

Or execute the `if __name__ == "__main__"` block in [`src/mace_eval_fitsnap_test.py`](src/mace_eval_fitsnap_test.py) after editing the example `MaceEvalConfig` there.

To build `test.xyz` from JSON elsewhere:

```bash
uv run python scripts/fitsnap_json_to_extxyz.py \
  --json-dir /path/to/NEWJSON/DEFAULT \
  --out-dir /path/to/xyz
```

## Beamer

[`beamer/main.tex`](beamer/main.tex) — proof-of-concept deck: external dataset catalog, FitSNAP baseline (`model_LiF/`), MACE architecture, zero-shot and finetuning notes.

```bash
cd beamer && pdflatex main.tex
```

## Prerequisites

- **FitSNAP 3** (`fitsnap3lib`) and, for SNAP bispectrum workflows, **LAMMPS** with SNAP support.
- **uv** (or another way to install from `pyproject.toml`) for MACE helpers and `src/mace_eval_fitsnap_test.py`.

## Run FitSNAP (external data)

Point `run_fitsnap3_patched.py` at a FitSNAP `.in` file on your machine (with `dataPath` set to your JSON):

```bash
uv run python scripts/run_fitsnap3_patched.py /path/to/your.deck.in
```

Outputs follow the `[OUTFILE]` section of that input (potential files, metrics, `perconfig.dat`, etc.).
