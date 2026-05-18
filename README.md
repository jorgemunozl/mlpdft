# mlpdft — FitSnap testing

This repository holds experiments and helper scripts for **[FitSNAP](https://github.com/FitSNAP/FitSNAP)** (SNAP interatomic potentials trained on quantum data). It is not a general-purpose library; it is a scratch space for running FitSNAP end-to-end and checking JSON-based training pipelines.

## Layout

```text
.
├── beamer/                   # Slides (e.g. MACE vs FitSNAP)
├── configs/fitsnap/          # FitSNAP input decks (.in)
├── configs/mace/             # MACE finetune YAML (LiF64 example)
├── examples/lifbf4/          # LiBF₄: QE inputs, QE→JSON converters, training JSON
├── logs/                     # Run logs (e.g. fitsnap_run.log)
├── scripts/                  # Runnable Python helpers (`fitsnap_json_scrape.py`, `toy_energy.py`, …)
├── LICENSE
└── README.md
```

## Scripts

| Path | Role |
|------|------|
| `scripts/run_fitsnap3_patched.py` | Runs FitSNAP 3 from a `.in` file. Resolves paths from the **repo root** (works whether you run from `.` or `scripts/`). Applies a small monkey-patch so `randint` uses integer bounds (needed on **Python 3.14+**, where FitSNAP 3.1.x may pass floats). Appends to `logs/fitsnap_run.log`. |
| `scripts/random_energy_demo.py` | Small **pedagogical** script: reads FitSNAP-style JSON frames, builds toy features, and prints energies with random linear weights. Default glob is under `examples/lifbf4/NEWJSON/` (repo-root relative). Not physical. |
| `scripts/fitsnap_json_scrape.py` | Minimal helper: patch `randint`, then `scrape_groups` → `divvy_up_configs` → `scrape_configs` for a given `.in`. |
| `scripts/toy_energy.py` | Pedagogy-only features + random linear energy (shared idea with `random_energy_demo.py`). |
| `scripts/random_energy_fitsnap.py` | Thin CLI: calls `fitsnap_json_scrape` + `toy_energy` (no bispectrum). |
| `scripts/fitsnap_snap_matrix.py` | Library helper: `snap_design_matrix()` runs `FitSnap` scrape → `process_configs` → returns design matrix `A` (SNAP bispectrum rows) and the `FitSnap` instance. |
| `scripts/snap_bispectrum.py` | CLI wrapper: prints `A.shape`, optional `-o` saves `A` as NumPy `.npy`. Requires LAMMPS with SNAP. |
| `scripts/mace_on_qe_out.py` | Reads a Quantum ESPRESSO **pw.x** `.out` (last frame), runs **MACE-MP** (`--model`, default `small`) on **CPU**, prints energy and forces; compares to QE forces if present in the OUT file. |
| `scripts/fitsnap_json_to_extxyz.py` | Converts FitSNAP-style JSON (sorted 80/20 split, FitSNAP `random_sampling=0` rule) to **extended XYZ** for MACE (`train.xyz` / `test.xyz`). Default `config_type=Default` matches `config_type_weights` in MACE configs. |
| `scripts/mace_eval_fitsnap_test.py` | Evaluate **MACE-MP** (`--model`) or a finetuned checkpoint (`--mace-model path/to/run.model`) on a JSON test split or on **`--test-extxyz`** (e.g. LiF64 `test.xyz`). Writes CSV with per-config energy/force errors (eV, eV/Å) for tables comparable to FitSNAP-style reporting. |
| `scripts/print_mace_model_arch.py` | Loads a MACE **`.model`** (or extensionless cached checkpoint) via `torch.load`, runs `extract_config_mace_model`, prints `r_max`, `num_interactions`, `hidden_irreps`, `correlation`, etc., for slides or debugging. Default search: `~/.cache/mace`. |

## Configs

| Path | Role |
|------|------|
| `configs/fitsnap/LiBF4-minimal.in` | Example FitSNAP deck (bispectrum / LAMMPSSNAP, JSON scraper). `dataPath` is relative to this file’s directory (see comment in file); with cwd at repo root it resolves to `LiBF4/NEWJSON/<GROUP>/`. |
| `configs/fitsnap/LiFB-example.in` | Larger PyTorch-focused example; `dataPath` points outside this repo (`../LiFB_kjpaw/JSON`) — adjust for your machine. |

## MACE (optional, CPU-only PyTorch via uv)

This repo includes a [`pyproject.toml`](pyproject.toml) that installs **`mace-torch`** with **`torch` from the official PyTorch CPU wheel index** so Linux resolves to **`torch…+cpu`** and does not pull NVIDIA CUDA wheels.

```bash
cd /path/to/mlpdft
uv sync
# First MACE-MP download needs a writable cache (default ~/.cache/mace):
uv run python scripts/mace_on_qe_out.py --qe-out /path/to/LiF64_kjpaw.out
```

Use `XDG_CACHE_HOME` if you want checkpoints under the repo, e.g. `export XDG_CACHE_HOME=$PWD/.cache`.

## LiF64: JSON to MACE finetune and evaluation

QE reference JSON lives under `examples/LiF64_kjpaw_v2/NEWJSON/DEFAULT/` (`output_*.json`). Generate MACE training files (same split as `configs/fitsnap/LiF64-NEWJSON.in`: 80% train / 20% test, sorted filenames):

```bash
uv run python scripts/fitsnap_json_to_extxyz.py \
  --json-dir examples/LiF64_kjpaw_v2/NEWJSON/DEFAULT \
  --out-dir examples/LiF64_kjpaw_v2/xyz
```

Finetune a foundation model (defaults in [`configs/mace/lif64_finetune.yaml`](configs/mace/lif64_finetune.yaml); override `device` / `default_dtype` for your hardware). From **mlpdft** repo root with `mace_run_train` on your `PATH` (e.g. after `uv run` / install):

```bash
mace_run_train --config=configs/mace/lif64_finetune.yaml --device=cuda
```

Evaluate the held-out XYZ (foundation or a trained `*.model` from the run directory):

```bash
uv run python scripts/mace_eval_fitsnap_test.py \
  --test-extxyz examples/LiF64_kjpaw_v2/xyz/test.xyz \
  --mace-model /path/to/checkpoints/your_final.model \
  --out-csv outputs/mace_lif64_eval.csv
```

Omit `--mace-model` to score **MACE-MP** zero-shot on the same frames. Energy **and** force units in the script output match the extended XYZ reference (eV and eV/Å here), useful for slide tables next to FitSNAP metrics.

## Beamer (MACE vs FitSNAP)

[`beamer/main.tex`](beamer/main.tex) is a short deck comparing MACE to the FitSNAP setup under `model/`. Build with:

```bash
cd beamer && pdflatex main.tex
```

## Prerequisites

- **FitSNAP 3** (`fitsnap3lib`) and its dependencies (see the upstream FitSNAP docs), including a **LAMMPS** build with SNAP support if you use the `LAMMPSSNAP` calculator as in the example inputs.
- Python version compatible with your FitSNAP install; use `run_fitsnap3_patched.py` if you hit the `randint` issue on newer Python.

## Run FitSNAP

From the **repository root**, after installing FitSNAP and ensuring JSON data exists where `dataPath` points (see `configs/fitsnap/LiBF4-minimal.in`):

```bash
python scripts/run_fitsnap3_patched.py configs/fitsnap/LiBF4-minimal.in
```

Check `logs/fitsnap_run.log` for status; successful runs also produce outputs named in the `[OUTFILE]` section of the input (e.g. metrics and potential files in the current working directory).

## Random energy demos (optional)

```bash
python scripts/random_energy_demo.py --max_frames 20
python scripts/random_energy_fitsnap.py --max-frames 20
python scripts/snap_bispectrum.py -o outputs/snap_A.npy
```

The demo’s default glob targets `examples/lifbf4/NEWJSON/output_*.json`. Override with `--glob` if your data lives elsewhere. The FitSNAP scraper demo uses `configs/fitsnap/LiBF4-minimal.in` by default.

## License

MIT — see [LICENSE](LICENSE).
