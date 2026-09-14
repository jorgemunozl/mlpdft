# Run Diffusion (MD) on the Cluster

## Prerequisites

- **Python ≥ 3.10** and **`uv`** (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **NVIDIA GPU + CUDA**
- **~10 GB free disk** and **internet**

## Step 1 — Setup

```bash
git clone https://github.com/jorgemunozl/mlpdft.git
cd mlpdft
uv sync
```

## Step 2 — Run (this is the only run step)

```bash
tmux new -s diffusion
uv run python src/mlpdft/utils/arrhenius.py
```

This runs NVT molecular dynamics at 300, 400, 500, and 600 K on the GPU,
extracts the diffusion coefficient at each temperature, and fits the Arrhenius
activation energy. Everything is written to `src/mlpdft/outputs/diffusion/`:

- `msd_<T>K.csv` — mean-squared displacement vs. time at each temperature
- `arrhenius.csv` — temperature and diffusion coefficient
- `summary.txt` — D values, activation energy E_a, and prefactor D_0

## Step 3 — Send me the results

```bash
tar czf diffusion_output.tar.gz -C src/mlpdft outputs/diffusion
```

Attach `diffusion_output.tar.gz`. It already contains the D values,
`E_a` and `D_0` in `summary.txt`.

---

## Troubleshooting

| Problem | Likely fix |
|---------|-----------|
| `uv` not found | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `torch` CUDA error | `uv sync` after pointing `pyproject.toml` at the `pytorch-cuda` index |
| `CUDA OOM` | Use a smaller cell / fewer atoms |
| MSD plateaus instead of growing | Unwrapping failed; confirm the cell is periodic (`pbc=True`) |
| Noisy/negative slope | Increase `--steps` and `--equil-steps` |
| `cuequivariance not available` | Optional speedup only; MACE still runs (slower) |
