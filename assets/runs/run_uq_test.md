# Run the UQ Test

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.11** | Pinned by `.python-version`; torch 2.6 has no 3.15 wheels |
| **`uv`** package manager | Install it via `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **~10 GB free disk** | Mostly for model weights + dataset |
| **Internet** | Only on the login node; compute nodes are offline (download the dataset, model and replay file first) |

> `uv` can also be installed via pip: `pip install uv`

### 1. Clone & enter the repo

```bash
git clone https://github.com/jorgemunozl/mlpdft.git
cd mlpdft
```

### 2. Create the environment

```bash
uv sync --locked
```

If `uv` picks the wrong interpreter, run `uv python pin 3.11` first.

### 3. Run training

> Use tmux or another tool so the run survives disconnects. Needs a CUDA GPU.

`uq_compare.py` is the runner (not `multi-train.py`):

```bash
uv run --locked python -m mlpdft.uq_compare --run-id lif_uq_001
```

The output lands in `src/mlpdft/outputs/uq_diffusion_comparison/lif_uq_001`.
## 4. Give me the results

```bash
tar czf uq_test_output.tar.gz \
  -C src/mlpdft/outputs/uq_diffusion_comparison lif_uq_001
```

Send me `uq_test_output.tar.gz`.
