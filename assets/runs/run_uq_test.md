# Run the UQ Test

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python ≥ 3.10** | `python3 --version` |
| **`uv`** package manager | Install it via `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **~10 GB free disk** | Mostly for model weights + dataset |
| **Internet** | Downloads the dataset (~20 MB) and foundation model (~20 MB) from HuggingFace |

> `uv` can also be installed via pip: `pip install uv`

### 1. Clone & enter the repo

```bash
git clone https://github.com/jorgemunozl/mlpdft.git
cd mlpdft
```

### 2. Create the environment

```bash
uv sync
```

### 3. Run training

> Use tmux or other tool to let the cluster running Needs a CUDA GPU.

```bash
uv run python src/mlpdft/multi-train.py
```
## 4. Give me the results

```bash
tar czf uq_test_output.tar.gz \
  -C src/mlpdft/outputs/uq_diffusion_comparison lif_uq_001
```

Send me `uq_test_output.tar.gz`.
