# Run Training

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python ≥ 3.10** | `python3 --version` |
| **`uv`** package manager | Install it via `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **~10 GB free disk** | Mostly for model weights + dataset |
| **Internet** | Downloads the dataset (~100 MB) and foundation model (~20 MB) from HuggingFace |

> `uv` can also be installed via pip: `pip install uv`

---

## 🪜 Step-by-step

### 1. Clone & enter the repo

```bash
git clone https://github.com/jorgemunozl/mlpdft.git
cd mlpdft
```

### 2. Create the environment

```bash
uv sync
```

This installs everything: `torch`, `mace-torch`, `huggingface-hub`, on a virtual environment created by `uv`.

### 3. Install the local dependencies

```bash
uv pip install -e .
```

### 4. Run training

> Use tmux to let the cluster running

```bash
uv run python src/mlpdft/train.py
```

Let the train finish. It will appear a directory `src/mlpdft/outputs/mace_omat_lora_v2`

### 5. Package the output

```bash
tar czf training_output.tar.gz -C src/mlpdft/outputs mace_omat_lora_v2
```

This produces a **~25 MB** tarball on the `mlpdft` directory.

### 6. Send me the tarball

Attach `training_output.tar.gz` via **WhatsApp** or any convenient
channel.

---
## Troubleshooting

| Problem | Likely fix |
|---------|-----------|
| `uv` not found | Install it: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `torch` CUDA error | Run `uv sync` after editing `pyproject.toml` to use `pytorch-cuda` index |
| Dataset download fails | Retry — HuggingFace may be rate-limited. Run the script again |
| `snapshot_download` hangs | Check your internet / proxy settings |
| Out of memory (`CUDA OOM`) | Reduce `batch_size` in `train.py` (try `batch_size=2` or `1`) |
| Permission error writing to `outputs/` | You're inside the repo — it should work. Check disk space |
