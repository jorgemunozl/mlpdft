# Run Inference

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python ≥ 3.10** | `python3 --version` |
| **`uv`** package manager | Install it via `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

> `uv` can also be installed via pip: `pip install uv`

---

## 🪜 Step-by-step

### 1. Clone & enter the repo (if not already done)

```bash
git clone https://github.com/jorgemunozl/mlpdft.git
cd mlpdft
```

### 2. Create the environment

```bash
uv sync
```

This installs everything: `torch`, `mace-torch`, `huggingface-hub`, `wandb`,
and the `mlpdft` package itself (editable) in a virtual environment created by `uv`.

### 3. Run inference

> Use tmux to let the cluster running or other similar tool.

```bash
uv run python src/mlpdft/inference.py
```

Let the inference finish.

### 4. Package the output

```bash
tar czf inference.tar.gz -C 
```

This produces `inference.tar.gz` in the repo root with everything I need

### 5. Send me the tarball

Attach `training_output.tar.gz` via **WhatsApp** or any convenient channel.

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
