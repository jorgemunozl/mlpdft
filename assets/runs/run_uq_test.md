# Run the UQ Test

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.11** | Pinned by `.python-version`; torch 2.6 has no 3.15 wheels |
| **`uv`** package manager | Install it via `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **~10 GB free disk** | Mostly for model weights + dataset |
| **Internet** | Only on the login node; compute nodes are offline (pre-download the splits and the foundation model) |

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

`uq_compare.py` is the runner (not `multi-train.py`). Run it **without flags**:

```bash
uv run --locked python -m mlpdft.uq_compare
```

With no flags this runs the corrected experiment: the frozen-body multi-head
committee (`mhc`) plus its point-estimate control (`frozen-probe`), 8
random-init heads on 80% overlapping data, force-only loss, 120 MHC epochs and
600 probe epochs. It writes to
`src/mlpdft/outputs/uq_diffusion_comparison/lif_uq_mhc_003`.

If the previous run directory `lif_uq_001` is still present, the committee and
snapshot results are merged into the new summary automatically; if it is absent
the run still completes with the two new arms plus a fresh zero-shot evaluation.

Optional smoke test before the production run (this one needs flags):

```bash
uv run --locked python -m mlpdft.uq_compare \
  --smoke-test --run-id lif_uq_mhc_smoke --overwrite
```

Delete or rename `lif_uq_mhc_003` before rerunning, otherwise the runner stops
with a clear "run directory already exists" error.

## 4. Give me the results

```bash
tar czf uq_mhc_output.tar.gz \
  -C src/mlpdft/outputs/uq_diffusion_comparison lif_uq_mhc_003
```

Send me `uq_mhc_output.tar.gz`.
