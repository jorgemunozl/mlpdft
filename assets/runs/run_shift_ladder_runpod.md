# Run the shift-ladder baseline on RunPod (12 h)

One A40 pod runs two things: a 30-minute smoke run, then the 1,000-label
baseline (about 8 h). Results go to the private Hugging Face repository after
each arm, so a run stopped early keeps everything that finished.

## Prerequisites

| Requirement | Notes |
|---|---|
| **RunPod pod** | 1× A40 (48 GB), PyTorch template (CUDA 12), container disk ≥ 50 GB |
| **Hugging Face token** | Write access to `jorgemunozl` (private dataset, model and results) |
| **tmux** | Keeps the run alive if SSH drops (`apt-get update && apt-get install -y tmux` if missing) |

### 1. Get the code

```bash
git clone https://github.com/jorgemunozl/mlpdft.git
cd mlpdft
git checkout orx/all-arms-1-000-bulk-labels
pip install uv
tmux new -s run
export HF_TOKEN=hf_...   # inside tmux; never commit it
```

### 2. Smoke run (about 30 min)

```bash
NODE_ENV=configs/shift_ladder/smoke.env RUN_ID=smoke-runpod bash scripts/run_shift_ladder.sh
```

It should end with:

- `failed: none`;
- a projection table ending with `deadline 630 min (10.5 h): fits`.

If anything failed, or the projection says `DOES NOT FIT`, stop here and send
me the end of the log:

```bash
tail -150 run_outputs/smoke-runpod/run.log
```

### 3. Baseline (about 8 h)

```bash
DEADLINE_MINUTES=630 RUN_ID=baseline-1000-s1 bash scripts/run_shift_ladder.sh
```

Detach with `Ctrl-b d` and reattach with `tmux attach -t run`.
`DEADLINE_MINUTES=630` stops the run at 10.5 h at the latest, even if it is
slower than projected; every arm that finished is already uploaded.

### 4. Stop the pod

When the output ends with `uploaded ... to jorgemunozl/li-f-uq-shift-ladder-runs/baseline-1000-s1`,
terminate the pod in RunPod. **RunPod keeps billing until you do.**

### 5. Tell me

Send me the `completed`, `failed` and `not run` lines printed just before the
last upload. Models, metrics and the full log are in
`jorgemunozl/li-f-uq-shift-ladder-runs`, under `smoke-runpod/` and
`baseline-1000-s1/`.

## Time budget

| Step | Time |
|---|---|
| Pod start, clone, first install | ~10 min |
| Smoke run | ~30 min |
| Baseline | ~8 h (stops at 10.5 h at the latest) |
| **Total** | **under 12 h** |
