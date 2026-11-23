#!/usr/bin/env bash
# Shift-ladder run (ADR-0007): train the UQ arms on one label rung of the 40 Ry
# bulk LiF pool on one GPU, evaluate every arm on every set, and upload models
# and results to a private Hugging Face repository (the GPU box is deleted when
# the run ends). Per-node settings: configs/shift_ladder/node.env.
set -euo pipefail
report() { printf '::orx:: %s\n' "$1"; }

set -a  # export the node settings to the Python steps
source configs/shift_ladder/node.env
set +a
run_id=${ORX_RUN_ID:-manual}
root="$PWD/run_outputs/$run_id"
data="$root/data"
mkdir -p "$root" "$data"
export PYTHONUNBUFFERED=1
: "${HF_TOKEN:?HF_TOKEN must be set (private dataset, model and results)}"
printf 'run %s: %s labels, seeds %s, arm groups: %s\n' "$run_id" "$LABELS" "$SEEDS" "$ARM_GROUPS"

report '{"phase":"install"}'
if command -v uv >/dev/null; then
  uv sync --locked --no-dev
else
  printf '%s\n' 'uv unavailable: using venv/pip; pip does not reproduce uv.lock.'
  python3 -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q .
fi
py="$PWD/.venv/bin/python"
"$py" -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('GPU:', torch.cuda.get_device_name(0))"

report '{"phase":"download"}'
"$py" - "$data" <<'EOF'
import hashlib, json, os, sys
from huggingface_hub import snapshot_download
data = sys.argv[1]
snapshot_download(repo_id=os.environ["DATA_REPO"], repo_type="dataset",
                  revision=os.environ["DATA_REVISION"], local_dir=data)
manifest = json.load(open(os.path.join(data, "manifest.json")))
for name, meta in manifest["sets"].items():
    digest = hashlib.sha256(open(os.path.join(data, f"{name}.extxyz"), "rb").read()).hexdigest()
    if digest != meta["sha256"]:
        raise SystemExit(f"{name}: checksum mismatch")
print(f"dataset {os.environ['DATA_REVISION'][:10]}: {len(manifest['sets'])} sets verified")
EOF
"$py" - "$data" "$LABELS" <<'EOF'
import sys
import numpy as np
from ase.io import read, write
data, labels = sys.argv[1], int(sys.argv[2])
pool = read(f"{data}/train_pool.extxyz", index=":")
if labels > len(pool):
    raise SystemExit(f"rung {labels} exceeds the pool of {len(pool)} frames")
idx = np.unique(np.linspace(0, len(pool) - 1, labels).round().astype(int))
write(f"{data}/train_rung.extxyz", [pool[i] for i in idx], format="extxyz")
print(f"training rung: {len(idx)} of {len(pool)} pool frames, evenly spaced")
# The runner's own evaluation files; capped evenly when UQC_MAX_EVAL_FRAMES is set (smoke runs).
import os
cap = int(os.environ.get("UQC_MAX_EVAL_FRAMES", "0") or 0)
for name, out in (("test_timeblock", "uqc_test"), ("monitor", "uqc_calibration")):
    frames = read(f"{data}/{name}.extxyz", index=":")
    if cap and len(frames) > cap:
        frames = [frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, cap).round().astype(int))]
    write(f"{data}/{out}.extxyz", frames, format="extxyz")
    print(f"runner {out}: {len(frames)} frames from {name}")
EOF

read -r -a groups <<< "$ARM_GROUPS"
report "{\"phase\":\"train\",\"step\":0,\"of\":${#groups[@]},\"unit\":\"arm groups\"}"
done_groups=0
for group in "${groups[@]}"; do
  start=$SECONDS
  # shellcheck disable=SC2086  # EXTRA_ARGS holds several options
  "$py" -m mlpdft.uq_compare \
    --train-file "$data/train_rung.extxyz" --valid-file "$data/monitor.extxyz" \
    --calibration-file "$data/uqc_calibration.extxyz" --test-file "$data/uqc_test.extxyz" \
    --arms "$group" --seeds "$SEEDS" --output-dir "$root/train" --run-id "${group//,/+}" \
    --device cuda $EXTRA_ARGS
  done_groups=$((done_groups + 1))
  minutes=$(( (SECONDS - start + 30) / 60 ))
  report "{\"step\":$done_groups,\"m\":{\"train_minutes_total\":$(( SECONDS / 60 ))}}"
  report "{\"row\":\"timing\",\"m\":{\"arm_group\":\"$group\",\"minutes\":$minutes}}"
done

# shellcheck disable=SC2086  # EVAL_ARGS holds several options
"$py" -m mlpdft.uq_evaluate --run-dir "$root/train" --dataset-dir "$data" --device cuda --report-orx ${EVAL_ARGS:-}

report '{"phase":"upload"}'
"$py" - "$root" "$run_id" <<'EOF'
import os, sys
from huggingface_hub import HfApi
root, run_id = sys.argv[1], sys.argv[2]
api = HfApi()
repo = os.environ["RESULTS_REPO"]
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
info = api.upload_folder(repo_id=repo, repo_type="model", folder_path=os.path.join(root, "train"),
                         path_in_repo=run_id, commit_message=f"run {run_id}",
                         ignore_patterns=["**/checkpoints/**", "**/*.pt"])
print(f"uploaded to {repo}/{run_id} at revision {info.oid}")
EOF
