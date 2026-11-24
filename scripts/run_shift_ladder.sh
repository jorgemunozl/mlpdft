#!/usr/bin/env bash
# Shift-ladder run (ADR-0007): train the UQ arms on one label rung of the 40 Ry
# bulk LiF pool on one GPU, evaluate every arm on every set, and upload models
# and results to a private Hugging Face repository.
#
# Each arm group is trained, evaluated and uploaded before the next starts, so
# a run stopped early keeps every finished group.
#
# Settings: configs/shift_ladder/node.env, or the file named by NODE_ENV (the
# smoke run uses configs/shift_ladder/smoke.env and ends with a projection of
# the full run's time). Optional environment:
#   DEADLINE_MINUTES  start no new step after this many minutes and stop a step
#                     that would run past it; finished groups stay uploaded
#   RUN_ID            name of the run folder and of its folder in RESULTS_REPO
set -euo pipefail
report() { printf '::orx:: %s\n' "$1"; }
timing() { printf '[timing] %s\n' "$1"; }

node_env=${NODE_ENV:-configs/shift_ladder/node.env}
set -a  # export the node settings to the Python steps
# shellcheck disable=SC1090
source "$node_env"
set +a
run_id=${ORX_RUN_ID:-${RUN_ID:-manual-$(date -u +%Y%m%d-%H%M%S)}}
root="$PWD/run_outputs/$run_id"
data="$root/data"
mkdir -p "$root" "$data"
exec > >(tee -a "$root/run.log") 2>&1
export PYTHONUNBUFFERED=1
: "${HF_TOKEN:?HF_TOKEN must be set (private dataset, model and results)}"
deadline=${DEADLINE_MINUTES:-0}
printf 'run %s (%s): %s labels, seeds %s, arm groups: %s, deadline %s min\n' \
  "$run_id" "$node_env" "$LABELS" "$SEEDS" "$ARM_GROUPS" "$deadline"

remaining() {  # seconds left before the deadline
  if (( deadline > 0 )); then echo $(( deadline * 60 - SECONDS )); else echo 999999; fi
}

run_step() {  # run_step <label> <command...>: the command's status; 124 at the deadline
  local label=$1 left start status=0
  shift
  left=$(remaining)
  if (( left < 120 )); then
    printf 'deadline: not starting %s\n' "$label"
    return 124
  fi
  start=$SECONDS
  if (( deadline > 0 )); then
    timeout --signal=TERM --kill-after=60 "$left" "$@" || status=$?
  else
    "$@" || status=$?
  fi
  timing "{\"kind\":\"step\",\"step\":\"$label\",\"seconds\":$((SECONDS - start)),\"status\":$status}"
  return "$status"
}

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
"$py" -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; p = torch.cuda.get_device_properties(0); print(f'GPU: {p.name}, {p.total_memory / 2**30:.0f} GiB, torch {torch.__version__}')"

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
import os, sys
import numpy as np
from ase.io import read, write
data, labels = sys.argv[1], int(sys.argv[2])
pool = read(f"{data}/train_pool.extxyz", index=":")
if labels > len(pool):
    raise SystemExit(f"rung {labels} exceeds the pool of {len(pool)} frames")
idx = np.unique(np.linspace(0, len(pool) - 1, labels).round().astype(int))
write(f"{data}/train_rung.extxyz", [pool[i] for i in idx], format="extxyz")
print(f"training rung: {len(idx)} of {len(pool)} pool frames, evenly spaced")
# The runner's own files (0 = every frame). uqc_test only feeds MACE's
# end-of-training error table (the evaluation below covers every set);
# uqc_valid is MACE's validation file, used only on the epochs whose
# checkpoint is kept; uqc_calibration (the monitor frames) calibrates LLPR.
# The runner refuses a calibration file identical to another split, so
# UQC_VALID_FRAMES must differ from UQC_CAL_FRAMES.
for name, out, key in (("test_timeblock", "uqc_test", "UQC_TEST_FRAMES"),
                       ("monitor", "uqc_valid", "UQC_VALID_FRAMES"),
                       ("monitor", "uqc_calibration", "UQC_CAL_FRAMES")):
    frames = read(f"{data}/{name}.extxyz", index=":")
    cap = int(os.environ.get(key, "0") or 0)
    if cap and len(frames) > cap:
        frames = [frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, cap).round().astype(int))]
    write(f"{data}/{out}.extxyz", frames, format="extxyz")
    print(f"runner {out}: {len(frames)} frames from {name}")
EOF
timing "{\"kind\":\"setup\",\"seconds\":$SECONDS}"

upload() {  # upload <local folder> <folder in RESULTS_REPO>
  "$py" - "$1" "$2" <<'EOF'
import os, sys
from huggingface_hub import HfApi
folder, path = sys.argv[1], sys.argv[2]
api = HfApi()
repo = os.environ["RESULTS_REPO"]
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
info = api.upload_folder(repo_id=repo, repo_type="model", folder_path=folder, path_in_repo=path,
                         commit_message=f"run {path}", ignore_patterns=["**/checkpoints/**", "**/*.pt"])
print(f"uploaded {folder} to {repo}/{path} at revision {info.oid}")
EOF
}

read -r -a groups <<< "$ARM_GROUPS"
report "{\"phase\":\"train\",\"step\":0,\"of\":${#groups[@]},\"unit\":\"arm groups\"}"
done_groups=0
completed=() failed=() skipped=()
stopped=0
for group in "${groups[@]}"; do
  gid=${group//,/+}
  if (( stopped )); then
    skipped+=("$group")
    continue
  fi
  start=$SECONDS
  status=0
  # shellcheck disable=SC2086  # EXTRA_ARGS holds several options
  run_step "train $group" "$py" -m mlpdft.uq_compare \
    --train-file "$data/train_rung.extxyz" --valid-file "$data/uqc_valid.extxyz" \
    --calibration-file "$data/uqc_calibration.extxyz" --test-file "$data/uqc_test.extxyz" \
    --arms "$group" --seeds "$SEEDS" --output-dir "$root/train" --run-id "$gid" \
    --device cuda $EXTRA_ARGS || status=$?
  if (( status == 0 )); then
    zero_shot=--no-zero-shot
    [[ $group == zero-shot ]] && zero_shot=""
    # shellcheck disable=SC2086  # EVAL_ARGS holds several options
    run_step "evaluate $group" "$py" -m mlpdft.uq_evaluate --run-dir "$root/train/$gid" \
      --dataset-dir "$data" --device cuda --report-orx --orx-rows-only $zero_shot ${EVAL_ARGS:-} \
      || status=$?
  fi
  if (( status == 0 )); then
    completed+=("$group")
  elif (( status == 124 )); then
    skipped+=("$group (stopped at the deadline)")
    stopped=1
  else
    failed+=("$group (exit $status)")
  fi
  if [[ -d "$root/train/$gid" ]]; then
    upload "$root/train/$gid" "$run_id/$gid" || printf 'upload of %s failed; retried at the end\n' "$gid"
  fi
  done_groups=$((done_groups + 1))
  minutes=$(( (SECONDS - start + 30) / 60 ))
  timing "{\"kind\":\"group\",\"group\":\"$group\",\"seconds\":$((SECONDS - start)),\"status\":$status}"
  report "{\"step\":$done_groups,\"m\":{\"train_minutes_total\":$(( SECONDS / 60 ))}}"
  report "{\"row\":\"timing\",\"m\":{\"arm_group\":\"$group\",\"minutes\":$minutes}}"
done

report '{"phase":"upload"}'
mkdir -p "$root/train"
"$py" - "$root/train" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
merged = {"arms": {}}
for path in sorted(root.glob("*/evaluation/metrics.json")):
    report = json.loads(path.read_text())
    for key in ("calibration_set", "id_reference_set", "id_reference_note", "conformal_target", "dtype"):
        merged.setdefault(key, report.get(key))
    merged["arms"].update(report["arms"])
(root / "metrics_all.json").write_text(json.dumps(merged, indent=2))
print(f"merged metrics of {len(merged['arms'])} arms: {', '.join(merged['arms']) or 'none'}")
EOF
if [[ -n "${PROJECT_TO:-}" ]]; then
  "$py" -m mlpdft.project_run_time --log "$root/run.log" --measured "$node_env" --target "$PROJECT_TO" \
    --dataset-dir "$data" --deadline-minutes "${PROJECT_DEADLINE_MINUTES:-630}" \
    | tee "$root/train/projection.txt" || printf 'projection failed\n'
fi
printf 'completed: %s\n' "${completed[*]:-none}"
printf 'failed: %s\n' "${failed[*]:-none}"
printf 'not run: %s\n' "${skipped[*]:-none}"
cp "$root/run.log" "$root/train/run.log"
upload "$root/train" "$run_id"
if (( ${#failed[@]} )); then
  exit 1
fi
