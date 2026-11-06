#!/usr/bin/env bash
set -euo pipefail

# One A40, eight wall-clock hours including setup, smoke, training and evaluation.
# An unfinished timed-out run is a failure, not an experimental result.
if [[ "${1:-}" != "--bounded" ]]; then
  if ! command -v timeout >/dev/null; then
    printf '%s\n' 'GNU timeout is required to enforce the eight-hour budget.' >&2
    exit 1
  fi
  exec timeout --signal=TERM --kill-after=60s 8h bash "$0" --bounded
fi

task_start_seconds=$SECONDS
task_budget_seconds=28800
task_reserve_seconds=900
task_id=${ORX_RUN_ID:-manual-preflight}
task_out="${PWD}/run_outputs/${task_id}"
task_data="${PWD}/dataset/xyz_files/uq_interim_two_env"
task_model="${PWD}/src/mlpdft/outputs/mace_omat_medium/mace_omat_medium.model"
task_old_model="${PWD}/assets/research/PHASE-2/experiment3_runs/mhc_pilot_2026-11-05/training/mhc_saved.model"
task_stage="${MHC_STAGE_ROOT:-${HOME}/.cache/mlpdft_mhc_4d}"
mkdir -p "$task_out"
printf 'Budget: 8h including setup, smoke, training and evaluation; run %s\n' "$task_id"
printf '%s  %s\n' 'dd6c542a4fc10d7a9e4a724cb06ba07b6a3fda22dc735f343bb8c1baa4b67a11' "$task_old_model" | sha256sum --check --status

if command -v uv >/dev/null; then
  uv sync --locked --no-dev
else
  printf '%s\n' 'uv unavailable: using venv/pip; pip does not reproduce uv.lock.'
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install .
fi
task_python="${PWD}/.venv/bin/python"
export PYTHONUNBUFFERED=1
PYTHONPATH="${PWD}/src" "$task_python" -m unittest discover -s tests -v

# Optional private inputs staged over SSH; the preflight checks every hash.
if [[ -d "$task_stage/dataset" && -f "$task_stage/mace_omat_medium.model" ]]; then
  mkdir -p "$task_data" "$(dirname "$task_model")"
  cp "$task_stage"/dataset/uq_{train,valid,calibration,test}.extxyz "$task_data/"
  cp "$task_stage/mace_omat_medium.model" "$task_model"
fi

"$task_python" scripts/prepare_mhc_4d.py \
  --data-dir "$task_data" --model-file "$task_model" --output-dir "$task_out"

task_common=(--arms mhc --train-file "$task_data/uq_train.extxyz"
  --valid-file "$task_data/uq_valid.extxyz" --calibration-file "$task_data/uq_calibration.extxyz"
  --output-dir "$task_out" --mhc-heads 4 --mhc-split disjoint
  --split-seed 20251027 --seeds 123 --batch-size 16 --valid-batch-size 1
  --device cuda --dtype float64 --forces-weight 100 --energy-weight 0
  --learning-rate 0.001 --patience 50)

printf '%s\n' 'Starting 20-epoch smoke on validation frames only.'
"$task_python" -m mlpdft.uq_compare "${task_common[@]}" --run-id smoke \
  --test-file "$task_out/smoke_validation.extxyz" --smoke-test --smoke-epochs 20 --smoke-frames 8

task_remaining=$((task_budget_seconds - (SECONDS - task_start_seconds) - task_reserve_seconds))
if (( task_remaining <= 0 )); then
  printf '%s\n' 'No budget remains after smoke; full run was not launched.' >&2
  exit 1
fi
printf 'Starting 90-epoch maximum full run, with %s seconds available.\n' "$task_remaining"
if timeout --signal=TERM --kill-after=60s "${task_remaining}s" \
  "$task_python" -m mlpdft.uq_compare "${task_common[@]}" --run-id full \
  --test-file "$task_data/uq_test.extxyz" --mhc-epochs 90 --reference-mhc-model "$task_old_model"; then
  "$task_python" - "$task_out/full/evaluation/uq_summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
print('FINAL_UQ_SUMMARY_JSON', json.dumps(json.loads(p.read_text()), sort_keys=True), flush=True)
PY
  printf 'Run completed. Outputs: %s/full\n' "$task_out"
else
  task_exit=$?
  printf 'Full run stopped or failed (exit %s); preserve logs and checkpoints in %s/full.\n' "$task_exit" "$task_out" >&2
  exit "$task_exit"
fi
