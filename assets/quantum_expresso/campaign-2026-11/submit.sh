#!/bin/bash
# Usage:
#   ./submit.sh all                        preflight, then stage 1 (starts only if the
#                                          preflight passes) and stage 2
#   ./submit.sh <stage folder> [job ...]   one stage, or some of its jobs, by hand
#   ./submit.sh stage1_pilot bulk_svr_s101
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
source "$root/config.env"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT in config.env}"
: "${ESPRESSO_PSEUDO:?set ESPRESSO_PSEUDO in config.env}"
: "${SCRATCH_BASE:?set SCRATCH_BASE in config.env}"
for upf in Li.pbe-sl-kjpaw_psl.1.0.0.UPF F.pbe-n-kjpaw_psl.1.0.0.UPF; do
  [ -f "$ESPRESSO_PSEUDO/$upf" ] || { echo "missing $ESPRESSO_PSEUDO/$upf" >&2; exit 1; }
done

submit() {  # submit <stage> <job> [sbatch options...]; prints the job id
  (cd "$root/$1/$2" && sbatch --parsable --account="$SLURM_ACCOUNT" "${@:3}" job.sbatch) | cut -d';' -f1
}
jobs_in() { (cd "$root/$1" && ls -d */ | tr -d /); }

if [ "${1:-}" = all ]; then
  pre=$(submit stage0_preflight "$(jobs_in stage0_preflight)")
  echo "stage0_preflight: job $pre"
  for j in $(jobs_in stage1_pilot); do
    echo "stage1_pilot/$j: job $(submit stage1_pilot "$j" --dependency=afterok:"$pre" --kill-on-invalid-dep=yes) (waits for the preflight)"
  done
  for j in $(jobs_in stage2_cutoff); do
    echo "stage2_cutoff/$j: job $(submit stage2_cutoff "$j")"
  done
  exit 0
fi

stage=$1; shift
if [ $# -gt 0 ]; then jobs=("$@"); else jobs=($(jobs_in "$stage")); fi
for j in "${jobs[@]}"; do
  echo "$stage/$j: job $(submit "$stage" "$j")"
done
