#!/bin/bash
# Usage: ./submit.sh <stage folder> [job folder ...]
#   ./submit.sh stage1_pilot
#   ./submit.sh stage2_cutoff interface
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
source "$root/config.env"
: "${SLURM_ACCOUNT:?set SLURM_ACCOUNT in config.env}"
: "${ESPRESSO_PSEUDO:?set ESPRESSO_PSEUDO in config.env}"
: "${SCRATCH_BASE:?set SCRATCH_BASE in config.env}"
stage=$1; shift
if [ $# -gt 0 ]; then jobs=("$@"); else jobs=($(cd "$root/$stage" && ls -d */ | tr -d /)); fi
for j in "${jobs[@]}"; do
  (cd "$root/$stage/$j" && sbatch --account="$SLURM_ACCOUNT" job.sbatch)
done
