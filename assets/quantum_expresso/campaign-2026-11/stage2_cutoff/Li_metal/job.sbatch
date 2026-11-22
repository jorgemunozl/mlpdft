#!/bin/bash
#SBATCH --job-name=cut_Li_metal
#SBATCH --time=4:00:00
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=48
#SBATCH --mem=300000M
#SBATCH -o slurm.o%j
#SBATCH -e slurm.e%j
# The account is passed by ../../submit.sh (sbatch --account); it is not stored here.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../../config.env
source ../../common.sh
load_qe
mkdir -p outputs  # git does not keep empty folders
status=0
for inp in inputs/*.in; do
  out="outputs/$(basename "${inp%.in}").out"
  if [ -s "$out" ] && grep -q "JOB DONE" "$out"; then continue; fi
  run_pw "$inp" "$out" "d2f1e9ef3385d70aefdbf456043e54ab" || status=$?
  [ "$status" -eq 3 ] && exit 3
done
exit "$status"
