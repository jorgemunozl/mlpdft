#!/bin/bash
#SBATCH --job-name=cut_interface
#SBATCH --time=9:00:00
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
  name=$(basename "${inp%.in}")
  out="outputs/$name.out"
  if [ -s "$out" ] && grep -q "JOB DONE" "$out"; then continue; fi
  run_pw "$inp" "$out" "4191b14a59696d8f3147dc7708df3c20 f73bf51f2e812fc84686424137293589" || status=$?
  [ "$status" -eq 3 ] && exit 3
  # only the text output is needed; drop this point's wavefunctions from scratch
  rm -rf "${ESPRESSO_TMPDIR:?}/$name".*
done
exit "$status"
