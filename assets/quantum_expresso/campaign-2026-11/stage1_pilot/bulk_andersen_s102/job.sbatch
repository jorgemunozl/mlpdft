#!/bin/bash
#SBATCH --job-name=pilot_and_s102
#SBATCH --time=30:00:00
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
# A restart (restart_mode = 'restart' in pw.in) writes pw.restart<N>.out,
# so the earlier part of the trajectory in pw.out is never overwritten.
out=pw.out
if [ -s pw.out ]; then n=1; while [ -e "pw.restart$n.out" ]; do n=$((n + 1)); done; out="pw.restart$n.out"; fi
run_pw pw.in "$out" "4191b14a59696d8f3147dc7708df3c20 f73bf51f2e812fc84686424137293589"
