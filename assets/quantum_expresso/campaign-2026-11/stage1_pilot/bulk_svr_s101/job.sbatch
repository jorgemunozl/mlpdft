#!/bin/bash
#SBATCH --job-name=pilot_svr_s101
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
run_pw pw.in "$out" "d2f1e9ef3385d70aefdbf456043e54ab 48ea186a387ffd7661a6cc7d97657bc3"
