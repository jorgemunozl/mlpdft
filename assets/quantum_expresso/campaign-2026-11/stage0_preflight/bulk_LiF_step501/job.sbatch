#!/bin/bash
#SBATCH --job-name=preflight
#SBATCH --time=1:00:00
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
run_pw pw.in pw.out "4191b14a59696d8f3147dc7708df3c20 f73bf51f2e812fc84686424137293589" || exit $?
TOL_F=1.0; TOL_MAX=3.0; TOL_E=0.0001; CONV=25711.0
awk -v tolF="$TOL_F" -v tolMax="$TOL_MAX" -v tolE="$TOL_E" -v conv="$CONV" '
function abs(v) { return v < 0 ? -v : v }
FNR == NR {
  if ($1 == "energy_Ry") eref = $2
  else if ($1 == "force") { n++; rx[n] = $2; ry[n] = $3; rz[n] = $4 }
  next
}
/^!/ && !gotE { e = $(NF - 1); gotE = 1 }
/Forces acting on atoms/ { inF = 1; next }
inF && /force =/ {
  m++; split($0, a, "="); split(a[2], f, " ")
  dx = f[1] - rx[m]; dy = f[2] - ry[m]; dz = f[3] - rz[m]
  s += dx * dx + dy * dy + dz * dz; c += 3
  if (abs(dx) > mx) mx = abs(dx); if (abs(dy) > mx) mx = abs(dy); if (abs(dz) > mx) mx = abs(dz)
  if (m == n) inF = 0
}
END {
  if (!gotE || m < n) { print "PREFLIGHT FAIL: pw.out has no complete energy and forces"; exit 1 }
  rmse = sqrt(s / c) * conv; mx = mx * conv; de = abs(e - eref)
  printf "force RMSE %.4f meV/A (tolerance %.1f), max %.4f meV/A (tolerance %.1f), |dE| %.2e Ry (tolerance %.0e)\n", rmse, tolF, mx, tolMax, de, tolE
  if (rmse <= tolF && mx <= tolMax && de <= tolE) {
    print "PREFLIGHT PASS: the repository pseudopotentials reproduce the existing 40 Ry data"; exit 0
  }
  print "PREFLIGHT FAIL: results differ from the existing 40 Ry data; the MD jobs will not start"; exit 1
}' reference.txt pw.out | tee preflight.log
exit ${PIPESTATUS[0]}
