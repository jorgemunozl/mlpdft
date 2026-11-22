# Sourced by every job script.
load_qe() {
  module purge
  module load GCC/13.3.0 OpenMPI/5.0.3 QuantumESPRESSO/7.4
}

# run_pw <input> <output> "<expected QE md5 sums>"
# Runs pw.x and stops it within ~10 minutes if the pseudopotentials it reads are
# not the files used for the existing 40 Ry data (QE prints their MD5 at start).
run_pw() {
  local inp=$1 out=$2 want=$3 pid n got expected
  export ESPRESSO_PSEUDO
  export ESPRESSO_TMPDIR="$SCRATCH_BASE/$(basename "$(dirname "$PWD")")/$(basename "$PWD")"
  mkdir -p "$ESPRESSO_TMPDIR"
  # -in, not "< file": a backgrounded mpirun does not forward stdin to pw.x.
  mpirun -np "$SLURM_NTASKS" pw.x -nk 2 -in "$inp" > "$out" &
  pid=$!
  for _ in $(seq 1 120); do
    n=$(grep -c "MD5 check sum" "$out" 2>/dev/null) || n=0
    [ "$n" -ge "$(wc -w <<< "$want")" ] && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  got=$(grep "MD5 check sum" "$out" 2>/dev/null | awk '{print $4}' | sort | xargs)
  expected=$(tr ' ' '\n' <<< "$want" | sort | xargs)
  if [ -z "$got" ] && ! kill -0 "$pid" 2>/dev/null; then
    wait "$pid"
    n=$?
    echo "pw.x stopped before reading the pseudopotentials for $inp (exit $n); see $out" >&2
    return $(( n == 0 ? 1 : n ))
  fi
  if [ "$got" != "$expected" ]; then
    echo "pseudopotential checksum mismatch for $inp: got [$got], expected [$expected]" >&2
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    return 3
  fi
  wait "$pid"
}
