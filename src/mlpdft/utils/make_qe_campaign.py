#!/usr/bin/env python3
"""Generate the QE job folders for the 2026-11 DFT campaign (study design C).

Every setting is copied from the existing 40 Ry runs so that new labels are
consistent with them. Only the thermostat and the starting structure change.

Stages
------
pilot     Two bulk LiF NVT runs (300 K, 40 Ry) started from the ideal lattice
          with independent random displacements: SVR and Andersen, both with
          nraise = 100. The thermostat with the shorter force decorrelation is
          used for the replicas; both pilots double as replicas.
cutoff    Single points on frames of the existing 40 Ry runs at 30/40/60/80 Ry
          (30 Ry only for the environments whose old data used it) and a
          4x4x4 k-point check, to justify 40 Ry and document why 30 Ry was
          dropped.
replicas  NVT replicas from the ideal lattice with the thermostat chosen after
          the pilot (bulk LiF, or F-doped Li as an optional shifted set).

The SLURM account, scratch folder and pseudopotential folder are not written
into any file here; ``submit.sh`` reads them from the bundle's ``config.env``.

Usage
-----
    python3 src/mlpdft/utils/make_qe_campaign.py --stage pilot \
        --dataset-root /home/jorge/project/mlpdft/dataset \
        --out assets/quantum_expresso/campaign-2026-11
    python3 src/mlpdft/utils/make_qe_campaign.py --stage cutoff ...
    python3 src/mlpdft/utils/make_qe_campaign.py --stage replicas \
        --env bulk_LiF --thermostat andersen --seeds 103 104 105 106 ...
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from diagnose_qe_drift import cell_from_out, parse

MASSES = {"Li": 6.941, "F": 18.998}
UPF = {"Li": "Li.pbe-sl-kjpaw_psl.1.0.0.UPF", "F": "F.pbe-n-kjpaw_psl.1.0.0.UPF"}
# MD5 sums QE printed when it read the pseudopotentials of the existing runs.
QE_MD5 = {"Li": "d2f1e9ef3385d70aefdbf456043e54ab", "F": "48ea186a387ffd7661a6cc7d97657bc3"}

# Settings read from the headers of the existing 40 Ry runs.
ENVS = {
    "bulk_LiF": {
        "source": "LIF64_KJPAW_FINAL", "ideal_input": "LIF64_KJPAW_FINAL/LiF64_kjpaw",
        "burn_in": 500, "mixing_beta": 0.7, "smearing": False, "s_per_md_step": 22,
        "old_cutoffs": [30],
    },
    "F_doped_Li": {
        "source": "LIWITHF_KJPAW_FINAL", "ideal_input": "LIWITHF_V3/LiwithF_kjpaw",
        "burn_in": 500, "mixing_beta": 0.6, "smearing": True, "s_per_md_step": 66,
        "old_cutoffs": [30],
    },
    "Li_metal": {
        "source": "BCC_54_KJPAW_FINAL", "ideal_input": None,
        "burn_in": 500, "mixing_beta": 0.4, "smearing": True, "s_per_md_step": 57,
        "old_cutoffs": [],
    },
    "interface": {
        "source": "LIFINTERFACE_KJPAW_V2", "ideal_input": None,
        "burn_in": 586, "mixing_beta": 0.6, "smearing": True, "s_per_md_step": 246,
        "old_cutoffs": [],
    },
}
ECUTRHO = {30: 320, 40: 320, 60: 480, 80: 640}  # 30/40 Ry as in the existing runs, then 8x
MD = {"dt": 10, "tempw": 300.0, "nraise": 100}
SIGMA_DISPLACEMENT = 0.06  # Angstrom, close to the thermal rms displacement of LiF at 300 K


# ---------------------------------------------------------------- structures
def read_ideal_input(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Cell (Angstrom), species and positions (Angstrom) from a pw.x input."""
    lines = path.read_text().splitlines()
    cell, species, positions = [], [], []
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            if "angstrom" not in line.lower():
                raise SystemExit(f"{path}: CELL_PARAMETERS must be in angstrom")
            cell = [[float(v) for v in lines[i + k].split()] for k in (1, 2, 3)]
        if line.strip().upper().startswith("ATOMIC_POSITIONS"):
            if "angstrom" not in line.lower():
                raise SystemExit(f"{path}: ATOMIC_POSITIONS must be in angstrom")
            for row in lines[i + 1:]:
                parts = row.split()
                if len(parts) < 4 or parts[0] not in MASSES:
                    break
                species.append(parts[0])
                positions.append([float(v) for v in parts[1:4]])
    return np.asarray(cell), species, np.asarray(positions)


def displaced(positions: np.ndarray, seed: int, sigma: float = SIGMA_DISPLACEMENT) -> np.ndarray:
    rng = np.random.default_rng(seed)
    disp = rng.normal(0.0, sigma, size=positions.shape)
    return positions + disp - disp.mean(axis=0)


def trajectory_frames(dataset_root: Path, env: str, count: int) -> list[dict]:
    """Evenly spaced post-burn-in configurations of an existing fixed-cell run.

    QE prints the new positions after each MD step, so ``pos[s]`` is the
    configuration of SCF block ``s + 1``. Frames are labelled with that SCF
    index, so the 40 Ry single point can be checked against the trajectory's
    own forces at the same step.
    """
    spec = ENVS[env]
    out = dataset_root / spec["source"] / f"{spec['source']}.out"
    data = parse(out)
    cell = cell_from_out(out)
    records = np.linspace(spec["burn_in"], len(data["pos"]) - 2, count).astype(int)
    return [{"step": int(s) + 1, "cell": cell, "species": [str(x) for x in data["species"]],
             "positions": data["pos"][s]} for s in records]


# ------------------------------------------------------------------- writers
def pw_input(*, env: str, calculation: str, prefix: str, cell: np.ndarray, species: list[str],
             positions: np.ndarray, ecutwfc: int = 40, kpts: tuple[int, int, int] = (3, 3, 3),
             thermostat: str | None = None, nstep: int | None = None,
             max_seconds: int | None = None) -> str:
    spec = ENVS[env]
    kinds = [k for k in ("Li", "F") if k in species]
    control = [f"calculation = '{calculation}'", "restart_mode = 'from_scratch'",
               f"prefix = '{prefix}'", "tstress = .true."]
    if calculation == "md":
        control += [f"dt = {MD['dt']}", f"nstep = {nstep}", "disk_io = 'high'"]
    else:
        control += ["tprnfor = .true."]
    if max_seconds:
        control.append(f"max_seconds = {max_seconds}")
    system = ["ibrav = 0", f"nat = {len(species)}", f"ntyp = {len(kinds)}",
              f"ecutwfc = {ecutwfc}", f"ecutrho = {ECUTRHO[ecutwfc]}", "nosym = .true."]
    if spec["smearing"]:
        system += ["occupations = 'smearing'", "smearing = 'mp'", "degauss = 0.02"]
    electrons = ["conv_thr = 1.0d-8", f"mixing_beta = {spec['mixing_beta']}", "electron_maxstep = 500"]
    blocks = [("control", control), ("system", system), ("electrons", electrons)]
    if calculation == "md":
        blocks.append(("ions", [f"tempw = {MD['tempw']}", f"ion_temperature = '{thermostat}'",
                                f"nraise = {MD['nraise']}", "pot_extrapolation = 'second-order'",
                                "wfc_extrapolation = 'second-order'"]))
    text = []
    for name, items in blocks:
        text.append(f" &{name}")
        text += [f"    {item}," for item in items[:-1]] + [f"    {items[-1]}"]
        text.append(" /")
    text.append("ATOMIC_SPECIES")
    text += [f" {k} {MASSES[k]} {UPF[k]}" for k in kinds]
    text.append("CELL_PARAMETERS angstrom")
    text += ["  " + "  ".join(f"{v:14.9f}" for v in row) for row in cell]
    text.append("ATOMIC_POSITIONS angstrom")
    text += [f"{s:3s} " + " ".join(f"{v:14.9f}" for v in p) for s, p in zip(species, positions)]
    text.append("K_POINTS {automatic}")
    text.append(" {} {} {} 0 0 0".format(*kpts))
    return "\n".join(text) + "\n"


def job_script(*, name: str, hours: int, body: list[str]) -> str:
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --time={hours}:00:00",
        "#SBATCH --ntasks=48",
        "#SBATCH --ntasks-per-node=48",
        "#SBATCH --mem=300000M",
        "#SBATCH -o slurm.o%j",
        "#SBATCH -e slurm.e%j",
        "# The account is passed by ../../submit.sh (sbatch --account); it is not stored here.",
        "",
        "set -uo pipefail",
        'cd "$SLURM_SUBMIT_DIR"',
        "source ../../config.env",
        "source ../../common.sh",
        "load_qe",
        *body,
        "",
    ])


def md5_list(species: list[str]) -> str:
    return " ".join(QE_MD5[k] for k in ("Li", "F") if k in species)


def write_md_job(folder: Path, *, env: str, prefix: str, thermostat: str, seed: int, nstep: int) -> dict:
    spec = ENVS[env]
    if spec["ideal_input"] is None:
        raise SystemExit(f"no ideal starting structure defined for {env}")
    cell, species, ideal = read_ideal_input(Path(ARGS.dataset_root) / spec["ideal_input"])
    hours = int(np.ceil(nstep * spec["s_per_md_step"] * 1.5 / 3600)) + 2
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pw.in").write_text(pw_input(
        env=env, calculation="md", prefix=prefix, cell=cell, species=species,
        positions=displaced(ideal, seed), thermostat=thermostat, nstep=nstep,
        max_seconds=(hours - 1) * 3600))
    body = [
        "# A restart (restart_mode = 'restart' in pw.in) writes pw.restart<N>.out,",
        "# so the earlier part of the trajectory in pw.out is never overwritten.",
        "out=pw.out",
        'if [ -s pw.out ]; then n=1; while [ -e "pw.restart$n.out" ]; do n=$((n + 1)); done; out="pw.restart$n.out"; fi',
        f'run_pw pw.in "$out" "{md5_list(species)}"',
    ]
    (folder / "job.sbatch").write_text(job_script(name=prefix, hours=hours, body=body))
    return {"job": folder.name, "env": env, "thermostat": thermostat, "seed": seed,
            "nstep": nstep, "hours": hours}


# -------------------------------------------------------------------- stages
def stage_pilot(root: Path) -> list[dict]:
    stage = root / "stage1_pilot"
    return [
        write_md_job(stage / "bulk_svr_s101", env="bulk_LiF", prefix="pilot_svr_s101",
                     thermostat="svr", seed=101, nstep=ARGS.nstep),
        write_md_job(stage / "bulk_andersen_s102", env="bulk_LiF", prefix="pilot_and_s102",
                     thermostat="andersen", seed=102, nstep=ARGS.nstep),
    ]


def stage_replicas(root: Path) -> list[dict]:
    stage = root / f"stage3_replicas_{ARGS.env}"
    return [write_md_job(stage / f"{ARGS.env}_{ARGS.thermostat}_s{seed}", env=ARGS.env,
                         prefix=f"rep_{ARGS.env}_s{seed}", thermostat=ARGS.thermostat,
                         seed=seed, nstep=ARGS.nstep) for seed in ARGS.seeds]


def stage_cutoff(root: Path) -> list[dict]:
    stage = root / "stage2_cutoff"
    rows = []
    for env, spec in ENVS.items():
        folder = stage / env
        (folder / "inputs").mkdir(parents=True, exist_ok=True)
        (folder / "outputs").mkdir(exist_ok=True)
        frames = trajectory_frames(Path(ARGS.dataset_root), env, ARGS.frames)
        cutoffs = sorted(spec["old_cutoffs"] + [40, 60, 80])
        cost = 0.0
        for i, frame in enumerate(frames):
            for ec in cutoffs:
                name = f"f{i}_step{frame['step']}_ec{ec}_k3"
                (folder / "inputs" / f"{name}.in").write_text(pw_input(
                    env=env, calculation="scf", prefix=name, cell=frame["cell"],
                    species=frame["species"], positions=frame["positions"], ecutwfc=ec))
                cost += spec["s_per_md_step"] * 2 * (ec / 40) ** 1.5
            # k-point check at 40 Ry on the first and last frame (first only for the insulator)
            if i == 0 or (spec["smearing"] and i == len(frames) - 1):
                name = f"f{i}_step{frame['step']}_ec40_k4"
                (folder / "inputs" / f"{name}.in").write_text(pw_input(
                    env=env, calculation="scf", prefix=name, cell=frame["cell"],
                    species=frame["species"], positions=frame["positions"], kpts=(4, 4, 4)))
                cost += spec["s_per_md_step"] * 2 * 2.4
        hours = int(np.ceil(cost * 1.5 / 3600)) + 2
        body = [
            "mkdir -p outputs  # git does not keep empty folders",
            "status=0",
            "for inp in inputs/*.in; do",
            '  out="outputs/$(basename "${inp%.in}").out"',
            '  if [ -s "$out" ] && grep -q "JOB DONE" "$out"; then continue; fi',
            f'  run_pw "$inp" "$out" "{md5_list(frames[0]["species"])}" || status=$?',
            '  [ "$status" -eq 3 ] && exit 3',
            "done",
            'exit "$status"',
        ]
        (folder / "job.sbatch").write_text(job_script(name=f"cut_{env}", hours=hours, body=body))
        rows.append({"job": env, "frames": [f["step"] for f in frames], "cutoffs": cutoffs,
                     "hours": hours, "est_compute_h": round(cost / 3600, 1)})
    return rows


# --------------------------------------------------------------- bundle files
COMMON_SH = r"""# Sourced by every job script.
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
"""

SUBMIT_SH = r"""#!/bin/bash
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
"""

CONFIG_EXAMPLE = """# Copy to config.env and fill in. config.env is not committed.
export SLURM_ACCOUNT=""     # the account used by the earlier QE_JOB scripts
export ESPRESSO_PSEUDO=""   # folder with the original Li/F PSLibrary 1.0.0 UPF files of the 40 Ry runs
export SCRATCH_BASE=""      # scratch folder for QE temporary files, e.g. /scratch/user/<you>/QE/campaign-2026-11
"""


def write_bundle_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "common.sh").write_text(COMMON_SH)
    (root / "submit.sh").write_text(SUBMIT_SH)
    (root / "submit.sh").chmod(0o755)
    (root / "config.env.example").write_text(CONFIG_EXAMPLE)
    (root / ".gitignore").write_text("config.env\nslurm.*\n*.out\n")


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=["pilot", "cutoff", "replicas"])
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--nstep", type=int, default=3000)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--env", choices=list(ENVS), default="bulk_LiF")
    ap.add_argument("--thermostat", choices=["andersen", "svr"], default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[])
    ap.add_argument("--clean", action="store_true", help="remove the stage folder first")
    ARGS = ap.parse_args()
    if ARGS.stage == "replicas" and (ARGS.thermostat is None or not ARGS.seeds):
        ap.error("--stage replicas needs --thermostat and --seeds")
    write_bundle_files(ARGS.out)
    folder = {"pilot": "stage1_pilot", "cutoff": "stage2_cutoff",
              "replicas": f"stage3_replicas_{ARGS.env}"}[ARGS.stage]
    if ARGS.clean and (ARGS.out / folder).exists():
        shutil.rmtree(ARGS.out / folder)
    rows = {"pilot": stage_pilot, "cutoff": stage_cutoff, "replicas": stage_replicas}[ARGS.stage](ARGS.out)
    for row in rows:
        print(row)


ARGS: argparse.Namespace

if __name__ == "__main__":
    main()
