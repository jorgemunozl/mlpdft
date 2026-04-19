# LiBF₄ example (QE → FitSNAP JSON)

End-to-end scratch data for a Li/B/F system: Quantum ESPresso inputs and job files, scripts that convert QE output into FitSNAP JSON, and the `NEWJSON/` frames used with `configs/fitsnap/LiBF4-minimal.in`.

| Path | Contents |
|------|----------|
| `qe/` | `run_example`, `QE_job.in`, scheduler snippets, and `LiBF4.out` from a finished QE run (used by the converters). |
| `converters/` | `qetofitsnap4n.py` (QE stdout → JSON in `NEWJSON/`), `qetodumptype.py` / `qetodump2.py` (QE → LAMMPS-style dump for visualization). |
| `data/` | Reference `output.dump` (and outputs from the dump converters when you run them). |
| `NEWJSON/` | FitSNAP-style `output_*.json` frames (may be gitignored via `*.json`). |

Run converters from the **repository root** (paths are resolved relative to this example directory):

```bash
python examples/lifbf4/converters/qetofitsnap4n.py
```

Place `BCC_54.out` under `qe/` before running `qetodump2.py`, if you use that workflow.
