import os
import sys
import random
import traceback
from pathlib import Path
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _patch_fitsnap_lammps_snap_bool_keywords() -> None:
    """
    FitSNAP builds ``compute snap ... bzeroflag True`` using Python bools.
    Recent LAMMPS ML-SNAP expects integer flags (0/1), not the strings True/False.
    """
    from fitsnap3lib.calculators import lammps_snap as ls  # type: ignore

    if getattr(ls.LammpsSnap, "_mlpdft_snap_bool_kw_patch", False):
        return

    def _set_computes(self):
        numtypes = self.config.sections["BISPECTRUM"].numtypes
        radelem = " ".join([f"${{radelem{i}}}" for i in range(1, numtypes + 1)])
        wj = " ".join([f"${{wj{i}}}" for i in range(1, numtypes + 1)])

        kw_options = {
            k: self.config.sections["BISPECTRUM"].__dict__[v]
            for k, v in {
                "rmin0": "rmin0",
                "bzeroflag": "bzeroflag",
                "quadraticflag": "quadraticflag",
                "switchflag": "switchflag",
                "chem": "chemflag",
                "bnormflag": "bnormflag",
                "wselfallflag": "wselfallflag",
                "bikflag": "bikflag",
                "switchinnerflag": "switchinnerflag",
                "switchflag": "switchflag",
                "sinner": "sinner",
                "dinner": "dinner",
                "dgradflag": "dgradflag",
            }.items()
            if v in self.config.sections["BISPECTRUM"].__dict__
        }

        if kw_options["chem"] == 0:
            kw_options.pop("chem")
        if kw_options["bikflag"] == 0:
            kw_options.pop("bikflag")
        if kw_options["switchinnerflag"] == 0:
            kw_options.pop("switchinnerflag")
        if kw_options["dgradflag"] == 0:
            kw_options.pop("dgradflag")
        kw_options["rmin0"] = self.config.sections["BISPECTRUM"].rmin0

        def _fmt(v):
            return int(v) if isinstance(v, bool) else v

        kw_substrings = [f"{k} {_fmt(v)}" for k, v in kw_options.items()]
        kwargs = " ".join(kw_substrings)
        base_snap = "compute snap all snap ${rcutfac} ${rfac0} ${twojmax}"
        command = f"{base_snap} {radelem} {wj} {kwargs}"
        self._lmp.command(command)

    ls.LammpsSnap._set_computes = _set_computes  # type: ignore[method-assign]
    ls.LammpsSnap._mlpdft_snap_bool_kw_patch = True  # type: ignore[attr-defined]


def _patch_torch_reduce_lr_on_plateau() -> None:
    """
    FitSNAP's PyTorch solver passes verbose= to ReduceLROnPlateau; that argument was
    removed in newer PyTorch (e.g. 2.x), which raises TypeError at FitSnap init.
    """
    import inspect

    import torch.optim.lr_scheduler as lrs

    try:
        if "verbose" in inspect.signature(lrs.ReduceLROnPlateau.__init__).parameters:
            return
    except (TypeError, ValueError):
        return

    _orig = lrs.ReduceLROnPlateau.__init__

    def _init(self, optimizer, *args, **kwargs):
        kwargs.pop("verbose", None)
        return _orig(self, optimizer, *args, **kwargs)

    lrs.ReduceLROnPlateau.__init__ = _init  # type: ignore[method-assign]


def _patch_fitsnap_randint() -> None:
    """
    FitSNAP 3.1.x uses randint(0, 1e5) which is a float in Python 3.14+.
    Monkey-patch the already-imported symbol in fitsnap3lib.parallel_tools.
    """
    import fitsnap3lib.parallel_tools as pt  # type: ignore

    def _randint_int_bounds(a, b):
        return random.randint(int(a), int(b))

    pt.randint = _randint_int_bounds


def _patch_fit_torch_atoms_per_structure_0dim() -> None:
    """
    FitSNAP's PyTorch solver evaluates one config with
    ``num_atoms = torch.tensor(config.natoms)`` (0-dim). FitTorch.forward then does
    ``torch.zeros(atoms_per_structure.size())`` -> scalar tensor, and ``index_add_``
    fails (PyTorch 2.x). A 1-d tensor ``[natoms]`` means one structure in the batch.
    """
    from fitsnap3lib.lib.neural_networks import pytorch as nnmod  # type: ignore

    if getattr(nnmod.FitTorch, "_mlpdft_apm_shape_patch", False):
        return

    _orig = nnmod.FitTorch.forward

    def forward(self, x, xd, indices, atoms_per_structure, types, xd_indx, unique_j, unique_i, device, dtype=torch.float32):
        apm = atoms_per_structure
        if apm.ndim == 0:
            apm = apm.unsqueeze(0)
        return _orig(self, x, xd, indices, apm, types, xd_indx, unique_j, unique_i, device, dtype)

    nnmod.FitTorch.forward = forward  # type: ignore[method-assign]
    nnmod.FitTorch._mlpdft_apm_shape_patch = True  # type: ignore[attr-defined]


def main(argv: list[str]) -> int:
    log_path = str(_REPO_ROOT / "logs" / "fitsnap_run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("run_fitsnap3_patched.py starting\n")
        log.write(f"argv={argv!r}\n")
        log.write(f"cwd={os.getcwd()}\n")

    if len(argv) != 2:
        print("usage: python scripts/run_fitsnap3_patched.py <input.in>", file=sys.stderr)
        return 2

    raw = argv[1]
    p = Path(raw)
    if not p.is_file():
        alt = _REPO_ROOT / raw
        if alt.is_file():
            p = alt
    if not p.is_file():
        print(f"error: input file not found: {raw}", file=sys.stderr)
        return 2

    os.chdir(_REPO_ROOT)
    infile_rel = os.path.relpath(p.resolve(), _REPO_ROOT)

    try:
        _patch_fitsnap_randint()
        _patch_torch_reduce_lr_on_plateau()
        _patch_fitsnap_lammps_snap_bool_keywords()
        _patch_fit_torch_atoms_per_structure_0dim()

        from fitsnap3lib.fitsnap import FitSnap  # type: ignore

        snap = FitSnap(input=infile_rel, comm=None, arglist=["--overwrite", "--verbose"])
        snap.scrape_configs(delete_scraper=True)
        snap.process_configs(delete_data=True)
        snap.perform_fit()
        try:
            snap.write_output()
        except TypeError as exc:
            # FitSNAP NN output formatting bug on some stacks:
            # TypeError: unsupported format string passed to numpy.ndarray.__format__
            # Fit/evaluation have already completed at this point.
            if "numpy.ndarray.__format__" not in str(exc):
                raise
            with open(log_path, "a", encoding="utf-8") as log:
                log.write("write_output warning (non-fatal):\n")
                log.write(traceback.format_exc())
            print("warning: FitSNAP write_output formatting bug; fit/evaluation completed.", file=sys.stderr)

        with open(log_path, "a", encoding="utf-8") as log:
            log.write("FitSNAP finished successfully\n")
        return 0
    except Exception:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("FitSNAP failed with exception:\n")
            log.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

