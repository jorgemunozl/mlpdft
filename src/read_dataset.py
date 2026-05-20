"""
Utils to read from generated dataset
"""

from pathlib import Path

from ase import Atoms


def load_json_as_atoms(json_path: Path):
    """Load generated JSON and return ASE Atoms with reference energy/forces."""
    data = read_json_allowing_header(str(json_path))
    if "Dataset" in data:
        data = data["Dataset"]
    if "Data" in data:
        frame = dict(data)
        frame.update(data["Data"][0])
    else:
        frame = data

    positions = np.asarray(frame["Positions"])
    lattice = np.asarray(frame["Lattice"])
    symbols = frame["AtomTypes"]
    energy = float(frame["Energy"])
    forces = np.asarray(frame["Forces"])

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=lattice,
        pbc=True,
    )
    atoms.info["energy_truth"] = energy
    atoms.arrays["forces_truth"] = forces
    return atoms
