"""
Utils to read from generated dataset and transform
it to xyz files expected by mace
Mace Internally uses its own data loader
"""
from dataclasses import dataclass
from annotation import Int
from ase import Atoms
from utils import parse_perconfig

@dataclass(frozen=True)
class Config_Row:
    """
    ConfigRow represents a single row in the file perconfig.dat
    """
    filename: str
    group: str
    natoms: in
    energy_truth: float
    energy_pred: Optional[float]
    testing_bool: bool


@dataclass
class Data_Config:
    json_root: Optional[Path] = None
    samples_num: Int = 2
    per_config: bool = False


class Data_Loader:
    def __init__(self, config: Data_config):
        self.config = config
        self.atoms_list = []
        self.per_config = parse_perconfig()


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
