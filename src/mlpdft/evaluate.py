from typing import Dict

import ase.io
import numpy as np
import torch
from mace import data
from mace.tools import torch_geometric, torch_tools, utils

from mlpdft.config import MaceConfig
from mlpdft.constants import LIF_KJPAW_GROUP


def get_model_output(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
) -> dict:
    return model(batch)


def _get_model_float_dtype(model: torch.nn.Module) -> torch.dtype:
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.get_default_dtype()


def _cast_batch_to_dtype(
    batch_dict: Dict[str, torch.Tensor], target_dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            out[key] = value.to(dtype=target_dtype)
        else:
            out[key] = value
    return out


def eval(config: MaceConfig) -> None:
    torch_tools.set_default_dtype(config.dtype)
    scrap = MACE_SCRAP(config=config)
    model = scrap.build_model()

    device = config.device
    model = model.to(device)
    model.eval()
    model_dtype = _get_model_float_dtype(model)

    # Load data and prepare input
    atoms_list = ase.io.read(config.data_out_path, index=":")

    head_name = "Default"
    configs = [
        data.config_from_atoms(atoms, head_name=head_name) for atoms in atoms_list
    ]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    heads = None

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for config in configs
        ],
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    forces_collection = []
    node_energies_list = []

    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = _cast_batch_to_dtype(batch.to_dict(), model_dtype)
        output = get_model_output(model, batch_dict)
        energies_list.append(torch_tools.to_numpy(output["energy"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])  # drop last as it's empty

        if config.node_energy:
            node_energies_list.append(
                np.split(
                    torch_tools.to_numpy(output["node_energy"]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )[:-1]  # drop last as its empty
            )

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [
        forces for forces_list in forces_collection for forces in forces_list
    ]

    if config.node_energy:
        node_energies = np.concatenate(node_energies_list, axis=0)
        assert len(atoms_list) == node_energies.shape[0]

    assert len(atoms_list) == len(energies) == len(forces_list)

    # Store data in atoms objects
    for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
        atoms.calc = None  # crucial
        total_energy_shift = config.resolved_energy_offset_per_atom * len(atoms)
        atoms.info[config.info_prefix + "energy"] = energy + total_energy_shift
        atoms.arrays[config.info_prefix + "forces"] = forces

        if config.node_energy:
            atoms.arrays[config.info_prefix + "node_energy"] = node_energies[i]

    # Write atoms to output path
    ase.io.write(str(config.model_output), images=atoms_list, format="extxyz")


def main() -> None:
    config = MaceConfig(
        model_key="0-omat-medium",
        group=LIF_KJPAW_GROUP,
        frame_stride=10,
        max_frames=20,
    )
    eval(config)


if __name__ == "__main__":
    main()
