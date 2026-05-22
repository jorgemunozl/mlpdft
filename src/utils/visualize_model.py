import torch

from config import MaceConfig
from constants import UTILS_DIR


def write_model_summary(model, f):
    print(model, file=f)
    print("Parameters:", file=f)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}", file=f)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}", file=f)

    scale_shift = getattr(model, "scale_shift", None)

    print("\nEnergy reference terms:", file=f)
    scale = getattr(scale_shift, "scale", None)
    shift = getattr(scale_shift, "shift", None)

    print(f"scale_shift.scale: {scale}", file=f)
    print(f"scale_shift.shift: {shift}", file=f)

    atomic_energies_fn = getattr(model, "atomic_energies_fn", None)
    atomic_energies = getattr(atomic_energies_fn, "atomic_energies", None)
    atomic_numbers = getattr(model, "atomic_numbers", None)

    if atomic_energies is None:
        print("atomic_energies_fn.atomic_energies: <not found>", file=f)
    else:
        print("atomic_energies_fn.atomic_energies (E0 table):", file=f)
        if atomic_numbers is not None and len(atomic_numbers) == len(atomic_energies):
            for z, e0 in zip(atomic_numbers, atomic_energies):
                print(f"  Z={int(z):3d}: E0={float(e0): .8f} eV", file=f)
        else:
            print(atomic_energies, file=f)


def visualize_model(config: MaceConfig):
    model = torch.load(config.model.path, weights_only=False)

    # Count the number of parameters
    write_model_summary(model, open(UTILS_DIR / f"{config.model.name}.txt", "w"))


if __name__ == "__main__":
    config = MaceConfig(model_key="0-small")
    visualize_model(config)
