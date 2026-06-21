#!/usr/bin/env python3
"""
Inspect a MACE model file and print all baked-in hyperparameters.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from mlpdft.config import MaceConfig
from mlpdft.constants import SRC_DIR


def fmt(v, max_width=70) -> str:
    """Pretty-format a value for display."""
    s = repr(v)
    return s if len(s) <= max_width else s[: max_width - 3] + "..."


def inspect_model(model_path: Path, quiet: bool = False):
    model = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.eval()

    # ── 1. Model identity ──
    class_name = model.__class__.__name__
    numel = sum(p.numel() for p in model.parameters())
    size_mb = numel * 4 / (1024 * 1024)  # approx with float32

    lines = []
    lines.append("=" * 70)
    lines.append(f"  Model Class        : {class_name}")
    lines.append(f"  Total parameters   : {numel:,}")
    lines.append(f"  Approx size (fp32) : {size_mb:.1f} MB")
    lines.append(f"  File               : {model_path.name}")
    lines.append(
        f"  File size          : {model_path.stat().st_size / (1024 * 1024):.1f} MB"
    )
    lines.append("")

    # ── 2. Architecture hyperparameters ──
    lines.append(f"  {'─' * 66}")
    lines.append("  ARCHITECTURE")
    lines.append(f"  {'─' * 66}")

    r_max = getattr(model, "r_max", None)
    num_interactions = getattr(model, "num_interactions", None)
    atomic_numbers = getattr(model, "atomic_numbers", None)

    if r_max is not None:
        lines.append(f"  r_max (cutoff)        : {float(r_max):.3f} Å")
    if num_interactions is not None:
        lines.append(f"  num_interactions      : {int(num_interactions)}")

    # heads
    heads = getattr(model, "heads", None)
    lines.append(f"  heads                 : {heads}")

    # hidden_irreps → channels + max_L
    try:
        prod = model.products[0]
        hidden = str(prod.linear.irreps_out)
        channels = prod.linear.irreps_out[0].mul  # first multiplicity
        max_l = prod.linear.irreps_out.lmax
        lines.append(f"  hidden_irreps         : {hidden}")
        lines.append(f"    → num_channels     : {channels}")
        lines.append(f"    → max_L            : {max_l}")
    except (AttributeError, IndexError):
        pass

    # max_ell (spherical harmonics)
    try:
        max_ell = model.spherical_harmonics._lmax
        lines.append(f"  max_ell (SH lm max)   : {max_ell}")
    except AttributeError:
        pass

    # correlation (body order)
    try:
        corr = len(model.products[0].symmetric_contractions.contractions[0].weights) + 1
        lines.append(f"  correlation           : {corr} ({corr + 1}-body order)")
    except (AttributeError, IndexError):
        try:
            corr = model.products[0].symmetric_contractions.contraction_degree
            lines.append(f"  correlation           : {corr} ({corr + 1}-body order)")
        except AttributeError:
            pass

    # radial basis
    try:
        num_bessel = len(model.radial_embedding.bessel_fn.bessel_weights)
        lines.append(f"  num_radial_basis      : {num_bessel}")
    except AttributeError:
        pass
    try:
        num_cutoff = model.radial_embedding.cutoff_fn.p.item()
        lines.append(f"  num_cutoff_basis      : {int(num_cutoff)}")
    except AttributeError:
        pass

    # radial type
    try:
        rtype = model.radial_embedding.bessel_fn.__class__.__name__
        lines.append(f"  radial_type           : {rtype}")
    except AttributeError:
        pass

    # distance transform
    try:
        dt = model.radial_embedding.distance_transform.__class__.__name__
        lines.append(f"  distance_transform    : {dt}")
    except AttributeError:
        pass

    # avg_num_neighbors
    try:
        ann = model.interactions[0].avg_num_neighbors
        lines.append(f"  avg_num_neighbors     : {float(ann):.2f}")
    except (AttributeError, IndexError):
        pass

    # readout class
    try:
        rcls = model.readouts[-1].__class__.__name__
        lines.append(f"  readout_cls           : {rcls}")
    except (AttributeError, IndexError):
        pass

    lines.append("")

    # ── 3. Elements ──
    lines.append(f"  {'─' * 66}")
    lines.append("  ELEMENTS")
    lines.append(f"  {'─' * 66}")

    if atomic_numbers is not None:
        zs = [int(z) for z in atomic_numbers]
        lines.append(f"  num_elements          : {len(zs)}")
        lines.append(f"  atomic_numbers (Z)    : {zs}")
        # map to symbols if available
        try:
            from ase.data import chemical_symbols

            syms = [chemical_symbols[z] for z in zs]
            lines.append(f"  symbols               : {syms}")
        except ImportError:
            pass
    else:
        lines.append("  atomic_numbers        : <not found>")

    lines.append("")

    # ── 4. Energy reference (E0 / atomic energies) ──
    lines.append(f"  {'─' * 66}")
    lines.append("  ENERGY REFERENCE (E0s)")
    lines.append(f"  {'─' * 66}")

    atomic_energies_fn = getattr(model, "atomic_energies_fn", None)
    e0s = getattr(atomic_energies_fn, "atomic_energies", None)
    scale_shift = getattr(model, "scale_shift", None)
    scale = getattr(scale_shift, "scale", None) if scale_shift else None
    shift = getattr(scale_shift, "shift", None) if scale_shift else None

    if scale is not None:
        lines.append(f"  scale_shift.scale     : {fmt(scale)}")
    if shift is not None:
        lines.append(f"  scale_shift.shift     : {fmt(shift)}")

    if e0s is not None and atomic_numbers is not None:
        e0s = e0s.squeeze()  # handle multi-dim
        if e0s.ndim == 0:
            e0s = e0s.unsqueeze(0)
        if len(e0s) == len(atomic_numbers):
            lines.append("  E0 table (per element):")
            try:
                from ase.data import chemical_symbols

                for z, e0 in zip(atomic_numbers, e0s):
                    sym = chemical_symbols[int(z)]
                    lines.append(f"    Z={int(z):3d} ({sym:2s}) : {float(e0): .8f} eV")
            except ImportError:
                for z, e0 in zip(atomic_numbers, e0s):
                    lines.append(f"    Z={int(z):3d}  : {float(e0): .8f} eV")
        else:
            lines.append(f"  atomic_energies       : shape {list(e0s.shape)}")
    elif e0s is not None:
        lines.append(f"  atomic_energies       : shape {list(e0s.shape)}")
    else:
        lines.append("  atomic_energies       : <not found>")

    lines.append("")

    # ── 5. Feature flags ──
    lines.append(f"  {'─' * 66}")
    lines.append("  FEATURE FLAGS")
    lines.append(f"  {'─' * 66}")

    for flag in [
        "apply_cutoff",
        "use_reduced_cg",
        "use_so3",
        "use_agnostic_product",
        "use_last_readout_only",
        "use_edge_irreps_first",
        "lammps_mliap",
    ]:
        val = getattr(model, flag, None)
        if val is not None:
            lines.append(f"  {flag:35s}: {val}")

    # cueq / oeq
    for attr in ["cueq_config", "oeq_config"]:
        val = getattr(model, attr, None)
        if val is not None:
            lines.append(f"  {attr:35s}: {fmt(val)}")

    # edge_irreps
    val = getattr(model, "edge_irreps", None)
    if val is not None:
        lines.append(f"  edge_irreps           : {val}")

    # embedding_specs
    val = getattr(model, "embedding_specs", None)
    if val is not None:
        lines.append(f"  embedding_specs       : {fmt(val)}")

    lines.append("")

    # ── 6. Parameter breakdown ──
    lines.append(f"  {'─' * 66}")
    lines.append("  PARAMETER BREAKDOWN (top 20 tensors by size)")
    lines.append(f"  {'─' * 66}")
    params_sorted = sorted(
        [(n, p) for n, p in model.named_parameters() if p.numel() > 0],
        key=lambda x: -x[1].numel(),
    )
    for name, param in params_sorted[:20]:
        lines.append(
            f"  {name:55s}  {list(param.shape)!s:>20s}  {param.numel():>8,} params"
        )
    if len(params_sorted) > 20:
        lines.append(f"  {'...':55s}  {'':>20s}  ({len(params_sorted) - 20} more)")

    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    if not quiet:
        print(text)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a MACE model and show all baked-in hyperparameters."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Direct path to .model or .compiled.model file",
    )
    parser.add_argument(
        "--model_key",
        "-k",
        type=str,
        default="mock_2_test",
        help="Model key in the registry (default: mock_2_test)",
    )
    parser.add_argument(
        "--compiled",
        "-c",
        action="store_true",
        help="Use the compiled version of the model",
    )
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        default=None,
        help="Optional path to save the output (default: print to stdout)",
    )

    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: model file not found: {model_path}", file=sys.stderr)
            sys.exit(1)
    else:
        config = MaceConfig(model_key=args.model_key)
        model_path = config.model.compiled_path if args.compiled else config.model.path

    inspect_model(model_path, quiet=False)

    if args.save:
        text = inspect_model(model_path, quiet=True)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(text)
        print(f"\nSaved to: {args.save}")


if __name__ == "__main__":
    main()
