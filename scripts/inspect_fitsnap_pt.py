#!/usr/bin/env python3
"""
Inspecciona un archivo .pt de FitSNAP para determinar si es:
  - TorchWrapper serializado (listo para cargar directo en LAMMPS)
  - State dict de entrenamiento (dict con model_state_dict)
  - Otro tipo de objeto
"""

import argparse
import sys

import torch


def inspect_pt(path: str, safe_mode: bool = True) -> None:
    print(f"Archivo: {path}")
    print(f"  Tamaño: {_human_size(path)}")
    print()

    # 1) Try loading with weights_only=True (safe)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        print("  ✅ Carga con weights_only=True: EXITOSA")
        _describe(obj)
        return
    except Exception as e:
        print(f"  ❌ Carga con weights_only=True FALLÓ:")
        print(f"     {e}")
        print()

    # 2) If safe mode is off or user allows, try loading with unsafe pickle
    if not safe_mode:
        print("  ⚠️  Intentando con weights_only=False (pickle inseguro)...")
        try:
            import fitsnap3lib  # noqa: F401 – needed for TorchWrapper deserialization
        except ImportError:
            print(
                "     ⚠️  fitsnap3lib no está instalado; la deserialización puede fallar"
            )
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            print("  ✅ Carga con pickle: EXITOSA")
            _describe(obj)
        except Exception as e:
            print(f"  ❌ Carga con pickle TAMBIÉN FALLÓ:")
            print(f"     {e}")
    else:
        print("  ℹ️  Usa --unsafe para intentar deserializar con pickle.")


def _describe(obj) -> None:
    print()
    if isinstance(obj, dict):
        print(f"  Tipo: dict (state_dict / checkpoint de entrenamiento)")
        print(f"  Keys principales: {list(obj.keys())}")
        for k in sorted(obj.keys()):
            v = obj[k]
            if isinstance(v, torch.Tensor):
                print(f"    {k}: Tensor shape={list(v.shape)} dtype={v.dtype}")
            elif isinstance(v, dict):
                print(f"    {k}: dict con {len(v)} sub-keys")
                # Show layer shapes if it's a model_state_dict
                if k == "model_state_dict" or any(
                    isinstance(vv, torch.Tensor) for vv in v.values()
                ):
                    for sk, sv in sorted(v.items()):
                        if isinstance(sv, torch.Tensor):
                            print(f"      {sk}: Tensor shape={list(sv.shape)}")
                        else:
                            print(f"      {sk}: {type(sv).__name__}")
            else:
                print(f"    {k}: {type(v).__name__} = {str(v)[:100]}")
    elif hasattr(obj, "n_descriptors") or hasattr(obj, "n_elements"):
        print(f"  Tipo: TorchWrapper (objeto serializado de FitSNAP)")
        n_desc = getattr(obj, "n_descriptors", "?")
        n_elem = getattr(obj, "n_elements", "?")
        dtype = getattr(obj, "dtype", "?")
        print(f"    n_descriptors: {n_desc}")
        print(f"    n_elements:    {n_elem}")
        print(f"    dtype:         {dtype}")
        if hasattr(obj, "model") and hasattr(obj.model, "elemwise_models"):
            emodels = obj.model.elemwise_models
            print(f"    elemwise_models: {len(emodels)} subred(es)")
            for i, m in enumerate(emodels):
                nparams = sum(p.numel() for p in m.parameters())
                print(f"      [{i}]: {type(m).__name__} — {nparams} parámetros")
                for k, v in m.state_dict().items():
                    print(f"        {k}: {list(v.shape)}")
    else:
        print(f"  Tipo: {type(obj).__name__}")
        print(f"  {str(obj)[:200]}")


def _human_size(path: str) -> str:
    import os

    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspecciona un .pt de FitSNAP")
    parser.add_argument(
        "path",
        help="Ruta al archivo .pt",
        default="fitsnap_models/LI_F/checkpoints/LiF_Pytorch.pt",
    )
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="Permite deserialización pickle (necesario para TorchWrapper)",
    )
    args = parser.parse_args()

    inspect_pt(args.path, safe_mode=not args.unsafe)


if __name__ == "__main__":
    main()
