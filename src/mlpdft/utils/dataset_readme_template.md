---
license: mit
tags:
  - mace
  - li-f
  - dft
  - qe
---

# Li–F MACE dataset

Merged DFT training data for Lithium–Fluoride systems computed with Quantum ESPRESSO (PAW).

## Composition

| # | Group | Frame count |
|---|-------|------------|
{{GROUP_TABLE}}

**Total frames:** {{TOTAL_FRAMES}}

## Generation parameters

- Frame stride: `{{FRAME_STRIDE}}`
- Energy key: `REF_energy`
- Force key: `REF_forces`
- Stress key: `stress` (when available)

## File format

Single multi-frame [extxyz](https://wiki.fysik.dtu.dk/ase/ase/io/formatoptions.html#extxyz) file.

## Usage

```python
from ase.io import read

atoms_list = read("{{MERGED_FILENAME}}", index=":")
print(f"Loaded {len(atoms_list)} configurations")
```
