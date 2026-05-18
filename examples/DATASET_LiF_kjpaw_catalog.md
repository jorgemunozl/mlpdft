# LiF_kjpaw / FitSNAP dataset catalog (Google Drive)

Summary of **dataset group folders** visible in shared Drive paths  
`…/data-fitsnap/Dataset-LUIS/LiF_kjpaw` (and related).  
The full tree is not shipped in this repo; this file is an **inventory of folder names** from your screenshots for consistent naming in scripts and slides.

System types suggested by names: **LiF** (bulk / interfaces / 64-atom cells), **LiBF₄**, **B–Li** (BLi, BLi₃, interfaces), **BCC-54** supercells, **ionic-liquid / electrolyte** (`EMIM_*`), plus **NPT**, **kjpaw**, **isolated**, **interface**, **relax**, version tags (**v1–v4**) and **final** variants.

---

## Distinct folder names (merged from both screenshots, A–Z)

Where the UI showed truncation (`…`), a **probable** full string is noted — confirm on Drive when you sync.

| # | Folder name |
|---|-------------|
| 1 | `BCC_54_kjpaw` |
| 2 | `BCC_54_kjpaw_2` |
| 3 | `BCC_54_kjpaw_final` |
| 4 | `BCC_54_kjpaw_NPT` |
| 5 | `BCC_54_kjpaw_NPT_final` (UI: `…NPT_f…`) |
| 6 | `BCC_54_kjpaw_NPT_v2` |
| 7 | `BCC54_isolated` |
| 8 | `BLi` |
| 9 | `BLi_interface` |
| 10 | `BLi_interface_NPT` |
| 11 | `BLi_interface_NPT_final` (UI: `…NPT_fi…`) |
| 12 | `BLi_isolated` |
| 13 | `BLi_NPT` |
| 14 | `BLi_v2` |
| 15 | `BLi3` |
| 16 | `BLi3_interface` |
| 17 | `BLi3_interface_final` |
| 18 | `BLi3_interface_NPT` |
| 19 | `BLi3_interface_NPT_final` (UI: `…NPT_fi…`) |
| 20 | `BLi3_isolated` |
| 21 | `BLi3_NPT` |
| 22 | `BLi3_v2` |
| 23 | `EMIM_TFSI_BF4` |
| 24 | `EMIMLiTFSI_anode_rv2` |
| 25 | `LiBF4` |
| 26 | `LiBF4_final` |
| 27 | `LiBF4_NPT` |
| 28 | `LiBF4_NPT_final` |
| 29 | `LiBF4_relax` |
| 30 | `LiBF4v2` |
| 31 | `LiBF4v3` |
| 32 | `LiBF4v4` |
| 33 | `LiF64_kjpaw` |
| 34 | `LiF64_kjpaw_final` |
| 35 | `LiF64_kjpaw_NPT` |
| 36 | `LiF64_kjpaw_NPT_final` |
| 37 | `LiF64_kjpaw_NPT_v2` |
| 38 | `LiF64_kjpaw_NPT_v3` |
| 39 | `LiF64_kjpaw_v2` |
| 40 | `LiFinterface_kjpaw_v1` |
| 41 | `LiFinterface_kjpaw_v2` |
| 42 | `LiFinterface_kjpaw_v3` |
| 43 | `LiFinterface_kjpaw_final` (inferred from UI `…kjpaw_fi…`) |
| 44 | `LiwithBF` |
| 45 | `LiwithBF_final` |
| 46 | `LiwithBF_isolated` |
| 47 | `LiwithBF_NPT` |
| 48 | `LiwithBF_NPT_final` |
| 49 | `LiwithBF_v2` |
| 50 | `LiwithF_kjpaw` |
| 51 | `LiwithF_kjpaw_final` |
| 52 | `LiwithF_isolated` |
| 53 | `LiwithF_NPT` |
| 54 | `LiwithF_NPT_final` |
| 55 | `LiwithF_v2` |
| 56 | `LiwithF_v3` |
| 57 | `special` |

Three **additional** folders in screenshot 1 were only visible as truncated `LiFinterface_kjpaw…` strings (`…fi…`, `…NP…`, `…N…`). They may coincide with rows 40–43 or be distinct — **open Drive and fill exact names** if they differ.

**Total:** 57 resolved rows above + up to 3 truncated `LiFinterface_*` variants to verify.

---

### Truncated labels only (screenshot 1)

These appeared as distinct tiles but the name was cut off in the UI:

- `LiFinterface_kjpaw_fi…` → usually `…_final`
- `LiFinterface_kjpaw_NP…` → often `…_NPT` or `…_NPT_*`
- `LiFinterface_kjpaw_N…` → second NPT-related interface folder (full string unknown from image)

---

## Related in-repo paths

- Checked-in example: [`LiF64_kjpaw_v2/`](LiF64_kjpaw_v2/) — matches catalog name `LiF64_kjpaw_v2`.
- FitSNAP `[GROUPS]` in decks such as [`model/LiF-example.in`](../model/LiF-example.in) list many of these names when `dataPath` points at the full JSON tree.

---

## Source

Folder names were transcribed from two Google Drive screenshots of `LiF_kjpaw` (shared `data-fitsnap` / `Dataset-LUIS`). Truncated labels in the UI were expanded where the full name is obvious; re-check on Drive for exact strings.

To regenerate from a local mirror:

```bash
find /path/to/LiF_kjpaw -maxdepth 1 -mindepth 1 -type d | sort
```
