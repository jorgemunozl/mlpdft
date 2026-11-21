import json
p = "/workspace/runs/lif_uq_mhc_002/analysis_mhc/uq_analysis.json"
d = json.load(open(p))["methods"]
m, c = d["mhc"], d["committee"]

def rho(x):
    return sum(g["spearman"] for g in x["groups"].values()) / len(x["groups"])

rm, rc = rho(m), rho(c)
ratio = m["overall"]["force_rmse_atom_weighted"] / c["overall"]["force_rmse_atom_weighted"]
print(f"overall force: mhc {m['overall']['force_rmse_atom_weighted']:.6f}  committee {c['overall']['force_rmse_atom_weighted']:.6f}  ratio {ratio:.3f}  (limit 1.25) -> {ratio <= 1.25}")
print(f"rho mean: mhc {rm:.4f}  committee {rc:.4f}  delta {rm-rc:+.4f}  (allowed -0.10) -> {rm >= rc - 0.10}")
group_ok = True
for g in sorted(m["groups"]):
    a = m["groups"][g]["force_rmse_atom_weighted"]
    b = c["groups"][g]["force_rmse_atom_weighted"]
    ok = a / b <= 1.25
    group_ok = group_ok and ok
    print(f"  {g}: mhc {a:.5f} / committee {b:.5f} = {a/b:.3f} -> {ok}")
passed = ratio <= 1.25 and group_ok and rm >= rc - 0.10
print("GATE PASSED:", passed)
