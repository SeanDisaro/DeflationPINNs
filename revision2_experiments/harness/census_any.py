"""Canonical flow census for a K=6 discovery checkpoint: gradient flow + Newton
per branch -> branch/state map (census6.json) + refined-column errors."""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import deepxde as dde  # noqa: F401
from harness_ldg import build_model, SOL_NAMES
import ldg_reference as LR
from flow_final import flow
import flow_final

RES = os.path.expanduser("~/DeflationPINNs_rev/results")
TAG = os.environ.get("TAG", "census6")
REF = os.path.join(RES, "ldg_reference")
N = 257

def main():
    flow_final.CKPT = os.environ.get("CKPT", flow_final.CKPT)
    fields = flow_final.nn_fields()
    refs = {nm: np.load(os.path.join(REF, f"{nm}_n{N}.npz")) for nm in SOL_NAMES}
    census, refined = [], {}
    disc = {}
    for k, (Q1, Q2) in enumerate(fields):
        E0 = LR.energy(Q1, Q2, N)
        F1, F2 = flow(Q1, Q2)
        T1, T2, rn, its = LR.newton_state("D1", N, seed=(F1, F2), tol=1e-12, itmax=80)
        E = LR.energy(T1, T2, N)
        best, bd = None, 1e9
        for nm, f in refs.items():
            d = np.sqrt(np.mean((T1 - f["Q1"]) ** 2 + (T2 - f["Q2"]) ** 2) /
                        np.mean(f["Q1"] ** 2 + f["Q2"] ** 2))
            if d < bd:
                best, bd = nm, d
        f = refs[best]
        den = np.mean(f["Q1"] ** 2 + f["Q2"] ** 2)
        disc[best] = float(np.sqrt(np.mean((Q1 - f["Q1"]) ** 2 + (Q2 - f["Q2"]) ** 2) / den))
        refined[best] = float(bd)
        census.append({"branch": k, "certified": bool(rn < 1e-9), "state": best,
                       "E": float(E), "E_nn": float(E0), "res_final": float(rn),
                       "method": "flow+newton"})
        np.savez_compressed(os.path.join(RES, f"{TAG}_endpoint_b{k}.npz"), Q1=T1, Q2=T2)
        print(f"branch {k}: E_nn={E0:.1f} -> E={E:.4f} res={rn:.1e} => {best} "
              f"(discovery rel {disc[best]:.4f}, refined rel {bd:.2e})", flush=True)
    json.dump(census, open(os.path.join(RES, f"{TAG}_census.json"), "w"), indent=1)
    json.dump({"discovery_rel": disc, "refined_rel": refined},
              open(os.path.join(RES, f"{TAG}_table1_cols.json"), "w"), indent=1)
    st = sorted(set(c["state"] for c in census))
    print(f"STATES: {st} ({len(st)}/6 distinct)", flush=True)
    print("DISCOVERY:", {k: round(v, 4) for k, v in sorted(disc.items())}, flush=True)
    print("REFINED:", {k: f"{v:.2e}" for k, v in sorted(refined.items())}, flush=True)

if __name__ == "__main__":
    main()
