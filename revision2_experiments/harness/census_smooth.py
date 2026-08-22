"""Flow+Newton census for a SmoothKState (DDR) checkpoint: how many DISTINCT
stable states do the branches occupy?  Mirrors census_any.py but for the smooth
per-branch architecture used by harness_ldg_multi.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import deepxde as dde  # noqa: E402
from harness_ldg import SOL_NAMES  # noqa: E402
from harness_ldg_multi import SmoothKState  # noqa: E402
import ldg_reference as LR  # noqa: E402
from flow_final import flow  # noqa: E402

RES = os.path.expanduser("~/DeflationPINNs_rev/results")
REF = os.path.join(RES, "ldg_reference")
CKPT = os.environ["CKPT"]
TAG = os.environ.get("TAG", "ddr")
K = int(os.environ.get("K", "6"))
N = 257


def main():
    geom = dde.geometry.geometry_2d.Rectangle([0., 0.], [1., 1.])
    m = SmoothKState(geom, K, 128, 3, 128, 16.0)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    m.eval()
    g = np.linspace(0, 1, N)
    X, Y = np.meshgrid(g, g, indexing="xy")
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32)
    refs = {nm: np.load(os.path.join(REF, f"{nm}_n{N}.npz")) for nm in SOL_NAMES}

    rows, states = [], []
    for k in range(K):
        with torch.no_grad():
            q1, q2 = m.fields(pts, k)
        Q1 = q1.numpy().reshape(N, N).astype(float)
        Q2 = q2.numpy().reshape(N, N).astype(float)
        E0 = LR.energy(Q1, Q2, N)
        # pre-refinement error against the closest reference
        pre = min(np.sqrt(np.mean((Q1 - f["Q1"]) ** 2 + (Q2 - f["Q2"]) ** 2) /
                          np.mean(f["Q1"] ** 2 + f["Q2"] ** 2)) for f in refs.values())
        F1, F2 = flow(Q1, Q2)
        T1, T2, rn, its = LR.newton_state("D1", N, seed=(F1, F2), tol=1e-12, itmax=80)
        best, bd = None, 1e9
        for nm, f in refs.items():
            d = np.sqrt(np.mean((T1 - f["Q1"]) ** 2 + (T2 - f["Q2"]) ** 2) /
                        np.mean(f["Q1"] ** 2 + f["Q2"] ** 2))
            if d < bd:
                best, bd = nm, d
        states.append(best if rn < 1e-9 else None)
        rows.append({"branch": k, "E_nn": float(E0), "pre_rel": float(pre),
                     "state": best, "res": float(rn), "post_rel": float(bd)})
        print(f"branch {k}: E_nn={E0:8.2f} pre_rel={pre:.4f} -> {best} "
              f"(res {rn:.1e}, post_rel {bd:.2e})", flush=True)

    # pairwise distances between branches (are they really distinct fields?)
    with torch.no_grad():
        F = [m.fields(pts, k) for k in range(K)]
    dmat = []
    for i in range(K):
        for j in range(i + 1, K):
            d = float(torch.sqrt(torch.mean((F[i][0] - F[j][0]) ** 2 +
                                            (F[i][1] - F[j][1]) ** 2)))
            dmat.append(d)
    st = sorted({s for s in states if s})
    print(f"STATES: {st} ({len(st)}/6 distinct)", flush=True)
    print(f"pairwise branch distance: min {min(dmat):.3f} max {max(dmat):.3f}", flush=True)
    json.dump({"rows": rows, "distinct": st, "pairwise_min": min(dmat),
               "pairwise_max": max(dmat)},
              open(os.path.join(RES, f"{TAG}_census.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
