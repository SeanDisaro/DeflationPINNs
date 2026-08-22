"""Certify an unstable wall state at eps=0.02 by eps-continuation inside the
invariant subspace Q12=0, then measure NN-branch distances to it."""
import os, sys, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import numpy as np, torch
import deepxde as dde  # noqa
from harness_ldg_delta import SmoothSixState
import ldg_reference as LR
from harness_ldg import HarmonicExt

RES = os.path.expanduser("~/DeflationPINNs_rev/results")
N = 257

g = np.linspace(0, 1, N); X, Y = np.meshgrid(g, g, indexing="xy")
hx = HarmonicExt().double()
with torch.no_grad():
    H = hx(torch.tensor(np.stack([X.ravel(), Y.ravel()], 1))).numpy().reshape(N, N)
Q1, Q2 = H.copy(), np.zeros((N, N))

eps_path = [0.08, 0.06, 0.045, 0.035, 0.028, 0.024, 0.022, 0.021, 0.0205, 0.02]
ok = True
for e in eps_path:
    LR.EPS = e
    Q1, Q2, rn, its = LR.newton_state("D1", N, seed=(Q1, Q2), tol=1e-11, itmax=100)
    E = LR.energy(Q1, Q2, N)
    print(f"eps={e}: res={rn:.1e} its={its} E={E:.4f} max|Q12|={np.abs(Q2).max():.1e} min|Q1|_int={np.abs(Q1[20:-20,20:-20]).min():.3f}", flush=True)
    if rn > 1e-9:
        ok = False
        break
LR.EPS = 0.02
if ok:
    np.savez_compressed(os.path.join(RES, "wall_certified_eps002.npz"), Q1=Q1, Q2=Q2)
    # NN branch distances (residual-refined run H, and discovery run C via 65-node eval? use H)
    geom = dde.geometry.geometry_2d.Rectangle([0., 0.], [1., 1.])
    m = SmoothSixState(geom).double()
    m.load_state_dict(torch.load(os.path.join(RES, "ldg_H_delta.pt"), map_location="cpu"))
    m.eval()
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float64)
    with torch.no_grad():
        out = m(pts)
    nT = np.sqrt(np.mean(Q1 ** 2 + Q2 ** 2))
    ds = []
    for k in range(6):
        q1 = out["out1"][k].numpy().reshape(N, N); q2 = out["out2"][k].numpy().reshape(N, N)
        ds.append(float(np.sqrt(np.mean((q1 - Q1) ** 2 + (q2 - Q2) ** 2))))
    k = int(np.argmin(ds))
    res = {"E": float(LR.energy(Q1, Q2, N)), "closest_branch": k, "abs": ds[k], "rel": ds[k] / float(nT), "all_abs": ds}
    json.dump(res, open(os.path.join(RES, "wall_certified.json"), "w"), indent=1)
    print("CERTIFIED", res, flush=True)
else:
    print("CONTINUATION_STALLED", flush=True)
