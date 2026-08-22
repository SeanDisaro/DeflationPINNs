"""Certify genuine UNSTABLE wall states via the invariant subspace Q12 = 0
(the boundary data has Q12=0, and Newton preserves Q2=0 exactly), then measure
how far the residual-refined NN branches sit from them. Emits JSON."""
import os, sys, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import numpy as np, torch
import deepxde as dde  # noqa
from harness_ldg_delta import SmoothSixState
from ldg_reference import newton_state, energy, boundary_fields
from harness_ldg import HarmonicExt

RES = os.path.expanduser("~/DeflationPINNs_rev/results")
N = 257

# seeds inside the Q2=0 subspace
seeds = {}
# (a) harmonic extension of the boundary data (smooth, wall-free start)
g = np.linspace(0, 1, N); X, Y = np.meshgrid(g, g, indexing="xy")
hx = HarmonicExt().double()
with torch.no_grad():
    H = hx(torch.tensor(np.stack([X.ravel(), Y.ravel()], 1))).numpy().reshape(N, N)
seeds["harmonic"] = (H.copy(), np.zeros((N, N)))
# (b) the stubborn residual-refined branch (H run, branch 1), Q2 zeroed
geom = dde.geometry.geometry_2d.Rectangle([0., 0.], [1., 1.])
m = SmoothSixState(geom).double()
m.load_state_dict(torch.load(os.path.join(RES, "ldg_H_delta.pt"), map_location="cpu"))
m.eval()
pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float64)
with torch.no_grad():
    out = m(pts)
nn_fields = [(out["out1"][k].numpy().reshape(N, N), out["out2"][k].numpy().reshape(N, N))
             for k in range(6)]
seeds["branch1_Q2zero"] = (nn_fields[1][0].copy(), np.zeros((N, N)))

walls = []
for name, (S1, S2) in seeds.items():
    T1, T2, rn, its = newton_state("D1", N, seed=(S1, S2), tol=1e-11, itmax=80)
    E = energy(T1, T2, N)
    q2max = float(np.abs(T2).max())
    print(f"{name}: res={rn:.1e} its={its} E={E:.4f} max|Q12|={q2max:.1e}", flush=True)
    if rn < 1e-9:
        walls.append({"seed": name, "E": float(E), "res": float(rn)})
        np.savez_compressed(os.path.join(RES, f"wall_{name}.npz"), Q1=T1, Q2=T2)

# distances of every NN branch (residual-refined run H) to each certified wall state
for w in walls:
    f = np.load(os.path.join(RES, f"wall_{w['seed']}.npz"))
    T1, T2 = f["Q1"], f["Q2"]
    nT = np.sqrt(np.mean(T1 ** 2 + T2 ** 2))
    ds = [float(np.sqrt(np.mean((q1 - T1) ** 2 + (q2 - T2) ** 2))) for q1, q2 in nn_fields]
    k = int(np.argmin(ds))
    w["closest_branch"] = k
    w["abs"] = ds[k]
    w["rel"] = ds[k] / nT
    print(f"wall E={w['E']:.3f}: closest NN branch {k} rel={w['rel']:.4f} abs={w['abs']:.4f}", flush=True)
json.dump(walls, open(os.path.join(RES, "walls.json"), "w"), indent=1)
print("WALLS_DONE", flush=True)
