"""Rel L2 / Linf / H1 errors of the classically refined branches (flow+Newton
endpoints of the canonical run) against the mesh-converged reference, evaluated
on the common h=1/256 grid (n=257) -- the same footing as Table 1's refined
block (2.5e-12)."""
import os
import numpy as np

ART = os.environ.get("ART", os.path.expanduser("~/DeflationPINN/artifacts"))
N = 257
h = 1.0 / (N - 1)


def metrics(Q1, Q2, R1, R2):
    e1, e2 = Q1 - R1, Q2 - R2
    pointwise = np.sqrt(e1 ** 2 + e2 ** 2)
    l2 = np.sqrt(np.mean(e1 ** 2 + e2 ** 2) / np.mean(R1 ** 2 + R2 ** 2))
    linf = pointwise.max() / np.sqrt((R1 ** 2 + R2 ** 2).max())
    gx = lambda F: (F[:, 1:] - F[:, :-1]) / h
    gy = lambda F: (F[1:, :] - F[:-1, :]) / h
    num = (np.mean(gx(e1) ** 2) + np.mean(gy(e1) ** 2) + np.mean(gx(e2) ** 2)
           + np.mean(gy(e2) ** 2) + np.mean(e1 ** 2 + e2 ** 2))
    den = (np.mean(gx(R1) ** 2) + np.mean(gy(R1) ** 2) + np.mean(gx(R2) ** 2)
           + np.mean(gy(R2) ** 2) + np.mean(R1 ** 2 + R2 ** 2))
    return l2, linf, np.sqrt(num / den)


rows = {}
for b in range(6):
    d = np.load(f"{ART}/dgx/checkpoints/flow_endpoint_b{b}.npz")
    state = str(d["state"])
    ref = np.load(f"{ART}/reference/{state}_n{N}.npz")
    l2, linf, h1 = metrics(d["Q1"], d["Q2"], ref["Q1"], ref["Q2"])
    rows[state] = (l2, linf, h1)

print(f"{'state':>6} {'rel L2':>12} {'rel Linf':>12} {'rel H1':>12}")
for s in ["D1", "D2", "R1", "R2", "R3", "R4"]:
    l2, linf, h1 = rows[s]
    print(f"{s:>6} {l2:12.2e} {linf:12.2e} {h1:12.2e}")
