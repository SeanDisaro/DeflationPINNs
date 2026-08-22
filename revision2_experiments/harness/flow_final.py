"""Classical validation of the discovery stage: run the LdG gradient flow
(semi-implicit) from each run-C branch field, then Newton, and identify which
stable state each branch flows to. Counts how many DISTINCT stable states the
unsupervised discovery actually seeds.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import deepxde as dde  # noqa: F401,E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402
from harness_ldg import build_model, SOL_NAMES  # noqa: E402
from ldg_reference import newton_state, energy, lap_matrix, boundary_fields, load_seed  # noqa: E402

EPS = 0.02
N = 257
CKPT = os.path.expanduser("~/DeflationPINNs_rev/results/ldg_C_full.pt")


def nn_fields():
    sd = torch.load(CKPT, map_location="cpu")
    width = sd["trunkNet_Lin.0.weight"].shape[0]
    K = sum(1 for key in sd if key.startswith("branchFeatures"))
    m = build_model(width=width, numsol=K)
    m.load_state_dict(sd)
    m.eval()
    g = np.linspace(0, 1, N)
    X, Y = np.meshgrid(g, g, indexing="xy")
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32)
    o1 = [[] for _ in range(K)]
    o2 = [[] for _ in range(K)]
    with torch.no_grad():
        for i in range(0, pts.shape[0], 4096):
            o = m(pts[i:i + 4096])
            for k in range(K):
                o1[k].append(o["out1"][k])
                o2[k].append(o["out2"][k])
    return [(torch.cat(a).numpy().reshape(N, N).astype(float),
             torch.cat(b).numpy().reshape(N, N).astype(float)) for a, b in zip(o1, o2)]


def flow(Q1, Q2, dt=0.1, nsteps=3000):
    """Semi-implicit gradient flow  Q_t = eps^2 Lap Q + 2(1-|Q|^2)Q."""
    Q1b, _ = boundary_fields(N)
    Q1 = Q1.copy(); Q2 = Q2.copy()
    Q1[0, :], Q1[-1, :], Q1[:, 0], Q1[:, -1] = Q1b[0, :], Q1b[-1, :], Q1b[:, 0], Q1b[:, -1]
    Q2[0, :], Q2[-1, :], Q2[:, 0], Q2[:, -1] = 0, 0, 0, 0
    m = N - 2
    L = lap_matrix(N)
    A = sparse.identity(m * m, format="csc") - dt * EPS ** 2 * L.tocsc()
    lu = splu(A)
    h2 = (N - 1) ** 2

    def bc_term(F):  # contribution of Dirichlet boundary values to the Laplacian
        out = np.zeros((m, m))
        out[0, :] += F[0, 1:-1]
        out[-1, :] += F[-1, 1:-1]
        out[:, 0] += F[1:-1, 0]
        out[:, -1] += F[1:-1, -1]
        return out * h2

    for s in range(nsteps):
        nq = Q1[1:-1, 1:-1] ** 2 + Q2[1:-1, 1:-1] ** 2
        r1 = Q1[1:-1, 1:-1] + dt * (2 * (1 - nq) * Q1[1:-1, 1:-1]) + dt * EPS ** 2 * bc_term(Q1)
        r2 = Q2[1:-1, 1:-1] + dt * (2 * (1 - nq) * Q2[1:-1, 1:-1]) + dt * EPS ** 2 * bc_term(Q2)
        Q1[1:-1, 1:-1] = lu.solve(r1.ravel()).reshape(m, m)
        Q2[1:-1, 1:-1] = lu.solve(r2.ravel()).reshape(m, m)
        if s % 200 == 0:
            print(f"    flow step {s} E={energy(Q1, Q2, N):.2f}", flush=True)
    return Q1, Q2


def classify(Q1, Q2):
    best = (None, 1e9)
    for nm in SOL_NAMES:
        gs, S1, S2 = load_seed(nm)
        stride = (N - 1) // (len(gs) - 1)
        R1, R2 = Q1[::stride, ::stride], Q2[::stride, ::stride]
        rel = np.sqrt(np.mean((R1 - S1) ** 2 + (R2 - S2) ** 2) / np.mean(S1 ** 2 + S2 ** 2))
        if rel < best[1]:
            best = (nm, rel)
    return best


def main():
    t0 = time.time()
    fields = nn_fields()
    endpoints = []
    for k, (Q1, Q2) in enumerate(fields):
        E0 = energy(Q1, Q2, N)
        F1, F2 = flow(Q1, Q2)
        T1, T2, rn, its = newton_state("D1", N, seed=(F1, F2), tol=1e-12, itmax=80)
        Ef = energy(T1, T2, N)
        nm, rel = classify(T1, T2)
        ref = np.load(os.path.expanduser(f"~/DeflationPINNs_rev/results/ldg_reference/{nm}_n257.npz"))
        rel_ref = np.sqrt(np.mean((T1-ref["Q1"])**2+(T2-ref["Q2"])**2)/np.mean(ref["Q1"]**2+ref["Q2"]**2))
        np.savez_compressed(os.path.expanduser(f"~/DeflationPINNs_rev/results/flow_endpoint_b{k}.npz"), Q1=T1, Q2=T2, state=nm)
        endpoints.append(nm)
        print(f"branch {k}: E_start={E0:.1f} -> after flow E={energy(F1, F2, N):.2f} "
              f"-> Newton res={rn:.1e} E={Ef:.4f} => {nm} (rel_vs_mat {rel:.4f} rel_vs_ref {rel_ref:.2e}) t={time.time()-t0:.0f}s", flush=True)
    print("DISTINCT STABLE ENDPOINTS:", sorted(set(endpoints)), f"({len(set(endpoints))}/6)", flush=True)


if __name__ == "__main__":
    main()
