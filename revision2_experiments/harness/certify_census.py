"""Certify ALL branches of a K-branch Deflation-PINN as PDE solutions.

For each branch: damped Newton (stability-blind) from the NN field; on stall,
Levenberg-Marquardt on the FD residual. Certified endpoints (residual < 1e-9)
are classified against the six stable references and clustered; for every
certified solution we record: type, energy, terminal residual, and the NN
branch's rel/abs L2 distance to it. Output: census.json + endpoint npz files.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import deepxde as dde  # noqa: F401,E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import spsolve  # noqa: E402
from harness_ldg import build_model, SOL_NAMES  # noqa: E402
import ldg_reference as LR  # noqa: E402

RES = os.path.expanduser("~/DeflationPINNs_rev/results")
N = 257
EPS = 0.02


def nn_fields(ckpt, numsol):
    m = build_model(numsol=numsol)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    g = np.linspace(0, 1, N)
    X, Y = np.meshgrid(g, g, indexing="xy")
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32)
    o1 = [[] for _ in range(numsol)]
    o2 = [[] for _ in range(numsol)]
    with torch.no_grad():
        for i in range(0, pts.shape[0], 4096):
            o = m(pts[i:i + 4096])
            for k in range(numsol):
                o1[k].append(o["out1"][k])
                o2[k].append(o["out2"][k])
    return [(torch.cat(a).cpu().numpy().reshape(N, N).astype(float),
             torch.cat(b).cpu().numpy().reshape(N, N).astype(float)) for a, b in zip(o1, o2)]


def fd_residual(Q1, Q2, n=None):
    n = n or N
    h2 = (n - 1) ** 2
    lap1 = (Q1[1:-1, :-2] + Q1[1:-1, 2:] + Q1[:-2, 1:-1] + Q1[2:, 1:-1] - 4 * Q1[1:-1, 1:-1]) * h2
    lap2 = (Q2[1:-1, :-2] + Q2[1:-1, 2:] + Q2[:-2, 1:-1] + Q2[2:, 1:-1] - 4 * Q2[1:-1, 1:-1]) * h2
    nq = Q1[1:-1, 1:-1] ** 2 + Q2[1:-1, 1:-1] ** 2
    r1 = EPS ** 2 * lap1 + 2 * (1 - nq) * Q1[1:-1, 1:-1]
    r2 = EPS ** 2 * lap2 + 2 * (1 - nq) * Q2[1:-1, 1:-1]
    return r1, r2


def jacobian(Q1, Q2, L):  # L carries the grid scale
    q1 = Q1[1:-1, 1:-1].ravel()
    q2 = Q2[1:-1, 1:-1].ravel()
    A11 = EPS ** 2 * L + sparse.diags(2 * (1 - 3 * q1 ** 2 - q2 ** 2))
    A22 = EPS ** 2 * L + sparse.diags(2 * (1 - q1 ** 2 - 3 * q2 ** 2))
    A12 = sparse.diags(-4 * q1 * q2)
    return sparse.bmat([[A11, A12], [A12, A22]], format="csc")


def refine2x(F):
    """Bilinear 129 -> 257 refinement on the unit square grid."""
    n = F.shape[0]
    out = np.zeros((2 * n - 1, 2 * n - 1))
    out[::2, ::2] = F
    out[1::2, ::2] = 0.5 * (F[:-1, :] + F[1:, :])
    out[::2, 1::2] = 0.5 * (F[:, :-1] + F[:, 1:])
    out[1::2, 1::2] = 0.25 * (F[:-1, :-1] + F[1:, :-1] + F[:-1, 1:] + F[1:, 1:])
    return out


def lm_polish(Q1, Q2, n=129, itmax=120, tol=1e-10):
    """Levenberg-Marquardt on the FD residual at grid n (coarse for speed);
    robust near degenerate walls."""
    if n < Q1.shape[0]:
        stride = (Q1.shape[0] - 1) // (n - 1)
        Q1 = Q1[::stride, ::stride].copy()
        Q2 = Q2[::stride, ::stride].copy()
    m = n - 2
    L = LR.lap_matrix(n)
    lam = 1e-4
    Q1, Q2 = Q1.copy(), Q2.copy()
    for it in range(itmax):
        r1, r2 = fd_residual(Q1, Q2, n)
        F = np.concatenate([r1.ravel(), r2.ravel()])
        rn = np.sqrt(np.mean(F ** 2))
        if rn < tol:
            break
        J = jacobian(Q1, Q2, L)
        A = (J.T @ J + lam * sparse.identity(2 * m * m)).tocsc()
        dq = spsolve(A, -J.T @ F)
        T1, T2 = Q1.copy(), Q2.copy()
        T1[1:-1, 1:-1] += dq[:m * m].reshape(m, m)
        T2[1:-1, 1:-1] += dq[m * m:].reshape(m, m)
        t1, t2 = fd_residual(T1, T2, n)
        tn = np.sqrt(np.mean(t1 ** 2 + t2 ** 2))
        if tn < rn:
            Q1, Q2 = T1, T2
            lam = max(lam * 0.3, 1e-9)
        else:
            lam *= 8
            if lam > 1e6:
                break
    r1, r2 = fd_residual(Q1, Q2, n)
    return Q1, Q2, float(np.sqrt(np.mean(r1 ** 2 + r2 ** 2))), it


def eps_continuation(Q1, Q2, path=(0.08, 0.06, 0.045, 0.035, 0.028, 0.024, 0.022, 0.0205, 0.02)):
    """Newton with eps-continuation from the NN field: regularizes the wall-type
    branches whose Jacobian is near-singular at eps = 0.02."""
    global EPS
    A, B = Q1.copy(), Q2.copy()
    rn = 1e9
    for e in path:
        LR.EPS = e
        A, B, rn, its = LR.newton_state("D1", N, seed=(A, B), tol=1e-11, itmax=80)
        if rn > 1e-9:
            LR.EPS = 0.02
            return A, B, rn
    LR.EPS = 0.02
    return A, B, rn


def classify(Q1, Q2, refs):
    best = ("wall", None, 1e9)
    for nm, (R1, R2) in refs.items():
        rel = np.sqrt(np.mean((Q1 - R1) ** 2 + (Q2 - R2) ** 2) / np.mean(R1 ** 2 + R2 ** 2))
        if rel < best[2]:
            best = ("stable", nm, rel)
    if best[2] < 0.05:
        return best[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--numsol", type=int, required=True)
    ap.add_argument("--tag", default="census")
    args = ap.parse_args()

    refs = {}
    for nm in SOL_NAMES:
        f = np.load(os.path.join(RES, f"ldg_reference/{nm}_n{N}.npz"))
        refs[nm] = (f["Q1"], f["Q2"])
    wallp = os.path.join(RES, "wall_certified_eps002.npz")
    known_wall = None
    if os.path.exists(wallp):
        w = np.load(wallp)
        known_wall = (w["Q1"], w["Q2"])

    fields = nn_fields(args.ckpt, args.numsol)
    census = []
    endpoints = []
    for k, (Q1, Q2) in enumerate(fields):
        E_nn = LR.energy(Q1, Q2, N)
        rn_nn = float(np.sqrt(np.mean(np.sum([r ** 2 for r in fd_residual(Q1, Q2)], axis=0))))
        # stage 1: damped Newton
        T1, T2, rn, its = LR.newton_state("D1", N, seed=(Q1.copy(), Q2.copy()), tol=1e-11, itmax=60)
        method = "newton"
        if rn > 1e-9:  # stage 2: coarse LM from the ORIGINAL NN field, then Newton
            C1, C2, rn_c, its = lm_polish(Q1, Q2, n=129)
            T1, T2 = refine2x(C1), refine2x(C2)
            T1, T2, rn, its = LR.newton_state("D1", N, seed=(T1, T2), tol=1e-11, itmax=60)
            method = f"LM129({rn_c:.1e})+newton"
        if rn > 1e-9:  # stage 3: eps-continuation Newton (certifies wall-type states)
            E1, E2, rn_e = eps_continuation(Q1, Q2)
            if rn_e < rn:
                T1, T2, rn = E1, E2, rn_e
                method = "eps-continuation"
        cert = bool(rn < 1e-9)
        E = LR.energy(T1, T2, N) if cert else None
        nm = classify(T1, T2, refs) if cert else None
        if cert and nm is None and known_wall is not None:
            dw = np.sqrt(np.mean((T1 - known_wall[0]) ** 2 + (T2 - known_wall[1]) ** 2) /
                         np.mean(known_wall[0] ** 2 + known_wall[1] ** 2))
            if dw < 0.05:
                nm = "W_X"   # the known X-wall state
        d = float(np.sqrt(np.mean((Q1 - T1) ** 2 + (Q2 - T2) ** 2))) if cert else None
        nT = float(np.sqrt(np.mean(T1 ** 2 + T2 ** 2))) if cert else None
        rec = {"branch": k, "E_nn": float(E_nn), "res_nn": rn_nn, "certified": cert,
               "method": method, "res_final": float(rn),
               "E": (float(E) if cert else None),
               "state": (nm if nm else ("wall" if cert else None)),
               "nn_rel_to_endpoint": (d / nT if cert else None),
               "nn_abs_to_endpoint": d}
        census.append(rec)
        print(rec, flush=True)
        if cert:
            np.savez_compressed(os.path.join(RES, f"{args.tag}_endpoint_b{k}.npz"), Q1=T1, Q2=T2)
            endpoints.append((k, T1, T2, rec))

    # distinctness clustering of certified endpoints
    groups = []
    for k, T1, T2, rec in endpoints:
        placed = False
        for gname, (G1, G2) in groups:
            if np.sqrt(np.mean((T1 - G1) ** 2 + (T2 - G2) ** 2)) < 0.05:
                rec["cluster"] = gname
                placed = True
                break
        if not placed:
            gname = f"S{len(groups)}"
            groups.append((gname, (T1, T2)))
            rec["cluster"] = gname
    distinct = len(groups)
    print(f"CENSUS: {sum(r['certified'] for r in census)}/{len(census)} certified, "
          f"{distinct} distinct solutions, "
          f"stable={sorted(set(r['state'] for r in census if r['state'] and r['state'] != 'wall'))}, "
          f"walls={[round(r['E'],2) for r in census if r['state'] == 'wall']}", flush=True)
    json.dump(census, open(os.path.join(RES, f"{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
