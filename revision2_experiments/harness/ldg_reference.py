"""Mesh-converged reference solutions for the reduced LdG system.

Solves  eps^2 * Lap(Q) + 2(1-|Q|^2) Q = 0  on [0,1]^2 with the trapezoidal
Dirichlet data (d = 3 eps, eps = 0.02) by 5-point finite differences + Newton,
seeded by bilinear interpolation of the six FEM states from the .mat files.
Runs at h = 1/256 and h = 1/512 for a convergence check; reports energies
E(Q) = int |grad Q|^2 + eps^-2 (|Q|^2-1)^2  (converged FEM literature values:
D ~ 78.011, R ~ 86.648) and the deviation of the 65x65 .mat export from the
converged reference at the shared nodes.
"""
import os
import numpy as np
from scipy.io import loadmat
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

EPS = 0.02
D_RAMP = 3 * EPS
NAMES = ["D1", "D2", "R1", "R2", "R3", "R4"]
DATA = os.path.join(os.getcwd(), "data", "trueSolution")
OUTDIR = os.path.expanduser("~/DeflationPINNs_rev/results/ldg_reference")
os.makedirs(OUTDIR, exist_ok=True)


def trap(t):
    return np.clip(np.minimum(t / D_RAMP, (1 - t) / D_RAMP), None, 1.0)


def boundary_fields(n):
    """Full-grid arrays with Q_b on the boundary, zero inside."""
    x = np.linspace(0, 1, n)
    Q1 = np.zeros((n, n))  # indexed [iy, ix]
    Q1[0, :] = trap(x)      # y = 0
    Q1[-1, :] = trap(x)     # y = 1
    Q1[:, 0] = -trap(x)     # x = 0  (=-T_d(y))
    Q1[:, -1] = -trap(x)    # x = 1
    # corners: T_d(0)=T_d(1)=0 so the two prescriptions agree (both 0)
    Q2 = np.zeros((n, n))
    return Q1, Q2


def load_seed(name):
    m = loadmat(os.path.join(DATA, f"data_LDG_{name}_solution.mat"))
    xy = m["c4n"]
    idx = np.lexsort((xy[:, 0], xy[:, 1]))
    nn = int(round(np.sqrt(xy.shape[0])))
    g = np.linspace(0, 1, nn)
    Q1 = m["p1"].reshape(-1)[idx].reshape(nn, nn)
    Q2 = m["q1"].reshape(-1)[idx].reshape(nn, nn)
    return g, Q1, Q2


def interp_to(gsrc, F, n):
    it = RegularGridInterpolator((gsrc, gsrc), F, method="linear")
    x = np.linspace(0, 1, n)
    Xg, Yg = np.meshgrid(x, x, indexing="xy")
    return it(np.stack([Yg.ravel(), Xg.ravel()], axis=1)).reshape(n, n)


def lap_matrix(n):
    """5-point Laplacian (times h^-2) on the (n-2)^2 interior, Dirichlet."""
    m = n - 2
    h2 = (n - 1) ** 2
    main = -4.0 * np.ones(m)
    off = np.ones(m - 1)
    T = sparse.diags([off, main, off], [-1, 0, 1])
    Im = sparse.identity(m)
    A = sparse.kron(Im, T) + sparse.kron(sparse.diags([off, off], [-1, 1]), Im)
    return (A * h2).tocsr()


def newton_state(name, n, seed=None, tol=1e-10, itmax=40):
    Q1b, Q2b = boundary_fields(n)
    if seed is None:
        gs, S1, S2 = load_seed(name)
        Q1 = interp_to(gs, S1, n)
        Q2 = interp_to(gs, S2, n)
    else:
        Q1, Q2 = seed
    Q1[0, :], Q1[-1, :], Q1[:, 0], Q1[:, -1] = Q1b[0, :], Q1b[-1, :], Q1b[:, 0], Q1b[:, -1]
    Q2[0, :], Q2[-1, :], Q2[:, 0], Q2[:, -1] = 0, 0, 0, 0

    m = n - 2
    L = lap_matrix(n)
    h2 = (n - 1) ** 2

    def lap_apply(F):
        """eps^2 * 5-point Laplacian of full-grid F, on the interior."""
        out = (F[1:-1, :-2] + F[1:-1, 2:] + F[:-2, 1:-1] + F[2:, 1:-1]
               - 4 * F[1:-1, 1:-1]) * h2
        return EPS ** 2 * out

    def residual(Q1, Q2):
        nq = Q1[1:-1, 1:-1] ** 2 + Q2[1:-1, 1:-1] ** 2
        r1 = lap_apply(Q1) + 2 * (1 - nq) * Q1[1:-1, 1:-1]
        r2 = lap_apply(Q2) + 2 * (1 - nq) * Q2[1:-1, 1:-1]
        return r1, r2

    for it in range(itmax):
        r1, r2 = residual(Q1, Q2)
        rn = np.sqrt(np.mean(r1 ** 2 + r2 ** 2))
        if rn < tol:
            break
        q1 = Q1[1:-1, 1:-1].ravel()
        q2 = Q2[1:-1, 1:-1].ravel()
        d11 = 2 * (1 - 3 * q1 ** 2 - q2 ** 2)
        d22 = 2 * (1 - q1 ** 2 - 3 * q2 ** 2)
        d12 = -4 * q1 * q2
        A11 = EPS ** 2 * L + sparse.diags(d11)
        A22 = EPS ** 2 * L + sparse.diags(d22)
        A12 = sparse.diags(d12)
        J = sparse.bmat([[A11, A12], [A12, A22]], format="csc")
        rhs = -np.concatenate([r1.ravel(), r2.ravel()])
        dq = spsolve(J, rhs)
        step = 1.0
        for _ in range(6):  # damped Newton fallback
            T1, T2 = Q1.copy(), Q2.copy()
            T1[1:-1, 1:-1] += step * dq[:m * m].reshape(m, m)
            T2[1:-1, 1:-1] += step * dq[m * m:].reshape(m, m)
            t1, t2 = residual(T1, T2)
            if np.sqrt(np.mean(t1 ** 2 + t2 ** 2)) < rn or step < 0.05:
                Q1, Q2 = T1, T2
                break
            step *= 0.5
    r1, r2 = residual(Q1, Q2)
    rn = np.sqrt(np.mean(r1 ** 2 + r2 ** 2))
    return Q1, Q2, rn, it


def energy(Q1, Q2, n):
    h = 1.0 / (n - 1)
    gx1 = (Q1[:, 1:] - Q1[:, :-1]) / h
    gy1 = (Q1[1:, :] - Q1[:-1, :]) / h
    gx2 = (Q2[:, 1:] - Q2[:, :-1]) / h
    gy2 = (Q2[1:, :] - Q2[:-1, :]) / h
    grad = np.sum(gx1 ** 2) * h * h + np.sum(gy1 ** 2) * h * h \
        + np.sum(gx2 ** 2) * h * h + np.sum(gy2 ** 2) * h * h
    bulk = np.sum((Q1 ** 2 + Q2 ** 2 - 1) ** 2) * h * h / EPS ** 2
    return grad + bulk


def main():
    for n in [257, 513]:
        print(f"===== grid {n} =====", flush=True)
        for name in NAMES:
            Q1, Q2, rn, it = newton_state(name, n)
            E = energy(Q1, Q2, n)
            np.savez_compressed(os.path.join(OUTDIR, f"{name}_n{n}.npz"),
                                Q1=Q1, Q2=Q2, resid=rn, energy=E)
            # deviation of the 65x65 export from this reference at shared nodes
            gs, S1, S2 = load_seed(name)
            stride = (n - 1) // (len(gs) - 1)
            R1 = Q1[::stride, ::stride]
            R2 = Q2[::stride, ::stride]
            num = np.mean((R1 - S1) ** 2 + (R2 - S2) ** 2)
            den = np.mean(R1 ** 2 + R2 ** 2)
            dev = np.sqrt(num / den)
            print(f"{name}: newton_its={it} resid={rn:.2e} E={E:.4f} "
                  f"mat_export_relL2_dev={dev:.4f}", flush=True)


if __name__ == "__main__":
    main()
