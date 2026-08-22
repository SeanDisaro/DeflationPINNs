"""Reviewer-1 requested diagnostics, computed locally from archived checkpoints:
  * contour maps of the pointwise error |Q_NN - Q_ref| for discovery and DDR
  * L2, L-infinity and H1 errors (relative), per state
  * where the error lives: corner-core region vs boundary layer vs bulk
Paths assume the artifacts tree (reference npz + checkpoints); adjust ART/OUT.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = os.environ.get("ART", os.path.expanduser("~/DeflationPINN/artifacts"))
OUT = os.environ.get("OUT", os.path.expanduser("~/DeflationPINN/figures"))
STATES = ["D1", "D2", "R1", "R2", "R3", "R4"]
N = 513
EPS = 0.02

# ---------------- models rebuilt from state dicts (no deepxde needed) ----------
sdD = {k: v.double() for k, v in torch.load(f"{ART}/dgx/checkpoints/ldg_C_full.pt", map_location="cpu").items()}  # discovery
sdJ = {k: v.double() for k, v in torch.load(f"{ART}/dgx/checkpoints/ldg_J_ritz1pct.pt", map_location="cpu").items()}  # DDR
n_pi, c_n = sdJ["ext.n_pi"], sdJ["ext.c_n"]


def sinh_ratio(z):
    return torch.exp(-n_pi * z) * (1 - torch.exp(-2 * n_pi * (1 - z))) / (1 - torch.exp(-2 * n_pi) + 1e-300)


def harm(xy):
    x, y = xy[:, :1], xy[:, 1:]
    return (torch.sum(c_n * torch.sin(x @ n_pi) * (sinh_ratio(y) + sinh_ratio(1 - y)), 1, keepdim=True)
            - torch.sum(c_n * torch.sin(y @ n_pi) * (sinh_ratio(x) + sinh_ratio(1 - x)), 1, keepdim=True))


def ddr_fields(xy, k):
    z = 2 * np.pi * (xy @ sdJ[f"branches.{k}.B"])
    h = torch.cat([xy, torch.sin(z), torch.cos(z)], 1)
    for i in (0, 2, 4):
        h = torch.tanh(h @ sdJ[f"branches.{k}.net.{i}.weight"].T + sdJ[f"branches.{k}.net.{i}.bias"])
    m = h @ sdJ["branches.%d.net.6.weight" % k].T + sdJ[f"branches.{k}.net.6.bias"]
    x, y = xy[:, :1], xy[:, 1:]
    w = 16 * x * (1 - x) * y * (1 - y)
    return w * m[:, :1] + harm(xy), w * m[:, 1:]


# discovery model: trunk (2->4000->32) x tanh, branch weights, star-domain extension.
# The star extension is reproduced from the repo formula for the unit square.
def disc_trunk(xy):
    h = torch.tanh(xy @ sdD["trunkNet_Lin.0.weight"].T + sdD["trunkNet_Lin.0.bias"])
    return h @ sdD["trunkNet_Lin.1.weight"].T + sdD["trunkNet_Lin.1.bias"]


def trap(t, d=0.06):
    return torch.clamp(torch.minimum(t / d, (1 - t) / d), max=1.0)


def star_ext(xy):
    """Repo DCBoundaryExtension for the unit square:
       f_b(x_b) * (r / r_b), with f_b(x_b) = T_d(x_b) - T_d(y_b) and x_b the
       boundary point hit by the ray from the centre (0.5, 0.5)."""
    dx, dy = xy[:, 0] - 0.5, xy[:, 1] - 0.5
    m = torch.maximum(torch.abs(dx), torch.abs(dy))
    t = (m / 0.5).clamp(max=1.0)                      # r / r_b
    scale = (2.0 * m).clamp(min=1e-12)                # maps the ray onto the boundary
    xb, yb = 0.5 + dx / scale, 0.5 + dy / scale
    fb = trap(xb) - trap(yb)                          # bfunc = T_d(x_b) - T_d(y_b)
    return (fb * t).view(-1, 1)


def disc_fields(xy, k):
    t = disc_trunk(xy)
    b = sdD[f"branchFeatures.{k}"]
    p = t.shape[1] // 2
    o1 = (t * b)[:, :p].sum(1, keepdim=True) + sdD["deepONet_biases"][0]
    o2 = (t * b)[:, p:].sum(1, keepdim=True) + sdD["deepONet_biases"][1]
    x, y = xy[:, :1], xy[:, 1:]
    w = 16 * x * (1 - x) * y * (1 - y)
    return o1 * w + star_ext(xy), o2 * w


# ---------------- references, matching, metrics --------------------------------
g = np.linspace(0, 1, N)
X, Y = np.meshgrid(g, g, indexing="xy")
xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float64)
h = 1.0 / (N - 1)
refs = {s: np.load(f"{ART}/reference/{s}_n{N}.npz") for s in STATES}


def eval_model(fn, K=6, chunk=60000):
    out = []
    for k in range(K):
        a, b = [], []
        with torch.no_grad():
            for i in range(0, xy.shape[0], chunk):
                q1, q2 = fn(xy[i:i + chunk], k)
                a.append(q1); b.append(q2)
        out.append((torch.cat(a).numpy().reshape(N, N), torch.cat(b).numpy().reshape(N, N)))
    return out


def metrics(Q1, Q2, ref):
    R1, R2 = ref["Q1"], ref["Q2"]
    e1, e2 = Q1 - R1, Q2 - R2
    pointwise = np.sqrt(e1 ** 2 + e2 ** 2)
    l2 = np.sqrt(np.mean(e1 ** 2 + e2 ** 2) / np.mean(R1 ** 2 + R2 ** 2))
    linf = pointwise.max() / np.sqrt((R1 ** 2 + R2 ** 2).max())
    gx = lambda F: (F[:, 1:] - F[:, :-1]) / h
    gy = lambda F: (F[1:, :] - F[:-1, :]) / h
    num = (np.mean(gx(e1) ** 2) + np.mean(gy(e1) ** 2) + np.mean(gx(e2) ** 2) + np.mean(gy(e2) ** 2)
           + np.mean(e1 ** 2 + e2 ** 2))
    den = (np.mean(gx(R1) ** 2) + np.mean(gy(R1) ** 2) + np.mean(gx(R2) ** 2) + np.mean(gy(R2) ** 2)
           + np.mean(R1 ** 2 + R2 ** 2))
    return l2, linf, np.sqrt(num / den), pointwise


def match(fields):
    cost = np.zeros((6, 6))
    for k, (Q1, Q2) in enumerate(fields):
        for j, s in enumerate(STATES):
            r = refs[s]
            cost[k, j] = np.sqrt(np.mean((Q1 - r["Q1"]) ** 2 + (Q2 - r["Q2"]) ** 2) /
                                 np.mean(r["Q1"] ** 2 + r["Q2"] ** 2))
    from scipy.optimize import linear_sum_assignment
    rw, cl = linear_sum_assignment(cost)
    return {STATES[j]: int(k) for k, j in zip(rw, cl)}


for tag, fn in (("discovery", disc_fields), ("DDR", ddr_fields)):
    F = eval_model(fn)
    perm = match(F)
    print(f"\n===== {tag} =====")
    print(f"{'state':>6} {'rel L2':>9} {'rel Linf':>9} {'rel H1':>9} {'corner%':>8} {'bdry%':>7} {'bulk%':>7}")
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.4))
    dist = np.minimum(np.minimum(X, 1 - X), np.minimum(Y, 1 - Y))
    corner = (np.minimum(X, 1 - X) < 0.1) & (np.minimum(Y, 1 - Y) < 0.1)
    bdry = (dist < 0.1) & (~corner)
    bulk = dist >= 0.1
    for i, s in enumerate(STATES):
        Q1, Q2 = F[perm[s]]
        l2, linf, h1, pw = metrics(Q1, Q2, refs[s])
        tot = (pw ** 2).sum()
        print(f"{s:>6} {l2:9.4f} {linf:9.4f} {h1:9.4f} "
              f"{100*(pw**2)[corner].sum()/tot:7.1f}% {100*(pw**2)[bdry].sum()/tot:6.1f}% "
              f"{100*(pw**2)[bulk].sum()/tot:6.1f}%")
        a = ax[i // 3][i % 3]
        cs = a.contourf(X, Y, pw, levels=24, cmap="magma")
        fig.colorbar(cs, ax=a, shrink=0.9)
        a.set_aspect("equal"); a.set_xticks([0, 1]); a.set_yticks([0, 1])
        a.set_title(f"{s}:  rel $L^2$={l2:.3f},  rel $L^\\infty$={linf:.3f}", fontsize=10)
    fig.suptitle(f"Pointwise error $|\\mathbf{{Q}}^{{NN}}-\\mathbf{{Q}}^{{ref}}|$ — {tag} stage",
                 fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/error_contours_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT}/error_contours_{tag}.png")
