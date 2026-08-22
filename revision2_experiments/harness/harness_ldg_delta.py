"""Run H: basin-safe high-accuracy refinement for the LdG six-state problem.

Pipeline (fully unsupervised end-to-end; no reference data enters training):
  0. Load the run-C Deflation-PINN checkpoint (all six basins discovered).
  1. SELF-DISTILL: fit six independent smooth per-branch nets, built on the
     smooth harmonic boundary extension, to run-C's own output fields (L2 fit).
     This projects away the derivative creases carried by the star extension.
  2. REFINE: per-branch squared PDE residual (eps^2-scaled), deflation frozen
     (basins fixed; true separations ~1.07 >> any drift), with a decaying L2
     tether to the distilled fields as a basin guard.
  3. RAR: inject the highest-residual points from a fine probe grid.
  4. Polish: float64 L-BFGS (strong Wolfe).
Evaluation against the 65^2 FEM export AND our h=1/512 FD-Newton reference.
"""
import argparse, copy, json, os, sys, time
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.getcwd())
import deepxde as dde  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from harness_ldg import (  # noqa: E402  (reuse: same dir on sys.path when run)
    HarmonicExt, build_points, build_model, SOL_NAMES,
)

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REF_DIR = os.path.expanduser("~/DeflationPINNs_rev/results/ldg_reference")


class BranchNet(nn.Module):
    def __init__(self, width=128, depth=3, fourier=0, fsigma=16.0):
        super().__init__()
        self.fourier = fourier
        in_dim = 2
        if fourier > 0:
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("B", torch.randn(2, fourier, generator=g, device="cpu") * fsigma)
            in_dim = 2 + 2 * fourier
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.fourier:
            z = 2 * np.pi * (x @ self.B)
            x = torch.cat([x, torch.sin(z), torch.cos(z)], dim=1)
        return self.net(x)


class SmoothSixState(nn.Module):
    """Six independent smooth branch nets under the harmonic hard constraint."""

    def __init__(self, geom, width=128, depth=3, fourier=0, fsigma=16.0):
        super().__init__()
        self.branches = nn.ModuleList(
            [BranchNet(width, depth, fourier, fsigma) for _ in range(6)])
        self.ext = HarmonicExt()
        self.geom = geom

    def fields(self, x, k):
        omega = self.geom.boundary_constraint_factor(x, smoothness="Cinf").view(-1, 1)
        m = self.branches[k](x)
        q1 = omega * m[:, 0].view(-1, 1) + self.ext(x)
        q2 = omega * m[:, 1].view(-1, 1)
        return q1, q2

    def forward(self, x):  # dict interface compatible with the eval helpers
        out1, out2 = [], []
        for k in range(6):
            q1, q2 = self.fields(x, k)
            out1.append(q1)
            out2.append(q2)
        return {"out1": out1, "out2": out2}


def ritz_energy(model, x, k, eps=0.02):
    """Deep-Ritz LdG energy of branch k (first derivatives only)."""
    q1, q2 = model.fields(x, k)
    ones = torch.ones_like(q1)
    g1 = torch.autograd.grad(q1, x, ones, create_graph=True)[0]
    g2 = torch.autograd.grad(q2, x, ones, create_graph=True)[0]
    nq = (q1 ** 2 + q2 ** 2).view(-1)
    e = (g1 ** 2).sum(1) + (g2 ** 2).sum(1) + (1 / eps ** 2) * (nq - 1) ** 2
    return torch.nanmean(e), (q1, q2)


def deflation_hinge(fields, dmin):
    """Pairwise hinge on full-field RMS distances; fields = [(q1,q2)]*6."""
    zero = torch.tensor(0., device=fields[0][0].device, dtype=fields[0][0].dtype)
    loss, np_ = zero.clone(), 0
    for i in range(6):
        for j in range(i + 1, 6):
            d1 = fields[i][0] - fields[j][0]
            d2 = fields[i][1] - fields[j][1]
            dist = torch.sqrt(torch.nanmean(d1 ** 2 + d2 ** 2))
            loss = loss + torch.maximum(1. - dist / dmin, zero)
            np_ += 1
    return loss / np_


def pde_residual(model, x, k, eps=0.02):
    q1, q2 = model.fields(x, k)
    ones = torch.ones_like(q1)
    g1 = torch.autograd.grad(q1, x, ones, create_graph=True)[0]
    g2 = torch.autograd.grad(q2, x, ones, create_graph=True)[0]
    q1xx = torch.autograd.grad(g1[:, 0].sum(), x, create_graph=True)[0][:, 0]
    q1yy = torch.autograd.grad(g1[:, 1].sum(), x, create_graph=True)[0][:, 1]
    q2xx = torch.autograd.grad(g2[:, 0].sum(), x, create_graph=True)[0][:, 0]
    q2yy = torch.autograd.grad(g2[:, 1].sum(), x, create_graph=True)[0][:, 1]
    nq = q1 ** 2 + q2 ** 2
    r1 = (2 * (1 - nq) * q1).view(-1) + eps ** 2 * (q1xx + q1yy)
    r2 = (2 * (1 - nq) * q2).view(-1) + eps ** 2 * (q2xx + q2yy)
    return r1, r2, (q1, q2)


def eval_all(model, data_dir):
    """Errors vs the 65^2 export and vs the h=1/512 FD reference; energies."""
    dt = next(model.parameters()).dtype
    mats = {nm: loadmat(os.path.join(data_dir, f"data_LDG_{nm}_solution.mat")) for nm in SOL_NAMES}
    xy = mats["D1"]["c4n"].astype(np.float64)
    x65 = torch.tensor(xy, dtype=dt, device=DEV)
    with torch.no_grad():
        out = model(x65)
    nn65 = [(out["out1"][k].cpu().numpy().reshape(-1), out["out2"][k].cpu().numpy().reshape(-1))
            for k in range(6)]
    cost = np.zeros((6, 6))
    for k in range(6):
        for j, nm in enumerate(SOL_NAMES):
            f1 = mats[nm]["p1"].reshape(-1)
            f2 = mats[nm]["q1"].reshape(-1)
            num = np.nanmean((nn65[k][0] - f1) ** 2 + (nn65[k][1] - f2) ** 2)
            den = np.nanmean(f1 ** 2 + f2 ** 2)
            cost[k, j] = np.sqrt(num / den)
    rows, cols = linear_sum_assignment(cost)
    rel65 = {SOL_NAMES[j]: float(cost[k, j]) for k, j in zip(rows, cols)}
    perm = {SOL_NAMES[j]: int(k) for k, j in zip(rows, cols)}

    # against the converged FD reference on its full 513^2 grid
    rel_ref = {}
    n = 513
    g = np.linspace(0, 1, n)
    Xg, Yg = np.meshgrid(g, g, indexing="xy")
    xs = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=dt, device=DEV)
    with torch.no_grad():
        big = model(xs)
    for nm in SOL_NAMES:
        f = np.load(os.path.join(REF_DIR, f"{nm}_n{n}.npz"))
        R1, R2 = f["Q1"], f["Q2"]  # indexed [iy, ix]
        k = perm[nm]
        q1 = big["out1"][k].cpu().numpy().reshape(n, n)  # rows follow y (meshgrid xy)
        q2 = big["out2"][k].cpu().numpy().reshape(n, n)
        num = np.mean((q1 - R1) ** 2 + (q2 - R2) ** 2)
        den = np.mean(R1 ** 2 + R2 ** 2)
        rel_ref[nm] = float(np.sqrt(num / den))
    return rel65, rel_ref, perm


def energies(model, n=257, eps=0.02):
    dt = next(model.parameters()).dtype
    g = np.linspace(0, 1, n)
    Xg, Yg = np.meshgrid(g, g, indexing="xy")
    xs = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=dt,
                      device=DEV).requires_grad_(True)
    E = []
    for k in range(6):
        q1, q2 = model.fields(xs, k)
        ones = torch.ones_like(q1)
        g1 = torch.autograd.grad(q1, xs, ones, create_graph=False, retain_graph=True)[0]
        g2 = torch.autograd.grad(q2, xs, ones, create_graph=False, retain_graph=True)[0]
        nq = (q1 ** 2 + q2 ** 2).view(-1)
        e = (g1 ** 2).sum(1) + (g2 ** 2).sum(1) + (1 / eps ** 2) * (nq - 1) ** 2
        E.append(float(e.mean().detach().cpu()))
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True)  # run-C checkpoint
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--distill-epochs", type=int, default=3000)
    ap.add_argument("--refine-epochs", type=int, default=6000)
    ap.add_argument("--rar-epochs", type=int, default=4000)
    ap.add_argument("--rar-points", type=int, default=3000)
    ap.add_argument("--lbfgs", type=int, default=150)
    ap.add_argument("--refine-mode", choices=["mse", "ritz"], default="mse")
    ap.add_argument("--polish", choices=["mse", "ritz"], default="mse")
    ap.add_argument("--fourier", type=int, default=0)
    ap.add_argument("--fsigma", type=float, default=16.0)
    ap.add_argument("--quad", type=int, default=0)        # cell-centered n x n quadrature grid
    ap.add_argument("--quad-final", type=int, default=0)  # finer quadrature for the L-BFGS stage
    ap.add_argument("--init-ckpt", type=str, default="")  # SmoothSixState checkpoint to continue from
    ap.add_argument("--dmin", type=float, default=0.8)
    ap.add_argument("--delta", type=float, default=100.)
    ap.add_argument("--tether", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="ldg_H_delta")
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    data_dir = os.path.join(os.getcwd(), "data", "trueSolution")
    t0 = time.time()

    # ---- source model (discovery stage output) ----
    src = build_model()  # original architecture: p=16, 1x4000, star extension
    src.load_state_dict(torch.load(args.source, map_location=DEV))
    src.eval()

    geom = dde.geometry.geometry_2d.Rectangle([0., 0.], [1., 1.])
    model = SmoothSixState(geom, args.width, args.depth, args.fourier, args.fsigma).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] device={DEV} params={n_par}", flush=True)
    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=DEV))
        print(f"[{args.tag}] initialized from {args.init_ckpt}", flush=True)

    def src_eval(pts, chunk=4096):
        """Chunked eval of the source model (its star-domain code has a
        quadratic broadcast that blows up on large batches)."""
        o1 = [[] for _ in range(6)]
        o2 = [[] for _ in range(6)]
        with torch.no_grad():
            for i in range(0, pts.shape[0], chunk):
                o = src(pts[i:i + chunk])
                for k in range(6):
                    o1[k].append(o["out1"][k])
                    o2[k].append(o["out2"][k])
        return [(torch.cat(a).detach().float(), torch.cat(b).detach().float())
                for a, b in zip(o1, o2)]

    def cell_grid(n):
        """Cell-centered n x n grid: the plain mean over it IS the composite
        midpoint rule, i.e., an unbiased quadrature of the Ritz energy."""
        c = (np.arange(n) + 0.5) / n
        Xc, Yc = np.meshgrid(c, c, indexing="xy")
        return torch.tensor(np.stack([Xc.ravel(), Yc.ravel()], 1),
                            dtype=torch.float32, device=DEV)

    # ---- stage 1: self-distillation on a dense point set ----
    xd = cell_grid(args.quad) if args.quad else build_points(129, corner_refine=True).to(DEV)
    targets = src_eval(xd)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(args.distill_epochs):
        opt.zero_grad()
        loss = 0.
        for k in range(6):
            q1, q2 = model.fields(xd, k)
            loss = loss + torch.mean((q1 - targets[k][0]) ** 2 + (q2 - targets[k][1]) ** 2)
        loss.backward()
        opt.step()
        if ep % 500 == 0:
            print(f"[{args.tag}] distill ep {ep} loss {loss.item():.3e} t {time.time()-t0:.0f}s", flush=True)
    rel65, rel_ref, perm = eval_all(model, data_dir)
    print(f"[{args.tag}] after distill rel65 {rel65}", flush=True)

    # training collocation/quadrature set (+ tether targets for mse mode)
    if args.quad:
        x = cell_grid(args.quad).requires_grad_(True)
    else:
        x = build_points(65, corner_refine=True).to(DEV).requires_grad_(True)
    tether = src_eval(x.detach()) if args.refine_mode == "mse" else None

    def refine(n_ep, lr, x_pts, tether_pts, w0, label, mode=None):
        mode = mode or args.refine_mode
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.7)
        for ep in range(n_ep):
            opt.zero_grad()
            w_t = w0 * (0.5 ** (ep / 1000.0))
            loss = 0.
            if mode == "ritz":
                fields, energies_now = [], []
                for k in range(6):
                    e_k, f_k = ritz_energy(model, x_pts, k)
                    loss = loss + 0.01 * e_k
                    fields.append(f_k)
                    energies_now.append(float(e_k.detach()))
                loss = loss + args.delta * deflation_hinge(fields, args.dmin)
            else:
                for k in range(6):
                    r1, r2, (q1, q2) = pde_residual(model, x_pts, k)
                    loss = loss + torch.nanmean(r1 ** 2) + torch.nanmean(r2 ** 2)
                    if tether_pts is not None and w_t > 1e-6:
                        loss = loss + w_t * torch.mean((q1 - tether_pts[k][0]) ** 2 +
                                                       (q2 - tether_pts[k][1]) ** 2)
            loss.backward()
            opt.step()
            sch.step()
            if ep % 500 == 0:
                extra = ""
                if mode == "ritz":
                    extra = " E=" + ",".join(f"{e:.0f}" for e in energies_now)
                print(f"[{args.tag}] {label} ep {ep} loss {loss.item():.4e} "
                      f"w_t {w_t:.1e}{extra} t {time.time()-t0:.0f}s", flush=True)

    # ---- stage 2: refinement with decaying tether ----
    refine(args.refine_epochs, 1e-4, x, tether, args.tether, "refine")
    rel65, rel_ref, perm = eval_all(model, data_dir)
    print(f"[{args.tag}] after refine rel65 {rel65}", flush=True)
    print(f"[{args.tag}] after refine relREF {rel_ref}", flush=True)
    torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}_postrefine.pt"))

    # ---- stage 3 (optional): residual-based point injection ----
    x_rar = x
    if args.rar_epochs > 0:
        gp = np.linspace(0.002, 0.998, 257)
        Xg, Yg = np.meshgrid(gp, gp, indexing="xy")
        probe = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], axis=1),
                             dtype=torch.float32, device=DEV).requires_grad_(True)
        res_tot = torch.zeros(probe.shape[0], device=DEV)
        for k in range(6):
            r1, r2, _ = pde_residual(model, probe, k)
            res_tot = res_tot + r1.detach() ** 2 + r2.detach() ** 2
        top = torch.topk(res_tot, args.rar_points).indices
        x_rar = torch.cat([x.detach(), probe[top].detach()], dim=0).requires_grad_(True)
        print(f"[{args.tag}] RAR added {args.rar_points} pts -> {x_rar.shape[0]}", flush=True)
        del probe, res_tot
        torch.cuda.empty_cache()
        refine(args.rar_epochs, 3e-5, x_rar, None, 0.0, "rar", mode="mse")
        rel65, rel_ref, perm = eval_all(model, data_dir)
        print(f"[{args.tag}] after RAR rel65 {rel65}", flush=True)
        print(f"[{args.tag}] after RAR relREF {rel_ref}", flush=True)

    # ---- stage 4: L-BFGS (fp64 unless --no-fp64) on an optional finer grid ----
    if not args.no_fp64:
        torch.set_default_dtype(torch.float64)
        model = model.double()
    x64_base = cell_grid(args.quad_final) if args.quad_final else x_rar.detach()
    x64 = (x64_base.detach() if args.no_fp64 else x64_base.detach().double()).requires_grad_(True)
    lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                           history_size=50, line_search_fn="strong_wolfe")
    best = {"loss": float("inf"), "state": None}
    for it in range(args.lbfgs):
        def closure():
            lb.zero_grad()
            loss = 0.
            if args.polish == "ritz":
                fields = []
                for k in range(6):
                    e_k, f_k = ritz_energy(model, x64, k)
                    loss = loss + 0.01 * e_k
                    fields.append(f_k)
                loss = loss + args.delta * deflation_hinge(fields, args.dmin)
            else:
                for k in range(6):
                    r1, r2, _ = pde_residual(model, x64, k)
                    loss = loss + torch.nanmean(r1 ** 2) + torch.nanmean(r2 ** 2)
            loss.backward()
            return loss
        l = lb.step(closure)
        if float(l) < best["loss"]:
            best = {"loss": float(l),
                    "state": copy.deepcopy({k_: v.detach().cpu() for k_, v in model.state_dict().items()})}
        if it % 10 == 0:
            print(f"[{args.tag}] lbfgs {it} loss {float(l):.6e} t {time.time()-t0:.0f}s", flush=True)
        if it % 30 == 0 and it:
            r65, rref, _ = eval_all(model, data_dir)
            print(f"[{args.tag}] lbfgs {it} rel65 {r65}", flush=True)
            print(f"[{args.tag}] lbfgs {it} relREF {rref}", flush=True)
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    model.eval()
    torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}.pt"))
    rel65, rel_ref, perm = eval_all(model, data_dir)
    E = energies(model)
    result = {"tag": args.tag, "args": vars(args), "rel_L2_vs_export65": rel65,
              "rel_L2_vs_reference513": rel_ref, "perm": perm,
              "energies": E, "wallclock_s": time.time() - t0}
    with open(os.path.join(args.outdir, f"{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{args.tag}] DONE rel65={rel65}", flush=True)
    print(f"[{args.tag}] DONE relREF={rel_ref}", flush=True)
    print(f"[{args.tag}] DONE energies={E}", flush=True)


if __name__ == "__main__":
    main()
