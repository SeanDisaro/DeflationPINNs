"""K-branch refinement to sub-1%: stable branches by deflated Deep-Ritz,
wall branches by residual least squares (their targets ARE saddles), finishing
with fp64 L-BFGS and a guarded all-MSE polish.

Branch modes come from the census JSON (Newton/LM certification of the
discovery run) -- an unsupervised source: it uses only the PDE.
Final acceptance guard per branch (branch nets are independent, so per-branch
revert is possible): accept the MSE-polished net iff its collocation residual
improved AND its energy stayed within a band of the census endpoint energy
(Lyapunov-style, unsupervised).
"""
import argparse, copy, json, os, sys, time
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import deepxde as dde  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from harness_ldg import HarmonicExt, build_model, SOL_NAMES  # noqa: E402
from harness_ldg_delta import BranchNet  # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RES = os.path.expanduser("~/DeflationPINNs_rev/results")
REF_DIR = os.path.join(RES, "ldg_reference")


class SmoothKState(nn.Module):
    def __init__(self, geom, K, width=128, depth=3, fourier=0, fsigma=16.0):
        super().__init__()
        self.K = K
        self.branches = nn.ModuleList(
            [BranchNet(width, depth, fourier, fsigma) for _ in range(K)])
        self.ext = HarmonicExt()
        self.geom = geom

    def fields(self, x, k):
        omega = self.geom.boundary_constraint_factor(x, smoothness="Cinf").view(-1, 1)
        m = self.branches[k](x)
        q1 = omega * m[:, 0].view(-1, 1) + self.ext(x)
        q2 = omega * m[:, 1].view(-1, 1)
        return q1, q2


def ritz_energy(model, x, k, eps=0.02, w=None):
    q1, q2 = model.fields(x, k)
    ones = torch.ones_like(q1)
    g1 = torch.autograd.grad(q1, x, ones, create_graph=True)[0]
    g2 = torch.autograd.grad(q2, x, ones, create_graph=True)[0]
    nq = (q1 ** 2 + q2 ** 2).view(-1)
    e = (g1 ** 2).sum(1) + (g2 ** 2).sum(1) + (1 / eps ** 2) * (nq - 1) ** 2
    val = (w * e).sum() if w is not None else torch.nanmean(e)
    return val, (q1, q2)


def mse_residual(model, x, k, eps=0.02, w=None):
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
    if w is not None:
        return (w * (r1 ** 2)).sum() + (w * (r2 ** 2)).sum(), (q1, q2)
    return torch.nanmean(r1 ** 2) + torch.nanmean(r2 ** 2), (q1, q2)


def deflation_hinge(fields, dmin, mode="hinge", p=2.0):
    """mode='hinge'   : max(1 - d/dmin, 0)          (the paper's deflation loss)
       mode='inverse' : (dmin/d)^p                  (scale-free repulsion, Farrell-like;
                        dmin is then only a units constant, not a threshold)"""
    K = len(fields)
    zero = torch.tensor(0., device=fields[0][0].device, dtype=fields[0][0].dtype)
    loss, np_ = zero.clone(), 0
    for i in range(K):
        for j in range(i + 1, K):
            d1 = fields[i][0] - fields[j][0]
            d2 = fields[i][1] - fields[j][1]
            dist = torch.sqrt(torch.nanmean(d1 ** 2 + d2 ** 2) + 1e-12)
            if mode == "inverse":
                loss = loss + (dmin / dist) ** p
            else:
                loss = loss + torch.maximum(1. - dist / dmin, zero)
            np_ += 1
    return loss / np_


def simpson_grid(n, dtype=torch.float32):
    """Nodal n x n grid (n odd) with tensorized Simpson weights (O(h^4))."""
    assert n % 2 == 1
    g = np.linspace(0, 1, n)
    w1 = np.ones(n); w1[1:-1:2] = 4.0; w1[2:-1:2] = 2.0
    w1 *= (1.0 / (n - 1)) / 3.0
    W = np.outer(w1, w1).ravel()
    Xg, Yg = np.meshgrid(g, g, indexing="xy")
    pts = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], 1), dtype=dtype, device=DEV)
    return pts, torch.tensor(W, dtype=dtype, device=DEV)


def cell_grid(n, dtype=torch.float32):
    c = (np.arange(n) + 0.5) / n
    Xc, Yc = np.meshgrid(c, c, indexing="xy")
    return torch.tensor(np.stack([Xc.ravel(), Yc.ravel()], 1), dtype=dtype, device=DEV)


def branch_metrics(model, x, k, w=None):
    """(rms residual, energy) of branch k on grid x -- unsupervised health."""
    r2, (q1, q2) = mse_residual(model, x, k, w=w)
    ones = torch.ones_like(q1)
    g1 = torch.autograd.grad(q1, x, ones, create_graph=False, retain_graph=True)[0]
    g2 = torch.autograd.grad(q2, x, ones, create_graph=False, retain_graph=True)[0]
    nq = (q1 ** 2 + q2 ** 2).view(-1)
    e = (g1 ** 2).sum(1) + (g2 ** 2).sum(1) + 2500. * (nq - 1) ** 2
    return float(torch.sqrt(r2).detach()), float(torch.nanmean(e).detach())


def load_refs(census, census_tag):
    """Union reference set: 6 stable (513 grid) + certified wall endpoints (257)."""
    refs = []
    n5 = 513
    for nm in SOL_NAMES:
        f = np.load(os.path.join(REF_DIR, f"{nm}_n{n5}.npz"))
        refs.append({"name": nm, "n": n5, "Q1": f["Q1"], "Q2": f["Q2"], "E": None})
    wi = 0
    seen = []
    for rec in census:
        if rec["certified"] and rec["state"] == "wall":
            f = np.load(os.path.join(RES, f"{census_tag}_endpoint_b{rec['branch']}.npz"))
            dup = False
            for s in seen:
                if np.sqrt(np.mean((f["Q1"] - s[0]) ** 2 + (f["Q2"] - s[1]) ** 2)) < 0.05:
                    dup = True
                    break
            if dup:
                continue
            wi += 1
            seen.append((f["Q1"], f["Q2"]))
            refs.append({"name": f"W{wi}", "n": 257, "Q1": f["Q1"], "Q2": f["Q2"], "E": rec["E"]})
    return refs


def eval_vs_refs(model, refs):
    grids = {}
    for n in {r["n"] for r in refs}:
        g = np.linspace(0, 1, n)
        Xg, Yg = np.meshgrid(g, g, indexing="xy")
        pts = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], 1),
                           dtype=next(model.parameters()).dtype, device=DEV)
        with torch.no_grad():
            out1, out2 = [], []
            for k in range(model.K):
                q1, q2 = model.fields(pts, k)
                out1.append(q1.cpu().numpy().reshape(n, n))
                out2.append(q2.cpu().numpy().reshape(n, n))
        grids[n] = (out1, out2)
    K = model.K
    cost = np.zeros((K, len(refs)))
    for k in range(K):
        for j, r in enumerate(refs):
            q1, q2 = grids[r["n"]][0][k], grids[r["n"]][1][k]
            num = np.mean((q1 - r["Q1"]) ** 2 + (q2 - r["Q2"]) ** 2)
            den = np.mean(r["Q1"] ** 2 + r["Q2"] ** 2)
            cost[k, j] = np.sqrt(num / den)
    rows, cols = linear_sum_assignment(cost)
    match = {refs[j]["name"]: {"branch": int(k), "rel": float(cost[k, j]),
                              "abs": float(cost[k, j] * np.sqrt(np.mean(refs[j]["Q1"] ** 2 + refs[j]["Q2"] ** 2)))}
             for k, j in zip(rows, cols)}
    return match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--census", required=True)
    ap.add_argument("--census-tag", default="census_K11")
    ap.add_argument("--numsol", type=int, default=11)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--fourier", type=int, default=128)
    ap.add_argument("--fsigma", type=float, default=16.0)
    ap.add_argument("--quad", type=int, default=192)
    ap.add_argument("--quad-final", type=int, default=384)
    ap.add_argument("--distill-epochs", type=int, default=4000)
    ap.add_argument("--adam-epochs", type=int, default=14000)
    ap.add_argument("--lbfgs", type=int, default=150)
    ap.add_argument("--final-mse-lbfgs", type=int, default=80)
    ap.add_argument("--dmin", type=float, default=0.4)
    ap.add_argument("--delta", type=float, default=100.)
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--no-fp64", action="store_true")  # consumer GPUs: fp64 is 1/64 rate
    ap.add_argument("--scratch", action="store_true")      # DDR from random init (no discovery stage)
    ap.add_argument("--dmin-final", type=float, default=None)  # ramp d_min over the Adam stage
    ap.add_argument("--repulsion", choices=["hinge", "inverse"], default="hinge")
    ap.add_argument("--rep-p", type=float, default=2.0)
    ap.add_argument("--quad-scheme", choices=["midpoint", "simpson"], default="midpoint")
    ap.add_argument("--tag", default="ldg_K11_refine")
    ap.add_argument("--outdir", default=RES)
    args = ap.parse_args()
    t0 = time.time()

    census = json.load(open(args.census)) if os.path.exists(args.census) else []
    K = args.numsol
    modes = {}
    e_target = {}
    if args.scratch:
        modes = {k: "ritz" for k in range(K)}
        e_target = {k: None for k in range(K)}
        census = []
    for rec in census:
        k = rec["branch"]
        if rec["certified"] and rec["state"] in SOL_NAMES:
            modes[k] = "ritz"
        else:
            modes[k] = "mse"  # wall or uncertified -> saddle candidate
        e_target[k] = rec["E"] if rec["certified"] else None
    print(f"[{args.tag}] modes={modes}", flush=True)

    geom = dde.geometry.geometry_2d.Rectangle([0., 0.], [1., 1.])
    model = SmoothKState(geom, K, args.width, args.depth, args.fourier, args.fsigma).to(DEV)
    src = None
    if args.scratch:
        print(f"[{args.tag}] SCRATCH: DDR from random initialization, no discovery stage", flush=True)
    elif args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=DEV))
        print(f"[{args.tag}] initialized from {args.init_ckpt}", flush=True)
    else:
        src = build_model(numsol=K)
        src.load_state_dict(torch.load(args.source, map_location=DEV))
        src.eval()
    refs = load_refs(census, args.census_tag)
    print(f"[{args.tag}] refs={[r['name'] for r in refs]}", flush=True)

    def src_eval(pts, chunk=4096):
        o1 = [[] for _ in range(K)]
        o2 = [[] for _ in range(K)]
        with torch.no_grad():
            for i in range(0, pts.shape[0], chunk):
                o = src(pts[i:i + chunk])
                for k in range(K):
                    o1[k].append(o["out1"][k])
                    o2[k].append(o["out2"][k])
        return [(torch.cat(a).detach().float(), torch.cat(b).detach().float())
                for a, b in zip(o1, o2)]

    # ---- distill ----
    if args.quad_scheme == "simpson":
        xq, wq = simpson_grid(args.quad if args.quad % 2 == 1 else args.quad + 1)
        xq = xq.requires_grad_(True)
    else:
        xq, wq = cell_grid(args.quad).requires_grad_(True), None
    targets = src_eval(xq.detach()) if (src is not None and args.distill_epochs > 0) else None
    if targets is None:
        args.distill_epochs = 0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(args.distill_epochs):
        opt.zero_grad()
        loss = 0.
        for k in range(K):
            q1, q2 = model.fields(xq, k)
            loss = loss + torch.mean((q1 - targets[k][0]) ** 2 + (q2 - targets[k][1]) ** 2)
        loss.backward()
        opt.step()
        if ep % 1000 == 0:
            print(f"[{args.tag}] distill ep {ep} loss {loss.item():.3e} t {time.time()-t0:.0f}s", flush=True)
    print(f"[{args.tag}] after distill {eval_vs_refs(model, refs)}", flush=True)

    def loss_and_backward(x, modes_map, w=None, dmin=None):
        """Accumulate gradients per branch so only ONE second-order graph is
        alive at a time (the joint closure OOMed at 384^2 fp64)."""
        total = 0.0
        for k in range(K):
            if modes_map[k] == "ritz":
                e_k, _ = ritz_energy(model, x, k, w=w)
                lk = 0.01 * e_k
            else:
                m_k, _ = mse_residual(model, x, k, w=w)
                lk = m_k
            lk.backward()
            total += float(lk.detach())
        fields = [model.fields(x, k) for k in range(K)]  # first-order graph only
        dl = args.delta * deflation_hinge(fields, dmin if dmin is not None else args.dmin,
                                          args.repulsion, args.rep_p)
        dl.backward()
        total += float(dl.detach())
        return total

    # ---- stage A: mixed Adam ----
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.6)
    for ep in range(args.adam_epochs):
        opt.zero_grad()
        if args.dmin_final is not None and args.adam_epochs > 1:
            frac = ep / (args.adam_epochs - 1)
            dmin_now = args.dmin + frac * (args.dmin_final - args.dmin)
        else:
            dmin_now = args.dmin
        loss_v = loss_and_backward(xq, modes, wq, dmin_now)
        opt.step()
        sch.step()
        if ep % 1000 == 0:
            print(f"[{args.tag}] adam ep {ep} loss {loss_v:.4e} dmin {dmin_now:.2f} "
                  f"t {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}_postadam.pt"))
    print(f"[{args.tag}] after adam {eval_vs_refs(model, refs)}", flush=True)

    # ---- stage B: mixed L-BFGS on finer quadrature (fp64 unless --no-fp64) ----
    if not args.no_fp64:
        torch.set_default_dtype(torch.float64)
        model = model.double()
    dt64 = torch.float32 if args.no_fp64 else torch.float64
    if args.quad_scheme == "simpson":
        nf = args.quad_final if args.quad_final % 2 == 1 else args.quad_final + 1
        x64, w64 = simpson_grid(nf, dt64)
        x64 = x64.requires_grad_(True)
    else:
        x64, w64 = cell_grid(args.quad_final, dt64).requires_grad_(True), None
    lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                           history_size=50, line_search_fn="strong_wolfe")
    for it in range(args.lbfgs):
        def closure():
            lb.zero_grad()
            v = loss_and_backward(x64, modes, w64, args.dmin_final or args.dmin)
            return torch.tensor(v, device=DEV)
        l = lb.step(closure)
        if it % 15 == 0:
            print(f"[{args.tag}] lbfgsB {it} loss {float(l):.6e} t {time.time()-t0:.0f}s", flush=True)
        if it % 45 == 0 and it:
            print(f"[{args.tag}] lbfgsB {it} {eval_vs_refs(model, refs)}", flush=True)
            torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}_Bpartial.pt"))
    torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}_postB.pt"))
    evB = eval_vs_refs(model, refs)
    print(f"[{args.tag}] after B {evB}", flush=True)

    # ---- stage C: all-MSE fp64 L-BFGS with per-branch guarded accept ----
    if args.final_mse_lbfgs > 0:
        before = {k: branch_metrics(model, x64, k, w64) for k in range(K)}
        state_before = {k: copy.deepcopy(model.branches[k].state_dict()) for k in range(K)}
        modes_c = {k: "mse" for k in range(K)}

        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                               history_size=50, line_search_fn="strong_wolfe")
        for it in range(args.final_mse_lbfgs):
            def closure():
                lb.zero_grad()
                v = loss_and_backward(x64, modes_c, w64, args.dmin_final or args.dmin)
                return torch.tensor(v, device=DEV)
            l = lb.step(closure)
            if it % 15 == 0:
                print(f"[{args.tag}] lbfgsC {it} loss {float(l):.6e} t {time.time()-t0:.0f}s", flush=True)
            if it % 40 == 0 and it:
                torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}_Cpartial.pt"))
                print(f"[{args.tag}] lbfgsC {it} {eval_vs_refs(model, refs)}", flush=True)
        after = {k: branch_metrics(model, x64, k, w64) for k in range(K)}
        for k in range(K):
            res_ok = after[k][0] < before[k][0]
            if e_target[k] is not None:
                band_ok = abs(after[k][1] - e_target[k]) <= abs(before[k][1] - e_target[k]) + 2.0
            else:
                band_ok = True
            if not (res_ok and band_ok):
                model.branches[k].load_state_dict(state_before[k])
                print(f"[{args.tag}] stage C: branch {k} REVERTED "
                      f"(res {before[k][0]:.3e}->{after[k][0]:.3e}, E {before[k][1]:.1f}->{after[k][1]:.1f})", flush=True)
            else:
                print(f"[{args.tag}] stage C: branch {k} accepted "
                      f"(res {before[k][0]:.3e}->{after[k][0]:.3e}, E {before[k][1]:.1f}->{after[k][1]:.1f})", flush=True)

    torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}.pt"))
    ev = eval_vs_refs(model, refs)
    mets = {k: branch_metrics(model, x64, k, w64) for k in range(K)}
    result = {"tag": args.tag, "modes": modes, "match": ev,
              "branch_rms_residual": {k: mets[k][0] for k in mets},
              "branch_energy": {k: mets[k][1] for k in mets},
              "wallclock_s": time.time() - t0}
    json.dump(result, open(os.path.join(args.outdir, f"{args.tag}.json"), "w"), indent=1)
    print(f"[{args.tag}] DONE {json.dumps(ev, indent=1)}", flush=True)
    print(f"[{args.tag}] energies {result['branch_energy']}", flush=True)
    print(f"[{args.tag}] residuals {result['branch_rms_residual']}", flush=True)


if __name__ == "__main__":
    main()
