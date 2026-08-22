"""Headless Allen-Cahn (unit disk) Deflation-PINN harness — second benchmark (R2.5).

Faithful to tests/AllenCahn/run.py:  -Delta u = 6u - u^3 (residual Delta u + 6u - u^3),
zero Dirichlet BC on the unit disk via hard constraint u = N(x) * (1 - r^2).
lambda = 6 > lambda_1(disk) ~ 5.783  =>  exactly three solutions: 0, +u*, -u*.
Reference u* computed with scipy.solve_bvp on the radial ODE.
"""
import argparse, json, os, sys, time, copy
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import deepxde as dde  # noqa: F401,E402
from src.architectures.DeflationPINN import two_dim_2_one_dim_DefPINN  # noqa: E402
from src.lossFunctions.AllenCahnPINNLoss import pimlLoss_w_AutoGrad_allenCahn  # noqa: E402
from src.starDomainExtrapolation.starDomain import Sphere  # noqa: E402
from scipy.integrate import solve_bvp  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def squaredRadial(x):
    return (1 - x * x).view(-1, 1)


def zeroOnBoundaryExtension(star, inp):
    radius, angles = star.getSphericalCoordinates(inp)
    return squaredRadial(radius / star.radiusDomainFunciton(angles)).view(-1, 1)


def disk_points(n, R=1.0):
    coords = torch.linspace(-R, R, n)
    xx, yy = torch.meshgrid(coords, coords, indexing="xy")
    r2 = xx ** 2 + yy ** 2
    mask = (r2 < R ** 2) & (r2 > 1e-8)
    return torch.stack([xx[mask], yy[mask]], dim=-1)


def radial_reference():
    """Positive radial solution of u'' + u'/r + 6u - u^3 = 0, u'(0)=0, u(1)=0.

    Shooting: integrate the IVP from r0 ~ 0 with u(r0)=a, u'(r0)=0 and find the
    amplitude a>0 with u(1)=0 by bisection.  (solve_bvp collapses onto the
    trivial branch here, so shooting is the robust choice.)  For small a,
    u(1) ~ a J_0(sqrt(6)) < 0 since sqrt(6) > j_{0,1}; for large a the cubic
    dominates and u(1) > 0, so a sign change brackets the ground state.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    r0 = 1e-8

    def shoot(a, dense=False):
        def rhs(r, y):
            return [y[1], -y[1] / r - 6 * y[0] + y[0] ** 3]
        sol = solve_ivp(rhs, [r0, 1.0], [a, 0.0], rtol=1e-11, atol=1e-12,
                        dense_output=dense, method="RK45")
        return sol if dense else sol.y[0][-1]

    grid = np.linspace(0.05, 6.0, 60)
    vals = [shoot(a) for a in grid]
    bracket = None
    for i in range(len(grid) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]):
            bracket = (grid[i], grid[i + 1])
            break
    if bracket is None:
        return None
    a_star = brentq(lambda a: shoot(a), *bracket, xtol=1e-13)
    return a_star, shoot(a_star, dense=True)


def pde_loss_ac(model, x, mse):
    """Sum over branches of mean(residual^2) (mse) or mean(residual^8) (repo)."""
    out = model(x)
    dt = x.dtype
    pde = torch.tensor(0., device=x.device, dtype=dt)
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=dt)
    for k in range(len(out["out"])):
        u = out["out"][k]
        g = torch.autograd.grad(u, x, ones, create_graph=True)[0]
        uxx = torch.autograd.grad(g[:, 0].view(-1, 1), x, ones, create_graph=True)[0][:, 0]
        uyy = torch.autograd.grad(g[:, 1].view(-1, 1), x, ones, create_graph=True)[0][:, 1]
        r = (uxx + uyy).view(-1, 1) + 6 * u - u ** 3
        pde = pde + (torch.mean(r ** 2) if mse else torch.mean(r ** 8))
    return out, pde


def deflation_loss(modelOut, dmin):
    out = modelOut["out"]
    K = len(out)
    zero = torch.tensor(0., device=out[0].device)
    loss, n_pairs = zero.clone(), 0
    for i in range(K):
        for j in range(i + 1, K):
            dist = torch.nanmean(torch.abs(out[i] - out[j]))  # code-faithful metric
            loss = loss + torch.maximum(1. - dist / dmin, zero)
            n_pairs += 1
    return loss / n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40000)
    ap.add_argument("--n", type=int, default=33)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=100.)
    ap.add_argument("--delta", type=float, default=1.)
    ap.add_argument("--dmin", type=float, default=0.4)
    ap.add_argument("--mse", action="store_true")
    ap.add_argument("--lbfgs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="allencahn")
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    x = disk_points(args.n).to(DEV).requires_grad_(True)
    star = Sphere(dim=2, center=torch.tensor([0, 0]), radius=torch.tensor([1.]))
    star.updateDevice(str(DEV))
    model = two_dim_2_one_dim_DefPINN(
        numSolutions=3, numBranchFeatures=32, trunk_layer=6, trunk_width=100,
        activationFunction=torch.nn.Tanh(), geom=None, DirichletHardConstraint=True,
        skipConnection=False,
        r_function=lambda p: zeroOnBoundaryExtension(star, [p[:, 0].view(-1, 1), p[:, 1].view(-1, 1)]),
        DirichletConditionFunc=None).to(DEV)
    print(f"[{args.tag}] device={DEV} pts={x.shape[0]}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.8)
    best = {"loss": float("inf"), "state": None}
    t0 = time.time()
    for ep in range(args.epochs):
        opt.zero_grad()
        out, pde = pde_loss_ac(model, x, args.mse)
        defl = deflation_loss(out, args.dmin)
        loss = args.alpha * pde + args.delta * defl
        if loss.item() < best["loss"]:
            best = {"loss": loss.item(),
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
        loss.backward()
        opt.step()
        sched.step()
        if ep % 500 == 0:
            print(f"[{args.tag}] ep {ep} loss {loss.item():.6f} pde {pde.item():.3e} "
                  f"defl {defl.item():.4f} t {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best["state"])

    if args.lbfgs > 0:
        torch.set_default_dtype(torch.float64)
        model = model.double()
        star.updateDevice(str(DEV))
        x = x.detach().double().requires_grad_(True)
        print(f"[{args.tag}] fp64 L-BFGS polish {args.lbfgs} outer steps", flush=True)
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                               history_size=50, line_search_fn="strong_wolfe")
        for it in range(args.lbfgs):
            def closure():
                lb.zero_grad()
                out_c, pde_c = pde_loss_ac(model, x, args.mse)
                l = args.alpha * pde_c + args.delta * deflation_loss(out_c, args.dmin)
                l.backward()
                return l
            l = lb.step(closure)
            if it % 10 == 0:
                print(f"[{args.tag}] lbfgs {it} loss {float(l):.6e}", flush=True)

    model.eval()
    # save the trained model FIRST so a failure in evaluation cannot lose it
    torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}.pt"))

    # ---- reference and branch matching ----
    ref = radial_reference()
    x_np = x.detach().cpu().numpy()
    r_np = np.sqrt((x_np ** 2).sum(1))
    with torch.no_grad():
        out = model(x.detach())
    branches = [out["out"][k].cpu().numpy().reshape(-1) for k in range(3)]

    match, ref_amp, ref_norm = {}, None, None
    if ref is not None:
        ref_amp, ref_sol = ref
        u_ref = ref_sol.sol(np.clip(r_np, 1e-8, 1.0))[0]
        refs = {"zero": np.zeros_like(r_np), "plus": u_ref, "minus": -u_ref}
        names = list(refs.keys())
        cost = np.zeros((3, 3))
        for k in range(3):
            for j, nm in enumerate(names):
                cost[k, j] = np.sqrt(np.nanmean((branches[k] - refs[nm]) ** 2))
        rows, cols = linear_sum_assignment(cost)
        ref_norm = float(np.sqrt(np.nanmean(u_ref ** 2)))
        for k, j in zip(rows, cols):
            nm = names[j]
            absL2 = float(cost[k, j])
            match[nm] = {"branch": int(k), "abs_L2": absL2,
                         "rel_L2": (absL2 / ref_norm) if nm != "zero" else None}

    # residual per branch on a finer disk grid
    xf = disk_points(65).to(DEV).to(next(model.parameters()).dtype).requires_grad_(True)
    outf = model(xf)
    ones = torch.ones((xf.shape[0], 1), device=DEV)
    resid = []
    for k in range(3):
        u = outf["out"][k]
        g = torch.autograd.grad(u, xf, ones, create_graph=True)[0]
        uxx = torch.autograd.grad(g[:, 0].sum(), xf, create_graph=True)[0][:, 0]
        uyy = torch.autograd.grad(g[:, 1].sum(), xf, create_graph=True)[0][:, 1]
        rr = (uxx + uyy).view(-1, 1) + 6 * u - u ** 3
        resid.append(float(rr.abs().mean().detach().cpu()))

    # pairwise distinctness in RMS metric
    pair = {f"{i}-{j}": float(np.sqrt(np.nanmean((branches[i] - branches[j]) ** 2)))
            for i in range(3) for j in range(i + 1, 3)}
    result = {"tag": args.tag, "args": vars(args), "best_loss": best["loss"],
              "ref_amplitude_at_0": ref_amp, "ref_L2_norm": ref_norm, "match": match,
              "pairwise_rms_dist": pair,
              "residual_per_branch": resid, "wallclock_s": time.time() - t0}
    with open(os.path.join(args.outdir, f"{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for k in range(3):
        sc = ax[k].tricontourf(x_np[:, 0], x_np[:, 1], branches[k], levels=30)
        fig.colorbar(sc, ax=ax[k])
        ax[k].set_title(f"branch {k}")
        ax[k].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"{args.tag}_solutions.png"), dpi=120)
    print(f"[{args.tag}] DONE match={match} resid={resid}", flush=True)


if __name__ == "__main__":
    main()
