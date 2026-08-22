"""Headless 1D steady reaction-diffusion Deflation-PINN harness (HomPINNs benchmark).

0.01 u'' + 0.7 tanh(u) = f(x),  f manufactured from u* = sin^3(6x),  x in [-1,1],
Dirichlet BC u(+-1) = sin^3(-+6) enforced by hard constraint.
Faithful to tests/OneD_reaction/run.py with the activation-tuple bug fixed.
"""
import argparse, json, os, sys, time, copy
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import deepxde as dde  # noqa: F401,E402
from src.architectures.DeflationPINN import one_dim_DefPINN  # noqa: E402
from src.lossFunctions.OneD_reactionPINNLoss import pimlLoss_w_AutoGrad_reaction, sourceTerm  # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def deflation_loss(modelOut, dmin):
    out = modelOut["out"]
    K = len(out)
    zero = torch.tensor(0., device=out[0].device)
    loss, n_pairs = zero.clone(), 0
    for i in range(K):
        for j in range(i + 1, K):
            dist = torch.nanmean(torch.abs(out[i] - out[j]))
            loss = loss + torch.maximum(1. - dist / dmin, zero)
            n_pairs += 1
    return loss / n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100000)
    ap.add_argument("--numsol", type=int, default=11)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=1.)
    ap.add_argument("--delta", type=float, default=100.)
    ap.add_argument("--dmin", type=float, default=3.)
    ap.add_argument("--omega", type=float, default=6.)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="reaction1d")
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    pts = torch.linspace(-1, 1, 100)
    pts = torch.cat((pts, torch.linspace(-1, -0.95, 50), torch.linspace(0.95, 1, 50)))
    x = pts.view(-1, 1).to(DEV).requires_grad_(True)

    om = args.omega
    model = one_dim_DefPINN(
        numSolutions=args.numsol, numBranchFeatures=8, trunk_layer=5, trunk_width=60,
        activationFunction=torch.nn.Tanh(), DirichletHardConstraint=True,
        skipConnection=False, useSwiGLU=False, fourierFeatures=False,
        DirichletConstAt1=-1., DirichletConstAt2=1.,
        DirichletConstValLeft=float(np.sin(-om) ** 3),
        DirichletConstValRight=float(np.sin(om) ** 3)).to(DEV)
    print(f"[{args.tag}] device={DEV} pts={x.shape[0]} K={args.numsol}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6000, gamma=0.8)
    best = {"loss": float("inf"), "state": None}
    t0 = time.time()
    for ep in range(args.epochs):
        opt.zero_grad()
        out = model(x)
        pde = pimlLoss_w_AutoGrad_reaction(modelOut=out, x=x, boundaryPoints=None,
                                           modelOutBoundary=None, alpha=1., beta=0., omega=om)
        defl = deflation_loss(out, args.dmin)
        loss = args.alpha * pde + args.delta * defl
        if loss.item() < best["loss"]:
            best = {"loss": loss.item(),
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
        loss.backward()
        opt.step()
        sched.step()
        if ep % 1000 == 0:
            print(f"[{args.tag}] ep {ep} loss {loss.item():.6f} pde {pde.item():.3e} "
                  f"defl {defl.item():.4f} t {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(best["state"])
    model.eval()

    xe = torch.linspace(-1, 1, 1001).view(-1, 1).to(DEV)
    with torch.no_grad():
        oute = model(xe)
    xs = xe.cpu().numpy().reshape(-1)
    branches = [oute["out"][k].cpu().numpy().reshape(-1) for k in range(args.numsol)]
    u_star = np.sin(om * xs) ** 3

    rel_to_star = [float(np.sqrt(np.mean((b - u_star) ** 2) / np.mean(u_star ** 2))) for b in branches]

    # residual per branch on eval grid
    xr = xe.clone().requires_grad_(True)
    outr = model(xr)
    ones = torch.ones((xr.shape[0], 1), device=DEV)
    resid = []
    for k in range(args.numsol):
        u = outr["out"][k]
        g = torch.autograd.grad(u, xr, ones, create_graph=True)[0]
        uxx = torch.autograd.grad(g[:, 0].sum(), xr, create_graph=True)[0][:, 0]
        rr = 0.01 * uxx.view(-1, 1) + 0.7 * torch.tanh(u) - sourceTerm(xr.detach(), omega=om)
        resid.append(float(rr.abs().mean().detach().cpu()))

    pair_dist = {f"{i}-{j}": float(np.mean(np.abs(branches[i] - branches[j])))
                 for i in range(args.numsol) for j in range(i + 1, args.numsol)}

    result = {"tag": args.tag, "args": vars(args), "best_loss": best["loss"],
              "rel_L2_to_manufactured": rel_to_star,
              "residual_per_branch": resid, "pairwise_dist_meanabs": pair_dist,
              "wallclock_s": time.time() - t0}
    with open(os.path.join(args.outdir, f"{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=1)
    torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}.pt"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, b in enumerate(branches):
        ax.plot(xs, b, lw=1, label=f"branch {k}")
    ax.plot(xs, u_star, "k--", lw=2, label="$\\sin^3(6x)$")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"{args.tag}_branches.png"), dpi=120)
    print(f"[{args.tag}] DONE min rel to u*={min(rel_to_star):.4f} resid={['%.2e' % r for r in resid]}", flush=True)


if __name__ == "__main__":
    main()
