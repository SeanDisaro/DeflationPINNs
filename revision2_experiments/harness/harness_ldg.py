"""Headless LDG Deflation-PINN training + evaluation harness (revision experiments).

Reuses the repo's model/losses verbatim; only the training loop and the
deflation-distance variant live here.  Run from the repo root.

Deflation variants (all use hinge max(1 - d_ij/dmin, 0), averaged over pairs):
  legacy : d_ij = mean|G2_i - G2_j|            (exact published-code metric)
  second : d_ij = ||G2_i - G2_j||_L2           (paper formula, 2nd component only)
  full   : d_ij = ||G_i - G_j||_L2(R^2)        (full vector-field norm)
"""
import argparse, json, os, sys, time, copy
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import deepxde as dde  # noqa: E402  (DDE_BACKEND=pytorch must be exported)
from src.architectures.DeflationPINN import two_dim_2_two_dim_DefPINN  # noqa: E402
from src.lossFunctions.LDGPINNLoss import pimlLoss_w_AutoGrad_LDG, bfunc, DCBoundaryExtension  # noqa: E402
from src.starDomainExtrapolation.starDomain import HyperCuboid  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SOL_NAMES = ["D1", "D2", "R1", "R2", "R3", "R4"]


class HarmonicExt(torch.nn.Module):
    """Closed-form harmonic extension of the trapezoidal boundary data Q_b^1.

    Port of the repo's src/harmonicTrapezoidalExtension.py (Fourier series with
    stable sinh ratios), with the coefficient tensors registered as buffers so
    .to(device)/.double() work, and a (N,2) input signature.
    """

    def __init__(self, d=0.06, num_terms=61):
        super().__init__()
        odds = np.arange(1, num_terms + 1, 2)
        n = torch.tensor(odds, dtype=torch.float64).unsqueeze(0)
        n_pi = n * np.pi
        self.register_buffer("n_pi", n_pi)
        self.register_buffer("c_n", (4.0 * torch.sin(n_pi * d)) / (d * n_pi ** 2))

    def _sinh_ratio(self, z):
        """sinh(n*pi*(1-z)) / sinh(n*pi), computed stably."""
        n_pi = self.n_pi
        return torch.exp(-n_pi * z) * (1.0 - torch.exp(-2.0 * n_pi * (1.0 - z))) \
            / (1.0 - torch.exp(-2.0 * n_pi) + 1e-300)

    def forward(self, inp):
        x = inp[:, 0].view(-1, 1).to(self.n_pi.dtype)
        y = inp[:, 1].view(-1, 1).to(self.n_pi.dtype)
        f_tb = torch.sum(self.c_n * torch.sin(x @ self.n_pi) *
                         (self._sinh_ratio(y) + self._sinh_ratio(1.0 - y)), dim=1, keepdim=True)
        f_lr = -torch.sum(self.c_n * torch.sin(y @ self.n_pi) *
                          (self._sinh_ratio(x) + self._sinh_ratio(1.0 - x)), dim=1, keepdim=True)
        # emit the caller's dtype: returning fp64 here silently promotes the
        # whole downstream autograd graph to double and ~3x-es the epoch time
        return (f_tb + f_lr).view(-1, 1).to(inp.dtype)


def build_points(n, corner_refine=False, safety=0.001):
    xs = torch.linspace(0 + safety, 1 - safety, n)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    if corner_refine:
        # extra collocation near the four corners (vortex cores) and edges (ramp)
        loc = torch.linspace(0 + safety, 0.12, 15)
        py, px = torch.meshgrid(loc, loc, indexing="ij")
        patch = torch.stack([px, py], dim=-1).reshape(-1, 2)
        patches = [patch,
                   torch.stack([1 - patch[:, 0], patch[:, 1]], dim=-1),
                   torch.stack([patch[:, 0], 1 - patch[:, 1]], dim=-1),
                   torch.stack([1 - patch[:, 0], 1 - patch[:, 1]], dim=-1)]
        # edge bands to resolve the trapezoidal ramp (d = 3*eps = 0.06)
        band = torch.linspace(0 + safety, 1 - safety, 60)
        depth = torch.linspace(0 + safety, 0.08, 8)
        by, bx = torch.meshgrid(depth, band, indexing="ij")
        bot = torch.stack([bx, by], dim=-1).reshape(-1, 2)
        top = torch.stack([bot[:, 0], 1 - bot[:, 1]], dim=-1)
        lef = torch.stack([bot[:, 1], bot[:, 0]], dim=-1)
        rig = torch.stack([1 - bot[:, 1], bot[:, 0]], dim=-1)
        pts = torch.cat([pts] + patches + [bot, top, lef, rig], dim=0)
    return pts


def build_model(p=16, layers=1, width=4000, harmonic=False, numsol=6):
    geom = dde.geometry.geometry_2d.Rectangle([0., 0.], [1., 1.])
    if harmonic:
        ext = HarmonicExt().to(DEV)
        func1 = lambda inp: ext(inp)  # noqa: E731
    else:
        star = HyperCuboid(2, torch.tensor([0.5, 0.5]), torch.tensor([1., 1.]))
        star.updateDevice(str(DEV))
        func1 = lambda inp: DCBoundaryExtension(  # noqa: E731
            star, [inp[:, 0].view(-1, 1), inp[:, 1].view(-1, 1)], bfunc)
    model = two_dim_2_two_dim_DefPINN(
        numSolutions=numsol, numBranchFeatures=p, trunk_layer=layers, trunk_width=width,
        activationFunction=torch.nn.Tanh(), geom=geom, DirichletHardConstraint=True,
        skipConnection=False,
        DirichletConditionFunc1=func1,
        DirichletConditionFunc2=lambda x: torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype),
    )
    return model.to(DEV)


def deflation_loss(modelOut, variant, dmin):
    o1, o2 = modelOut["out1"], modelOut["out2"]
    K = len(o1)
    zero = torch.tensor(0., device=o1[0].device)
    loss, n_pairs = zero.clone(), 0
    for i in range(K):
        for j in range(i + 1, K):
            d1 = o1[i] - o1[j]
            d2 = o2[i] - o2[j]
            if variant == "legacy":
                dist = torch.nanmean(torch.abs(d2))
            elif variant == "second":
                dist = torch.sqrt(torch.nanmean(d2 ** 2))
            elif variant == "full":
                dist = torch.sqrt(torch.nanmean(d1 ** 2 + d2 ** 2))
            else:
                raise ValueError(variant)
            loss = loss + torch.maximum(1. - dist / dmin, zero)
            n_pairs += 1
    return loss / n_pairs


def feature_deflation(model, dmin):
    """Deflation in feature space: hinge on ||beta_i - beta_j|| / sqrt(dim),
    i.e. on the branch vectors (the network's per-solution representation)
    instead of the output fields."""
    B = list(model.branchFeatures)
    K = len(B)
    zero = torch.tensor(0., device=B[0].device, dtype=B[0].dtype)
    loss, n = zero.clone(), 0
    for i in range(K):
        for j in range(i + 1, K):
            dist = torch.norm(B[i] - B[j]) / (B[i].numel() ** 0.5)
            loss = loss + torch.maximum(1. - dist / dmin, zero)
            n += 1
    return loss / n


def total_loss(model, x, args, eps=0.02):
    """Reference (per-branch loop) loss; |r| or r^2 residual aggregation."""
    out = model(x)
    if not args.mse:
        pde = pimlLoss_w_AutoGrad_LDG(modelOut=out, x=x, boundaryPoints=None,
                                      modelOutBoundary=None, eps=eps, alpha=1., beta=0.)
    else:
        pde = torch.tensor(0., device=x.device, dtype=x.dtype)
        ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        for k in range(len(out["out1"])):
            q1, q2 = out["out1"][k], out["out2"][k]
            g1 = torch.autograd.grad(q1.view(-1, 1), x, ones, create_graph=True)[0]
            g2 = torch.autograd.grad(q2.view(-1, 1), x, ones, create_graph=True)[0]
            q1xx = torch.autograd.grad(g1[:, 0].view(-1, 1), x, ones, create_graph=True)[0][:, 0]
            q1yy = torch.autograd.grad(g1[:, 1].view(-1, 1), x, ones, create_graph=True)[0][:, 1]
            q2xx = torch.autograd.grad(g2[:, 0].view(-1, 1), x, ones, create_graph=True)[0][:, 0]
            q2yy = torch.autograd.grad(g2[:, 1].view(-1, 1), x, ones, create_graph=True)[0][:, 1]
            nq = q1 ** 2 + q2 ** 2
            r1 = (2 * (1 - nq) * q1).view(-1) + eps ** 2 * (q1xx + q1yy)
            r2 = (2 * (1 - nq) * q2).view(-1) + eps ** 2 * (q2xx + q2yy)
            pde = pde + torch.nanmean(r1 ** 2) + torch.nanmean(r2 ** 2)
    if args.deflation == "feature":
        defl = feature_deflation(model, args.dmin)
    else:
        defl = deflation_loss(out, args.deflation, args.dmin)
    return args.alpha * pde + args.delta * defl, pde, defl


class FastLdG:
    """Value-identical reimplementation of forward+PDE loss.

    Replicates the collocation set once per branch (x_big has 6N rows, block k
    holding branch k), so ONE trunk pass, ONE omega, ONE boundary extension and
    6 autograd.grad calls replace the per-branch python loops of the repo code.
    """

    def __init__(self, model, x, K=None):
        K = K or model.numSolutions
        self.m, self.K, self.N = model, K, x.shape[0]
        self.x_big = x.detach().repeat(K, 1).requires_grad_(True)

    def forward_big(self):
        m, K, N = self.m, self.K, self.N
        t = self.x_big
        out = torch.zeros_like(t) + t
        for i in range(m.trunk_layer):
            out = m.activationFunction(m.trunkNet_Lin[i](out))
        out = m.trunkNet_Lin[-1](out)                       # (KN, 2p)
        B = torch.stack(list(m.branchFeatures), dim=0)      # (K, 2p)
        Brows = torch.repeat_interleave(B, N, dim=0)        # (KN, 2p)
        prod = out * Brows
        p = m.numBranchFeatures
        q1 = prod[:, :p].sum(1) + m.deepONet_biases[0]
        q2 = prod[:, p:2 * p].sum(1) + m.deepONet_biases[1]
        omega = m.geom.boundary_constraint_factor(t, smoothness="Cinf").view(-1)
        g1 = q1 * omega + m.DirichletConditionFunc1(t).view(-1)
        g2 = q2 * omega + m.DirichletConditionFunc2(t).view(-1)
        return g1.view(-1, 1), g2.view(-1, 1)

    def out_dict(self, g1, g2):
        N = self.N
        return {"out1": [g1[k * N:(k + 1) * N] for k in range(self.K)],
                "out2": [g2[k * N:(k + 1) * N] for k in range(self.K)]}

    def loss(self, args, eps=0.02):
        g1, g2 = self.forward_big()
        t = self.x_big
        ones = torch.ones_like(g1)
        d1 = torch.autograd.grad(g1, t, ones, create_graph=True)[0]
        d2 = torch.autograd.grad(g2, t, ones, create_graph=True)[0]
        g1xx = torch.autograd.grad(d1[:, 0].sum(), t, create_graph=True)[0][:, 0]
        g1yy = torch.autograd.grad(d1[:, 1].sum(), t, create_graph=True)[0][:, 1]
        g2xx = torch.autograd.grad(d2[:, 0].sum(), t, create_graph=True)[0][:, 0]
        g2yy = torch.autograd.grad(d2[:, 1].sum(), t, create_graph=True)[0][:, 1]
        nq = g1 ** 2 + g2 ** 2
        r1 = ((2 * (1 - nq) * g1).view(-1) + eps ** 2 * (g1xx + g1yy))
        r2 = ((2 * (1 - nq) * g2).view(-1) + eps ** 2 * (g2xx + g2yy))
        if args.mse:
            pde = self.K * (torch.nanmean(r1 ** 2) + torch.nanmean(r2 ** 2))
        else:
            pde = self.K * (torch.nanmean(r1.abs()) + torch.nanmean(r2.abs()))
        if args.deflation == "feature":
            defl = feature_deflation(self.m, args.dmin)
        else:
            defl = deflation_loss(self.out_dict(g1, g2), args.deflation, args.dmin)
        return args.alpha * pde + args.delta * defl, pde, defl


@torch.no_grad()
def eval_vs_fem(model, data_dir):
    dt = next(model.parameters()).dtype
    mats = {nm: loadmat(os.path.join(data_dir, f"data_LDG_{nm}_solution.mat")) for nm in SOL_NAMES}
    xy = mats["D1"]["c4n"].astype(np.float64)
    x_eval = torch.tensor(xy, dtype=dt, device=DEV)
    with torch.no_grad():
        parts = [model(x_eval[i:i + 2048]) for i in range(0, x_eval.shape[0], 2048)]
    out = {key: [torch.cat([p[key][k] for p in parts]) for k in range(len(parts[0][key]))]
           for key in ("out1", "out2")}
    K = len(out["out1"])
    nn_q11 = [out["out1"][k].cpu().numpy().reshape(-1) for k in range(K)]
    nn_q12 = [out["out2"][k].cpu().numpy().reshape(-1) for k in range(K)]
    fem_q11 = {nm: mats[nm]["p1"].reshape(-1) for nm in SOL_NAMES}
    fem_q12 = {nm: mats[nm]["q1"].reshape(-1) for nm in SOL_NAMES}

    cost = np.zeros((K, 6))
    for k in range(K):
        for j, nm in enumerate(SOL_NAMES):
            num = np.nanmean((nn_q11[k] - fem_q11[nm]) ** 2 + (nn_q12[k] - fem_q12[nm]) ** 2)
            den = np.nanmean(fem_q11[nm] ** 2 + fem_q12[nm] ** 2)
            cost[k, j] = np.sqrt(num / den)
    rows, cols = linear_sum_assignment(cost)
    rel = {SOL_NAMES[j]: float(cost[k, j]) for k, j in zip(rows, cols)}
    perm = {SOL_NAMES[j]: int(k) for k, j in zip(rows, cols)}

    # pairwise NN distances (distinctness) in the three metrics
    pair = {}
    for i in range(K):
        for j in range(i + 1, K):
            d1 = nn_q11[i] - nn_q11[j]
            d2 = nn_q12[i] - nn_q12[j]
            pair[f"{i}-{j}"] = {
                "legacy": float(np.nanmean(np.abs(d2))),
                "second": float(np.sqrt(np.nanmean(d2 ** 2))),
                "full": float(np.sqrt(np.nanmean(d1 ** 2 + d2 ** 2))),
            }
    return rel, perm, pair, (nn_q11, nn_q12, xy)


def eval_residual_energy(model, n=129, eps=0.02, chunk=1024):
    """Mean |PDE residual| and energy E(Q) per branch on an n x n interior grid.

    Chunked: second-order autograd over all points at once needs tens of GB for
    a wide trunk; we accumulate per-chunk sums instead.
    """
    dt = next(model.parameters()).dtype
    xs = torch.linspace(0.001, 0.999, n, dtype=dt)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    pts_all = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    K = model.numSolutions
    res_sum = [0.0] * K
    ene_sum = [0.0] * K
    cnt = 0
    for i in range(0, pts_all.shape[0], chunk):
        x = pts_all[i:i + chunk].to(DEV).requires_grad_(True)
        m = x.shape[0]
        out = model(x)
        ones = torch.ones((m, 1), device=DEV, dtype=dt)
        for k in range(K):
            q1, q2 = out["out1"][k], out["out2"][k]
            g1 = torch.autograd.grad(q1, x, ones, create_graph=True)[0]
            g2 = torch.autograd.grad(q2, x, ones, create_graph=True)[0]
            q1xx = torch.autograd.grad(g1[:, 0].sum(), x, create_graph=True)[0][:, 0]
            q1yy = torch.autograd.grad(g1[:, 1].sum(), x, create_graph=True)[0][:, 1]
            q2xx = torch.autograd.grad(g2[:, 0].sum(), x, create_graph=True)[0][:, 0]
            q2yy = torch.autograd.grad(g2[:, 1].sum(), x, create_graph=True)[0][:, 1]
            nq = q1 ** 2 + q2 ** 2
            r1 = (0.02 ** 2) * (q1xx + q1yy) + (2 * (1 - nq) * q1).view(-1)
            r2 = (0.02 ** 2) * (q2xx + q2yy) + (2 * (1 - nq) * q2).view(-1)
            res_sum[k] += float((r1.abs().sum() + r2.abs().sum()).detach().cpu())
            e = (g1 ** 2).sum(1) + (g2 ** 2).sum(1) + (1 / eps ** 2) * ((nq.view(-1) - 1) ** 2)
            ene_sum[k] += float(e.sum().detach().cpu())
        cnt += m
        del out
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
    return [r / cnt for r in res_sum], [e / cnt for e in ene_sum]


def quiver_plot(nn_q11, nn_q12, xy, perm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = len(nn_q11)
    rows = (K + 2) // 3
    fig, ax = plt.subplots(rows, 3, figsize=(15, 5 * rows), squeeze=False)
    inv = {v: nm for nm, v in perm.items()}
    for k in range(K):
        a = ax[k // 3][k % 3]
        a.quiver(xy[:, 0], xy[:, 1], nn_q11[k], nn_q12[k],
                 angles="xy", scale_units="xy", scale=20, width=0.0015)
        a.set_title(f"branch {k} -> {inv.get(k, 'extra')}")
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deflation", choices=["legacy", "second", "full", "feature"], required=True)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--grid", type=int, default=33)
    ap.add_argument("--corner-refine", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=100.)
    ap.add_argument("--dmin", type=float, default=0.4)
    ap.add_argument("--sched", type=str, default="none")  # "step,5000,0.8"
    ap.add_argument("--lbfgs", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--fp64-polish", action="store_true")  # Adam in fp32, L-BFGS in fp64
    ap.add_argument("--mse", action="store_true")
    ap.add_argument("--harmonic-ext", action="store_true")
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--width", type=int, default=4000)
    ap.add_argument("--eps-stages", type=str, default="")  # "0.04:6000,0.028:6000,0.02:10000"
    ap.add_argument("--numsol", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--eval-only", type=str, default="")  # path to a .pt state dict
    ap.add_argument("--warm-start", type=str, default="")  # state dict to start from
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    out_json = os.path.join(args.outdir, f"{args.tag}.json")
    data_dir = os.path.join(os.getcwd(), "data", "trueSolution")

    x = build_points(args.grid, args.corner_refine)
    model = build_model(p=args.p, layers=args.layers, width=args.width,
                        harmonic=args.harmonic_ext, numsol=args.numsol)
    if args.fp64:
        torch.set_default_dtype(torch.float64)
        model = model.double()
        x = x.double()
    x = x.to(DEV).requires_grad_(True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] device={DEV} pts={x.shape[0]} params={n_par} "
          f"dtype={next(model.parameters()).dtype}", flush=True)

    if args.warm_start:
        model.load_state_dict(torch.load(args.warm_start, map_location=DEV))
        print(f"[{args.tag}] warm-started from {args.warm_start}", flush=True)

    # epsilon-continuation schedule: list of (eps, epochs); default single stage
    if args.eps_stages:
        stages = [(float(s.split(":")[0]), int(s.split(":")[1]))
                  for s in args.eps_stages.split(",")]
    else:
        stages = [(0.02, args.epochs)]
    final_eps = stages[-1][0]

    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=DEV))
        model.eval()
        rel, perm, pair, (q11, q12, xy) = eval_vs_fem(model, data_dir)
        res, ene = eval_residual_energy(model)
        result = {"tag": args.tag, "checkpoint": args.eval_only, "rel_L2": rel,
                  "perm": perm, "pairwise_nn_dist": pair,
                  "residual_per_branch": res, "energy_per_branch": ene}
        with open(out_json, "w") as f:
            json.dump(result, f, indent=1)
        print(f"[{args.tag}] EVAL-ONLY relL2={rel} resid={res} energy={ene}", flush=True)
        return

    fast = FastLdG(model, x) if args.fast else None
    if fast is not None:  # verify value-identical losses before trusting fast mode
        # run the check on a small subsample so the two loss graphs never
        # coexist at full size (that combination OOMed on a shared GPU)
        x_chk = x[:1089].detach().clone().requires_grad_(True)
        ls, _, _ = total_loss(model, x_chk, args, eps=final_eps)
        slow_val = ls.item()
        del ls
        torch.cuda.empty_cache()
        chk = FastLdG(model, x_chk)
        lf, _, _ = chk.loss(args, eps=final_eps)
        rel = abs(slow_val - lf.item()) / max(abs(slow_val), 1e-12)
        print(f"[{args.tag}] fast-check slow={slow_val:.8f} fast={lf.item():.8f} rel={rel:.2e}", flush=True)
        assert rel < 1e-4, "fast mode does not reproduce reference loss"
        del lf, chk, x_chk
        torch.cuda.empty_cache()

    def loss_fn(eps):
        return fast.loss(args, eps=eps) if fast is not None else total_loss(model, x, args, eps=eps)

    best = {"loss": float("inf"), "state": None}
    hist = []
    t0 = time.time()
    ep_global = 0
    for eps_now, n_ep in stages:
        is_final = (eps_now == final_eps)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = None
        if args.sched.startswith("step"):
            _, size, gamma = args.sched.split(",")
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=int(size), gamma=float(gamma))
        print(f"[{args.tag}] === stage eps={eps_now} epochs={n_ep} final={is_final} ===", flush=True)
        for ep in range(n_ep):
            opt.zero_grad()
            loss, pde, defl = loss_fn(eps_now)
            # best-checkpoint only meaningful at the target epsilon
            if is_final and loss.item() < best["loss"]:
                best = {"loss": loss.item(),
                        "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            if ep % 200 == 0:
                print(f"[{args.tag}] eps {eps_now} ep {ep} loss {loss.item():.6e} pde {pde.item():.6e} "
                      f"defl {defl.item():.5f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"t {time.time()-t0:.0f}s", flush=True)
            if is_final and args.eval_every and ep and ep % args.eval_every == 0:
                model.eval()
                rel, perm, _, _ = eval_vs_fem(model, data_dir)
                model.train()
                hist.append({"epoch": ep_global + ep, "loss": loss.item(), "rel": rel})
                print(f"[{args.tag}] ep {ep_global+ep} relL2 {rel}", flush=True)
        ep_global += n_ep

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    if args.lbfgs > 0 and args.fp64_polish and not args.fp64:
        # cast to double for the polish phase; rebuild the fast-path caches
        torch.set_default_dtype(torch.float64)
        model = model.double()
        x = x.detach().double().requires_grad_(True)
        if fast is not None:
            fast = FastLdG(model, x)
        print(f"[{args.tag}] switched to float64 for polish", flush=True)

        def loss_fn(eps):  # noqa: F811 - rebind over the fp32 closure
            return fast.loss(args, eps=eps) if fast is not None else total_loss(model, x, args, eps=eps)
        best = {"loss": float("inf"), "state": None}

    if args.lbfgs > 0:
        print(f"[{args.tag}] L-BFGS polish {args.lbfgs} outer steps at eps={final_eps}", flush=True)
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                               history_size=50, line_search_fn="strong_wolfe")
        for it in range(args.lbfgs):
            def closure():
                lb.zero_grad()
                l, _, _ = loss_fn(final_eps)
                l.backward()
                return l
            l = lb.step(closure)
            if it % 10 == 0:
                print(f"[{args.tag}] lbfgs {it} loss {float(l):.6e} t {time.time()-t0:.0f}s", flush=True)
            if float(l) < best["loss"]:
                best = {"loss": float(l),
                        "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            if it % 25 == 0 and it:
                model.eval()
                rel, _, _, _ = eval_vs_fem(model, data_dir)
                model.train()
                print(f"[{args.tag}] lbfgs {it} relL2 {rel}", flush=True)
        model.load_state_dict(best["state"])

    model.eval()
    torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}.pt"))
    print(f"[{args.tag}] checkpoint saved", flush=True)
    rel, perm, pair, (q11, q12, xy) = eval_vs_fem(model, data_dir)
    res, ene = eval_residual_energy(model)
    result = {"tag": args.tag, "args": vars(args), "best_loss": best["loss"],
              "rel_L2": rel, "perm": perm, "pairwise_nn_dist": pair,
              "residual_per_branch": res, "energy_per_branch": ene,
              "wallclock_s": time.time() - t0, "history": hist}
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    torch.save(model.state_dict(), os.path.join(args.outdir, f"{args.tag}.pt"))
    quiver_plot(q11, q12, xy, perm, os.path.join(args.outdir, f"{args.tag}_quiver.png"))
    print(f"[{args.tag}] DONE relL2={rel} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
