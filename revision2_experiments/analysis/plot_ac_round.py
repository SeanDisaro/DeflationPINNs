"""Publication plot for the Allen-Cahn benchmark: evaluate the trained model on
a fine polar grid so the disk renders as an exact circle (no polygonal rim)."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
import deepxde as dde  # noqa: F401,E402
from src.architectures.DeflationPINN import two_dim_2_one_dim_DefPINN  # noqa: E402
from src.starDomainExtrapolation.starDomain import Sphere  # noqa: E402

CKPT = os.path.expanduser("~/DeflationPINNs_rev/results/allencahn_v2.pt")
OUT = os.path.expanduser("~/DeflationPINNs_rev/results/AC_DefPINN_solutions.png")
DEV = "cpu"


def squaredRadial(x):
    return (1 - x * x).view(-1, 1)


def zeroOnBoundaryExtension(star, inp):
    radius, angles = star.getSphericalCoordinates(inp)
    return squaredRadial(radius / star.radiusDomainFunciton(angles)).view(-1, 1)


star = Sphere(dim=2, center=torch.tensor([0, 0]), radius=torch.tensor([1.]))
star.updateDevice(DEV)
model = two_dim_2_one_dim_DefPINN(
    numSolutions=3, numBranchFeatures=32, trunk_layer=6, trunk_width=100,
    activationFunction=torch.nn.Tanh(), geom=None, DirichletHardConstraint=True,
    skipConnection=False,
    r_function=lambda p: zeroOnBoundaryExtension(star, [p[:, 0].view(-1, 1), p[:, 1].view(-1, 1)]),
    DirichletConditionFunc=None).double()
model.load_state_dict(torch.load(CKPT, map_location=DEV))
model.eval()

# fine polar grid -> exactly circular boundary
r = np.linspace(0.0, 1.0, 241)
th = np.linspace(0.0, 2 * np.pi, 481)
R, TH = np.meshgrid(r, th, indexing="ij")
X, Y = R * np.cos(TH), R * np.sin(TH)
pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float64)
# nudge the exact center to avoid 0/0 in the radial construction
pts[(pts[:, 0] == 0) & (pts[:, 1] == 0)] = 1e-12
with torch.no_grad():
    out = model(pts)
U = [out["out"][k].numpy().reshape(R.shape) for k in range(3)]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

order = [0, 1, 2]  # branch matching from allencahn_v2.json: minus, zero, plus
titles = [r"$u_-$", r"$u_0$", r"$u_+$"]
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
for i, (k, ttl) in enumerate(zip(order, titles)):
    cs = ax[i].contourf(X, Y, U[k], levels=60, cmap="viridis")
    for c in cs.collections:
        c.set_edgecolor("face")  # avoid contour banding artifacts in PDF
    cb = fig.colorbar(cs, ax=ax[i], shrink=0.9)
    cb.formatter.set_powerlimits((-2, 3))
    cb.update_ticks()
    ax[i].set_title(ttl, fontsize=15)
    ax[i].set_aspect("equal")
    ax[i].set_xticks([-1, 0, 1])
    ax[i].set_yticks([-1, 0, 1])
fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("saved", OUT)
