import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import numpy as np, torch
import deepxde as dde
from harness_ldg_delta import SmoothSixState

RES = os.path.expanduser("~/DeflationPINNs_rev/results")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
geom = dde.geometry.geometry_2d.Rectangle([0., 0.], [1., 1.])
m = SmoothSixState(geom, 128, 3, 128, 16.0).double()
m.load_state_dict(torch.load(os.path.join(RES, "ldg_J_ritz1pct.pt"), map_location="cpu"))
m.eval()
NAMES = {0: "R3", 1: "D2", 2: "R4", 3: "R2", 4: "R1", 5: "D1"}
nq, na = 129, 27
gq = np.linspace(0, 1, nq); Xq, Yq = np.meshgrid(gq, gq, indexing="xy")
ga = np.linspace(0.02, 0.98, na); Xa, Ya = np.meshgrid(ga, ga, indexing="xy")
pq = torch.tensor(np.stack([Xq.ravel(), Yq.ravel()], 1), dtype=torch.float64)
pa = torch.tensor(np.stack([Xa.ravel(), Ya.ravel()], 1), dtype=torch.float64)
order = sorted(range(6), key=lambda k: ["D1", "D2", "R1", "R2", "R3", "R4"].index(NAMES[k]))
fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.6))
with torch.no_grad():
    for i, k in enumerate(order):
        a = ax[i // 3][i % 3]
        q1h, q2h = m.fields(pq, k)
        mod = np.sqrt(q1h.numpy() ** 2 + q2h.numpy() ** 2).reshape(nq, nq)
        im = a.imshow(mod, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1.02, cmap="viridis", alpha=0.85)
        q1a, q2a = m.fields(pa, k)
        th = 0.5 * np.arctan2(q2a.numpy().ravel(), q1a.numpy().ravel())
        s = np.sqrt(np.sqrt(q1a.numpy().ravel() ** 2 + q2a.numpy().ravel() ** 2))
        a.quiver(Xa.ravel(), Ya.ravel(), s * np.cos(th), s * np.sin(th),
                 angles="xy", scale_units="xy", scale=38, width=0.003,
                 headwidth=1, headlength=0, headaxislength=0, color="white", pivot="mid")
        a.set_title(NAMES[k], fontsize=12)
        a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_aspect("equal")
        fig.colorbar(im, ax=a, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "DDR_solutions.png"), dpi=150)
print("saved")
