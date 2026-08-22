"""Render |Q| maps of J3's step-120 drifted fields (the Cpartial checkpoint)."""
import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import numpy as np, torch
import deepxde as dde
from harness_ldg_multi import SmoothKState

RES = os.path.expanduser("~/DeflationPINNs_rev/results")
geom = dde.geometry.geometry_2d.Rectangle([0., 0.], [1., 1.])
m = SmoothKState(geom, 6, 128, 3, 128, 16.0).double()
m.load_state_dict(torch.load(os.path.join(RES, "ldg_J3_msefinish_Cpartial.pt"), map_location="cpu"))
m.eval()
n = 257
g = np.linspace(0, 1, n); X, Y = np.meshgrid(g, g, indexing="xy")
pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float64)
NAMES = {0: "R3", 1: "D2", 2: "R4", 3: "R2", 4: "R1", 5: "D1"}  # J's assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.6), facecolor="#fcfcfb")
with torch.no_grad():
    for k in range(6):
        q1, q2 = m.fields(pts, k)
        mod = np.sqrt(q1.numpy() ** 2 + q2.numpy() ** 2).reshape(n, n)
        a = ax[k // 3][k % 3]
        im = a.imshow(mod, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1.05, cmap="viridis")
        a.set_title(f"branch {k} (was {NAMES[k]}), $|Q|$", fontsize=10)
        fig.colorbar(im, ax=a, shrink=0.85)
fig.suptitle("J3 step-120 drifted fields: $|Q|$ maps (dark lines = emerging order-reconstruction walls)",
             fontsize=12, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(RES, "J3_step120_fields.png"), dpi=150, facecolor="#fcfcfb")
print("saved")
