import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import numpy as np, torch
import deepxde as dde
from harness_ldg_delta import SmoothSixState

RES = os.path.expanduser("~/DeflationPINNs_rev/results")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# MATLAB-parula-like colormap to match the FEM director figures
parula = LinearSegmentedColormap.from_list("parula_like", [
    "#352a87", "#0f5cdd", "#1481d6", "#06a4ca", "#2eb7a4",
    "#87bf77", "#d1bb59", "#fec832", "#f9fb0e"])

geom = dde.geometry.geometry_2d.Rectangle([0., 0.], [1., 1.])
m = SmoothSixState(geom, 128, 3, 128, 16.0).double()
m.load_state_dict(torch.load(os.path.join(RES, "ldg_J_ritz1pct.pt"), map_location="cpu"))
m.eval()
NAMES = {0: "R3", 1: "D2", 2: "R4", 3: "R2", 4: "R1", 5: "D1"}
nq, na = 161, 33
gq = np.linspace(0, 1, nq); Xq, Yq = np.meshgrid(gq, gq, indexing="xy")
ga = np.linspace(0.015, 0.985, na); Xa, Ya = np.meshgrid(ga, ga, indexing="xy")
pq = torch.tensor(np.stack([Xq.ravel(), Yq.ravel()], 1), dtype=torch.float64)
pa = torch.tensor(np.stack([Xa.ravel(), Ya.ravel()], 1), dtype=torch.float64)
order = sorted(range(6), key=lambda k: ["D1", "D2", "R1", "R2", "R3", "R4"].index(NAMES[k]))
letters = "abcdef"
fig, ax = plt.subplots(2, 3, figsize=(13.2, 8.8))
with torch.no_grad():
    for i, k in enumerate(order):
        a = ax[i // 3][i % 3]
        q1h, q2h = m.fields(pq, k)
        mod = np.sqrt(q1h.numpy() ** 2 + q2h.numpy() ** 2).reshape(nq, nq)
        im = a.imshow(mod, origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=1, cmap=parula)
        q1a, q2a = m.fields(pa, k)
        th = 0.5 * np.arctan2(q2a.numpy().ravel(), q1a.numpy().ravel())
        a.quiver(Xa.ravel(), Ya.ravel(), np.cos(th), np.sin(th),
                 angles="xy", scale_units="xy", scale=46, width=0.0022,
                 headwidth=1, headlength=0, headaxislength=0, color="black", pivot="mid")
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
        a.set_xlabel(f"({letters[i]}) {NAMES[k]}", fontsize=12)
        cb = fig.colorbar(im, ax=a, shrink=0.9, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "DDR_solutions.png"), dpi=150)
print("saved")
