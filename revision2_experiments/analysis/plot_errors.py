"""Bar chart: rel/abs L2 errors for stable states (discovery + refined) and for
the certified UNSTABLE solutions that residual refinement approached."""
import os, sys, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/DeflationPINNs_rev/harness"))
import numpy as np, torch
import deepxde as dde  # noqa
from harness_ldg_delta import SmoothSixState
from ldg_reference import newton_state, energy

RES = os.path.expanduser("~/DeflationPINNs_rev/results")
N = 257

# ---- stable-state data (discovery errors from run C vs 513-reference; refined from flow census)
rel_disc = {"D1": 0.5839, "D2": 0.5948, "R1": 0.3252, "R2": 0.3022, "R3": 0.3026, "R4": 0.2988}
rel_refi = {"D1": 2.45e-12, "D2": 2.45e-12, "R1": 2.46e-12, "R2": 2.46e-12, "R3": 2.46e-12, "R4": 2.46e-12}
rms = {}
for nm in rel_disc:
    f = np.load(os.path.join(RES, f"ldg_reference/{nm}_n513.npz"))
    rms[nm] = float(np.sqrt(np.mean(f["Q1"] ** 2 + f["Q2"] ** 2)))

# ---- certify unstable endpoints from the residual-refined (run H) fields
geom = dde.geometry.geometry_2d.Rectangle([0., 0.], [1., 1.])
m = SmoothSixState(geom).double()
m.load_state_dict(torch.load(os.path.join(RES, "ldg_H_delta.pt"), map_location="cpu"))
m.eval()
g = np.linspace(0, 1, N); X, Y = np.meshgrid(g, g, indexing="xy")
pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float64)
with torch.no_grad():
    out = m(pts)
E_STABLE = (79.455, 88.090)
unstable = []
for k in range(6):
    Q1 = out["out1"][k].numpy().reshape(N, N).copy()
    Q2 = out["out2"][k].numpy().reshape(N, N).copy()
    T1, T2, rn, its = newton_state("D1", N, seed=(Q1, Q2), tol=1e-11, itmax=60)
    if rn > 1e-8:
        print(f"branch {k}: Newton not certified (res {rn:.1e}) - skipped", flush=True)
        continue
    E = energy(T1, T2, N)
    d = np.sqrt(np.mean((Q1 - T1) ** 2 + (Q2 - T2) ** 2))
    nT = np.sqrt(np.mean(T1 ** 2 + T2 ** 2))
    stab = any(abs(E - e) < 0.5 for e in E_STABLE)
    print(f"branch {k}: certified E={E:.3f} stable={stab} rel={d/nT:.4f}", flush=True)
    if not stab:
        unstable.append({"E": E, "rel": d / nT, "abs": d})
unstable.sort(key=lambda u: u["E"])

# ---- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
states = ["D1", "D2", "R1", "R2", "R3", "R4"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURF)
for ax, kind in zip(axes, ["rel", "abs"]):
    ax.set_facecolor(SURF)
    xs = np.arange(len(states))
    if kind == "rel":
        v1 = [rel_disc[s] for s in states]
        v2 = [rel_refi[s] for s in states]
        vu = [u["rel"] for u in unstable]
    else:
        v1 = [rel_disc[s] * rms[s] for s in states]
        v2 = [rel_refi[s] * rms[s] for s in states]
        vu = [u["abs"] for u in unstable]
    ax.bar(xs - 0.19, v1, 0.34, color=BLUE, edgecolor=SURF, linewidth=1.5,
           label="Deflation-PINN (discovery)")
    ax.bar(xs + 0.19, v2, 0.34, color=ORANGE, edgecolor=SURF, linewidth=1.5,
           label="after classical refinement")
    xu = np.arange(len(unstable)) + len(states) + 0.6
    ax.bar(xu, vu, 0.34, color=AQUA, edgecolor=SURF, linewidth=1.5,
           label="NN vs certified unstable solution")
    for x, u in zip(xu, unstable):
        ax.annotate(f"E={u['E']:.0f}", (x, ax.get_ylim()[0]), xytext=(x, max(vu) * 1.6),
                    ha="center", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(list(xs) + list(xu))
    ax.set_xticklabels(states + [f"U{i+1}" for i in range(len(unstable))],
                       fontsize=9, color=INK)
    ax.set_title("Relative $L^2$ error" if kind == "rel" else "Absolute $L^2$ error",
                 fontsize=11, color=INK, loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.axvline(len(states) - 0.2 + 0.4, color=INK2, alpha=0.3, linewidth=0.8, linestyle=":")
axes[0].legend(fontsize=8.5, frameon=False, loc="center left")
fig.suptitle("LdG solution errors vs mesh-converged references: stable states and certified unstable solutions",
             fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
outp = os.path.join(RES, "error_summary.png")
fig.savefig(outp, dpi=170, facecolor=SURF)
json.dump({"unstable": unstable}, open(os.path.join(RES, "unstable_summary.json"), "w"), indent=1)
print("saved", outp, flush=True)
