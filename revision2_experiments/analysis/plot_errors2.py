import os, numpy as np
RES = os.path.expanduser("~/DeflationPINNs_rev/results")
rel_disc = {"D1": 0.5839, "D2": 0.5948, "R1": 0.3252, "R2": 0.3022, "R3": 0.3026, "R4": 0.2988}
rel_refi = {s: 2.46e-12 for s in rel_disc}; rel_refi["D1"] = rel_refi["D2"] = 2.45e-12
rms = {}
for nm in rel_disc:
    f = np.load(os.path.join(RES, f"ldg_reference/{nm}_n513.npz"))
    rms[nm] = float(np.sqrt(np.mean(f["Q1"] ** 2 + f["Q2"] ** 2)))
U = {"E": 366.07, "rel": 0.4536, "abs": 0.4279}  # certified wall state, closest NN branch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
states = ["D1", "D2", "R1", "R2", "R3", "R4"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=SURF)
for ax, kind in zip(axes, ["rel", "abs"]):
    ax.set_facecolor(SURF)
    xs = np.arange(len(states))
    v1 = [rel_disc[s] * (1 if kind == "rel" else rms[s]) for s in states]
    v2 = [rel_refi[s] * (1 if kind == "rel" else rms[s]) for s in states]
    vu = U["rel"] if kind == "rel" else U["abs"]
    ax.bar(xs - 0.19, v1, 0.34, color=BLUE, edgecolor=SURF, linewidth=1.5,
           label="Deflation-PINN (discovery stage)")
    ax.bar(xs + 0.19, v2, 0.34, color=ORANGE, edgecolor=SURF, linewidth=1.5,
           label="after classical refinement")
    xu = len(states) + 0.7
    ax.bar([xu], [vu], 0.38, color=AQUA, edgecolor=SURF, linewidth=1.5,
           label="closest NN branch vs certified\nunstable wall state (E=366)")
    ax.axvline(len(states) - 0.25 + 0.5, color=INK2, alpha=0.3, linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylim(1e-13, 3)
    ax.set_xticks(list(xs) + [xu])
    ax.set_xticklabels(states + ["U1"], fontsize=9.5, color=INK)
    ax.set_title("Relative $L^2$ error" if kind == "rel" else "Absolute $L^2$ error",
                 fontsize=11, color=INK, loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
axes[0].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.42))
fig.suptitle("LdG: errors vs mesh-converged references — six stable states and the certified unstable X-wall state",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(RES, "error_summary.png"), dpi=170, facecolor=SURF)
print("saved")
