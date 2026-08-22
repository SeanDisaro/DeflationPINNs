import ast, os, re
import numpy as np
LOG = os.path.expanduser("~/DeflationPINNs_rev/logs/ldg_J3_msefinish.log")
lines = open(LOG).read().splitlines()

loss_steps, loss_vals = [], []
evals = {}  # step -> {state: rel}
for ln in lines:
    m = re.search(r"lbfgsC (\d+) loss ([0-9.e+-]+)", ln)
    if m:
        loss_steps.append(int(m.group(1))); loss_vals.append(float(m.group(2)))
    m = re.search(r"lbfgsC (\d+) (\{.*\})$", ln)
    if m:
        d = ast.literal_eval(m.group(2))
        evals[int(m.group(1))] = {k: v["rel"] for k, v in d.items()}
    m = re.search(r"after distill (\{.*\})$", ln)
    if m:
        d = ast.literal_eval(m.group(1))
        evals[0] = {k: v["rel"] for k, v in d.items()}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
COLORS = {"D1": "#2a78d6", "D2": "#eb6834", "R1": "#1baf7a",
          "R2": "#eda100", "R3": "#e87ba4", "R4": "#008300"}
steps = sorted(evals)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=SURF)
for ax in (a1, a2):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(INK2)
for nm, c in COLORS.items():
    ys = [evals[s][nm] for s in steps]
    a1.plot(steps, ys, color=c, lw=2, marker="o", ms=6, label=nm)
a1.axhline(0.01, color=INK2, lw=1, ls="--", alpha=0.6)
a1.annotate("1% target", (steps[-1], 0.011), fontsize=8, color=INK2, ha="right")
a1.set_title("Relative $L^2$ error vs converged references", fontsize=11, color=INK, loc="left")
a1.set_xlabel("residual L-BFGS outer step", fontsize=9.5, color=INK2)
a1.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper left")
a1.set_ylim(0, 0.115)
a2.plot(loss_steps, loss_vals, color="#4a3aa7", lw=2, marker="o", ms=5)
a2.set_title("Residual least-squares objective (the loss being minimized)", fontsize=11, color=INK, loc="left")
a2.set_xlabel("residual L-BFGS outer step", fontsize=9.5, color=INK2)
a2.set_ylim(0, 0.14)
fig.suptitle("Run J3: residual training from 2%-accurate stable states — the objective falls while every branch drifts away",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.91])
out = os.path.expanduser("~/DeflationPINNs_rev/results/J3_drift.png")
fig.savefig(out, dpi=170, facecolor=SURF)
print("saved", out, "steps:", steps)
