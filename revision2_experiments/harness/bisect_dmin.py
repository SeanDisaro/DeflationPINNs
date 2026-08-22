"""Bisection on d_min to localize the trainability edge d*.

AC  (unit disk, K=3): endpoints 0.2 (good) / 0.4 (bad). The over-separation
    pathology satisfies the hinge by inflating the fields, so the discriminator
    is the final PDE residual, not the hinge. Prediction: transition near the
    minimal true separation ~0.28 in the deflation metric.
LdG (unit square, K=6): endpoints 0.4 (good) / 0.85 (bad). Same discriminator
    (at 0.85 the hinge is satisfied but the residual plateaus ~7x higher).

Classifier: query is GOOD iff its final pde value is below the geometric mean
of the two current endpoint pde values (endpoints re-measured by the driver
itself under the identical reduced-budget protocol). Everything is logged to
results/bisect_results.json after every query, so partial runs are usable.
"""
import json
import os
import re
import subprocess
import sys
import time

H = os.path.expanduser("~/DeflationPINNs_rev/harness")
R = os.path.expanduser("~/DeflationPINNs_rev/results")
L = os.path.expanduser("~/DeflationPINNs_rev/logs")
CWD = os.path.expanduser("~/DeflationPINNs_rev/DeflationPINNs-dev")
PY = sys.executable
OUT = os.path.join(R, "bisect_results.json")
ENV = dict(os.environ, DDE_BACKEND="pytorch", MPLBACKEND="Agg", LDG_DEVICE="cuda:0",
           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           PYTORCH_ALLOC_CONF="expandable_segments:True")

STATE = {"queries": []}


def save():
    json.dump(STATE, open(OUT, "w"), indent=1)


def last_float(pat, text):
    m = re.findall(pat, text)
    return float(m[-1]) if m else float("nan")


def run_ac(dmin, seed=0):
    tag = f"bisect_ac_d{int(round(dmin * 1000)):03d}_s{seed}"
    log = os.path.join(L, tag + ".log")
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run([PY, os.path.join(H, "harness_ac.py"), "--dmin", str(dmin),
                        "--epochs", "15000", "--lbfgs", "30", "--mse",
                        "--seed", str(seed), "--tag", tag, "--outdir", R],
                       stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=CWD)
    text = open(log).read()
    return {"problem": "AC", "dmin": dmin, "seed": seed,
            "pde": last_float(r"pde ([0-9.e+-]+)", text),
            "defl": last_float(r"defl ([0-9.]+)", text),
            "loss": last_float(r"loss ([0-9.e+-]+)", text),
            "wall_s": round(time.time() - t0), "tag": tag}


def run_ldg(dmin, seed=0):
    tag = f"bisect_ldg_d{int(round(dmin * 1000)):03d}_s{seed}"
    log = os.path.join(L, tag + ".log")
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run([PY, os.path.join(H, "harness_ldg.py"), "--deflation", "full",
                        "--fast", "--epochs", "10000", "--lr", "1e-4",
                        "--dmin", str(dmin), "--eval-every", "0",
                        "--seed", str(seed), "--tag", tag, "--outdir", R],
                       stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=CWD)
    text = open(log).read()
    return {"problem": "LdG", "dmin": dmin, "seed": seed,
            "pde": last_float(r"pde ([0-9.e+-]+)", text),
            "defl": last_float(r"defl ([0-9.]+)", text),
            "loss": last_float(r"loss ([0-9.e+-]+)", text),
            "wall_s": round(time.time() - t0), "tag": tag}


def bisect(runner, name, a, b, tol, max_queries=8, c_floor=2.5, ambig=3.0):
    """Floor-test classifier: a query is GOOD iff its final residual stays within
    a factor c_floor of the good-endpoint residual (the good regime is a tight
    multiplicative floor; the bad regime is heavy-tailed, so midpoint rules are
    fragile there). Queries landing within a factor `ambig` of the threshold are
    confirmed with a second seed, and BOTH seeds must pass -- misclassifying a
    bad d_min as good is the costly error (it ends in an unusable recommendation),
    so ties break conservatively."""
    ra = runner(a)
    STATE["queries"].append(ra); save()
    rb = runner(b)
    STATE["queries"].append(rb); save()
    hist = [{"a": a, "b": b, "pde_a": ra["pde"], "pde_b": rb["pde"]}]
    print(f"[{name}] endpoints: pde({a})={ra['pde']:.3e}  pde({b})={rb['pde']:.3e}", flush=True)
    n = 0
    while (b - a) > tol and n < max_queries:
        m = round((a + b) / 2, 4)
        rm = runner(m)
        STATE["queries"].append(rm); save()
        thresh = c_floor * ra["pde"]
        good = rm["pde"] < thresh
        repeated = False
        if good and rm["pde"] > thresh / ambig:
            rm2 = runner(m, seed=1)
            STATE["queries"].append(rm2); save()
            good = rm2["pde"] < thresh
            repeated = True
            print(f"[{name}] query d={m}: ambiguous (pde={rm['pde']:.3e}), "
                  f"seed-1 confirm pde={rm2['pde']:.3e}", flush=True)
        print(f"[{name}] query d={m}: pde={rm['pde']:.3e} defl={rm['defl']:.4f} "
              f"thresh={thresh:.3e} -> {'GOOD' if good else 'BAD'}"
              f"{' (2-seed)' if repeated else ''}", flush=True)
        if good:
            a, ra = m, rm
        else:
            b, rb = m, rm
        hist.append({"a": a, "b": b, "query": m, "pde": rm["pde"], "good": good,
                     "two_seed": repeated})
        n += 1
    STATE[name] = {"interval": [a, b], "history": hist}
    save()
    print(f"[{name}] FINAL interval: [{a}, {b}]", flush=True)


if __name__ == "__main__":
    bisect(run_ac, "AC", 0.2, 0.4, tol=0.02)
    bisect(run_ldg, "LdG", 0.4, 0.85, tol=0.05)
    print("BISECT_ALL_DONE", flush=True)
