# DeflationPIML

This is the repository to the paper ["Deflation-PINNs and Deflation–Deep-Ritz: Multi-Solution Discovery, Certification, and Refinement for Nonlinear PDEs"](http://arxiv.org/abs/2603.27936) in which we present a PINN and DeepONet based model to approximate multiple solutions at once for a Landau de Gennes problem from liquid crystal theory.

The paper also tests the method on a second problem, an Allen–Cahn equation on the unit disk whose solutions are known exactly.

## The method in short

A Deflation-PINN learns K solutions of the same PDE with a single network. A shared trunk net τ takes the spatial point x, and each solution branch k has its own learnable vector of branch weights β<sup>k</sup> ∈ ℝ<sup>p</sup>:

$$u_k(x) \approx \sum_{i=1}^{p} \tau_i(x)\,\beta^k_i, \qquad k = 1, \dots, K.$$

The Dirichlet boundary data is built into the architecture (hard constraint), so the loss contains only the physics-informed residuals of all branches and a **deflation loss**. The deflation loss is a pairwise hinge that is zero exactly when every two branches are at least d<sub>min</sub> apart:

$$\mathcal{L}_{\mathrm{Def}} = \frac{2}{K(K-1)} \sum_{i < j} \max\Big(1 - \frac{\lVert u_i - u_j \rVert_{L^2}}{d_{\min}},\ 0\Big).$$

Training uses no solution data. For a PDE that is the Euler–Lagrange equation of an energy, replacing the residual by that energy gives a **Deflation–Deep-Ritz** method, which we use to refine the discovered solutions.

## Experiments at a glance

**Landau–de Gennes (LdG).** The reduced LdG model of nematic liquid crystals on a square has six stable states. We compute them in three stages:

1. **Discovery.** A single Deflation-PINN run with K = 6 branches. The branches are coarse, but in the run reported in the paper each one lies in the basin of attraction of a different stable state.
2. **Certification and classical refinement.** Each branch is evolved by the LdG gradient flow and then polished by Newton's method. This uses no reference data: it certifies which basin each branch lies in and refines the branch to the accuracy of the reference solution.
3. **Neural refinement.** Deflation–Deep-Ritz, started from the discovered branches, refines all six to 1.2–1.3% relative error with networks alone.

| State | Discovery | Deflation–Deep-Ritz | Classical refinement |
|---|---|---|---|
| D1 | 0.58 | 0.013 | 2.5·10⁻¹² |
| D2 | 0.59 | 0.013 | 2.5·10⁻¹² |
| R1 | 0.33 | 0.013 | 2.5·10⁻¹² |
| R2 | 0.30 | 0.013 | 2.5·10⁻¹² |
| R3 | 0.30 | 0.012 | 2.5·10⁻¹² |
| R4 | 0.30 | 0.013 | 2.5·10⁻¹² |

*Relative L² errors of the six branches against a mesh-converged finite-difference reference (Table 1 of the paper).*

Not every run finds all six states: eight further runs with the same settings found between two and four each. Repeating the discovery from fresh random initializations and certifying the branches of every run covered all six states after five runs.

**Allen–Cahn.** A single run with K = 3 recovers all three solutions, 0 and ±u\*, without a refinement stage. The relative L² errors are 4.7·10⁻⁵ for +u\* and 1.4·10⁻⁴ for −u\*, and the trivial branch is zero up to 2.0·10⁻⁵.

## Benchmark 1: Landau–de Gennes

$$-\Delta \mathbf{Q} = 2\epsilon^{-2}\,(1 - |\mathbf{Q}|^2)\,\mathbf{Q} \ \text{ in } [0,1]^2, \qquad \mathbf{Q} = \mathbf{Q}_b \ \text{ on } \partial[0,1]^2,$$

with **Q** = (Q₁₁, Q₁₂), ε = 0.02 and tangent boundary data built from a trapezoidal profile (paper, Sec. 3.1). The six stable states are two diagonal states (D1, D2) and four rotated states (R1–R4). FEM solutions of all six from Maity et al. (2021) are in `data/trueSolution/`. Unless a row says otherwise, discovery runs use the settings of Appendix A: K = 6, d<sub>min</sub> = 0.4, 33 × 33 collocation points and 10,000 Adam epochs.

| Experiment | Paper | Finding | Code in `revision2_experiments/` |
|---|---|---|---|
| Discovery | Sec. 3.2, Table 1, Fig. 2 | One run finds all six states with relative L² errors of 0.30–0.59; the closest two branches are ≈ 0.5 apart | [`harness/harness_ldg.py`](revision2_experiments/harness/harness_ldg.py) `--deflation full` |
| Mesh-converged reference | Sec. 3.3 | Finite differences and Newton at h = 1/256 and 1/512; the extrapolated energies agree with the FEM literature to 0.06% | [`harness/ldg_reference.py`](revision2_experiments/harness/ldg_reference.py) |
| Certification and classical refinement | Sec. 3.3, Table 1 | Gradient flow and Newton take each branch to a different stable state, 2.5·10⁻¹² from the reference | [`harness/census6_from_flow.py`](revision2_experiments/harness/census6_from_flow.py), [`harness/flow_final.py`](revision2_experiments/harness/flow_final.py) |
| Deflation–Deep-Ritz refinement | Sec. 3.4, Table 1, Fig. 5 | Relative L² errors of 1.2–1.3%; switching the energy quadrature from midpoint to Simpson lowered them from 1.9–2.4% | [`harness/harness_ldg_delta.py`](revision2_experiments/harness/harness_ldg_delta.py), then [`harness/harness_ldg_multi.py`](revision2_experiments/harness/harness_ldg_multi.py) `--quad-scheme simpson` |
| Error norms and localization | Table 1, Figs. 3–4 | The discovery error is an amplitude deficit in the bulk; after Deflation–Deep-Ritz, 59–77% of the squared error sits in the four corner defect cores | [`analysis/error_analysis.py`](revision2_experiments/analysis/error_analysis.py), [`analysis/refined_norms.py`](revision2_experiments/analysis/refined_norms.py) |
| Restarts with certification | Sec. 3.3, App. C, Table 4 | Eight independent runs find 2–4 states each; together they cover all six after five runs | [`launchers/launch_popos_seeds.sh`](revision2_experiments/launchers/launch_popos_seeds.sh), [`harness/census_any.py`](revision2_experiments/harness/census_any.py) |
| Choice of d<sub>min</sub> | Sec. 3.3, App. C, Table 5 | The stable states are at least 1.086 apart, but training fails for d<sub>min</sub> above ≈ 0.5; a bisection finds this edge without reference data, and the paper uses 0.4 | [`harness/bisect_dmin.py`](revision2_experiments/harness/bisect_dmin.py) |
| Over-generation, K = 11 | Sec. 3.3, App. C | Nine branches certify but occupy only four distinct states: deflation separates fields, not basins | [`launchers/launch_gpu0_K11.sh`](revision2_experiments/launchers/launch_gpu0_K11.sh), [`harness/certify_census.py`](revision2_experiments/harness/certify_census.py) |
| Residual-only post-training | Sec. 3.3 | Drives under-saturated branches toward unstable wall states; an X-shaped wall state with energy 366.07 is certified by Newton's method with ε-continuation | [`launchers/launch_gpu0_H.sh`](revision2_experiments/launchers/launch_gpu0_H.sh), [`analysis/certify_walls2.py`](revision2_experiments/analysis/certify_walls2.py) |
| Convergence in collocation and width | App. C, Table 3 | The best branch error stays at 0.29–0.30 for 17²–49² points and widths 500–4000; only the number of states found varies | [`launchers/launch_popos_convergence.sh`](revision2_experiments/launchers/launch_popos_convergence.sh), [`launchers/launch_popos_takeover.sh`](revision2_experiments/launchers/launch_popos_takeover.sh) |
| Feature-space deflation | App. C | A hinge on the branch vectors β<sup>k</sup> instead of the fields separates the vectors but not the solutions: three states only | [`harness/harness_ldg.py`](revision2_experiments/harness/harness_ldg.py) `--deflation feature` |
| Deep-Ritz without discovery | Sec. 3.4, App. C | With a fixed, an annealed (0.4 → 0.9) or a scale-free deflation term, all six branches collapse onto the two lowest-energy states D1 and D2 | [`launchers/launch_popos_ddr_scratch.sh`](revision2_experiments/launchers/launch_popos_ddr_scratch.sh) |

## Benchmark 2: Allen–Cahn on the unit disk

$$-\Delta u = 6u - u^3 \ \text{ in } B(0,1) \subset \mathbb{R}^2, \qquad u = 0 \ \text{ on } \partial B(0,1).$$

Because 6 lies between the first two Dirichlet eigenvalues of the disk (≈ 5.783 and ≈ 14.68), the solutions are exactly 0 and ±u\*, where u\* is positive and radial (paper, Sec. 3.5). The number of branches, K = 3, is therefore known in advance. The reference u\* is computed by shooting on the radial ODE, so no PDE discretization enters the comparison.

| Experiment | Paper | Finding | Code in `revision2_experiments/` |
|---|---|---|---|
| Recovery of all three solutions | Sec. 3.5, Table 2, Fig. 6 | One run finds 0 and ±u\* with relative L² errors of 4.7·10⁻⁵ and 1.4·10⁻⁴ | [`harness/harness_ac.py`](revision2_experiments/harness/harness_ac.py) `--dmin 0.2`, [`analysis/plot_ac_round.py`](revision2_experiments/analysis/plot_ac_round.py) |
| d<sub>min</sub> above the true separation | Sec. 3.5 | With d<sub>min</sub> = 0.4 > 0.28 the hinge cannot reach zero and the learned fields are inflated | [`harness/harness_ac.py`](revision2_experiments/harness/harness_ac.py) `--dmin 0.4` |
| Bisection on d<sub>min</sub> | App. C, Table 5 | Finds the interval [0.275, 0.288], which contains the true minimal separation 0.28, from training behavior alone | [`harness/bisect_dmin.py`](revision2_experiments/harness/bisect_dmin.py) |

## Installation

Set up a Python 3.12 environment on a machine with a CUDA GPU. From the root of this repository, install the packages and set DeepXDE to its PyTorch backend:

```bash
pip install -r ./requirements.txt
python -m deepxde.backend.set_default_backend pytorch
```

The notebook `metricsAndPics4Paper.ipynb` additionally needs `pandas` and Jupyter. Run all scripts from the repository root, since `config.py` resolves paths relative to the working directory.

## Running the original experiment (`main.py`)

```bash
python main.py
```

This trains one Deflation-PINN on the LdG problem. It saves the model with the lowest loss to `models/DeflationPINN.pkl` and writes plots of the six branches to `tests/pictures/deflationPINNTest/`: files ending in `_zero.png` show the random initialization, files ending in `_AfterTraining.png` the trained branches, and `Learning_Epoch_Plot.png` the loss curve. The notebook `metricsAndPics4Paper.ipynb` loads the saved model, plots the branches next to the FEM solutions and computes their relative L² errors.

The hyperparameters are set in `tests/deflationPINNTest/run.py`; change them there and rerun `python main.py`.

| Setting | Value | Variable in `run.py` |
|---|---|---|
| Branches K | 6 | `numSolutions` |
| Trunk net | 1 hidden layer of width 4000, tanh | `trunk_layer`, `trunk_width` |
| Branch weights p | 16 per output component | `numBranchFeatures` |
| Collocation points | 33 × 33 grid on [0.001, 0.999]² | `n`, `safetySpace` |
| Optimizer | Adam, learning rate 10⁻⁴, 10,000 epochs | `learningRate`, `epochs` |
| Loss weights | 0.01 for the residual, 100 for the deflation loss | `alpha`, `delta` |
| d<sub>min</sub> | 0.4 | second entry of `deflationLossPoints` |

> **Note:** `main.py` measures the deflation distance as the mean absolute difference of the Q₁₂ components only (the `legacy` variant of the revision harness). The LdG results in the paper use the L² distance of the full field (Q₁₁, Q₁₂); to reproduce them, run `revision2_experiments/harness/harness_ldg.py --deflation full` as described below.

<p align="center">
  <img src="tests/pictures/deflationPINNTest/Reduced2DimLDG_Results_AfterTraining.png" width="700" alt="The six branches learned by one run of main.py">
</p>

*The six branches of one `main.py` run, drawn as (Q₁₁, Q₁₂) vector fields: two diagonal and four rotated states.*

## Running the paper experiments (`revision2_experiments/`)

This folder contains the code that produced the numbers, tables and figures of the paper; [revision2_experiments/README.md](revision2_experiments/README.md) maps it to the manuscript file by file. Before running it, note:

- The scripts in `launchers/` record the exact arguments of every run, together with paths of the machines they ran on (`launch_gpu*` on a DGX with V100 GPUs, `launch_popos_*` on a workstation with an RTX 4060 Ti). Use them as a record of the settings rather than running them as they are.
- Reference solutions, checkpoints and certification results are exchanged through `~/DeflationPINNs_rev/results`, which several harness scripts hard-code. Pass `--outdir ~/DeflationPINNs_rev/results` to the training scripts, or edit the path constants at the top of the files.
- Some launchers belong to exploratory runs that are not part of the paper, for example a 1D reaction–diffusion benchmark (`analysis/harness_reaction.py`).

The Table 1 pipeline, in order:

```bash
export DDE_BACKEND=pytorch
H=revision2_experiments/harness
R=~/DeflationPINNs_rev/results

# 1. Mesh-converged FD reference at h = 1/256 and 1/512, written to $R/ldg_reference
python $H/ldg_reference.py

# 2. Discovery: one Deflation-PINN run with K = 6 and full vector-field deflation
python $H/harness_ldg.py --deflation full --fast --epochs 10000 --grid 33 \
    --lr 1e-4 --dmin 0.4 --tag ldg_C_full --outdir $R

# 3. Certification and classical refinement: gradient flow + Newton per branch
#    (CPU, a few minutes per branch)
CUDA_VISIBLE_DEVICES= python $H/census6_from_flow.py

# 4. Deflation-Deep-Ritz: self-distillation, Adam on a 192^2 midpoint grid,
#    fp64 L-BFGS on 384^2 (a few GPU hours on a V100)
python $H/harness_ldg_delta.py --source $R/ldg_C_full.pt --refine-mode ritz --polish ritz \
    --fourier 128 --fsigma 16 --quad 192 --quad-final 384 \
    --distill-epochs 4000 --refine-epochs 16000 --rar-epochs 0 --lbfgs 150 \
    --tag ldg_J_ritz1pct --outdir $R

# 5. Simpson-quadrature continuation in fp64 on a 321^2 grid (about 30 h on a 32-core CPU)
CUDA_VISIBLE_DEVICES= python $H/harness_ldg_multi.py --source $R/ldg_C_full.pt \
    --init-ckpt $R/ldg_J_ritz1pct.pt --census $R/census6.json --census-tag census6 \
    --numsol 6 --distill-epochs 0 --adam-epochs 0 --lbfgs 200 --final-mse-lbfgs 0 \
    --quad 193 --quad-final 321 --quad-scheme simpson --dmin 0.8 \
    --tag ldg_J5_simpson --outdir $R
```

The Allen–Cahn benchmark is a single run: AdamW for 40,000 epochs, then 100 L-BFGS steps in double precision (under an hour on a V100).

```bash
python $H/harness_ac.py --dmin 0.2 --mse --lbfgs 100 --tag allencahn_v2 --outdir $R
```

The launchers of all other experiments are listed in the two benchmark tables above.

## Repository layout

```
📦DeflationPINNs
 ┣ 📜main.py                      original experiment (see above)
 ┣ 📜config.py                    paths, relative to the working directory
 ┣ 📜metricsAndPics4Paper.ipynb   evaluation of a main.py model against the FEM data
 ┣ 📂data
 ┃ ┗ 📂trueSolution               FEM solutions of the six LdG states (data_LDG_*_solution.mat)
 ┣ 📂src
 ┃ ┣ 📂architectures
 ┃ ┃ ┗ 📜DeflationPINN.py         shared trunk, learnable branch weights, hard Dirichlet constraint
 ┃ ┣ 📂lossFunctions
 ┃ ┃ ┣ 📜DeflationLoss.py         hinge deflation loss
 ┃ ┃ ┣ 📜DeflationPINNLoss.py     residual + deflation loss
 ┃ ┃ ┗ 📜LDGPINNLoss.py           LdG residual, trapezoidal boundary data and its radial extension
 ┃ ┣ 📂starDomainExtrapolation
 ┃ ┃ ┗ 📜starDomain.py            star-shaped domains for the radial extension (paper, App. B)
 ┃ ┣ 📜harmonicTrapezoidalExtension.py   harmonic extension of the boundary data
 ┃ ┗ 📜molifiedTrapezoidExtension.py     mollified extension of the boundary data
 ┣ 📂tests
 ┃ ┣ 📂deflationPINNTest          training loop, hyperparameters (run.py) and plotting for main.py
 ┃ ┗ 📂pictures
 ┃   ┗ 📂deflationPINNTest        plots written by main.py
 ┣ 📂models                       main.py saves trained models here
 ┗ 📂revision2_experiments        paper experiments
   ┣ 📂harness                    training, reference solver, certification, d_min bisection
   ┣ 📂launchers                  exact run scripts of every experiment
   ┣ 📂analysis                   error norms and contours, wall certification, paper plots
   ┗ 📜README.md                  file-by-file mapping to the manuscript
```

## Cite the paper


[Link to arXiv](http://arxiv.org/abs/2603.27936)
BibTeX:

```
@misc{disarò2026deflationpinnslearningmultiplesolutions,
      title={Deflation-PINNs: Learning Multiple Solutions for PDEs and Landau-de Gennes}, 
      author={Sean Disarò and Ruma Rani Maity and Aras Bacho},
      year={2026},
      eprint={2603.27936},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2603.27936}, 
}
```

## Team

| Name        | Email                 |
|-------------|-----------------------|
| Sean Disarò | seandisaro@gmail.com  |
| Aras Bacho  | bacho@caltech.edu     |
| Ruma Maity  | rumamaity081@gmail.com|

## Questions
If you have any questions, please write an email to seandisaro@gmail.com

## License
This project is licensed under the GNU license. You may use it however you want!


