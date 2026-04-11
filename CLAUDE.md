# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires Python 3.12 and a CUDA GPU.

```bash
pip install -r ./requirements.txt
pip install -e .
```

## Running Experiments

Run the main experiment (currently configured to the 1D reaction-diffusion problem):

```bash
python main.py
```

To switch between experiments, edit [main.py](main.py) — comment/uncomment the relevant `from tests.<experiment>.run import run` line. Currently available experiments:
- `tests/LDG/` — 2D Landau-de Gennes (LDG) liquid crystal problem
- `tests/OneD_reaction/` — 1D steady-state reaction-diffusion problem

To adjust hyperparameters (learning rate, epochs, number of solutions, deflation loss parameters, etc.), edit the `run()` function in the corresponding `tests/<experiment>/run.py`.

## Architecture Overview

The project implements **Deflation-PINNs**: a DeepONet-based architecture that simultaneously learns multiple distinct solutions to a PDE by penalizing solutions that are too similar to each other (deflation loss).

### Model Architecture (`src/architectures/DeflationPINN.py`)

Two model variants exist:
- `two_dim_2_two_dim_DefPINN` — 2D input → 2D output (for LDG)
- `one_dim_DefPINN` — 1D input → 1D output (for reaction-diffusion)

Both share the same design pattern: a shared **trunk network** (MLP) processes spatial coordinates, and per-solution **branch features** (learned vectors, one per solution) are multiplied elementwise with the trunk output to produce each solution. The `numSolutions` parameter controls how many solutions are learned simultaneously.

Optional **Dirichlet hard constraints** can be enforced by multiplying outputs by a boundary-vanishing factor and adding a boundary extension function, ensuring boundary conditions are satisfied exactly rather than through a soft penalty.

### Loss Functions (`src/lossFunctions/`)

- `LDGPINNLoss.py` — PDE residual loss for the 2D LDG problem (computes Laplacian via `torch.autograd.grad` and enforces the LDG Euler-Lagrange equation)
- `OneD_reactionPINNLoss.py` — PDE residual loss for the 1D reaction-diffusion problem
- `DeflationLoss.py` — Linear deflation loss: penalizes pairs of solutions that are too similar. Loss decays to 0 once solutions are separated by `maxDistance`
- `DeflationPINNLoss.py` — Combines PDE loss + deflation loss with weighting coefficients `alpha` (PDE weight) and `delta` (deflation weight)

### Key Loss Parameters

| Parameter | Meaning |
|-----------|---------|
| `alpha` | Weight for PDE residual loss (typically small, e.g. 0.01) |
| `beta` | Weight for soft boundary loss (when not using hard constraints) |
| `delta` | Weight for deflation loss (typically large, e.g. 100) |
| `deflationLossPoints` | `(maxLoss, maxDistance)` — shape of the linear deflation penalty |

### Star Domain Extrapolation (`src/starDomainExtrapolation/starDomain.py`)

Implements the boundary hard constraint via star-shaped domain geometry. The `HyperCuboid` class represents the square domain as a star domain, enabling computation of boundary-vanishing factors in polar coordinates for smooth Dirichlet enforcement.

### Training Loop (`tests/<experiment>/training.py`)

Uses Adam optimizer. If `loadBestModel=True`, saves the best checkpoint (by loss) to `models/` using `dill` pickle and reloads it at the end. Models are saved as `.pkl` files (not `.pt`) due to use of `dill`.

### Data (`data/trueSolution/`)

FEM reference solutions for the LDG problem stored as `.mat` files (`data_LDG_R1–R4_solution.mat` for radial, `data_LDG_D1–D2_solution.mat` for diagonal). Used for error comparison in `metricsAndPics4Paper.ipynb`.

### Paths (`config.py`)

All paths (models directory, data directory, plot output directory) are defined in [config.py](config.py) relative to `os.getcwd()`. Run scripts from the repository root.
