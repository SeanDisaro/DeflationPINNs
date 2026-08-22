# Revision-2 experiment code (CMAME-D-26-00791)

Code for the round-2 revision experiments of *"Deflation-PINNs: Learning Multiple
Solutions for PDEs and Landau-de Gennes"*. Everything here was used to produce the
numbers, tables and figures added in the second revision; raw run outputs
(checkpoints, censuses, reference solutions) are archived separately.

## Layout

- `harness/` — training and certification code
  - `harness_ldg.py` — LdG discovery stage (Deflation-PINN; deflation variants
    `legacy`/`second`/`full`/`feature`; `--grid`, `--width`, `--seed` sweeps)
  - `harness_ldg_multi.py` — Deflation--Deep-Ritz refinement (per-branch smooth
    networks, midpoint/Simpson quadrature, `--scratch` single-stage mode,
    annealed `--dmin-final`, `--repulsion inverse` variant)
  - `harness_ac.py` — Allen--Cahn benchmark (unit disk, K=3)
  - `ldg_reference.py` — mesh-converged FD reference solver (damped Newton)
  - `flow_final.py` — semi-implicit LdG gradient flow (basin census backbone)
  - `census_any.py` / `census_smooth.py` / `census6_from_flow.py` — flow+Newton
    basin censuses for the two architectures (Tables 1, 5, 6)
  - `certify_census.py` — 3-stage certification (Newton -> coarse LM ->
    eps-continuation; wall-state identification)
  - `bisect_dmin.py` — automated d_min selection by bisection (Table 7 /
    Appendix D; floor-test classifier with two-seed confirmation)
- `launchers/` — the exact run scripts for every experiment set (DGX `launch_gpu*`,
  pop-os `launch_popos_*`, DDR chain `chain_ddr.sh`, canonical set `run_canonical.sh`)
- `analysis/` — post-processing: error contours + L2/Linf/H1 + localization
  (`error_analysis.py`), refined-stage norms (`refined_norms.py`), wall
  certification (`certify_walls*.py`), paper plots (`plot_*.py`)

## Mapping to the revised manuscript

| Result | Code |
|---|---|
| Table 1 (three-stage errors) | `harness_ldg.py`, `harness_ldg_multi.py`, `census6_from_flow.py`, `analysis/refined_norms.py` |
| Figs. 3-4 (error contours, localization) | `analysis/error_analysis.py` |
| Table 4 (collocation/width convergence) + feature-space deflation | `launchers/launch_popos_takeover.sh`, `harness_ldg.py --deflation feature` |
| Table 5 (restart-with-certification) | `launchers/launch_popos_seeds.sh`, `census_any.py` |
| Table 6 (d_min bisection) | `harness/bisect_dmin.py` |
| Single-stage DDR / annealed d_min / repulsion (Appendix D) | `launchers/launch_popos_ddr_scratch.sh` |
| K=11 over-generation census | `certify_census.py` |
| Simpson continuation (DDR 1.2-1.3%) | `launchers/launch_popos_J5cpu.sh` |
| Allen--Cahn benchmark (Sec. 3.5) | `harness_ac.py` |

Machine notes: discovery/DDR runs in fp32 on a V100 or RTX 4060 Ti; double-precision
L-BFGS stages on V100 or 32-core CPU. Launchers carry the exact hyperparameters used.
