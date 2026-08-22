#!/bin/bash
# GPU 2: run J = Deflation-Deep-Ritz at 1%-target.
# Unbiased midpoint quadrature (192^2 Adam, 384^2 fp64 L-BFGS), Fourier features
# for the eps-scale structures, energy descent end-to-end.
cd /home/bacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export CUDA_VISIBLE_DEVICES=2
export DDE_BACKEND=pytorch
export MPLBACKEND=Agg
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/bacho/anaconda3/bin/python
H=/home/bacho/DeflationPINNs_rev/harness
R=/home/bacho/DeflationPINNs_rev/results
L=/home/bacho/DeflationPINNs_rev/logs

$PY $H/harness_ldg_delta.py --source $R/ldg_C_full.pt \
  --refine-mode ritz --polish ritz --fourier 128 --fsigma 16 \
  --quad 192 --quad-final 384 \
  --distill-epochs 4000 --refine-epochs 16000 --rar-epochs 0 --lbfgs 150 \
  --tag ldg_J_ritz1pct --outdir $R > $L/ldg_J_ritz1pct.log 2>&1
echo "GPU2 J DONE" > $L/gpu2_J_done.marker
