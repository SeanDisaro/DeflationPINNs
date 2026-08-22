#!/bin/bash
# GPU 0: run I = self-distill + deflated Deep-Ritz energy descent + MSE/L-BFGS polish.
cd /home/bacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export CUDA_VISIBLE_DEVICES=0
export DDE_BACKEND=pytorch
export MPLBACKEND=Agg
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/bacho/anaconda3/bin/python
H=/home/bacho/DeflationPINNs_rev/harness
R=/home/bacho/DeflationPINNs_rev/results
L=/home/bacho/DeflationPINNs_rev/logs

$PY $H/harness_ldg_delta.py --source $R/ldg_C_full.pt --refine-mode ritz \
  --refine-epochs 8000 --rar-epochs 3000 --lbfgs 120 \
  --tag ldg_I_ritz --outdir $R > $L/ldg_I_ritz.log 2>&1
echo "GPU0 I DONE" > $L/gpu0_I_done.marker
