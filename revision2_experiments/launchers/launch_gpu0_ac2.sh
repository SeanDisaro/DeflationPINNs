#!/bin/bash
# GPU 0: AllenCahn rerun with corrected dmin, MSE loss, fp64 L-BFGS polish.
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

$PY $H/harness_ac.py --dmin 0.2 --mse --lbfgs 100 --tag allencahn_v2 --outdir $R > $L/allencahn_v2.log 2>&1
echo "GPU0 AC2 DONE" > $L/gpu0_ac2_done.marker
