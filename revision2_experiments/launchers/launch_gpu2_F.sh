#!/bin/bash
# GPU 2 (exclusive): run F = direct eps=0.02, harmonic ext, MSE, p=64, dmin=0.6,
# fp32 Adam then fp64 L-BFGS.
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

$PY $H/harness_ldg.py --deflation full --fast --fp64-polish --mse --harmonic-ext \
  --p 64 --layers 2 --width 1024 --grid 65 --corner-refine --dmin 0.6 \
  --lr 1e-3 --sched step,3000,0.6 --epochs 20000 \
  --lbfgs 200 --eval-every 2000 \
  --tag ldg_F_accurate --outdir $R > $L/ldg_F_accurate.log 2>&1
echo "GPU2 F DONE" > $L/gpu2_F_done.marker
