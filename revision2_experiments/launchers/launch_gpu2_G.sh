#!/bin/bash
# GPU 2: run G = basin-safe refinement. Warm-start from ldg_C_full.pt (all six
# basins correct), then MSE loss + denser collocation + fp64 L-BFGS polish.
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

$PY $H/harness_ldg.py --deflation full --fast --fp64-polish --mse \
  --warm-start $R/ldg_C_full.pt --grid 65 --corner-refine --dmin 0.4 \
  --lr 1e-4 --sched step,2000,0.7 --epochs 8000 \
  --lbfgs 100 --eval-every 1000 \
  --tag ldg_G_refine --outdir $R > $L/ldg_G_refine.log 2>&1
echo "GPU2 G DONE" > $L/gpu2_G_done.marker
