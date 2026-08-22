#!/bin/bash
# GPU 2 (exclusive): accuracy run D with memory headroom.
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

$PY $H/harness_ldg.py --deflation full --fast --epochs 50000 --grid 65 --corner-refine \
  --lr 1e-3 --sched step,5000,0.8 --lbfgs 300 --eval-every 5000 \
  --tag ldg_D_accurate --outdir $R > $L/ldg_D_accurate.log 2>&1
echo "GPU2 D DONE" > $L/gpu2_D_done.marker
