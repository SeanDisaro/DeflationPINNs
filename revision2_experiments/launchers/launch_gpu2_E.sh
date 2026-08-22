#!/bin/bash
# GPU 2 (exclusive): high-accuracy run E.
# fp32 Adam with eps-continuation -> fp64 L-BFGS polish; harmonic extension;
# squared-residual loss; p=64, 2x1024 trunk; corner/edge-refined collocation.
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
  --p 64 --layers 2 --width 1024 --grid 65 --corner-refine \
  --lr 1e-3 --sched step,2000,0.6 --eps-stages "0.04:6000,0.028:6000,0.02:10000" \
  --lbfgs 150 --eval-every 2000 \
  --tag ldg_E_accurate --outdir $R > $L/ldg_E_accurate.log 2>&1
echo "GPU2 E DONE" > $L/gpu2_E_done.marker
