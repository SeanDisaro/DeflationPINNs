#!/bin/bash
# GPU 0: K=11 discovery — over-generate branches so surplus ones must settle on
# unstable (wall) solutions; census certification follows.
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

$PY $H/harness_ldg.py --deflation full --fast --numsol 11 --epochs 12000 \
  --grid 33 --lr 1e-4 --dmin 0.4 --eval-every 3000 \
  --tag ldg_K11_discovery --outdir $R > $L/ldg_K11_discovery.log 2>&1

$PY $H/certify_census.py --ckpt $R/ldg_K11_discovery.pt --numsol 11 \
  --tag census_K11 > $L/census_K11.log 2>&1
echo "GPU0 K11 DONE" > $L/gpu0_K11_done.marker
