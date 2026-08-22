#!/bin/bash
# GPU 2: J3 = guarded all-MSE fp64 L-BFGS finish from run J's 2% checkpoint.
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

$PY $H/harness_ldg_multi.py --source $R/ldg_C_full.pt --init-ckpt $R/ldg_J_ritz1pct.pt \
  --census $R/census6.json --census-tag census6 --numsol 6 \
  --distill-epochs 0 --adam-epochs 0 --lbfgs 0 --final-mse-lbfgs 150 \
  --quad 192 --quad-final 320 --dmin 0.8 \
  --tag ldg_J3_msefinish --outdir $R > $L/ldg_J3_msefinish.log 2>&1
echo "GPU2 J3 DONE" > $L/gpu2_J3_done.marker
