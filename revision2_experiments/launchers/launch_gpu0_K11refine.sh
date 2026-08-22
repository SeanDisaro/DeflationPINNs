#!/bin/bash
# GPU 0: waits for the K11 census, then runs the mixed sub-1% refinement of all 11 branches.
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

# wait (up to 3h) for the census to finish
for i in $(seq 1 360); do
  [ -f "$R/census_K11.json" ] && break
  sleep 30
done
[ -f "$R/census_K11.json" ] || { echo "census never appeared" > $L/K11refine_error; exit 1; }

$PY $H/harness_ldg_multi.py --source $R/ldg_K11_discovery.pt --census $R/census_K11.json \
  --numsol 11 --quad 192 --quad-final 384 \
  --distill-epochs 4000 --adam-epochs 14000 --lbfgs 150 --final-mse-lbfgs 80 \
  --tag ldg_K11_refine --outdir $R > $L/ldg_K11_refine.log 2>&1
echo "GPU0 K11REFINE DONE" > $L/gpu0_K11refine_done.marker
