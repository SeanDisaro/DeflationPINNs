#!/bin/bash
# GPU 0: re-evals of A/B/C checkpoints (residual/energy fix), then benchmarks.
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

for t in A_legacy B_second C_full; do
  $PY $H/harness_ldg.py --deflation full --eval-only $R/ldg_$t.pt \
    --tag ldg_${t}_reeval --outdir $R > $L/ldg_${t}_reeval.log 2>&1
done
$PY $H/harness_ac.py --tag allencahn --outdir $R > $L/allencahn.log 2>&1
$PY $H/harness_reaction.py --epochs 50000 --dmin 0.5 --tag reaction1d_dmin05 --outdir $R > $L/reaction1d_dmin05.log 2>&1
echo "GPU0 BENCH DONE" > $L/gpu0_bench_done.marker
