#!/bin/bash
# GPU 2: ablation arms A/B (R2.2) + second benchmarks (R2.5), sequential.
cd /home/bacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export CUDA_VISIBLE_DEVICES=2
export DDE_BACKEND=pytorch
export MPLBACKEND=Agg
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/home/bacho/anaconda3/bin/python
H=/home/bacho/DeflationPINNs_rev/harness
R=/home/bacho/DeflationPINNs_rev/results
L=/home/bacho/DeflationPINNs_rev/logs

$PY $H/harness_ldg.py --deflation legacy --tag ldg_A_legacy  --outdir $R > $L/ldg_A_legacy.log 2>&1
$PY $H/harness_ldg.py --deflation second --tag ldg_B_second --outdir $R > $L/ldg_B_second.log 2>&1
$PY $H/harness_ac.py       --tag allencahn  --outdir $R > $L/allencahn.log 2>&1
$PY $H/harness_reaction.py --epochs 50000 --tag reaction1d --outdir $R > $L/reaction1d.log 2>&1
echo "ALL GPU2 RUNS DONE" > $L/gpu2_done.marker
