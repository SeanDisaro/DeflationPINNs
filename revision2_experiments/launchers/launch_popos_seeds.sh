#!/bin/bash
# pop-os: how many of the six states does ONE K=6 run at the published protocol
# recover? Distribution over seeds + union coverage.
cd /home/arasbacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export DDE_BACKEND=pytorch MPLBACKEND=Agg LDG_DEVICE=cuda:0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/arasbacho/miniconda3/bin/python
H=/home/arasbacho/DeflationPINNs_rev/harness
R=/home/arasbacho/DeflationPINNs_rev/results
L=/home/arasbacho/DeflationPINNs_rev/logs
for s in 2 3 4 5 6 7; do
  T=ldg_C_seed$s
  $PY $H/harness_ldg.py --deflation full --fast --epochs 10000 --grid 33 --lr 1e-4 \
    --dmin 0.4 --seed $s --eval-every 0 --tag $T --outdir $R > $L/$T.log 2>&1
  LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= CKPT=$R/$T.pt TAG=$T $PY $H/census_any.py > $L/${T}_census.log 2>&1
  echo "seed $s: $(grep STATES $L/${T}_census.log | tail -1)" >> $L/seed_study_summary.log
done
echo SEEDS_DONE > $L/seed_study_done.marker
