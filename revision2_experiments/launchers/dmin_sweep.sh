#!/bin/bash
# pop-os: d_min sensitivity of the discovery stage.
# The six true states are pairwise >= 1.075 apart (full vector-field RMS metric),
# so d_min = 0.4 (published value) permits two branches to satisfy deflation while
# sitting in the SAME basin. Sweep d_min and count DISTINCT certified states.
cd /home/arasbacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export DDE_BACKEND=pytorch MPLBACKEND=Agg LDG_DEVICE=cuda:0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/arasbacho/miniconda3/bin/python
H=/home/arasbacho/DeflationPINNs_rev/harness
R=/home/arasbacho/DeflationPINNs_rev/results
L=/home/arasbacho/DeflationPINNs_rev/logs

run_one () {  # $1 = dmin, $2 = seed, $3 = tag
  $PY $H/harness_ldg.py --deflation full --fast --epochs 10000 --grid 33 --lr 1e-4 \
    --dmin $1 --seed $2 --eval-every 0 --tag $3 --outdir $R > $L/$3.log 2>&1
  LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= CKPT=$R/$3.pt TAG=$3 \
    $PY $H/census_any.py > $L/${3}_census.log 2>&1
  echo "$3 (dmin=$1 seed=$2): $(grep STATES $L/${3}_census.log | tail -1)" >> $L/dmin_sweep_summary.log
}

# wait for the GPU to be free of the ablation arms
for i in $(seq 1 240); do pgrep -f "[h]arness_ldg.py" >/dev/null || break; sleep 30; done

run_one 0.9 0 ldg_C_dmin09
run_one 0.6 0 ldg_C_dmin06
run_one 0.4 1 ldg_C_dmin04_s1
echo SWEEP_DONE > $L/dmin_sweep_done.marker
