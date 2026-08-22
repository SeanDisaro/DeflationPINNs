#!/bin/bash
# J5cpu: Simpson L-BFGS continuation of the J DDR solution, CPU fp64
# (GPU fp32 attempt OOMed at 321^2 on the 8GB card; fp64 on this card is 1/64-rate,
#  but the box has 32 cores -- CPU fp64 matches J4's double-precision semantics).
cd /home/arasbacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export DDE_BACKEND=pytorch MPLBACKEND=Agg
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
PY=/home/arasbacho/miniconda3/bin/python
H=/home/arasbacho/DeflationPINNs_rev/harness
R=/home/arasbacho/DeflationPINNs_rev/results
L=/home/arasbacho/DeflationPINNs_rev/logs
SUM=$L/convergence_summary.log

LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= $PY $H/harness_ldg_multi.py --source $R/ldg_C_full.pt \
  --init-ckpt $R/ldg_J_ritz1pct.pt --census $R/census6_dgx.json --census-tag census6 \
  --numsol 6 --distill-epochs 0 --adam-epochs 0 --lbfgs 200 --final-mse-lbfgs 0 \
  --quad 193 --quad-final 321 --quad-scheme simpson --dmin 0.8 \
  --tag ldg_J5_simpson --outdir $R > $L/ldg_J5_simpson.log 2>&1

if [ -f $R/ldg_J5_simpson.pt ]; then
  LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= CKPT=$R/ldg_J5_simpson.pt TAG=ldg_J5_simpson K=6 \
    $PY $H/census_smooth.py > $L/ldg_J5_simpson_census.log 2>&1
  echo "J5 census: $(grep STATES $L/ldg_J5_simpson_census.log | tail -1)" >> $SUM
  echo "J5 rels: $(grep -o "pre_rel=[0-9.]*" $L/ldg_J5_simpson_census.log | tr "\n" " ")" >> $SUM
else
  echo "J5cpu: NO CHECKPOINT ($(tail -1 $L/ldg_J5_simpson.log | cut -c1-100))" >> $SUM
fi
echo J5_DONE > $L/j5cpu_done.marker
