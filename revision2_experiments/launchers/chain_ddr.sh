#!/bin/bash
# GPU 2: run exclusive Deep-Ritz rounds until all six stable states are below 1%
# relative error. Escalates Simpson quadrature when a round plateaus.
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

maxrel () {  # max stable-state rel error from a run json
  $PY - "$1" << 'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))["match"]
print(max(v["rel"] for k, v in m.items() if k in ("D1","D2","R1","R2","R3","R4")))
PYEOF
}

# wait for J4 (up to 12h)
for i in $(seq 1 1440); do
  [ -f "$L/gpu2_J4_done.marker" ] && break
  sleep 30
done
CKPT=$R/ldg_J4_simpson.pt
JSON=$R/ldg_J4_simpson.json
PREV=$(maxrel $JSON)
echo "chain: J4 maxrel=$PREV" >> $L/chain_ddr.log

for r in 1 2 3 4 5 6; do
  ok=$($PY -c "print(1 if float('$PREV') <= 0.01 else 0)")
  [ "$ok" = "1" ] && { echo "chain: target reached at $PREV" >> $L/chain_ddr.log; break; }
  QF=385; [ $r -ge 3 ] && QF=513; [ $r -ge 5 ] && QF=641
  TAG=ldg_J5_r$r
  $PY $H/harness_ldg_multi.py --source $R/ldg_C_full.pt --init-ckpt $CKPT \
    --census $R/census6.json --census-tag census6 --numsol 6 \
    --distill-epochs 0 --adam-epochs 0 --lbfgs 160 --final-mse-lbfgs 0 \
    --quad 193 --quad-final $QF --quad-scheme simpson --dmin 0.8 \
    --tag $TAG --outdir $R > $L/$TAG.log 2>&1
  CKPT=$R/$TAG.pt
  CUR=$(maxrel $R/$TAG.json)
  echo "chain: round $r quad=$QF maxrel=$CUR (prev $PREV)" >> $L/chain_ddr.log
  # plateau -> force escalation next round by skipping ahead
  imp=$($PY -c "print(1 if float('$CUR') > 0.92*float('$PREV') else 0)")
  [ "$imp" = "1" ] && [ $r -lt 3 ] && r=2
  PREV=$CUR
done
echo "CHAIN DONE maxrel=$PREV" >> $L/chain_ddr.log
echo done > $L/chain_ddr_done.marker
