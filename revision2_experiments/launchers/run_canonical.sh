#!/bin/bash
# pop-os: canonical run set. Everything keyed on real output files.
cd /home/arasbacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export DDE_BACKEND=pytorch MPLBACKEND=Agg LDG_DEVICE=cuda:0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/arasbacho/miniconda3/bin/python
H=/home/arasbacho/DeflationPINNs_rev/harness
R=/home/arasbacho/DeflationPINNs_rev/results
L=/home/arasbacho/DeflationPINNs_rev/logs
wait_file () { for i in $(seq 1 1440); do [ -f "$1" ] && return 0; sleep 30; done; return 1; }

wait_file $R/ldg_C_full.pt || exit 1
# 1) flow census on the canonical discovery run (CPU): Table 1 cols 2 and 4
LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= $PY $H/census6_from_flow.py > $L/census6.log 2>&1

wait_file $R/ldg_K11_discovery.pt || true
# 2) ablation arms for the appendix (canonical, same machine)
$PY $H/harness_ldg.py --deflation legacy --fast --epochs 10000 --grid 33 --lr 1e-4 \
  --dmin 0.4 --eval-every 0 --tag ldg_A_legacy --outdir $R > $L/ldg_A_legacy.log 2>&1
$PY $H/harness_ldg.py --deflation second --fast --epochs 10000 --grid 33 --lr 1e-4 \
  --dmin 0.4 --eval-every 0 --tag ldg_B_second --outdir $R > $L/ldg_B_second.log 2>&1

# 3) DDR (run-J equivalent) with per-branch accumulation, fp32
$PY $H/harness_ldg_multi.py --source $R/ldg_C_full.pt --census $R/census6.json \
  --census-tag census6 --numsol 6 --no-fp64 --fourier 128 --fsigma 16 \
  --quad 192 --quad-final 320 --quad-scheme simpson \
  --distill-epochs 4000 --adam-epochs 16000 --lbfgs 200 --final-mse-lbfgs 0 \
  --dmin 0.8 --tag ldg_DDR --outdir $R > $L/ldg_DDR.log 2>&1

# 4) chain pure-Ritz rounds until every stable state is below 1%
CKPT=$R/ldg_DDR.pt
for r in 1 2 3 4 5; do
  MAX=$($PY -c "
import json,sys
try:
    m=json.load(open('$R/ldg_DDR${SUF}.json'))['match']
    print(max(v['rel'] for k,v in m.items() if k in ('D1','D2','R1','R2','R3','R4')))
except Exception: print(1.0)" 2>/dev/null)
  echo "before round $r: maxrel=$MAX" >> $L/canonical_chain.log
  $PY -c "import sys;sys.exit(0 if float('$MAX')<=0.01 else 1)" && break
  QF=320; [ $r -ge 2 ] && QF=384; [ $r -ge 4 ] && QF=448
  SUF=_r$r
  $PY $H/harness_ldg_multi.py --source $R/ldg_C_full.pt --init-ckpt $CKPT \
    --census $R/census6.json --census-tag census6 --numsol 6 --no-fp64 \
    --fourier 128 --fsigma 16 --quad 192 --quad-final $QF --quad-scheme simpson \
    --distill-epochs 0 --adam-epochs 0 --lbfgs 200 --final-mse-lbfgs 0 \
    --dmin 0.8 --tag ldg_DDR$SUF --outdir $R > $L/ldg_DDR$SUF.log 2>&1
  CKPT=$R/ldg_DDR$SUF.pt
done
echo CANONICAL_DONE > $L/canonical_done.marker
