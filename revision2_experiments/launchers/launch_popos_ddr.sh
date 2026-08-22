#!/bin/bash
# pop-os phase 2: DDR (run-J equivalent) once the K11 GPU work is done, then
# repeat Ritz rounds until every stable state is below 1% relative error.
cd /home/arasbacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export DDE_BACKEND=pytorch MPLBACKEND=Agg LDG_DEVICE=cuda:0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/arasbacho/miniconda3/bin/python
H=/home/arasbacho/DeflationPINNs_rev/harness
R=/home/arasbacho/DeflationPINNs_rev/results
L=/home/arasbacho/DeflationPINNs_rev/logs

for i in $(seq 1 240); do [ -f "$L/popos_K11_done.marker" ] && break; sleep 30; done

# run-J equivalent: self-distill -> deflated Deep-Ritz -> Ritz L-BFGS (fp32)
$PY $H/harness_ldg_delta.py --source $R/ldg_C_full.pt \
  --refine-mode ritz --polish ritz --no-fp64 --fourier 128 --fsigma 16 \
  --quad 192 --quad-final 320 \
  --distill-epochs 4000 --refine-epochs 16000 --rar-epochs 0 --lbfgs 200 \
  --tag ldg_J_popos --outdir $R > $L/ldg_J_popos.log 2>&1
echo "J DONE" > $L/popos_J_done.marker

CKPT=$R/ldg_J_popos.pt
for r in 1 2 3 4; do
  QF=320; [ $r -ge 2 ] && QF=384; [ $r -ge 3 ] && QF=448
  $PY $H/harness_ldg_delta.py --source $R/ldg_C_full.pt --init-ckpt $CKPT \
    --refine-mode ritz --polish ritz --no-fp64 --fourier 128 --fsigma 16 \
    --quad 192 --quad-final $QF \
    --distill-epochs 0 --refine-epochs 0 --rar-epochs 0 --lbfgs 200 \
    --tag ldg_J_popos_r$r --outdir $R > $L/ldg_J_popos_r$r.log 2>&1
  CKPT=$R/ldg_J_popos_r$r.pt
  MAX=$($PY -c "
import json;d=json.load(open('$R/ldg_J_popos_r$r.json'))['rel_L2_vs_reference513'];print(max(d.values()))" 2>/dev/null || echo 1)
  echo "round $r quad=$QF maxrel=$MAX" >> $L/popos_ddr_chain.log
  $PY -c "import sys;sys.exit(0 if float('$MAX')<=0.01 else 1)" && break
done
echo "DDR CHAIN DONE" > $L/popos_ddr_done.marker
