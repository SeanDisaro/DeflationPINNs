#!/bin/bash
# pop-os (RTX 4060 Ti): rebuild the discovery stage after the DGX went offline.
# 1) K=6 discovery (feeds the DDR pipeline)  2) K=11 discovery (wall states)  3) census
cd /home/arasbacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export DDE_BACKEND=pytorch MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/arasbacho/miniconda3/bin/python
H=/home/arasbacho/DeflationPINNs_rev/harness
R=/home/arasbacho/DeflationPINNs_rev/results
L=/home/arasbacho/DeflationPINNs_rev/logs

$PY $H/harness_ldg.py --deflation full --fast --epochs 10000 --grid 33 \
  --lr 1e-4 --dmin 0.4 --eval-every 2500 --tag ldg_C_full --outdir $R > $L/ldg_C_full.log 2>&1
echo "C DONE" > $L/popos_C_done.marker

$PY $H/harness_ldg.py --deflation full --fast --numsol 11 --epochs 12000 --grid 33 \
  --lr 1e-4 --dmin 0.4 --eval-every 4000 --tag ldg_K11_discovery --outdir $R > $L/ldg_K11_discovery.log 2>&1
echo "K11 DONE" > $L/popos_K11_done.marker

$PY $H/certify_census.py --ckpt $R/ldg_K11_discovery.pt --numsol 11 --tag census_K11 > $L/census_K11.log 2>&1
echo "CENSUS DONE" > $L/popos_census_done.marker
