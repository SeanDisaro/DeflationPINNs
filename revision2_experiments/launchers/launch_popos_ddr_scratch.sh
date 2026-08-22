#!/bin/bash
# pop-os: can the DISCOVERY stage be done directly with Deflation-Deep-Ritz?
#
# Rationale: duplicates survive residual discovery because the branches are far
# from the solutions, so two coarse fields in one basin can still satisfy the
# hinge. At DDR convergence a duplicate would have distance -> 0, which the
# deflation term forbids. Three variants:
#   S1  fixed d_min = 0.4        (published value, no discovery stage)
#   S2  d_min ramped 0.4 -> 0.9  (undistorted descent first, separation later)
#   S3  scale-free inverse-power repulsion (NO d_min threshold at all)
cd /home/arasbacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export DDE_BACKEND=pytorch MPLBACKEND=Agg LDG_DEVICE=cuda:0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/arasbacho/miniconda3/bin/python
H=/home/arasbacho/DeflationPINNs_rev/harness
R=/home/arasbacho/DeflationPINNs_rev/results
L=/home/arasbacho/DeflationPINNs_rev/logs

common="--scratch --census $R/none.json --numsol 6 --no-fp64 --fourier 128 --fsigma 16 \
  --quad 192 --quad-final 320 --quad-scheme simpson \
  --distill-epochs 0 --adam-epochs 20000 --lbfgs 120 --final-mse-lbfgs 0 --source $R/ldg_C_full.pt"

$PY $H/harness_ldg_multi.py $common --dmin 0.4 \
  --tag ddr_scratch_S1 --outdir $R > $L/ddr_scratch_S1.log 2>&1
LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= CKPT=$R/ddr_scratch_S1.pt TAG=ddr_scratch_S1 \
  $PY $H/census_smooth.py > $L/ddr_scratch_S1_census.log 2>&1

$PY $H/harness_ldg_multi.py $common --dmin 0.4 --dmin-final 0.9 \
  --tag ddr_scratch_S2 --outdir $R > $L/ddr_scratch_S2.log 2>&1
LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= CKPT=$R/ddr_scratch_S2.pt TAG=ddr_scratch_S2 \
  $PY $H/census_smooth.py > $L/ddr_scratch_S2_census.log 2>&1

$PY $H/harness_ldg_multi.py $common --dmin 0.5 --repulsion inverse --rep-p 2 --delta 1 \
  --tag ddr_scratch_S3 --outdir $R > $L/ddr_scratch_S3.log 2>&1
LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= CKPT=$R/ddr_scratch_S3.pt TAG=ddr_scratch_S3 \
  $PY $H/census_smooth.py > $L/ddr_scratch_S3_census.log 2>&1

echo DDR_SCRATCH_DONE > $L/ddr_scratch_done.marker
