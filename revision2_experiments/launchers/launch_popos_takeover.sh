#!/bin/bash
# pop-os takeover of the DGX workload (DGX driver re-wedged; box needs hardware attention).
# Remaining convergence runs + featdefl + S1-S3 reruns.
# Fixes vs the OOMed first attempt: correct torch-2.10 allocator env name
# (PYTORCH_ALLOC_CONF) and quad-final 320 -> 256.
# Censuses run pipelined on CPU in the background while the next training holds the GPU.
# K16 is intentionally skipped: 16 branches at width 4000 need the 32GB V100.
cd /home/arasbacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export DDE_BACKEND=pytorch MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/arasbacho/miniconda3/bin/python
H=/home/arasbacho/DeflationPINNs_rev/harness
R=/home/arasbacho/DeflationPINNs_rev/results
L=/home/arasbacho/DeflationPINNs_rev/logs
SUM=$L/convergence_summary.log
echo "== pop-os takeover queue started $(date) ==" >> "$SUM"

census_bg () { # $1 tag  $2 census script  $3 K
  ( LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= CKPT=$R/$1.pt TAG=$1 K=$3 \
      $PY $H/$2 > "$L/$1_census.log" 2>&1
    echo "$1 census: $(grep STATES "$L/$1_census.log" | tail -1)" >> "$SUM" ) &
}

conv () { # $1 tag  $2.. extra args
  local tag=$1; shift
  LDG_DEVICE=cuda:0 $PY $H/harness_ldg.py --deflation full --fast --epochs 10000 \
    --lr 1e-4 --dmin 0.4 --eval-every 0 --tag "$tag" --outdir $R "$@" > "$L/$tag.log" 2>&1
  echo "$tag: $(grep DONE "$L/$tag.log" | tail -1 | cut -c1-200)" >> "$SUM"
  if [ -f "$R/$tag.pt" ]; then
    census_bg "$tag" census_any.py 6
  else
    echo "$tag: NO CHECKPOINT ($(tail -1 "$L/$tag.log" | cut -c1-100))" >> "$SUM"
  fi
}

conv conv_grid25 --grid 25
conv conv_grid49 --grid 49
conv conv_grid65 --grid 65
conv conv_w2000 --grid 33 --width 2000

LDG_DEVICE=cuda:0 $PY $H/harness_ldg.py --deflation feature --fast --epochs 10000 --grid 33 \
  --lr 1e-4 --dmin 0.4 --eval-every 0 --tag conv_featdefl --outdir $R > "$L/conv_featdefl.log" 2>&1
echo "conv_featdefl: $(grep DONE "$L/conv_featdefl.log" | tail -1 | cut -c1-200)" >> "$SUM"
if [ -f "$R/conv_featdefl.pt" ]; then
  census_bg conv_featdefl census_any.py 6
else
  echo "conv_featdefl: NO CHECKPOINT ($(tail -1 "$L/conv_featdefl.log" | cut -c1-100))" >> "$SUM"
fi

common="--scratch --census $R/none.json --numsol 6 --no-fp64 --fourier 128 --fsigma 16 \
  --quad 192 --quad-final 256 --quad-scheme simpson \
  --distill-epochs 0 --adam-epochs 20000 --lbfgs 120 --final-mse-lbfgs 0 --source $R/ldg_C_full.pt"

srun () { # $1 tag  $2.. extra args
  local tag=$1; shift
  LDG_DEVICE=cuda:0 $PY $H/harness_ldg_multi.py $common "$@" --tag "$tag" --outdir $R > "$L/$tag.log" 2>&1
  if [ -f "$R/$tag.pt" ]; then
    census_bg "$tag" census_smooth.py 6
  else
    echo "$tag: NO CHECKPOINT ($(tail -1 "$L/$tag.log" | cut -c1-100))" >> "$SUM"
  fi
}

srun pop_scratch_S1 --dmin 0.4
srun pop_scratch_S2 --dmin 0.4 --dmin-final 0.9
srun pop_scratch_S3 --dmin 0.5 --repulsion inverse --rep-p 2 --delta 1

echo "K16 skipped on pop-os (16 branches x width 4000 needs 32GB; pending DGX repair)" >> "$SUM"
wait
echo POPOS_TAKEOVER_DONE > "$L/popos_takeover_done.marker"
