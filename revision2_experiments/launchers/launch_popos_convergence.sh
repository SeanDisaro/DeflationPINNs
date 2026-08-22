#!/bin/bash
# pop-os: Referee-1 convergence study + feature-space deflation + K16 over-generation.
# Moved here from the DGX (driver wedged in D-state since ~Aug 10; needs reboot).
# Waits for the S1/S2/S3 DDR-scratch chain to finish, then runs sequentially on cuda:0.
# Single 8GB card: grid65 and K16 may OOM -> each run is independent, chain continues.
cd /home/arasbacho/DeflationPINNs_rev/DeflationPINNs-dev || exit 1
export DDE_BACKEND=pytorch MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/arasbacho/miniconda3/bin/python
H=/home/arasbacho/DeflationPINNs_rev/harness
R=/home/arasbacho/DeflationPINNs_rev/results
L=/home/arasbacho/DeflationPINNs_rev/logs

# ---- wait for the S-chain (marker written by launch_popos_ddr_scratch.sh) ----
miss=0
while true; do
  [ -f "$L/ddr_scratch_done.marker" ] && break
  if ! pgrep -f "harness_ldg_multi|census_smooth" > /dev/null; then
    miss=$((miss+1))
    if [ $miss -ge 3 ]; then
      echo "WARN: S-chain not running and no marker; starting anyway" >> "$L/convergence_summary.log"
      break
    fi
  else
    miss=0
  fi
  sleep 300
done
echo "convergence queue started $(date)" >> "$L/convergence_summary.log"

run () {  # $1 tag  $2.. extra args
  local tag=$1; shift
  LDG_DEVICE=cuda:0 $PY $H/harness_ldg.py --deflation full --fast \
    --epochs 10000 --lr 1e-4 --dmin 0.4 --eval-every 0 --tag "$tag" --outdir $R \
    "$@" > "$L/$tag.log" 2>&1
  if [ -f "$R/$tag.pt" ]; then
    LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= CKPT=$R/$tag.pt TAG=$tag \
      $PY $H/census_any.py > "$L/${tag}_census.log" 2>&1
    echo "$tag: $(grep DONE "$L/$tag.log" | tail -1 | cut -c1-160)" >> "$L/convergence_summary.log"
    echo "$tag census: $(grep STATES "$L/${tag}_census.log" | tail -1)" >> "$L/convergence_summary.log"
  else
    echo "$tag: NO CHECKPOINT (see $tag.log: $(tail -1 "$L/$tag.log" | cut -c1-120))" >> "$L/convergence_summary.log"
  fi
}

# feature-space deflation first (R1 point 7, pairs with S1-S3)
LDG_DEVICE=cuda:0 $PY $H/harness_ldg.py --deflation feature --fast \
  --epochs 10000 --grid 33 --lr 1e-4 --dmin 0.4 --eval-every 0 \
  --tag conv_featdefl --outdir $R > "$L/conv_featdefl.log" 2>&1
if [ -f "$R/conv_featdefl.pt" ]; then
  LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= CKPT=$R/conv_featdefl.pt TAG=conv_featdefl \
    $PY $H/census_any.py > "$L/conv_featdefl_census.log" 2>&1
  echo "featdefl census: $(grep STATES "$L/conv_featdefl_census.log" | tail -1)" >> "$L/convergence_summary.log"
else
  echo "featdefl: NO CHECKPOINT" >> "$L/convergence_summary.log"
fi

# collocation sweep (33^2 = existing canonical run C)
run conv_grid17 --grid 17
run conv_grid25 --grid 25
run conv_grid49 --grid 49
# width sweep (4000 = existing canonical run C)
run conv_w500  --grid 33 --width 500
run conv_w1000 --grid 33 --width 1000
run conv_w2000 --grid 33 --width 2000
# OOM-risk tail on the 8GB card
run conv_grid65 --grid 65
echo CONVERGENCE_DONE >> "$L/convergence_summary.log"

# K=16 over-generation at d_min = 0.4 (moved from dead DGX chain)
LDG_DEVICE=cuda:0 $PY $H/harness_ldg.py --deflation full --fast --numsol 16 \
  --epochs 12000 --grid 33 --lr 1e-4 --dmin 0.4 --eval-every 0 \
  --tag ldg_K16_dmin04 --outdir $R > "$L/ldg_K16_dmin04.log" 2>&1
if [ -f "$R/ldg_K16_dmin04.pt" ]; then
  LDG_DEVICE=cpu CUDA_VISIBLE_DEVICES= CKPT=$R/ldg_K16_dmin04.pt TAG=census_K16 \
    $PY $H/census_any.py > "$L/census_K16.log" 2>&1
  echo "K16 census: $(grep STATES "$L/census_K16.log" | tail -1)" >> "$L/convergence_summary.log"
else
  echo "K16: NO CHECKPOINT (likely OOM on 8GB; rerun on DGX after reboot)" >> "$L/convergence_summary.log"
fi
echo QUEUE_DONE > "$L/popos_convergence_done.marker"
