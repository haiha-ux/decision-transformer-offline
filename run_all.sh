#!/bin/bash
# Whole grid, strictly sequential, low priority. Seed-major so that a time cut drops seeds, not datasets.
cd "$(dirname "$0")"
PY=.venv/bin/python
for seed in 0 1 2; do
  for ds in hopper-medium halfcheetah-medium walker2d-medium hopper-simple hopper-medium-expert; do
    for algo in dt bc pbc; do
      [ -f ckpt/${ds}_${algo}_s${seed}.pt ] && continue
      while [ "$(awk '{print int($1)}' /proc/loadavg)" -gt 30 ]; do sleep 60; done
      nice -n 19 ionice -c3 timeout 1500 $PY train.py --algo $algo --dataset $ds --seed $seed >> logs/run_all.log 2>&1 \
        || nice -n 19 ionice -c3 timeout 1500 $PY train.py --algo $algo --dataset $ds --seed $seed >> logs/run_all.log 2>&1
    done
  done
  nice -n 19 ionice -c3 $PY eval.py >> logs/eval_all.log 2>&1
done
