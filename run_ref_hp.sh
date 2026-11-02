#!/bin/bash
# Side experiment: reference DT hyperparameters (lr 1e-4, 100k steps) on hopper-medium-expert, 2 seeds, sequential.
# Writes ckpt/*_dtref_s*.pt and results/eval_ref_hp.json; the 45-run grid in results/eval.json is untouched.
cd "$(dirname "$0")"
PY=.venv/bin/python
for seed in 0 1; do
  nice -n 19 ionice -c3 $PY train.py --algo dt --dataset hopper-medium-expert --seed $seed --steps 100000 --lr 1e-4 --tag ref >> logs/ref_hp.log 2>&1
done
nice -n 19 ionice -c3 $PY eval.py --glob 'ckpt/hopper-medium-expert_dtref_s*.pt' --out results/eval_ref_hp.json >> logs/ref_hp.log 2>&1
