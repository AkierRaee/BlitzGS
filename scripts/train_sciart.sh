#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-/path/to/sci-art-pixsfm}"
OUT="${OUT:-/path/to/output/sciart}"
NPROC="${NPROC:-4}"

mkdir -p "$OUT"

torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" train.py \
    --bsz 4 -s "$DATA" --resolution 4 --images train/rgbs \
    --model_path "$OUT" --iterations 90000 \
    --test_iterations 999999 --save_iterations 90000 \
    --densify_grad_threshold 0.0001 --densify_until_iter 32000 --simp_iteration2 34000 \
    --spawn_gate models/spawn_gate_LOSO_sciart.pkl --spawn_gate_k 0.60 --spawn_gate_start 24000 \
    --sampling_factor 0.7

torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" render_c.py \
    -m "$OUT" -s "$DATA" --images train/rgbs --resolution 4 --skip_train --iteration 90000

python metrics.py -m "$OUT" --mode test --num_gpus "$NPROC"
