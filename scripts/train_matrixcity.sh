#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-/path/to/matrix_city/small_city/aerial/train/block_all}"
TEST_DATA="${TEST_DATA:-/path/to/matrix_city/small_city/aerial/test/block_all_test}"
OUT="${OUT:-/path/to/output/matrixcity}"
NPROC="${NPROC:-4}"

mkdir -p "$OUT"

torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" train.py \
    --bsz 4 -s "$DATA" --resolution -1 --images images \
    --model_path "$OUT" --iterations 120000 \
    --test_iterations 999999 --save_iterations 120000 \
    --densify_grad_threshold 0.0001 --densify_until_iter 32000 --simp_iteration2 34000 \
    --spawn_gate models/spawn_gate_ALL4.pkl --spawn_gate_k 0.60 --spawn_gate_start 24000 \
    --sampling_factor 0.7

torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" render_c.py \
    -m "$OUT" -s "$TEST_DATA" --images images --resolution -1 --skip_train --eval --iteration 120000

python metrics.py -m "$OUT" --mode test --num_gpus "$NPROC"
