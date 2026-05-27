#!/usr/bin/env bash
# BlitzGS — official training + evaluation script

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="${DATASET:-residence}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"
OUTPUT_ROOT="${SCRIPT_DIR}/output"
LOG_ROOT="${SCRIPT_DIR}/logs"
NPROC="${NPROC_PER_NODE:-4}"
MAX_ITERS="${MAX_ITERS:-100000}"
RENDER_SPLIT="${RENDER_SPLIT:-test}"

DATASET_PATH="${DATA_ROOT}/${DATASET}"
MODEL_PATH="${OUTPUT_ROOT}/${DATASET}2_$(printf '%dk' $((MAX_ITERS/1000)))"
TRAIN_LOG="${LOG_ROOT}/${DATASET}_train.log"
EVAL_LOG="${LOG_ROOT}/${DATASET}_eval.log"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

if [[ ! -d "${DATASET_PATH}/${RENDER_SPLIT}" ]]; then
    echo "[ERROR] ${DATASET_PATH}/${RENDER_SPLIT} not found" >&2
    exit 1
fi


echo "[TRAIN] ${DATASET}  ->  ${MODEL_PATH}"


DEBUG_CUDA="${DEBUG_CUDA:-0}"
LAUNCH_BLOCKING_ENV=""
if [[ "${DEBUG_CUDA}" == "1" ]]; then
    LAUNCH_BLOCKING_ENV="CUDA_LAUNCH_BLOCKING=1 BLITZGS_FINITE_CHECK=1"
fi

NCCL_ASYNC_ERROR_HANDLING=1 TORCH_SHOW_CPP_STACKTRACES=1 ${LAUNCH_BLOCKING_ENV} \
torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC}" train.py \
    --bsz 4 \
    -s "${DATASET_PATH}" \
    --resolution 4 \
    --model_path "${MODEL_PATH}" \
    --iterations "${MAX_ITERS}" \
    --images train/rgbs \
    --test_iterations 999999 \
    --densify_until_iter 24000 \
    > "${TRAIN_LOG}" 2>&1 \
    && echo "[TRAIN OK]" \
    || { echo "[TRAIN FAILED] see ${TRAIN_LOG}"; exit 1; }


echo "[RENDER] split=${RENDER_SPLIT}"


torchrun --nproc_per_node="${NPROC}" render_c.py \
    -m "${MODEL_PATH}" \
    -s "${DATASET_PATH}/${RENDER_SPLIT}" \
    --images images \
    --resolution 4 \
    --iteration "${MAX_ITERS}" \
    --skip_train --eval \
    >> "${EVAL_LOG}" 2>&1 \
    || { echo "[RENDER FAILED] see ${EVAL_LOG}"; exit 1; }

python metrics.py -m "${MODEL_PATH}" --mode "${RENDER_SPLIT}" --num_gpus "${NPROC}" \
    >> "${EVAL_LOG}" 2>&1 \
    || { echo "[METRICS FAILED] see ${EVAL_LOG}"; exit 1; }

echo "Done."
