#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/qwen3-4B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
GPUS="${GPUS:-0}"
TP="${TP:-1}"
# Reserve room for CUDA Graph capture; this produces higher decode throughput
# than a larger KV cache with graphs disabled on a 32 GB RTX 5090.
MEM_FRACTION="${MEM_FRACTION:-0.80}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-256}"
SGLANG_MODEL_NAME="${SGLANG_MODEL_NAME:-qwen3-4b}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 2 4 8 16}"

CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
if [[ ! -x "${SGLANG_PYTHON}" ]]; then
  echo "Missing SGLang Python: ${SGLANG_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing Qwen3-4B model directory: ${MODEL_DIR}" >&2
  exit 1
fi

export PATH="$(dirname "${SGLANG_PYTHON}"):${PATH}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_FORCE_NATIVE_CUDA_OPS="${SGLANG_FORCE_NATIVE_CUDA_OPS:-1}"

# Torch 2.9.1 loads CUDA wheels from both the service venv and the base
# environment. Make those libraries visible before importing SGLang.
CUDA_PYTHON_LIB_PATHS=()
for CUDA_PYTHON_LIB_ROOT in \
  "${ROOT_DIR}/.venv/lib/python3.12/site-packages/nvidia" \
  "/root/miniconda3/lib/python3.12/site-packages/nvidia"; do
  for CUDA_PYTHON_LIB_DIR in "${CUDA_PYTHON_LIB_ROOT}"/*/lib; do
    [[ -d "${CUDA_PYTHON_LIB_DIR}" ]] || continue
    CUDA_PYTHON_LIB_PATHS+=("${CUDA_PYTHON_LIB_DIR}")
  done
done
if ((${#CUDA_PYTHON_LIB_PATHS[@]})); then
  CUDA_PYTHON_LIB_PATH="$(IFS=:; echo "${CUDA_PYTHON_LIB_PATHS[*]}")"
  export LD_LIBRARY_PATH="${CUDA_PYTHON_LIB_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

read -r -a CUDA_GRAPH_BATCH_SIZES <<< "${CUDA_GRAPH_BS}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${SGLANG_PYTHON}" -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SGLANG_MODEL_NAME}" \
  --tp "${TP}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --max-queued-requests "${MAX_QUEUED_REQUESTS}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --cuda-graph-bs "${CUDA_GRAPH_BATCH_SIZES[@]}" \
  --enable-tokenizer-batch-encode \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --sampling-backend "${SAMPLING_BACKEND}" \
  --tool-call-parser qwen25 \
  --reasoning-parser qwen3
