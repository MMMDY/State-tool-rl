#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TAU_BENCH_OUT_DIR="${TAU_BENCH_OUT_DIR:-${SCRIPT_DIR}/../outputs}"

MODEL_DIR="${MODEL_DIR:-${TAU_BENCH_OUT_DIR}/models/Qwen3-4B-Instruct-2507}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30001}"
# Keep these GPUs separate from training CUDA_VISIBLE_DEVICES.
GPUS="${GPUS:-0}"
TP="${TP:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${SGLANG_PYTHON}" ]]; then
  echo "Missing SGLang Python: ${SGLANG_PYTHON}" >&2
  exit 1
fi
export PATH="$(dirname "${SGLANG_PYTHON}"):${PATH}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

if [ ! -d "${MODEL_DIR}" ]; then
  echo "Missing model directory: ${MODEL_DIR}"
  echo "Download first (example): huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir \"${MODEL_DIR}\""
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPUS}" "${SGLANG_PYTHON}" -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP}" \
  --mem-fraction-static "${MEM_FRACTION}"
