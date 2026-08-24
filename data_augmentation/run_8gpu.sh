#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-text}"
AGENT_MODEL="${2:-${AUG_AGENT_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}}"
INPUT_DIR="${3:-${AUG_BASE_DATASET:-dataset/train_original}}"
OUTPUT_DIR="${4:-dataset/train_round1_${MODE//-/_}}"
WORK_DIR="${5:-augmentation_runs/round1_${MODE//-/_}}"
SOLVER_MODEL="${6:-${AUG_SOLVER_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}}"

if [[ "${MODE}" != "text" && "${MODE}" != "image-edit" ]]; then
  echo "MODE must be 'text' or 'image-edit'." >&2
  exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "Missing setsid (normally provided by util-linux); it is required for leak-free 8-GPU cleanup." >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-8}"
ROUND_NAME="${ROUND_NAME:-round1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROCESSOR_MODEL="${PROCESSOR_MODEL:-}"
SOLVER_PROCESSOR_MODEL="${SOLVER_PROCESSOR_MODEL:-}"
IMAGE_MODEL="${IMAGE_MODEL:-${QWEN_IMAGE_EDIT_LOCAL_DIR:-Qwen/Qwen-Image-Edit-2511}}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TRANSFER="${TRANSFER:-hardlink}"
AGENT_MAX_MODEL_LEN="${AGENT_MAX_MODEL_LEN:-16384}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-384}"
AGENT_BATCH_SIZE="${AGENT_BATCH_SIZE:-32}"
AGENT_GPU_MEMORY_UTILIZATION="${AGENT_GPU_MEMORY_UTILIZATION:-0.80}"
AGENT_MAX_NUM_BATCHED_TOKENS="${AGENT_MAX_NUM_BATCHED_TOKENS:-16384}"
AGENT_MAX_GENERATION_ATTEMPTS="${AGENT_MAX_GENERATION_ATTEMPTS:-1}"
AGENT_MIN_PIXELS="${AGENT_MIN_PIXELS:-200704}"
AGENT_MAX_PIXELS="${AGENT_MAX_PIXELS:-1605632}"
TTRL_VOTES_PER_ROUND="${TTRL_VOTES_PER_ROUND:-5}"
TTRL_MIN_VOTE_ROUNDS="${TTRL_MIN_VOTE_ROUNDS:-1}"
TTRL_MAX_VOTE_ROUNDS="${TTRL_MAX_VOTE_ROUNDS:-1}"
TTRL_MIN_VALID_VOTES="${TTRL_MIN_VALID_VOTES:-2}"
TTRL_MIN_AGREE_VOTES="${TTRL_MIN_AGREE_VOTES:-2}"
TTRL_CONSENSUS_THRESHOLD="${TTRL_CONSENSUS_THRESHOLD:-0.4}"
SOLVER_BATCH_SIZE="${SOLVER_BATCH_SIZE:-51}"
SOLVER_GPU_MEMORY_UTILIZATION="${SOLVER_GPU_MEMORY_UTILIZATION:-0.90}"
SOLVER_MAX_MODEL_LEN="${SOLVER_MAX_MODEL_LEN:-8192}"
SOLVER_MAX_NUM_SEQS="${SOLVER_MAX_NUM_SEQS:-256}"
SOLVER_MAX_NUM_BATCHED_TOKENS="${SOLVER_MAX_NUM_BATCHED_TOKENS:-8192}"
SOLVER_MAX_TOKENS="${SOLVER_MAX_TOKENS:-1024}"
SOLVER_DTYPE="${SOLVER_DTYPE:-bfloat16}"
SOLVER_TEMPERATURE="${SOLVER_TEMPERATURE:-0.7}"
SOLVER_TOP_P="${SOLVER_TOP_P:-0.95}"
IMAGE_NUM_INFERENCE_STEPS="${IMAGE_NUM_INFERENCE_STEPS:-20}"
IMAGE_TRUE_CFG_SCALE="${IMAGE_TRUE_CFG_SCALE:-4.0}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.0}"

if [[ -d "${AGENT_MODEL}" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
    "${AGENT_MODEL}" --expected-model-type qwen2_5_vl
fi
if [[ "${MODE}" == "text" && -d "${SOLVER_MODEL}" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
    "${SOLVER_MODEL}" --expected-model-type qwen2_5_vl
fi
if [[ "${MODE}" == "image-edit" && -d "${IMAGE_MODEL}" && ! -s "${IMAGE_MODEL}/model_index.json" ]]; then
  echo "Local Qwen image-edit checkpoint is incomplete: ${IMAGE_MODEL}/model_index.json" >&2
  exit 1
fi

if [[ "${AUG_SKIP_RUNTIME_CHECK:-0}" != "1" ]]; then
  "${PYTHON_BIN}" data_augmentation/check_runtime.py \
    --require-gpus "${NUM_GPUS}" \
    --gpu-profile a800
fi

mkdir -p "${WORK_DIR}/logs"

all_pids=()
cleanup_jobs() {
  trap - EXIT INT TERM
  local pid live
  for pid in "${all_pids[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
  for ((attempt = 1; attempt <= 50; attempt++)); do
    live=0
    for pid in "${all_pids[@]}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then
        live=1
        break
      fi
    done
    [[ "${live}" == "0" ]] && break
    sleep 0.1
  done
  for pid in "${all_pids[@]}"; do
    if kill -0 -- "-${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
  # Catch vLLM/Ray descendants that changed process title or escaped their
  # original process group, and wait for their CUDA contexts to disappear.
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/cleanup_stale_question_runtime.py" \
    --repo-root "${ROOT_DIR}"
}
trap cleanup_jobs EXIT
trap 'cleanup_jobs; exit 130' INT TERM

wait_for_jobs() {
  local stage="$1"
  shift
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  # Every PID passed above has now been reaped. Remove it from the EXIT-owned
  # set so a later OS PID reuse cannot target an unrelated process group.
  local tracked completed matched
  local -a retained_pids=()
  for tracked in "${all_pids[@]}"; do
    matched=0
    for completed in "$@"; do
      if [[ "${tracked}" == "${completed}" ]]; then
        matched=1
        break
      fi
    done
    if [[ "${matched}" == "0" ]]; then
      retained_pids+=("${tracked}")
    fi
  done
  all_pids=("${retained_pids[@]}")
  if [[ "${failed}" -ne 0 ]]; then
    echo "${stage} failed; inspect ${WORK_DIR}/logs and rerun the same command to resume." >&2
    return 1
  fi
}

agent_pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  agent_args=(
    "${PYTHON_BIN}" data_augmentation/augment_dataset.py generate
    --mode "${MODE}"
    --input-dir "${INPUT_DIR}"
    --work-dir "${WORK_DIR}"
    --agent-model "${AGENT_MODEL}"
    --num-shards "${NUM_GPUS}"
    --shard-index "${gpu}"
    --tensor-parallel-size 1
    --gpu-memory-utilization "${AGENT_GPU_MEMORY_UTILIZATION}"
    --batch-size "${AGENT_BATCH_SIZE}"
    --max-num-batched-tokens "${AGENT_MAX_NUM_BATCHED_TOKENS}"
    --max-model-len "${AGENT_MAX_MODEL_LEN}"
    --max-tokens "${AGENT_MAX_TOKENS}"
    --max-generation-attempts "${AGENT_MAX_GENERATION_ATTEMPTS}"
    --min-pixels "${AGENT_MIN_PIXELS}"
    --max-pixels "${AGENT_MAX_PIXELS}"
  )
  if [[ -n "${PROCESSOR_MODEL}" ]]; then
    agent_args+=(--processor-model "${PROCESSOR_MODEL}")
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    agent_args+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [[ "${RESET_SHARDS:-0}" == "1" ]]; then
    agent_args+=(--reset-shard)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" \
  VLLM_ENABLE_V1_MULTIPROCESSING="${AUGMENTATION_VLLM_MULTIPROCESSING:-0}" \
  setsid "${agent_args[@]}" \
    >"${WORK_DIR}/logs/agent_${gpu}.log" 2>&1 &
  agent_pids+=("$!")
  all_pids+=("$!")
done
wait_for_jobs "Agent generation" "${agent_pids[@]}"

if [[ "${MODE}" == "text" ]]; then
  solver_pids=()
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    solver_args=(
      "${PYTHON_BIN}" data_augmentation/augment_dataset.py solve-text
      --input-dir "${INPUT_DIR}"
      --work-dir "${WORK_DIR}"
      --solver-model "${SOLVER_MODEL}"
      --num-shards "${NUM_GPUS}"
      --shard-index "${gpu}"
      --tensor-parallel-size 1
      --dtype "${SOLVER_DTYPE}"
      --gpu-memory-utilization "${SOLVER_GPU_MEMORY_UTILIZATION}"
      --batch-size "${SOLVER_BATCH_SIZE}"
      --max-model-len "${SOLVER_MAX_MODEL_LEN}"
      --max-num-seqs "${SOLVER_MAX_NUM_SEQS}"
      --max-num-batched-tokens "${SOLVER_MAX_NUM_BATCHED_TOKENS}"
      --max-tokens "${SOLVER_MAX_TOKENS}"
      --votes-per-round "${TTRL_VOTES_PER_ROUND}"
      --min-vote-rounds "${TTRL_MIN_VOTE_ROUNDS}"
      --max-vote-rounds "${TTRL_MAX_VOTE_ROUNDS}"
      --min-valid-votes "${TTRL_MIN_VALID_VOTES}"
      --min-agree-votes "${TTRL_MIN_AGREE_VOTES}"
      --consensus-threshold "${TTRL_CONSENSUS_THRESHOLD}"
      --temperature "${SOLVER_TEMPERATURE}"
      --top-p "${SOLVER_TOP_P}"
    )
    if [[ -n "${SOLVER_PROCESSOR_MODEL}" ]]; then
      solver_args+=(--processor-model "${SOLVER_PROCESSOR_MODEL}")
    fi
    if [[ "${TTRL_STORE_RAW_OUTPUTS:-0}" == "1" ]]; then
      solver_args+=(--store-raw-outputs)
    fi
    if [[ "${RESET_SOLVER_SHARDS:-0}" == "1" ]]; then
      solver_args+=(--reset-shard)
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" \
    VLLM_ENABLE_V1_MULTIPROCESSING="${AUGMENTATION_VLLM_MULTIPROCESSING:-0}" \
    setsid "${solver_args[@]}" \
      >"${WORK_DIR}/logs/solver_${gpu}.log" 2>&1 &
    solver_pids+=("$!")
    all_pids+=("$!")
  done
  wait_for_jobs "TTRL answer voting" "${solver_pids[@]}"
fi

if [[ "${MODE}" == "image-edit" ]]; then
  image_pids=()
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    image_args=(
      "${PYTHON_BIN}" data_augmentation/augment_dataset.py edit-images
      --input-dir "${INPUT_DIR}"
      --work-dir "${WORK_DIR}"
      --image-model "${IMAGE_MODEL}"
      --round-name "${ROUND_NAME}"
      --num-shards "${NUM_GPUS}"
      --shard-index "${gpu}"
      --num-inference-steps "${IMAGE_NUM_INFERENCE_STEPS}"
      --true-cfg-scale "${IMAGE_TRUE_CFG_SCALE}"
      --guidance-scale "${IMAGE_GUIDANCE_SCALE}"
    )
    if [[ "${IMAGE_CPU_OFFLOAD:-0}" == "1" ]]; then
      image_args+=(--cpu-offload)
    fi
    if [[ "${RESET_IMAGE_SHARDS:-0}" == "1" ]]; then
      image_args+=(--reset-shard)
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" setsid "${image_args[@]}" \
      >"${WORK_DIR}/logs/image_${gpu}.log" 2>&1 &
    image_pids+=("$!")
    all_pids+=("$!")
  done
  wait_for_jobs "Qwen image editing" "${image_pids[@]}"
fi

finalize_args=(
  "${PYTHON_BIN}" data_augmentation/augment_dataset.py finalize
  --mode "${MODE}"
  --input-dir "${INPUT_DIR}"
  --work-dir "${WORK_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --round-name "${ROUND_NAME}"
  --transfer "${TRANSFER}"
)
if [[ "${ALLOW_PARTIAL:-0}" == "1" ]]; then
  finalize_args+=(--allow-partial)
fi
if [[ "${INCLUDE_ORIGINAL:-0}" == "0" ]]; then
  finalize_args+=(--no-include-original)
else
  finalize_args+=(--include-original)
fi
if [[ "${OVERWRITE_OUTPUT:-0}" == "1" ]]; then
  finalize_args+=(--overwrite)
fi

"${finalize_args[@]}"
echo "New training dataset: ${OUTPUT_DIR}"
