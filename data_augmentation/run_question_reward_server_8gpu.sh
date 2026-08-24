#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}"

ANSWER_MODEL="${1:-${AUG_SOLVER_MODEL:-${ROOT_DIR}/models/Qwen2.5-VL-7B-Instruct}}"
BASE_PORT="${2:-8765}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-augmentation/bin/python}"
IMAGE_MODEL="${IMAGE_MODEL:-${ROOT_DIR}/models/Qwen-Image-Edit-2511}"
CLIP_MODEL="${QUESTION_CLIP_MODEL:-${ROOT_DIR}/models/clip-vit-large-patch14}"
PROCESSOR_MODEL="${PROCESSOR_MODEL:-${ANSWER_MODEL}}"
CACHE_DIR="${QUESTION_REWARD_CACHE:-${ROOT_DIR}/augmentation_runs/question_reward_cache}"
NUM_GPUS="${NUM_GPUS:-8}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing repository environment. Run: source ./augmentation_env.sh" >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "Missing setsid (normally provided by util-linux); it is required for leak-free reward cleanup." >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
  "${ANSWER_MODEL}" --expected-model-type qwen2_5_vl
if [[ -d "${PROCESSOR_MODEL}" && ! -s "${PROCESSOR_MODEL}/preprocessor_config.json" ]]; then
  echo "Answer processor checkpoint is incomplete: ${PROCESSOR_MODEL}/preprocessor_config.json" >&2
  exit 1
fi
if [[ -d "${IMAGE_MODEL}" && ! -s "${IMAGE_MODEL}/model_index.json" ]]; then
  echo "Qwen image-edit checkpoint is incomplete: ${IMAGE_MODEL}/model_index.json" >&2
  exit 1
fi
if [[ -d "${CLIP_MODEL}" && ! -s "${CLIP_MODEL}/config.json" ]]; then
  echo "CLIP checkpoint is incomplete: ${CLIP_MODEL}/config.json" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/EasyQ1${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${CACHE_DIR}/logs"
pids=()
cleanup() {
  trap - EXIT INT TERM
  local pid live
  for pid in "${pids[@]}"; do
    # Every shard is a session/process-group leader. Kill the complete group so
    # vLLM EngineCore multiprocessing children cannot survive the HTTP parent.
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
  # Bound shutdown time even if a CUDA child ignores SIGTERM.
  for ((attempt = 1; attempt <= 50; attempt++)); do
    live=0
    for pid in "${pids[@]}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then
        live=1
        break
      fi
    done
    if [[ "${live}" == "0" ]]; then
      break
    fi
    sleep 0.1
  done
  for pid in "${pids[@]}"; do
    if kill -0 -- "-${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  port=$((BASE_PORT + gpu))
  # Reward inference is a separate GPU phase: actor vLLM/FSDP and reference
  # weights are asleep/offloaded until the HTTP response returns.  Use most of
  # each A800 here, then release it again before actor/ref computation resumes.
  # A separate session gives cleanup ownership of every vLLM descendant. V1
  # multiprocessing is unnecessary for tensor_parallel_size=1 and disabling it
  # avoids a second orphanable Python process per reward shard.
  CUDA_VISIBLE_DEVICES="${gpu}" \
  VLLM_ENABLE_V1_MULTIPROCESSING="${QUESTION_REWARD_VLLM_MULTIPROCESSING:-0}" \
  setsid "${PYTHON_BIN}" -m data_augmentation.question_reward_server \
    --answer-model "${ANSWER_MODEL}" \
    --processor-model "${PROCESSOR_MODEL}" \
    --image-model "${IMAGE_MODEL}" \
    --clip-model "${CLIP_MODEL}" \
    --cache-dir "${CACHE_DIR}" \
    --host "${QUESTION_REWARD_HOST:-127.0.0.1}" \
    --port "${port}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${QUESTION_REWARD_GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-model-len "${QUESTION_REWARD_MAX_MODEL_LEN:-16384}" \
    --max-num-seqs "${QUESTION_REWARD_MAX_NUM_SEQS:-256}" \
    --max-num-batched-tokens "${QUESTION_REWARD_MAX_NUM_BATCHED_TOKENS:-16384}" \
    --max-tokens "${QUESTION_REWARD_MAX_TOKENS:-1024}" \
    --votes 5 \
    --min-agree-votes 2 \
    --consensus-threshold 0.4 \
    --image-num-inference-steps "${QUESTION_IMAGE_STEPS:-20}" \
    >"${CACHE_DIR}/logs/server_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

echo "Started ${NUM_GPUS} question-reward shards on ports ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))."
echo "Logs: ${CACHE_DIR}/logs/server_gpu_*.log"
echo "Keep this process running; Ctrl-C stops every shard."

# A serving shard should never exit on its own.  If any shard stops, the EXIT
# trap terminates the remaining shards so the trainer cannot silently run with
# only part of the advertised endpoint list.
set +e
wait -n "${pids[@]}"
status=$?
set -e
if [[ "${status}" -eq 0 ]]; then
  status=1
fi
exit "${status}"
