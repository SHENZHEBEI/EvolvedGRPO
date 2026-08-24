#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
EASYR1_ROOT="${ROOT_DIR}/EasyR1"

# Keep the standalone entry point consistent with every augmentation launcher.
# shellcheck disable=SC1091
source "${ROOT_DIR}/augmentation_env.sh"

MODEL_PATH="${1:-${AUG_SOLVER_MODEL:-${ROOT_DIR}/models/Qwen2.5-VL-7B-Instruct}}"
DATASET="${2:-${ROOT_DIR}/dataset/train_original}"
TEST_DATASET="${3:-${ROOT_DIR}/dataset/validation}"
OUTPUT_DIR="${4:-${ROOT_DIR}/answer_model_runs/round_1}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-augmentation/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
ANSWER_EXPECTED_STEPS="${ANSWER_EXPECTED_STEPS:-30}"
ANSWER_MAX_PROMPT_LENGTH="${ANSWER_MAX_PROMPT_LENGTH:-3072}"
ANSWER_MAX_RESPONSE_LENGTH="${ANSWER_MAX_RESPONSE_LENGTH:-2048}"
ANSWER_MAX_MODEL_LEN="${ANSWER_MAX_MODEL_LEN:-$((ANSWER_MAX_PROMPT_LENGTH + ANSWER_MAX_RESPONSE_LENGTH))}"
ANSWER_MAX_NUM_BATCHED_TOKENS="${ANSWER_MAX_NUM_BATCHED_TOKENS:-8192}"

for positive_integer in \
  "${ANSWER_MAX_PROMPT_LENGTH}" "${ANSWER_MAX_RESPONSE_LENGTH}" \
  "${ANSWER_MAX_MODEL_LEN}" "${ANSWER_MAX_NUM_BATCHED_TOKENS}"; do
  if ! [[ "${positive_integer}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Answer context settings must be positive integers; got ${positive_integer}." >&2
    exit 2
  fi
done
if ((ANSWER_MAX_MODEL_LEN < ANSWER_MAX_PROMPT_LENGTH + ANSWER_MAX_RESPONSE_LENGTH)); then
  echo "ANSWER_MAX_MODEL_LEN must cover prompt + response: ${ANSWER_MAX_PROMPT_LENGTH} + ${ANSWER_MAX_RESPONSE_LENGTH}." >&2
  exit 2
fi
if ((ANSWER_MAX_NUM_BATCHED_TOKENS < ANSWER_MAX_MODEL_LEN)); then
  echo "ANSWER_MAX_NUM_BATCHED_TOKENS must be at least ANSWER_MAX_MODEL_LEN." >&2
  exit 2
fi

if [[ "${NUM_GPUS}" != "8" ]]; then
  echo "The answer-model profile requires exactly 8 A800 GPUs; got NUM_GPUS=${NUM_GPUS}." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing repository environment. Run: source ./augmentation_env.sh" >&2
  exit 1
fi
for dataset_dir in "${DATASET}" "${TEST_DATASET}"; do
  if [[ ! -f "${dataset_dir}/data.json" ]]; then
    echo "Dataset directory must contain data.json: ${dataset_dir}" >&2
    exit 1
  fi
done
answer_dataset_validation_args=(
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_multimodal_dataset.py"
  "${DATASET}"
)
if [[ -n "${ANSWER_EXPECTED_DATASET_ROWS:-}" ]]; then
  answer_dataset_validation_args+=(--expected-rows "${ANSWER_EXPECTED_DATASET_ROWS}")
fi
"${answer_dataset_validation_args[@]}"
"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_multimodal_dataset.py" \
  "${TEST_DATASET}"
"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
  "${MODEL_PATH}" --expected-model-type qwen2_5_vl

export CUDA_VISIBLE_DEVICES="${ANSWER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${EASYR1_ROOT}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

runtime_cleaned=0
training_local_tmp=""

stop_answer_runtime() {
  if [[ -x "${ROOT_DIR}/.venv-augmentation/bin/ray" ]]; then
    "${ROOT_DIR}/.venv-augmentation/bin/ray" stop \
      --grace-period "${ANSWER_RAY_STOP_GRACE_PERIOD:-30}" \
      >/dev/null 2>&1 || \
      "${ROOT_DIR}/.venv-augmentation/bin/ray" stop --force >/dev/null 2>&1 || true
  fi
  "${PYTHON_BIN}" -m data_augmentation.cleanup_stale_question_runtime \
    --repo-root "${ROOT_DIR}"
}

cleanup_local_training_tmp() {
  if [[ -z "${training_local_tmp}" || ! -e "${training_local_tmp}" ]]; then
    return
  fi
  resolved_training_tmp="$(realpath -m -- "${training_local_tmp}")"
  case "${resolved_training_tmp}" in
    /tmp/nips-answer-runtime-*) rm -rf -- "${resolved_training_tmp}" ;;
    *) echo "Refusing to remove unexpected answer temp path: ${resolved_training_tmp}" >&2 ;;
  esac
  training_local_tmp=""
}

cleanup_answer_runtime() {
  local cleanup_status=0
  if [[ "${runtime_cleaned}" == "1" ]]; then
    return
  fi
  runtime_cleaned=1
  set +e
  stop_answer_runtime || cleanup_status=$?
  cleanup_local_training_tmp
  if command -v nvidia-smi >/dev/null 2>&1; then
    remaining_gpu_processes="$(nvidia-smi \
      --query-compute-apps=pid,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "${remaining_gpu_processes//[[:space:]]/}" ]]; then
      echo "Other GPU compute processes remain after repository answer-runtime cleanup (PID, MiB):" >&2
      echo "${remaining_gpu_processes}" >&2
    else
      echo "Answer-model GPU cleanup verified: no compute process remains."
    fi
  fi
  set -e
  return "${cleanup_status}"
}

on_exit() {
  local status=$?
  local cleanup_status=0
  trap - EXIT INT TERM
  cleanup_answer_runtime || cleanup_status=$?
  if [[ "${status}" -eq 0 && "${cleanup_status}" -ne 0 ]]; then
    status="${cleanup_status}"
  fi
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT TERM

# An eight-GPU answer phase is exclusive. Clear a failed previous Ray/vLLM
# invocation before allocating eight actor/rollout copies.
stop_answer_runtime

"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/check_runtime.py" \
  --require-gpus "${NUM_GPUS}" \
  --gpu-profile a800

mkdir -p "${OUTPUT_DIR}"
run_log_dir="${ANSWER_TRAINING_LOG_ROOT:-${TRAINING_LOG_ROOT:-${ROOT_DIR}/training_logs}}/answer_model/$(basename "${OUTPUT_DIR}")"
export WANDB_MODE=offline
export WANDB_DIR="${run_log_dir}"
export STEP_REWARD_LOG_PATH="$(realpath -m -- "${OUTPUT_DIR}/step_rewards.jsonl")"
export STEP_REWARD_LOG_APPEND=0
mkdir -p "${WANDB_DIR}"

echo "Starting EasyR1 answer training."
echo "  start model=${MODEL_PATH}"
echo "  train data=${DATASET}"
echo "  validation=${TEST_DATASET}"
echo "  output=${OUTPUT_DIR}"
echo "  rollout=512 prompts x 5 responses on 8 A800 GPUs"
echo "  context=${ANSWER_MAX_PROMPT_LENGTH} prompt + ${ANSWER_MAX_RESPONSE_LENGTH} response = ${ANSWER_MAX_MODEL_LEN} tokens"
echo "  validation=once after the round (not after every training step)"
echo "  offline logs=${run_log_dir}"
echo "  per-step rewards=${STEP_REWARD_LOG_PATH}"

cd "${EASYR1_ROOT}"
training_local_tmp="$(mktemp -d "/tmp/nips-answer-runtime-${UID:-$(id -u)}-XXXXXX")"
echo "Node-local multiprocessing temp: ${training_local_tmp}"
TMPDIR="${training_local_tmp}" \
TMP="${training_local_tmp}" \
TEMP="${training_local_tmp}" \
"${PYTHON_BIN}" -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${DATASET}" \
  data.val_files="${TEST_DATASET}" \
  data.seed="${ANSWER_TRAIN_SEED:-1}" \
  data.max_prompt_length="${ANSWER_MAX_PROMPT_LENGTH}" \
  data.max_response_length="${ANSWER_MAX_RESPONSE_LENGTH}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.rollout.tensor_parallel_size=1 \
  worker.rollout.max_model_len="${ANSWER_MAX_MODEL_LEN}" \
  worker.rollout.max_num_batched_tokens="${ANSWER_MAX_NUM_BATCHED_TOKENS}" \
  trainer.experiment_name="$(basename "${OUTPUT_DIR}")" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.val_before_train=false \
  trainer.val_freq=-1 \
  trainer.max_steps="${ANSWER_EXPECTED_STEPS}" \
  trainer.load_checkpoint_path=null \
  trainer.save_checkpoint_path="${OUTPUT_DIR}"

# Release all CUDA/Ray state before the CPU-only FSDP merger.
cleanup_answer_runtime

"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_step_reward_log.py" \
  "${STEP_REWARD_LOG_PATH}" \
  --expected-steps "${ANSWER_EXPECTED_STEPS}" \
  --expected-model-path "${MODEL_PATH}"

checkpoint_tracker="${OUTPUT_DIR}/latest_global_step.txt"
if [[ ! -f "${checkpoint_tracker}" ]]; then
  echo "Training completed but no checkpoint tracker was written: ${checkpoint_tracker}" >&2
  exit 1
fi
trained_step="$(tr -d '[:space:]' <"${checkpoint_tracker}")"
if [[ "${trained_step}" != "${ANSWER_EXPECTED_STEPS}" ]]; then
  echo "Expected answer training to save global_step_${ANSWER_EXPECTED_STEPS}; tracker contains: ${trained_step}" >&2
  exit 1
fi
actor_dir="${OUTPUT_DIR}/global_step_${trained_step}/actor"
if [[ ! -d "${actor_dir}" ]]; then
  echo "Latest answer actor checkpoint is missing: ${actor_dir}" >&2
  exit 1
fi
for ((rank = 0; rank < NUM_GPUS; rank++)); do
  if [[ ! -s "${actor_dir}/model_world_size_${NUM_GPUS}_rank_${rank}.pt" ]]; then
    echo "Answer checkpoint is incomplete; missing/empty actor model shard ${rank}: ${actor_dir}" >&2
    exit 1
  fi
done
"${PYTHON_BIN}" scripts/model_merger.py --local_dir "${actor_dir}"
"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
  "${actor_dir}/huggingface" --expected-model-type qwen2_5_vl
echo "Merged answer model: ${actor_dir}/huggingface"
