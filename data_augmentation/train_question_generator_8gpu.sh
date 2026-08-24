#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

SOURCE_ROUND="${1:?Usage: train_question_generator_8gpu.sh <source_round> [generator_model] [source_dataset] [output_dir] [answer_model]}"
GENERATOR_MODEL="${2:-${AUG_AGENT_MODEL:-${ROOT_DIR}/models/Qwen2.5-VL-7B-Instruct}}"
SOURCE_DATASET="${3:-${ROOT_DIR}/dataset/iterative_rounds/train_round_${SOURCE_ROUND}}"
OUTPUT_DIR="${4:-${ROOT_DIR}/question_generator_runs/round_${SOURCE_ROUND}}"
ANSWER_MODEL="${5:-${AUG_SOLVER_MODEL:-${ROOT_DIR}/models/Qwen2.5-VL-7B-Instruct}}"
TRAIN_DATASET="${QUESTION_TRAIN_DATASET:-${ROOT_DIR}/dataset/question_generator/round_${SOURCE_ROUND}}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-augmentation/bin/python}"
QUESTION_REWARD_ENDPOINTS="${QUESTION_REWARD_ENDPOINTS:-http://127.0.0.1:8765,http://127.0.0.1:8766,http://127.0.0.1:8767,http://127.0.0.1:8768,http://127.0.0.1:8769,http://127.0.0.1:8770,http://127.0.0.1:8771,http://127.0.0.1:8772}"
TRAIN_SEED="${QUESTION_TRAIN_SEED:-$(date +%s)}"
NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_ROLLOUT_GPU_UTIL="${QUESTION_ACTOR_ROLLOUT_GPU_UTIL:-0.80}"
ACTOR_UPDATE_MICRO_BATCH="${QUESTION_ACTOR_UPDATE_MICRO_BATCH:-8}"
ACTOR_EXPERIENCE_MICRO_BATCH="${QUESTION_ACTOR_EXPERIENCE_MICRO_BATCH:-16}"
export QUESTION_REWARD_GPU_MEMORY_UTILIZATION="${QUESTION_REWARD_GPU_MEMORY_UTILIZATION:-0.85}"
export QUESTION_REWARD_MAX_NUM_SEQS="${QUESTION_REWARD_MAX_NUM_SEQS:-256}"
export QUESTION_REWARD_MAX_NUM_BATCHED_TOKENS="${QUESTION_REWARD_MAX_NUM_BATCHED_TOKENS:-16384}"

if [[ "${NUM_GPUS}" != "8" ]]; then
  echo "The A800 question-generator profile requires NUM_GPUS=8; got ${NUM_GPUS}." >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing repository environment. Run: source ./augmentation_env.sh" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_DATASET}/data.json" ]]; then
  echo "Missing previous-round dataset: ${SOURCE_DATASET}/data.json" >&2
  exit 1
fi

question_source_validation_args=(
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_multimodal_dataset.py"
  "${SOURCE_DATASET}"
)
if [[ -n "${QUESTION_EXPECTED_DATASET_ROWS:-}" ]]; then
  question_source_validation_args+=(--expected-rows "${QUESTION_EXPECTED_DATASET_ROWS}")
fi
"${question_source_validation_args[@]}"

"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
  "${GENERATOR_MODEL}" --expected-model-type qwen2_5_vl
"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
  "${ANSWER_MODEL}" --expected-model-type qwen2_5_vl

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/EasyQ1${PYTHONPATH:+:${PYTHONPATH}}"
export QUESTION_REWARD_ENDPOINTS
export QUESTION_ANSWER_MODEL="${ANSWER_MODEL}"
export CUDA_VISIBLE_DEVICES="${QUESTION_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

reward_server_pid=""
runtime_cleaned=0
training_local_tmp=""

cleanup_reward_server() {
  if [[ -n "${reward_server_pid}" ]]; then
    kill "${reward_server_pid}" 2>/dev/null || true
    wait "${reward_server_pid}" 2>/dev/null || true
    reward_server_pid=""
  fi
}

stop_ray_and_stale_processes() {
  if [[ -x "${ROOT_DIR}/.venv-augmentation/bin/ray" ]]; then
    # SIGTERM first lets multiprocessing close open files before Python removes
    # its temp directory. Ray force-kills only processes that outlive the grace
    # period, avoiding noisy NFS .nfs* "resource busy" finalizer tracebacks.
    "${ROOT_DIR}/.venv-augmentation/bin/ray" stop \
      --grace-period "${QUESTION_RAY_STOP_GRACE_PERIOD:-30}" \
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
    /tmp/nips-question-runtime-*) rm -rf -- "${resolved_training_tmp}" ;;
    *) echo "Refusing to remove unexpected training temp path: ${resolved_training_tmp}" >&2 ;;
  esac
  training_local_tmp=""
}

cleanup_training_runtime() {
  local cleanup_status=0
  if [[ "${runtime_cleaned}" == "1" ]]; then
    return
  fi
  runtime_cleaned=1
  set +e
  cleanup_reward_server
  stop_ray_and_stale_processes || cleanup_status=$?
  cleanup_local_training_tmp
  if command -v nvidia-smi >/dev/null 2>&1; then
    remaining_gpu_processes="$(nvidia-smi \
      --query-compute-apps=pid,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "${remaining_gpu_processes//[[:space:]]/}" ]]; then
      echo "Other GPU compute processes remain after repository question-runtime cleanup (PID, MiB):" >&2
      echo "${remaining_gpu_processes}" >&2
    else
      echo "GPU cleanup verified: no compute process remains."
    fi
  fi
  set -e
  return "${cleanup_status}"
}

on_exit() {
  local status=$?
  local cleanup_status=0
  trap - EXIT INT TERM
  cleanup_training_runtime || cleanup_status=$?
  if [[ "${status}" -eq 0 && "${cleanup_status}" -ne 0 ]]; then
    status="${cleanup_status}"
  fi
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT TERM

# This profile exclusively occupies all eight local GPUs.  Remove the Ray
# cluster and repository-local vLLM children left by an interrupted earlier
# invocation before loading another eight actor copies.  This prevents old
# ~30-GiB EngineCore processes from pushing a 1-TiB node over Ray's 95% RAM
# safety threshold.  Set CLEAN_STALE_QUESTION_RUNTIME=0 only when deliberately
# managing the reward services and Ray lifecycle outside this script.
if [[ "${CLEAN_STALE_QUESTION_RUNTIME:-1}" == "1" ]]; then
  stop_ray_and_stale_processes
fi

"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/check_runtime.py" \
  --require-gpus "${NUM_GPUS}" \
  --gpu-profile a800

reward_health() {
  "${PYTHON_BIN}" -c '
import json, os, urllib.request
urls = [value.strip().rstrip("/") + "/health" for value in os.environ["QUESTION_REWARD_ENDPOINTS"].split(",") if value.strip()]
assert urls, "QUESTION_REWARD_ENDPOINTS is empty"
for url in urls:
    with urllib.request.urlopen(url, timeout=3) as response:
        assert response.status == 200, (url, response.status)
        status = json.load(response)
        assert status.get("answer_model") == os.environ["QUESTION_ANSWER_MODEL"], (url, status.get("answer_model"))
        assert status.get("gpu_idle") is True, (url, "reward model still occupies GPU")
' >/dev/null 2>&1
}

if ! reward_health; then
  if [[ "${AUTO_START_QUESTION_REWARD:-1}" != "1" ]]; then
    echo "Question reward service is unavailable and automatic startup is disabled." >&2
    exit 1
  fi
  case "${QUESTION_REWARD_ENDPOINTS}" in
    *127.0.0.1*|*localhost*) ;;
    *)
      echo "Remote question reward endpoints are unavailable; refusing to start a local replacement." >&2
      exit 1
      ;;
  esac

  echo "Starting single-node reward shards. They release GPU memory before training begins."
  bash "${ROOT_DIR}/data_augmentation/run_question_reward_server_8gpu.sh" \
    "${ANSWER_MODEL}" \
    "${QUESTION_REWARD_BASE_PORT:-8765}" &
  reward_server_pid="$!"

  ready=0
  for ((attempt = 1; attempt <= 180; attempt++)); do
    if reward_health; then
      ready=1
      break
    fi
    if ! kill -0 "${reward_server_pid}" 2>/dev/null; then
      echo "Question reward service exited during startup. Inspect question_reward_cache/logs/." >&2
      exit 1
    fi
    sleep 5
  done
  if [[ "${ready}" != "1" ]]; then
    echo "Question reward service did not become ready within 15 minutes." >&2
    exit 1
  fi
fi

if [[ ! -f "${TRAIN_DATASET}/data.json" || "${REBUILD_QUESTION_DATA:-0}" == "1" ]]; then
  prepare_args=(
    "${PYTHON_BIN}" -m data_augmentation.prepare_question_generator_dataset
    --input-dir "${SOURCE_DATASET}"
    --output-dir "${TRAIN_DATASET}"
    --round-name "round_${SOURCE_ROUND}"
    --image-ratio 0.1
    --seed "$((2025 + SOURCE_ROUND))"
    --transfer "${TRANSFER:-hardlink}"
  )
  if [[ "${REBUILD_QUESTION_DATA:-0}" == "1" ]]; then
    prepare_args+=(--overwrite)
  fi
  "${prepare_args[@]}"
else
  echo "Reuse question-generator dataset: ${TRAIN_DATASET}"
fi

question_train_validation_args=(
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_multimodal_dataset.py"
  "${TRAIN_DATASET}"
)
if [[ -n "${QUESTION_EXPECTED_DATASET_ROWS:-}" ]]; then
  question_train_validation_args+=(--expected-rows "${QUESTION_EXPECTED_DATASET_ROWS}")
fi
"${question_train_validation_args[@]}"

reward_health
run_log_dir="${TRAINING_LOG_ROOT:-${ROOT_DIR}/training_logs}/question_generator/round_${SOURCE_ROUND}"
export WANDB_MODE=offline
export WANDB_DIR="${run_log_dir}"
export TENSORBOARD_DIR="${run_log_dir}/tensorboard"
export STEP_REWARD_LOG_PATH="$(realpath -m -- "${OUTPUT_DIR}/step_rewards.jsonl")"
export STEP_REWARD_LOG_APPEND=0
mkdir -p "${WANDB_DIR}" "${TENSORBOARD_DIR}"

echo "Reward service shards are idle and ready; starting 8-GPU A800 GRPO."
echo "Random 512-row training seed: ${TRAIN_SEED}"
echo "GPU phase plan: actor rollout=${ACTOR_ROLLOUT_GPU_UTIL}, reward=${QUESTION_REWARD_GPU_MEMORY_UTILIZATION}, update micro-batch=${ACTOR_UPDATE_MICRO_BATCH}, experience micro-batch=${ACTOR_EXPERIENCE_MICRO_BATCH}"
echo "Offline logs: ${run_log_dir}"
echo "Per-step rewards: ${STEP_REWARD_LOG_PATH}"

cd "${ROOT_DIR}/EasyQ1"
# Python multiprocessing temp files must be node-local. The repository itself
# can be NFS; deleting an open NFS file creates .nfs* placeholders and caused
# harmless but alarming finalizer tracebacks after successful checkpointing.
training_local_tmp="$(mktemp -d "/tmp/nips-question-runtime-${UID:-$(id -u)}-XXXXXX")"
echo "Node-local multiprocessing temp: ${training_local_tmp}"
TMPDIR="${training_local_tmp}" \
TMP="${training_local_tmp}" \
TEMP="${training_local_tmp}" \
"${PYTHON_BIN}" -m verl.trainer.main \
  config=examples/question_generator.yaml \
  data.train_files="${TRAIN_DATASET}" \
  data.val_files="${TRAIN_DATASET}" \
  data.seed="${TRAIN_SEED}" \
  worker.actor.model.model_path="${GENERATOR_MODEL}" \
  worker.actor.micro_batch_size_per_device_for_update="${ACTOR_UPDATE_MICRO_BATCH}" \
  worker.actor.micro_batch_size_per_device_for_experience="${ACTOR_EXPERIENCE_MICRO_BATCH}" \
  worker.rollout.gpu_memory_utilization="${ACTOR_ROLLOUT_GPU_UTIL}" \
  worker.reward.score_function="${ROOT_DIR}/data_augmentation/question_generator_reward.py:score_batch" \
  worker.reward.score_function_kwargs.endpoint="${QUESTION_REWARD_ENDPOINTS}" \
  trainer.experiment_name="round_${SOURCE_ROUND}" \
  trainer.save_checkpoint_path="${OUTPUT_DIR}" \
  trainer.max_steps=1 \
  trainer.n_gpus_per_node="${NUM_GPUS}"

# Release actor/ref/reward CUDA contexts before the CPU-memory-intensive model
# merge. The EXIT trap performs the same cleanup if training raised an error.
cleanup_training_runtime

"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_step_reward_log.py" \
  "${STEP_REWARD_LOG_PATH}" \
  --expected-steps 1 \
  --expected-model-path "${GENERATOR_MODEL}"

checkpoint_tracker="${OUTPUT_DIR}/latest_global_step.txt"
if [[ ! -f "${checkpoint_tracker}" ]]; then
  echo "Training completed but ${checkpoint_tracker} was not written." >&2
  exit 1
fi
trained_step="$(tr -d '[:space:]' <"${checkpoint_tracker}")"
if [[ "${trained_step}" != "1" ]]; then
  echo "Expected this one-step run to write global_step_1; tracker contains: ${trained_step}" >&2
  exit 1
fi
latest_checkpoint="${OUTPUT_DIR}/global_step_${trained_step}"
if [[ ! -d "${latest_checkpoint}/actor" ]]; then
  echo "Training tracker points to a missing actor checkpoint: ${latest_checkpoint}/actor" >&2
  exit 1
fi
for ((rank = 0; rank < NUM_GPUS; rank++)); do
  if [[ ! -s "${latest_checkpoint}/actor/model_world_size_${NUM_GPUS}_rank_${rank}.pt" ]]; then
    echo "Question checkpoint is incomplete; missing/empty actor model shard ${rank}: ${latest_checkpoint}/actor" >&2
    exit 1
  fi
done
"${PYTHON_BIN}" scripts/model_merger.py --local_dir "${latest_checkpoint}/actor"
"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
  "${latest_checkpoint}/actor/huggingface" --expected-model-type qwen2_5_vl
echo "Merged question-generator model: ${latest_checkpoint}/actor/huggingface"
