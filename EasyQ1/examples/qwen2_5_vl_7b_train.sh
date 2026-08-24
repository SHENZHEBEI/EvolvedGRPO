#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
EASYQ1_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd -- "${EASYQ1_ROOT}/.." && pwd -P)"

MODEL_PATH="${1:-${AUG_SOLVER_MODEL:-${REPO_ROOT}/models/Qwen2.5-VL-7B-Instruct}}"
DATASET="${2:-${REPO_ROOT}/dataset/train_original}"
TEST_DATASET="${3:-${REPO_ROOT}/dataset/validation}"
OUTPUT_DIR="${4:-${REPO_ROOT}/answer_model_runs/qwen2_5_vl_7b}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-augmentation/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing repository environment. Run: source ./augmentation_env.sh" >&2
  exit 1
fi
if [[ ! -f "${DATASET}/data.json" || ! -f "${TEST_DATASET}/data.json" ]]; then
  echo "Training and validation directories must each contain data.json." >&2
  echo "train=${DATASET}" >&2
  echo "validation=${TEST_DATASET}" >&2
  exit 1
fi

# Keep the global step unchanged: 512 prompts x 5 responses = 2560
# trajectories across eight A800 80GB GPUs.
export CUDA_VISIBLE_DEVICES="${ANSWER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${EASYQ1_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" "${REPO_ROOT}/data_augmentation/check_runtime.py" \
  --require-gpus 8 \
  --gpu-profile a800

mkdir -p "${OUTPUT_DIR}"
run_log_dir="${TRAINING_LOG_ROOT:-${REPO_ROOT}/training_logs}/answer_model/$(basename "${OUTPUT_DIR}")"
export WANDB_MODE=offline
export WANDB_DIR="${run_log_dir}"
export TENSORBOARD_DIR="${run_log_dir}/tensorboard"
mkdir -p "${WANDB_DIR}" "${TENSORBOARD_DIR}"
echo "Offline logs: ${run_log_dir}"

cd "${EASYQ1_ROOT}"
"${PYTHON_BIN}" -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="${DATASET}" \
  data.val_files="${TEST_DATASET}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.rollout.tensor_parallel_size=1 \
  trainer.experiment_name="$(basename "${OUTPUT_DIR}")" \
  trainer.n_gpus_per_node=8 \
  trainer.load_checkpoint_path=null \
  trainer.save_checkpoint_path="${OUTPUT_DIR}"

checkpoint_tracker="${OUTPUT_DIR}/latest_global_step.txt"
if [[ ! -f "${checkpoint_tracker}" ]]; then
  echo "Training finished but no checkpoint tracker was written: ${checkpoint_tracker}" >&2
  exit 1
fi
trained_step="$(tr -d '[:space:]' <"${checkpoint_tracker}")"
actor_dir="${OUTPUT_DIR}/global_step_${trained_step}/actor"
if [[ ! -d "${actor_dir}" ]]; then
  echo "Latest actor checkpoint is missing: ${actor_dir}" >&2
  exit 1
fi
"${PYTHON_BIN}" scripts/model_merger.py --local_dir "${actor_dir}"
echo "Merged answer model: ${actor_dir}/huggingface"
