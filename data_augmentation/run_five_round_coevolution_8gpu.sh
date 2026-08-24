#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}"

# shellcheck disable=SC1091
source "${ROOT_DIR}/augmentation_env.sh"

TOTAL_ROUNDS="${1:-5}"
if ! [[ "${TOTAL_ROUNDS}" =~ ^[1-5]$ ]]; then
  echo "Usage: run_five_round_coevolution_8gpu.sh [rounds: 1-5]" >&2
  exit 2
fi
if [[ "${NUM_GPUS:-8}" != "8" ]]; then
  echo "Five-round coevolution requires the exclusive 8xA800 profile." >&2
  exit 2
fi

BASE_ANSWER_MODEL="${COEVOLUTION_BASE_ANSWER_MODEL:-${AUG_SOLVER_MODEL}}"
BASE_QUESTION_MODEL="${COEVOLUTION_BASE_QUESTION_MODEL:-${AUG_AGENT_MODEL}}"
BASE_DATASET="${COEVOLUTION_BASE_DATASET:-${AUG_BASE_DATASET}}"
VALIDATION_DATASET="${COEVOLUTION_VALIDATION_DATASET:-${ROOT_DIR}/dataset/validation}"
COEVOLUTION_ROOT="${COEVOLUTION_ROOT:-${ROOT_DIR}/coevolution_5rounds}"
DATA_ROOT="${COEVOLUTION_DATA_ROOT:-${ROOT_DIR}/dataset/coevolution_5rounds}"
WORK_ROOT="${COEVOLUTION_WORK_ROOT:-${ROOT_DIR}/augmentation_runs/coevolution_5rounds}"
ANSWER_RUN_ROOT="${COEVOLUTION_ANSWER_ROOT:-${COEVOLUTION_ROOT}/answer_models}"
QUESTION_RUN_ROOT="${COEVOLUTION_QUESTION_ROOT:-${COEVOLUTION_ROOT}/question_models}"
QUESTION_DATA_ROOT="${COEVOLUTION_QUESTION_DATA_ROOT:-${COEVOLUTION_ROOT}/question_datasets}"
REWARD_CACHE_ROOT="${COEVOLUTION_REWARD_CACHE:-${COEVOLUTION_ROOT}/question_reward_cache}"
LOG_ROOT="${COEVOLUTION_LOG_ROOT:-${ROOT_DIR}/training_logs/coevolution_5rounds}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-augmentation/bin/python}"

cleanup_coevolution_runtime() {
  if [[ -x "${ROOT_DIR}/.venv-augmentation/bin/ray" ]]; then
    "${ROOT_DIR}/.venv-augmentation/bin/ray" stop \
      --grace-period "${COEVOLUTION_RAY_STOP_GRACE_PERIOD:-30}" \
      >/dev/null 2>&1 || \
      "${ROOT_DIR}/.venv-augmentation/bin/ray" stop --force >/dev/null 2>&1 || true
  fi
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/cleanup_stale_question_runtime.py" \
    --repo-root "${ROOT_DIR}"
}

on_exit() {
  local status=$?
  local cleanup_status=0
  trap - EXIT INT TERM
  cleanup_coevolution_runtime || cleanup_status=$?
  if [[ "${status}" -eq 0 && "${cleanup_status}" -ne 0 ]]; then
    status="${cleanup_status}"
  fi
  exit "${status}"
}

trap on_exit EXIT
trap 'exit 130' INT TERM

# The complete workflow owns all eight local GPUs.  Start from a clean state;
# the EXIT trap repeats this for success, failure, and Ctrl-C.
cleanup_coevolution_runtime

for required_file in \
  "${BASE_DATASET}/data.json" \
  "${VALIDATION_DATASET}/data.json"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required coevolution input: ${required_file}" >&2
    exit 1
  fi
done
"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
  "${BASE_ANSWER_MODEL}" --expected-model-type qwen2_5_vl
"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
  "${BASE_QUESTION_MODEL}" --expected-model-type qwen2_5_vl
"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_multimodal_dataset.py" \
  "${BASE_DATASET}" --expected-rows 15616

mkdir -p \
  "${COEVOLUTION_ROOT}" "${DATA_ROOT}" "${WORK_ROOT}" \
  "${ANSWER_RUN_ROOT}" "${QUESTION_RUN_ROOT}" "${QUESTION_DATA_ROOT}" \
  "${REWARD_CACHE_ROOT}" "${LOG_ROOT}"

canonical_path() {
  realpath -m -- "$1"
}

model_is_ready() {
  local model_dir="$1"
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_hf_checkpoint.py" \
    "${model_dir}" --expected-model-type qwen2_5_vl >/dev/null 2>&1
}

latest_actor_dir() {
  local run_dir="$1"
  local expected_step="${2:-}"
  local tracker="${run_dir}/latest_global_step.txt"
  [[ -f "${tracker}" ]] || return 1
  local step
  step="$(tr -d '[:space:]' <"${tracker}")"
  [[ "${step}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid checkpoint tracker: ${tracker}" >&2
    return 1
  }
  if [[ -n "${expected_step}" && "${step}" != "${expected_step}" ]]; then
    echo "Unexpected checkpoint step in ${tracker}: got ${step}, expected ${expected_step}" >&2
    return 1
  fi
  local actor_dir="${run_dir}/global_step_${step}/actor"
  [[ -d "${actor_dir}" ]] || {
    echo "Checkpoint tracker points to a missing actor directory: ${actor_dir}" >&2
    return 1
  }
  printf '%s\n' "${actor_dir}"
}

latest_model_dir() {
  local actor_dir
  actor_dir="$(latest_actor_dir "$1" "${2:-}")" || return 1
  local model_dir="${actor_dir}/huggingface"
  model_is_ready "${model_dir}" || return 1
  printf '%s\n' "${model_dir}"
}

merge_saved_checkpoint_if_needed() {
  local framework_root="$1"
  local run_dir="$2"
  local expected_step="$3"
  local actor_dir
  actor_dir="$(latest_actor_dir "${run_dir}" "${expected_step}")" || return 1
  if model_is_ready "${actor_dir}/huggingface"; then
    return 0
  fi
  for ((rank = 0; rank < 8; rank++)); do
    if [[ ! -s "${actor_dir}/model_world_size_8_rank_${rank}.pt" ]]; then
      echo "Checkpoint is incomplete; missing rank ${rank}: ${actor_dir}" >&2
      return 1
    fi
  done
  echo "Training checkpoint is complete; resume only the model merge: ${actor_dir}"
  (
    cd "${framework_root}"
    "${PYTHON_BIN}" scripts/model_merger.py --local_dir "${actor_dir}"
  )
  model_is_ready "${actor_dir}/huggingface"
}

ensure_lineage() {
  local expected_file="$1"
  local artifact="$2"
  local expected_content="$3"
  mkdir -p "$(dirname -- "${expected_file}")"
  if [[ -f "${expected_file}" ]]; then
    local existing_content
    existing_content="$(<"${expected_file}")"
    if [[ "${existing_content}" != "${expected_content}" ]]; then
      echo "Coevolution lineage mismatch: ${expected_file}" >&2
      echo "Use a new COEVOLUTION_ROOT/DATA_ROOT; do not mix checkpoints from different chains." >&2
      exit 2
    fi
  else
    if [[ -e "${artifact}" && "${COEVOLUTION_ADOPT_UNMARKED:-0}" != "1" ]]; then
      echo "Unmarked existing artifact cannot be safely reused: ${artifact}" >&2
      echo "Choose a new coevolution root or explicitly set COEVOLUTION_ADOPT_UNMARKED=1." >&2
      exit 2
    fi
    printf '%s\n' "${expected_content}" >"${expected_file}"
  fi
}

mark_complete() {
  local expected_file="$1"
  cp -f -- "${expected_file}" "${expected_file%.expected}.complete"
}

join_models() {
  local -n values_ref="$1"
  local joined=""
  local value
  for value in "${values_ref[@]}"; do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${value}"
  done
  printf '%s\n' "${joined}"
}

declare -a ANSWER_MODELS=()
declare -a QUESTION_MODELS=()

echo "Five-round coevolution plan (single node, 8xA800, sequential phases):"
echo "  D1 -> A1 -> Q1 -> D2(Q1,A1) -> A2 -> Q2 -> ... -> D${TOTAL_ROUNDS} -> A${TOTAL_ROUNDS} -> Q${TOTAL_ROUNDS}"
echo "  answer round: one full 15616-row EasyR1 epoch (512 prompts/step, normally 30 steps)"
echo "  question round: one EasyQ1 step (512 prompts x 5 candidates)"
echo "  data root=${DATA_ROOT}"
echo "  model root=${COEVOLUTION_ROOT}"

# D1 is an exact hardlinked materialization of the original dataset.
round_one_dataset="${DATA_ROOT}/train_round_1"
printf -v data_manifest \
  'protocol=aq_coevolution_v1\nkind=data\nround=1\nsource=%s' \
  "$(canonical_path "${BASE_DATASET}")"
data_expected="${DATA_ROOT}/lineage_round_1.expected"
ensure_lineage "${data_expected}" "${round_one_dataset}/data.json" "${data_manifest}"
if [[ ! -f "${round_one_dataset}/data.json" ]]; then
  EXPECTED_DATASET_ROWS=15616 bash "${ROOT_DIR}/data_augmentation/build_round_k.sh" \
    1 "${BASE_QUESTION_MODEL}" "${BASE_DATASET}" "${DATA_ROOT}" "${WORK_ROOT}" "${BASE_ANSWER_MODEL}"
fi
"${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_multimodal_dataset.py" \
  "${round_one_dataset}" --expected-rows 15616
mark_complete "${data_expected}"

for ((round = 1; round <= TOTAL_ROUNDS; round++)); do
  echo
  echo "========== Coevolution round ${round}/${TOTAL_ROUNDS} =========="

  current_dataset="${DATA_ROOT}/train_round_${round}"
  if ((round > 1)); then
    previous_round=$((round - 1))
    previous_dataset="${DATA_ROOT}/train_round_${previous_round}"
    previous_answer="${ANSWER_MODELS[previous_round]}"
    previous_question="${QUESTION_MODELS[previous_round]}"
    printf -v data_manifest \
      'protocol=aq_coevolution_v1\nkind=data\nround=%d\nsource_dataset=%s\nquestion_model=Q%d:%s\nsolver_model=A%d:%s\nimage_ratio=0.1\nimage_steps=20\nttrl_votes=5' \
      "${round}" "$(canonical_path "${previous_dataset}")" \
      "${previous_round}" "$(canonical_path "${previous_question}")" \
      "${previous_round}" "$(canonical_path "${previous_answer}")"
    data_expected="${DATA_ROOT}/lineage_round_${round}.expected"
    ensure_lineage "${data_expected}" "${current_dataset}/data.json" "${data_manifest}"
    if [[ ! -f "${current_dataset}/data.json" ]]; then
      agent_models="$(join_models QUESTION_MODELS)"
      solver_models="$(join_models ANSWER_MODELS)"
      echo "Build D${round} from D${previous_round} using Q${previous_round} and A${previous_round}."
      AGENT_MODELS="${agent_models}" \
      SOLVER_MODELS="${solver_models}" \
      IMAGE_EDIT_RATIO=0.1 \
      IMAGE_NUM_INFERENCE_STEPS=20 \
      TTRL_VOTES_PER_ROUND=5 \
      TTRL_MIN_VOTE_ROUNDS=1 \
      TTRL_MAX_VOTE_ROUNDS=1 \
      PARTITION_SEED=2025 \
      EXPECTED_DATASET_ROWS=15616 \
      REBUILD_ROUNDS=0 \
        bash "${ROOT_DIR}/data_augmentation/build_round_k.sh" \
          "${round}" "${BASE_QUESTION_MODEL}" "${BASE_DATASET}" \
          "${DATA_ROOT}" "${WORK_ROOT}" "${BASE_ANSWER_MODEL}"
    else
      echo "Reuse completed D${round}: ${current_dataset}"
    fi
    mark_complete "${data_expected}"
  fi

  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_multimodal_dataset.py" \
    "${current_dataset}" --expected-rows 15616

  # A_k starts from A_{k-1}; A_1 starts from base Qwen2.5-VL-7B.
  if ((round == 1)); then
    answer_parent="${BASE_ANSWER_MODEL}"
  else
    answer_parent="${ANSWER_MODELS[round - 1]}"
  fi
  answer_run="${ANSWER_RUN_ROOT}/A${round}"
  printf -v answer_manifest \
    'protocol=aq_coevolution_v1\nkind=answer\nround=%d\nparent=%s\ndataset=%s\nvalidation=%s\nframework=EasyR1\nrollout_batch=512\nrollouts_per_prompt=5' \
    "${round}" "$(canonical_path "${answer_parent}")" \
    "$(canonical_path "${current_dataset}")" "$(canonical_path "${VALIDATION_DATASET}")"
  answer_expected="${answer_run}/lineage.expected"
  ensure_lineage "${answer_expected}" "${answer_run}/latest_global_step.txt" "${answer_manifest}"
  if ! latest_model_dir "${answer_run}" 30 >/dev/null 2>&1; then
    if ! merge_saved_checkpoint_if_needed "${ROOT_DIR}/EasyR1" "${answer_run}" 30; then
      echo "Train A${round} from $([[ ${round} == 1 ]] && echo base || echo A$((round - 1))) on D${round}."
      ANSWER_TRAIN_SEED="$((3100 + round))" \
      ANSWER_EXPECTED_STEPS=30 \
      ANSWER_EXPECTED_DATASET_ROWS=15616 \
      ANSWER_TRAINING_LOG_ROOT="${LOG_ROOT}" \
        bash "${ROOT_DIR}/data_augmentation/train_answer_model_8gpu.sh" \
          "${answer_parent}" "${current_dataset}" "${VALIDATION_DATASET}" "${answer_run}"
    fi
  else
    echo "Reuse completed A${round}."
  fi
  ANSWER_MODELS[round]="$(latest_model_dir "${answer_run}" 30)"
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_step_reward_log.py" \
    "${answer_run}/step_rewards.jsonl" \
    --expected-steps 30 \
    --expected-model-path "${answer_parent}"
  mark_complete "${answer_expected}"
  echo "A${round}=${ANSWER_MODELS[round]}"

  # Q_k starts from Q_{k-1} and uses the newly trained A_k for online reward.
  if ((round == 1)); then
    question_parent="${BASE_QUESTION_MODEL}"
  else
    question_parent="${QUESTION_MODELS[round - 1]}"
  fi
  question_run="${QUESTION_RUN_ROOT}/Q${round}"
  question_dataset="${QUESTION_DATA_ROOT}/round_${round}"
  printf -v question_manifest \
    'protocol=aq_coevolution_v1\nkind=question\nround=%d\nparent=%s\ndataset=%s\nanswer_model=A%d:%s\nframework=EasyQ1\ntraining_steps=1\nrollout_batch=512\ncandidates_per_prompt=5\nimage_steps=20' \
    "${round}" "$(canonical_path "${question_parent}")" \
    "$(canonical_path "${current_dataset}")" "${round}" \
    "$(canonical_path "${ANSWER_MODELS[round]}")"
  question_expected="${question_run}/lineage.expected"
  ensure_lineage "${question_expected}" "${question_run}/latest_global_step.txt" "${question_manifest}"
  if ! latest_model_dir "${question_run}" 1 >/dev/null 2>&1; then
    if ! merge_saved_checkpoint_if_needed "${ROOT_DIR}/EasyQ1" "${question_run}" 1; then
      echo "Train Q${round} from $([[ ${round} == 1 ]] && echo base || echo Q$((round - 1))) with A${round} as reward solver."
      QUESTION_TRAIN_DATASET="${question_dataset}" \
      QUESTION_REWARD_CACHE="${REWARD_CACHE_ROOT}/round_${round}" \
      QUESTION_TRAIN_SEED="$((4100 + round))" \
      QUESTION_IMAGE_STEPS=20 \
      QUESTION_EXPECTED_DATASET_ROWS=15616 \
      TRAINING_LOG_ROOT="${LOG_ROOT}" \
        bash "${ROOT_DIR}/data_augmentation/train_question_generator_8gpu.sh" \
          "${round}" "${question_parent}" "${current_dataset}" \
          "${question_run}" "${ANSWER_MODELS[round]}"
    fi
  else
    echo "Reuse completed Q${round}."
  fi
  QUESTION_MODELS[round]="$(latest_model_dir "${question_run}" 1)"
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_step_reward_log.py" \
    "${question_run}/step_rewards.jsonl" \
    --expected-steps 1 \
    --expected-model-path "${question_parent}"
  mark_complete "${question_expected}"
  echo "Q${round}=${QUESTION_MODELS[round]}"
done

summary_file="${COEVOLUTION_ROOT}/models.env"
{
  echo "COEVOLUTION_ROUNDS=${TOTAL_ROUNDS}"
  echo "A0=$(canonical_path "${BASE_ANSWER_MODEL}")"
  for ((round = 1; round <= TOTAL_ROUNDS; round++)); do
    echo "A${round}=${ANSWER_MODELS[round]}"
    echo "A${round}_REWARDS=${ANSWER_RUN_ROOT}/A${round}/step_rewards.jsonl"
    echo "Q${round}=${QUESTION_MODELS[round]}"
    echo "Q${round}_REWARDS=${QUESTION_RUN_ROOT}/Q${round}/step_rewards.jsonl"
    echo "D${round}=${DATA_ROOT}/train_round_${round}"
  done
} >"${summary_file}"

echo
echo "Five-round coevolution completed through round ${TOTAL_ROUNDS}."
echo "Final answer model A${TOTAL_ROUNDS}: ${ANSWER_MODELS[TOTAL_ROUNDS]}"
echo "Final question model Q${TOTAL_ROUNDS}: ${QUESTION_MODELS[TOTAL_ROUNDS]}"
echo "Auditable model map: ${summary_file}"
