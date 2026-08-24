#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}"

# Make this entry point self-contained. Environment variables can survive a
# `conda deactivate` while PATH/python no longer points at the augmentation
# venv, so always activate and validate the repository-local runtime here.
# shellcheck disable=SC1091
source "${ROOT_DIR}/augmentation_env.sh"

K="${1:?Usage: build_round_k.sh <k> [agent_model] [base_dataset] [output_root] [work_root] [solver_model]}"
DEFAULT_AGENT_MODEL="${2:-${AUG_AGENT_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}}"
BASE_DATASET="${3:-${AUG_BASE_DATASET:-dataset/train_original}}"
OUTPUT_ROOT="${4:-${AUG_OUTPUT_ROOT:-dataset/iterative_rounds}}"
WORK_ROOT="${5:-${AUG_WORK_ROOT:-augmentation_runs/iterative_rounds}}"
DEFAULT_SOLVER_MODEL="${6:-${AUG_SOLVER_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}}"

if ! [[ "${K}" =~ ^[1-9][0-9]*$ ]]; then
  echo "k must be a positive integer." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRANSFER="${TRANSFER:-hardlink}"
REBUILD_ROUNDS="${REBUILD_ROUNDS:-0}"
IMAGE_EDIT_RATIO="${IMAGE_EDIT_RATIO:-0.1}"
PARTITION_SEED="${PARTITION_SEED:-2025}"
EXPECTED_DATASET_ROWS="${EXPECTED_DATASET_ROWS:-}"

# Each round is a fixed-size, disjoint mixed-edit successor of the previous round:
#   D_1 = original data
#   D_r = 90% text_edit(D_{r-1}) union 10% image_edit(D_{r-1}), r >= 2
# Comma-separated values are selected by augmentation depth. Examples:
#   AGENT_MODELS=/checkpoint/round1/actor/huggingface,/checkpoint/round2/actor/huggingface
#   SOLVER_MODELS=/answer/round1/actor/huggingface,/answer/round2/actor/huggingface
IFS=',' read -r -a model_sequence <<< "${AGENT_MODELS:-${DEFAULT_AGENT_MODEL}}"
IFS=',' read -r -a solver_sequence <<< "${SOLVER_MODELS:-${DEFAULT_SOLVER_MODEL}}"

if [[ "${#model_sequence[@]}" -eq 0 || "${#solver_sequence[@]}" -eq 0 ]]; then
  echo "AGENT_MODELS and SOLVER_MODELS must not be empty." >&2
  exit 2
fi
if ! [[ "${PARTITION_SEED}" =~ ^-?[0-9]+$ ]]; then
  echo "PARTITION_SEED must be an integer." >&2
  exit 2
fi
if [[ -z "${EXPECTED_DATASET_ROWS}" ]]; then
  EXPECTED_DATASET_ROWS="$("${PYTHON_BIN}" -c '
import json, pathlib, sys
value = json.loads((pathlib.Path(sys.argv[1]) / "data.json").read_text(encoding="utf-8"))
if not isinstance(value, list) or not value:
    raise SystemExit("base data.json must be a non-empty JSON list")
print(len(value))
' "${BASE_DATASET}")"
fi
if ! [[ "${EXPECTED_DATASET_ROWS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_DATASET_ROWS must be a positive integer; got ${EXPECTED_DATASET_ROWS}." >&2
  exit 2
fi

validate_dataset() {
  "${PYTHON_BIN}" "${ROOT_DIR}/data_augmentation/validate_multimodal_dataset.py" \
    "$1" --expected-rows "${EXPECTED_DATASET_ROWS}"
}

validate_dataset "${BASE_DATASET}"

mkdir -p "${OUTPUT_ROOT}" "${WORK_ROOT}"

ROUND_STRATEGY="mixed_single_edit_chain_v3:ttrl5_fallback:image=${IMAGE_EDIT_RATIO}:seed=${PARTITION_SEED}"
strategy_file="${OUTPUT_ROOT}/.round_strategy"
if [[ -f "${strategy_file}" ]]; then
  existing_strategy="$(<"${strategy_file}")"
  if [[ "${existing_strategy}" != "${ROUND_STRATEGY}" && "${REBUILD_ROUNDS}" != "1" ]]; then
    echo "Output root uses round strategy '${existing_strategy}', expected '${ROUND_STRATEGY}'." >&2
    echo "Choose a new OUTPUT_ROOT or set REBUILD_ROUNDS=1." >&2
    exit 2
  fi
elif [[ -f "${OUTPUT_ROOT}/train_round_2/data.json" && "${REBUILD_ROUNDS}" != "1" ]]; then
  echo "Existing round 2 has no fixed-size chain marker and may be a cumulative dataset." >&2
  echo "Choose a new OUTPUT_ROOT or set REBUILD_ROUNDS=1." >&2
  exit 2
fi
printf '%s\n' "${ROUND_STRATEGY}" >"${strategy_file}"

round_one="${OUTPUT_ROOT}/train_round_1"
if [[ ! -f "${round_one}/data.json" || "${REBUILD_ROUNDS}" == "1" ]]; then
  merge_args=(
    "${PYTHON_BIN}" data_augmentation/augment_dataset.py merge-datasets
    --input-dirs "${BASE_DATASET}"
    --output-dir "${round_one}"
    --round-name round_1
    --transfer "${TRANSFER}"
  )
  if [[ "${REBUILD_ROUNDS}" == "1" ]]; then
    merge_args+=(--overwrite)
  fi
  "${merge_args[@]}"
else
  echo "Reuse existing round 1 dataset: ${round_one}"
fi
validate_dataset "${round_one}"

previous_round_dataset="${round_one}"

for ((depth = 1; depth < K; depth++)); do
  target_round=$((depth + 1))

  if ((depth - 1 < ${#model_sequence[@]})); then
    model="${model_sequence[depth - 1]}"
  else
    last_model_index=$((${#model_sequence[@]} - 1))
    model="${model_sequence[last_model_index]}"
  fi
  if ((depth - 1 < ${#solver_sequence[@]})); then
    solver_model="${solver_sequence[depth - 1]}"
  else
    last_solver_index=$((${#solver_sequence[@]} - 1))
    solver_model="${solver_sequence[last_solver_index]}"
  fi

  round_output="${OUTPUT_ROOT}/train_round_${target_round}"
  stage_root="${WORK_ROOT}/round_${target_round}_mixed"
  text_input="${stage_root}/inputs/text"
  image_input="${stage_root}/inputs/image_edit"
  text_output="${stage_root}/outputs/text"
  image_output="${stage_root}/outputs/image_edit"
  text_work="${stage_root}/work/text"
  image_work="${stage_root}/work/image_edit"
  round_partition_seed=$((PARTITION_SEED + target_round))

  echo "Round ${target_round}: one disjoint edit per row from round $((target_round - 1)); image-edit ratio=${IMAGE_EDIT_RATIO}, remaining rows=text, agent=${model}, solver=${solver_model}"

  if [[ ! -f "${round_output}/data.json" || "${REBUILD_ROUNDS}" == "1" ]]; then
    if [[ ! -f "${text_input}/data.json" || ! -f "${image_input}/data.json" || "${REBUILD_ROUNDS}" == "1" ]]; then
      partition_args=(
        "${PYTHON_BIN}" data_augmentation/augment_dataset.py partition-dataset
        --input-dir "${previous_round_dataset}"
        --text-output-dir "${text_input}"
        --image-output-dir "${image_input}"
        --image-ratio "${IMAGE_EDIT_RATIO}"
        --seed "${round_partition_seed}"
        --transfer "${TRANSFER}"
        --overwrite
      )
      "${partition_args[@]}"
    else
      echo "Reuse round ${target_round} text/image partitions."
    fi

    if [[ ! -f "${text_output}/data.json" || "${REBUILD_ROUNDS}" == "1" ]]; then
      INCLUDE_ORIGINAL=0 \
      OVERWRITE_OUTPUT="${REBUILD_ROUNDS}" \
      RESET_SHARDS="${REBUILD_ROUNDS}" \
      RESET_SOLVER_SHARDS="${REBUILD_ROUNDS}" \
      ROUND_NAME="round_${target_round}" \
        bash data_augmentation/run_8gpu.sh \
          text \
          "${model}" \
          "${text_input}" \
          "${text_output}" \
          "${text_work}" \
          "${solver_model}"
    else
      echo "Reuse round ${target_round} text-edit output."
    fi

    if [[ ! -f "${image_output}/data.json" || "${REBUILD_ROUNDS}" == "1" ]]; then
      INCLUDE_ORIGINAL=0 \
      OVERWRITE_OUTPUT="${REBUILD_ROUNDS}" \
      RESET_SHARDS="${REBUILD_ROUNDS}" \
      RESET_IMAGE_SHARDS="${REBUILD_ROUNDS}" \
      ROUND_NAME="round_${target_round}" \
        bash data_augmentation/run_8gpu.sh \
          image-edit \
          "${model}" \
          "${image_input}" \
          "${image_output}" \
          "${image_work}" \
          "${solver_model}"
    else
      echo "Reuse round ${target_round} image-edit output."
    fi

    merge_args=(
      "${PYTHON_BIN}" data_augmentation/augment_dataset.py merge-datasets
      --input-dirs "${text_output}" "${image_output}"
      --output-dir "${round_output}"
      --round-name "round_${target_round}"
      --transfer "${TRANSFER}"
    )
    if [[ "${REBUILD_ROUNDS}" == "1" ]]; then
      merge_args+=(--overwrite)
    fi
    "${merge_args[@]}"
  else
    echo "Reuse existing round ${target_round} dataset: ${round_output}"
  fi

  validate_dataset "${round_output}"

  previous_round_dataset="${round_output}"
done

echo "Round ${K} training dataset: ${previous_round_dataset}"
