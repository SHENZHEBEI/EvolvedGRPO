#!/usr/bin/env bash

# Source this file before running data augmentation:
#   source ./augmentation_env.sh
#
# Cluster-specific model/data paths can be supplied before sourcing this file.
# PYTHON_BIN is intentionally replaced with the repository-local interpreter.

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  aug_script_source="${BASH_SOURCE[0]}"
  if [[ "${aug_script_source}" == */* ]]; then
    aug_script_dir="${aug_script_source%/*}"
  else
    aug_script_dir="."
  fi
  AUG_REPO_ROOT="$(builtin cd -- "${aug_script_dir}" && builtin pwd -P)"
else
  AUG_REPO_ROOT="$(builtin pwd -P)"
fi
case "${AUG_REPO_ROOT}" in
  /*) ;;
  *)
    echo "Failed to resolve an absolute repository path: ${AUG_REPO_ROOT}" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
export AUG_REPO_ROOT
export AUG_OFFLINE="${AUG_OFFLINE:-0}"
if [[ "${AUG_OFFLINE}" != "0" && "${AUG_OFFLINE}" != "1" ]]; then
  echo "AUG_OFFLINE must be 0 or 1; got ${AUG_OFFLINE}." >&2
  return 1 2>/dev/null || exit 1
fi

# Resolve external programs to their actual executables.  This script is
# sourced from an interactive shell, where a same-named alias or function can
# otherwise corrupt path output (for example, prefixing realpath with a prompt
# fragment such as '\)').
AUG_REALPATH_BIN="$(type -P realpath 2>/dev/null || true)"
AUG_READLINK_BIN="$(type -P readlink 2>/dev/null || true)"
AUG_CKSUM_BIN="$(type -P cksum 2>/dev/null || true)"
AUG_AWK_BIN="$(type -P awk 2>/dev/null || true)"
for required_program in \
  "${AUG_REALPATH_BIN}" "${AUG_READLINK_BIN}" \
  "${AUG_CKSUM_BIN}" "${AUG_AWK_BIN}"; do
  if [[ -z "${required_program}" || ! -x "${required_program}" ]]; then
    echo "Missing a required system utility (realpath, readlink, cksum, or awk)." >&2
    return 1 2>/dev/null || exit 1
  fi
done

# This is the single supported entry point. Ignore stale values inherited from
# previous attempts or another checkout so one source command is sufficient.
unset \
  AUG_BOOTSTRAP_PYTHON PYTHONHOME PYTHONPATH PYTHONUSERBASE \
  PIP_TARGET PIP_PREFIX PIP_USER \
  TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export PIP_CONFIG_FILE=/dev/null

require_path_inside_repo() {
  local requested_path="$1"
  local label="$2"
  local resolved_path="${requested_path}"
  resolved_path="$("${AUG_REALPATH_BIN}" -m -- "${requested_path}")"
  case "${resolved_path}" in
    "${AUG_REPO_ROOT}"/*) ;;
    *)
      echo "${label} must stay inside the current repository: ${resolved_path}" >&2
      return 1
      ;;
  esac
}

# Keep caches and package extraction on the same filesystem as this checkout.
# Disable pip's wheel cache to avoid storing a second copy of large CUDA wheels.
export AUG_CACHE_ROOT="${AUG_REPO_ROOT}/.cache"
export XDG_CACHE_HOME="${AUG_CACHE_ROOT}"
export PIP_CACHE_DIR="${AUG_CACHE_ROOT}/pip"
export PIP_NO_CACHE_DIR=1
export TMPDIR="${AUG_REPO_ROOT}/.tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export TORCH_HOME="${AUG_CACHE_ROOT}/torch"
export TORCH_EXTENSIONS_DIR="${AUG_CACHE_ROOT}/torch_extensions"
export TRITON_CACHE_DIR="${AUG_CACHE_ROOT}/triton"
export NUMBA_CACHE_DIR="${AUG_CACHE_ROOT}/numba"
export CUDA_CACHE_PATH="${AUG_CACHE_ROOT}/cuda"
export VLLM_CACHE_ROOT="${AUG_CACHE_ROOT}/vllm"
export AUG_RAY_STORAGE_DIR="${TMPDIR}/ray"
require_path_inside_repo "${AUG_CACHE_ROOT}" "Cache directory" || {
  return 1 2>/dev/null || exit 1
}
require_path_inside_repo "${TMPDIR}" "Temporary directory" || {
  return 1 2>/dev/null || exit 1
}
require_path_inside_repo "${AUG_RAY_STORAGE_DIR}" "Ray storage directory" || {
  return 1 2>/dev/null || exit 1
}
mkdir -p \
  "${AUG_CACHE_ROOT}" "${TMPDIR}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" \
  "${TRITON_CACHE_DIR}" "${NUMBA_CACHE_DIR}" "${CUDA_CACHE_PATH}" \
  "${VLLM_CACHE_ROOT}" "${AUG_RAY_STORAGE_DIR}"

# AF_UNIX limits socket paths to 107 bytes.  Ray appends
# ray/session_<timestamp>_<pid>/sockets/plasma_store, so exposing the long
# repository .tmp through a short /tmp symlink is required on this cluster.
# The symlink is only a short pathname: all actual Ray files remain under the
# repository-owned ${AUG_RAY_STORAGE_DIR} directory.
ray_repo_key="$(printf '%s' "${AUG_REPO_ROOT}" | "${AUG_CKSUM_BIN}" | "${AUG_AWK_BIN}" '{print $1}')"
ray_user_id="${UID:-$(id -u)}"
export RAY_TMPDIR="/tmp/nips-ray-${ray_user_id}-${ray_repo_key}"
# This cluster's shared filesystem is very large (about 103 TB) and can have
# several TB free while exceeding Ray's percentage-only 95% spill guard.  A
# 99% threshold still reserves roughly 1 TB on that filesystem and lets this
# one-step job spill objects when necessary.  EasyQ1 passes this explicitly to
# ray.init; the name is repository-specific rather than a Ray environment key.
export AUG_RAY_LOCAL_FS_CAPACITY_THRESHOLD=0.99
if [[ -L "${RAY_TMPDIR}" ]]; then
  ray_link_target="$("${AUG_READLINK_BIN}" -f -- "${RAY_TMPDIR}" 2>/dev/null || true)"
  ray_expected_target="$("${AUG_REALPATH_BIN}" -m -- "${TMPDIR}")"
  if [[ "${ray_link_target}" != "${ray_expected_target}" ]]; then
    echo "Ray short-path link points to the wrong directory: ${RAY_TMPDIR} -> ${ray_link_target}" >&2
    return 1 2>/dev/null || exit 1
  fi
elif [[ -e "${RAY_TMPDIR}" ]]; then
  echo "Ray short path exists but is not a symbolic link: ${RAY_TMPDIR}" >&2
  return 1 2>/dev/null || exit 1
else
  ln -s -- "${TMPDIR}" "${RAY_TMPDIR}" || {
    echo "Failed to create Ray short-path link: ${RAY_TMPDIR} -> ${TMPDIR}" >&2
    return 1 2>/dev/null || exit 1
  }
fi
ray_socket_probe="${RAY_TMPDIR}/ray/session_2026-08-04_10-58-13_746345_4179257/sockets/plasma_store"
if ((${#ray_socket_probe} > 107)); then
  echo "Ray short path is still too long for AF_UNIX (${#ray_socket_probe} bytes): ${ray_socket_probe}" >&2
  return 1 2>/dev/null || exit 1
fi

echo "Repository storage target:"
df -hP "${AUG_REPO_ROOT}" | tail -n 1

# Keep every Python package inside this repository. The base/host Python is
# used only once to create an isolated Python 3.10 venv; it is never a pip
# installation target. Override AUG_BOOTSTRAP_PYTHON only to select which
# Python 3.10 executable creates the local environment.
# Keep this augmentation stack separate from any existing .conda_envs used by
# other projects in the same checkout.
export AUG_ENV_DIR="${AUG_REPO_ROOT}/.venv-augmentation"
export AUG_CREATE_LOCAL_ENV=1
export AUG_GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"

AUG_ENV_DIR="$("${AUG_REALPATH_BIN}" -m -- "${AUG_ENV_DIR}")"
case "${AUG_ENV_DIR}" in
  "${AUG_REPO_ROOT}"/*) ;;
  *)
    echo "AUG_ENV_DIR must stay inside the repository: ${AUG_REPO_ROOT}" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
export AUG_ENV_DIR

if [[ ! -x "${AUG_ENV_DIR}/bin/python" || ! -f "${AUG_ENV_DIR}/bin/activate" ]]; then
  if [[ "${AUG_CREATE_LOCAL_ENV}" != "1" ]]; then
    echo "Local environment does not exist and AUG_CREATE_LOCAL_ENV=0: ${AUG_ENV_DIR}" >&2
    return 1 2>/dev/null || exit 1
  fi

  bootstrap_python="$(command -v python3 2>/dev/null || true)"
  if [[ ! -x "${bootstrap_python}" ]]; then
    bootstrap_python="$(command -v python3.10 2>/dev/null || true)"
  fi
  if [[ ! -x "${bootstrap_python}" ]]; then
    echo "No executable python3 was found; detected: ${bootstrap_python:-missing}." >&2
    return 1 2>/dev/null || exit 1
  fi
  echo "Using bootstrap interpreter: ${bootstrap_python} ($("${bootstrap_python}" --version 2>&1))"

  echo "Creating repository-local Python environment: ${AUG_ENV_DIR}"
  "${bootstrap_python}" -m venv --without-pip "${AUG_ENV_DIR}" || {
    echo "Failed to create the local venv with --without-pip." >&2
    return 1 2>/dev/null || exit 1
  }
fi

if [[ ! -f "${AUG_ENV_DIR}/bin/activate" ]]; then
  echo "Invalid local venv (missing bin/activate): ${AUG_ENV_DIR}" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "${AUG_ENV_DIR}/bin/activate"
export PYTHON_BIN="${AUG_ENV_DIR}/bin/python"
export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=1

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  if [[ "${AUG_OFFLINE}" == "1" ]]; then
    echo "Offline mode cannot bootstrap pip; the local environment is incomplete: ${AUG_ENV_DIR}" >&2
    return 1 2>/dev/null || exit 1
  fi
  get_pip_dir="${AUG_ENV_DIR}/.bootstrap"
  get_pip_script="${get_pip_dir}/get-pip.py"
  mkdir -p "${get_pip_dir}"
  echo "The local venv has no pip; downloading ${AUG_GET_PIP_URL}" >&2
  if ! "${PYTHON_BIN}" -c '
import pathlib, sys, urllib.request
url, destination = sys.argv[1], pathlib.Path(sys.argv[2])
with urllib.request.urlopen(url, timeout=120) as response:
    destination.write_bytes(response.read())
' "${AUG_GET_PIP_URL}" "${get_pip_script}"; then
    echo "Failed to download get-pip.py. Check outbound HTTPS access and source this file again." >&2
    return 1 2>/dev/null || exit 1
  fi
  "${PYTHON_BIN}" "${get_pip_script}" --disable-pip-version-check --no-cache-dir || {
    echo "Failed to install pip inside ${AUG_ENV_DIR} with get-pip.py." >&2
    return 1 2>/dev/null || exit 1
  }
fi

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  echo "The repository-local venv still has no working pip: ${AUG_ENV_DIR}" >&2
  return 1 2>/dev/null || exit 1
fi

if ! "${PYTHON_BIN}" -c '
import pathlib, site, sys, sysconfig
expected = pathlib.Path(sys.argv[1]).resolve()
actual = pathlib.Path(sys.prefix).resolve()
assert actual == expected, f"active prefix {actual} is not repository-local prefix {expected}"
assert not site.ENABLE_USER_SITE, "user site-packages must be disabled"
for scheme_name in ("purelib", "platlib", "scripts"):
    install_path = pathlib.Path(sysconfig.get_path(scheme_name)).resolve()
    install_path.relative_to(expected)
' "${AUG_ENV_DIR}"; then
  echo "Refusing to install because Python package paths are not inside ${AUG_ENV_DIR}." >&2
  return 1 2>/dev/null || exit 1
fi

# Download the two Qwen checkpoints and the CLIP scorer into this repository
# on first source.
# Repository-owned paths are intentionally not overrideable. This prevents a
# shell that previously sourced another checkout from reusing its dependency
# files, cache, models, or output directories.
export AUG_MODEL_ROOT="${AUG_REPO_ROOT}/models"
export HF_HOME="${AUG_REPO_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_TOKEN_FILE="${AUG_REPO_ROOT}/.hf_token"
export AUG_RUNTIME_REQUIREMENTS="${AUG_REPO_ROOT}/data_augmentation/requirements-runtime.txt"
export AUG_RUNTIME_CHECK="${AUG_REPO_ROOT}/data_augmentation/check_runtime.py"
export AUG_CONSTRAINTS="${AUG_REPO_ROOT}/data_augmentation/constraints-a800-cu124.txt"
require_path_inside_repo "${AUG_MODEL_ROOT}" "Model directory" || {
  return 1 2>/dev/null || exit 1
}
require_path_inside_repo "${HF_HOME}" "Hugging Face cache" || {
  return 1 2>/dev/null || exit 1
}
export AUG_INSTALL_DEPS=1
export AUG_PYTORCH_INDEX_URL="${AUG_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
export AUG_FLASH_ATTN_WHEEL_URL="${AUG_FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
export AUG_FLASHINFER_WHEEL_URL="${AUG_FLASHINFER_WHEEL_URL:-https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
# Public Qwen snapshots can fail with a CAS/Xet 401 on some clusters. This must
# be exported before importing huggingface_hub so downloads use regular HTTP.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-2}"
export HF_DOWNLOAD_RETRIES="${HF_DOWNLOAD_RETRIES:-3}"
export AUG_DOWNLOAD_MODELS=1
export AUG_FORCE_MODEL_DOWNLOAD=0
export QWEN_VL_REPO="Qwen/Qwen2.5-VL-7B-Instruct"
export QWEN_IMAGE_EDIT_REPO="Qwen/Qwen-Image-Edit-2511"
export QUESTION_CLIP_REPO="openai/clip-vit-large-patch14"
export QWEN_VL_LOCAL_DIR="${AUG_MODEL_ROOT}/Qwen2.5-VL-7B-Instruct"
export QWEN_IMAGE_EDIT_LOCAL_DIR="${AUG_MODEL_ROOT}/Qwen-Image-Edit-2511"
export QUESTION_CLIP_LOCAL_DIR="${AUG_MODEL_ROOT}/clip-vit-large-patch14"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

for required_file in "${AUG_RUNTIME_REQUIREMENTS}" "${AUG_RUNTIME_CHECK}" "${AUG_CONSTRAINTS}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required standalone augmentation file is missing: ${required_file}" >&2
    return 1 2>/dev/null || exit 1
  fi
done

# Prefer a token already exported by a secret manager. Otherwise load the
# repository-local, git-ignored token file without printing its contents.
if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_FILE}" ]]; then
  chmod 600 "${HF_TOKEN_FILE}" 2>/dev/null || true
  IFS= read -r HF_TOKEN <"${HF_TOKEN_FILE}"
  export HF_TOKEN
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "Hugging Face token file is empty: ${HF_TOKEN_FILE}" >&2
    return 1 2>/dev/null || exit 1
  fi
  echo "Loaded Hugging Face credentials from ${HF_TOKEN_FILE}"
fi

if ! "${PYTHON_BIN}" -c '
import platform, sys
assert sys.version_info[:2] == (3, 10), f"Python 3.10 required, got {sys.version}"
assert platform.system() == "Linux", f"Linux required, got {platform.system()}"
assert platform.machine() == "x86_64", f"x86_64 required, got {platform.machine()}"
'; then
  echo "The standalone augmentation CUDA stack requires Linux x86_64 and Python 3.10." >&2
  echo "Active Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')" >&2
  return 1 2>/dev/null || exit 1
fi

check_augmentation_runtime() {
  "${PYTHON_BIN}" "${AUG_RUNTIME_CHECK}" --quiet >/dev/null 2>&1
}

if ! check_augmentation_runtime; then
  if [[ "${AUG_OFFLINE}" == "1" ]]; then
    echo "Offline mode cannot install missing runtime packages in ${AUG_ENV_DIR}." >&2
    "${PYTHON_BIN}" "${AUG_RUNTIME_CHECK}" || true
    return 1 2>/dev/null || exit 1
  fi
  if [[ "${AUG_INSTALL_DEPS}" != "1" ]]; then
    echo "Augmentation runtime is absent or does not match ${AUG_CONSTRAINTS}." >&2
    echo "Set AUG_INSTALL_DEPS=1 and source this file again." >&2
    return 1 2>/dev/null || exit 1
  fi
  echo "Installing the standalone augmentation CUDA 12.4 runtime with ${PYTHON_BIN}."
  "${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
    --index-url "${AUG_PYTORCH_INDEX_URL}" \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" || {
    return 1 2>/dev/null || exit 1
  }
  "${PYTHON_BIN}" -m pip install \
    --no-cache-dir \
    --upgrade \
    --upgrade-strategy only-if-needed \
    -r "${AUG_RUNTIME_REQUIREMENTS}" || {
    return 1 2>/dev/null || exit 1
  }
  # Pure-Python additions (for example TensorBoard) must not trigger another
  # download of the large CUDA kernel wheels. Reinstall the pinned kernels only
  # when the complete ABI/version check still fails after normal dependencies.
  if ! check_augmentation_runtime; then
    "${PYTHON_BIN}" -m pip install --no-cache-dir --no-deps --force-reinstall \
      "${AUG_FLASH_ATTN_WHEEL_URL}" \
      "${AUG_FLASHINFER_WHEEL_URL}" || {
      return 1 2>/dev/null || exit 1
    }
  fi
  "${PYTHON_BIN}" -m pip check || {
    echo "pip detected incompatible packages in the active Python environment." >&2
    return 1 2>/dev/null || exit 1
  }
fi

if ! check_augmentation_runtime; then
  "${PYTHON_BIN}" "${AUG_RUNTIME_CHECK}" || true
  echo "Runtime compatibility check still fails after installation." >&2
  echo "Use a clean Python 3.10 environment, then source augmentation_env.sh again." >&2
  return 1 2>/dev/null || exit 1
fi

"${PYTHON_BIN}" "${AUG_RUNTIME_CHECK}"

download_hf_snapshot() {
  local repo_id="$1"
  local local_dir="$2"
  local required_file="$3"
  local completion_marker="${local_dir}/.download_complete"

  if [[ "${AUG_FORCE_MODEL_DOWNLOAD}" != "1" && -f "${completion_marker}" && -f "${local_dir}/${required_file}" ]]; then
    echo "Reuse local model: ${local_dir}"
    return 0
  fi

  mkdir -p "${local_dir}" "${HF_HOME}"
  echo "Downloading ${repo_id} to ${local_dir}"
  local attempt
  local downloaded=0
  for ((attempt = 1; attempt <= HF_DOWNLOAD_RETRIES; attempt++)); do
    if "${PYTHON_BIN}" -c \
      'import os, sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], max_workers=int(sys.argv[3]), token=os.environ.get("HF_TOKEN") or None)' \
      "${repo_id}" "${local_dir}" "${HF_DOWNLOAD_WORKERS}"; then
      downloaded=1
      break
    fi
    if ((attempt < HF_DOWNLOAD_RETRIES)); then
      echo "Download attempt ${attempt}/${HF_DOWNLOAD_RETRIES} failed; retrying existing partial files." >&2
      sleep $((attempt * 5))
    fi
  done
  if [[ "${downloaded}" != "1" ]]; then
    echo "Failed to download ${repo_id} after ${HF_DOWNLOAD_RETRIES} attempts." >&2
    return 1
  fi
  if [[ ! -f "${local_dir}/${required_file}" ]]; then
    echo "Download finished but required file is missing: ${local_dir}/${required_file}" >&2
    return 1
  fi
  touch "${completion_marker}"
}

if [[ "${AUG_DOWNLOAD_MODELS}" == "1" ]]; then
  all_local_models_ready=1
  for local_model_spec in \
    "${QWEN_VL_LOCAL_DIR}/.download_complete:${QWEN_VL_LOCAL_DIR}/config.json" \
    "${QWEN_IMAGE_EDIT_LOCAL_DIR}/.download_complete:${QWEN_IMAGE_EDIT_LOCAL_DIR}/model_index.json" \
    "${QUESTION_CLIP_LOCAL_DIR}/.download_complete:${QUESTION_CLIP_LOCAL_DIR}/config.json"; do
    marker_path="${local_model_spec%%:*}"
    required_path="${local_model_spec#*:}"
    if [[ ! -f "${marker_path}" || ! -f "${required_path}" ]]; then
      all_local_models_ready=0
      break
    fi
  done

  if [[ "${AUG_OFFLINE}" == "1" && "${all_local_models_ready}" != "1" ]]; then
    echo "Offline mode requires complete local Qwen-VL, Qwen-Image-Edit, and CLIP checkpoints under ${AUG_MODEL_ROOT}." >&2
    return 1 2>/dev/null || exit 1
  fi

  # Authentication is needed only for an actual download. Avoid an external
  # whoami API request every time an already-complete local environment is sourced.
  if [[ "${all_local_models_ready}" != "1" && -n "${HF_TOKEN:-}" ]]; then
    "${PYTHON_BIN}" -c \
      'import os; from huggingface_hub import HfApi; user = HfApi().whoami(token=os.environ["HF_TOKEN"]); print("Authenticated with Hugging Face as " + str(user.get("name", "unknown")))' || {
      echo "Hugging Face authentication failed. Replace ${HF_TOKEN_FILE} with a valid token." >&2
      return 1 2>/dev/null || exit 1
    }
  elif [[ "${all_local_models_ready}" != "1" ]]; then
    echo "No HF_TOKEN or ${HF_TOKEN_FILE}; attempting anonymous public-model download."
  fi
  download_hf_snapshot "${QWEN_VL_REPO}" "${QWEN_VL_LOCAL_DIR}" "config.json" || {
    return 1 2>/dev/null || exit 1
  }
  download_hf_snapshot "${QWEN_IMAGE_EDIT_REPO}" "${QWEN_IMAGE_EDIT_LOCAL_DIR}" "model_index.json" || {
    return 1 2>/dev/null || exit 1
  }
  download_hf_snapshot "${QUESTION_CLIP_REPO}" "${QUESTION_CLIP_LOCAL_DIR}" "config.json" || {
    return 1 2>/dev/null || exit 1
  }
fi

# Default models and data stay in this checkout. Pass trained model paths to
# build_round_k.sh (or export overrides after sourcing) when needed.
export AUG_AGENT_MODEL="${QWEN_VL_LOCAL_DIR}"
export AUG_SOLVER_MODEL="${QWEN_VL_LOCAL_DIR}"
export IMAGE_MODEL="${QWEN_IMAGE_EDIT_LOCAL_DIR}"
export QUESTION_CLIP_MODEL="${QUESTION_CLIP_LOCAL_DIR}"
export QUESTION_REWARD_CACHE="${AUG_REPO_ROOT}/augmentation_runs/question_reward_cache"
export QUESTION_REWARD_ENDPOINT="http://127.0.0.1:8765"
export QUESTION_REWARD_ENDPOINTS="http://127.0.0.1:8765,http://127.0.0.1:8766,http://127.0.0.1:8767,http://127.0.0.1:8768,http://127.0.0.1:8769,http://127.0.0.1:8770,http://127.0.0.1:8771,http://127.0.0.1:8772"
export AUG_BASE_DATASET="${AUG_REPO_ROOT}/dataset/train_original"
export AUG_OUTPUT_ROOT="${AUG_REPO_ROOT}/dataset/iterative_rounds"
export AUG_WORK_ROOT="${AUG_REPO_ROOT}/augmentation_runs/iterative_rounds"

# W&B is always offline on this cluster.  Its raw run files and the parallel
# TensorBoard event stream both stay in the repository for local inspection.
export TRAINING_LOG_ROOT="${AUG_REPO_ROOT}/training_logs"
export WANDB_MODE=offline
export WANDB_DIR="${TRAINING_LOG_ROOT}"
export WANDB_CACHE_DIR="${AUG_CACHE_ROOT}/wandb"
export WANDB_CONFIG_DIR="${AUG_CACHE_ROOT}/wandb_config"
export WANDB_SILENT=true
export WANDB_DISABLE_GIT=true
export WANDB_DISABLE_CODE=true
export TENSORBOARD_DIR="${TRAINING_LOG_ROOT}/tensorboard"
require_path_inside_repo "${TRAINING_LOG_ROOT}" "Training log directory" || {
  return 1 2>/dev/null || exit 1
}
mkdir -p "${TRAINING_LOG_ROOT}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${TENSORBOARD_DIR}"

# Runtime and fixed-size mixed augmentation policy.
# The supported local profile is exactly eight visible NVIDIA A800 80GB GPUs.
# Force stale values from another hardware profile out when sourced again.
export NUM_GPUS=8
export AUG_GPU_PROFILE=a800
export TRANSFER="${TRANSFER:-hardlink}"
export IMAGE_EDIT_RATIO="${IMAGE_EDIT_RATIO:-0.1}"
export PARTITION_SEED="${PARTITION_SEED:-2025}"
export REBUILD_ROUNDS="${REBUILD_ROUNDS:-0}"
export ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"

# The local base processor also supports merged actor checkpoints that contain
# only model weights/configuration.
export PROCESSOR_MODEL="${QWEN_VL_LOCAL_DIR}"
export SOLVER_PROCESSOR_MODEL="${QWEN_VL_LOCAL_DIR}"

# Qwen2.5-VL answer voting: exactly five candidates, accept a pair (2/5).
export TTRL_VOTES_PER_ROUND="${TTRL_VOTES_PER_ROUND:-5}"
export TTRL_MIN_VOTE_ROUNDS="${TTRL_MIN_VOTE_ROUNDS:-1}"
export TTRL_MAX_VOTE_ROUNDS="${TTRL_MAX_VOTE_ROUNDS:-1}"
export TTRL_MIN_VALID_VOTES="${TTRL_MIN_VALID_VOTES:-2}"
export TTRL_MIN_AGREE_VOTES="${TTRL_MIN_AGREE_VOTES:-2}"
export TTRL_CONSENSUS_THRESHOLD="${TTRL_CONSENSUS_THRESHOLD:-0.4}"
export TTRL_STORE_RAW_OUTPUTS="${TTRL_STORE_RAW_OUTPUTS:-0}"

# Qwen2.5-VL question Agent capacity. The image is resized in memory for
# inference only; the dataset image on disk is never modified here.
export AGENT_MAX_MODEL_LEN="${AGENT_MAX_MODEL_LEN:-16384}"
export AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-384}"
export AGENT_BATCH_SIZE=32
export AGENT_GPU_MEMORY_UTILIZATION=0.80
export AGENT_MAX_NUM_BATCHED_TOKENS=16384
if [[ "${AGENT_MAX_GENERATION_ATTEMPTS:-}" == "3" ]]; then
  unset AGENT_MAX_GENERATION_ATTEMPTS
fi
export AGENT_MAX_GENERATION_ATTEMPTS="${AGENT_MAX_GENERATION_ATTEMPTS:-1}"
export AGENT_MIN_PIXELS="${AGENT_MIN_PIXELS:-200704}"
export AGENT_MAX_PIXELS="${AGENT_MAX_PIXELS:-1605632}"

# 51 questions x 5 candidates = 255 vLLM sequences per A800.
export SOLVER_BATCH_SIZE=51
export SOLVER_MAX_NUM_SEQS=256
export SOLVER_MAX_NUM_BATCHED_TOKENS=8192
export SOLVER_GPU_MEMORY_UTILIZATION=0.90
# Migrate the old exported default when this file is sourced again in the same
# shell. A deliberate override can still be supplied after `source`, or as an
# environment assignment on the build command.
if [[ "${SOLVER_MAX_MODEL_LEN:-}" == "4096" ]]; then
  unset SOLVER_MAX_MODEL_LEN
fi
export SOLVER_MAX_MODEL_LEN="${SOLVER_MAX_MODEL_LEN:-8192}"
if [[ "${SOLVER_MAX_TOKENS:-}" == "2048" ]]; then
  unset SOLVER_MAX_TOKENS
fi
export SOLVER_MAX_TOKENS="${SOLVER_MAX_TOKENS:-1024}"
export SOLVER_DTYPE="${SOLVER_DTYPE:-bfloat16}"

# Later coevolution rounds concatenate additional mathematical questions onto
# the previous prompt. A 2048-token answer context removed 259 D5 rows and left
# only 15357 examples, three short of the 30 x 512 training schedule. Keep the
# 2048-token response budget while expanding only the answer prompt capacity.
export ANSWER_MAX_PROMPT_LENGTH="${ANSWER_MAX_PROMPT_LENGTH:-3072}"
export ANSWER_MAX_RESPONSE_LENGTH="${ANSWER_MAX_RESPONSE_LENGTH:-2048}"
export ANSWER_MAX_MODEL_LEN="${ANSWER_MAX_MODEL_LEN:-$((ANSWER_MAX_PROMPT_LENGTH + ANSWER_MAX_RESPONSE_LENGTH))}"
export ANSWER_MAX_NUM_BATCHED_TOKENS="${ANSWER_MAX_NUM_BATCHED_TOKENS:-8192}"
export SOLVER_TEMPERATURE="${SOLVER_TEMPERATURE:-0.7}"
export SOLVER_TOP_P="${SOLVER_TOP_P:-0.95}"

# Image editing defaults.
export IMAGE_NUM_INFERENCE_STEPS=20
export QUESTION_IMAGE_STEPS=20
export IMAGE_CPU_OFFLOAD="${IMAGE_CPU_OFFLOAD:-0}"

# Keep tokenizer worker logs quiet and avoid accidental CPU oversubscription.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "Loaded augmentation environment:"
echo "  local Python environment=${AUG_ENV_DIR}"
echo "  Python=${PYTHON_BIN}"
echo "  GPUs=${NUM_GPUS}, image-edit ratio=${IMAGE_EDIT_RATIO}"
echo "  agent=${AUG_AGENT_MODEL}"
echo "  Agent context=${AGENT_MAX_MODEL_LEN}, TTRL context=${SOLVER_MAX_MODEL_LEN}"
echo "  solver=${AUG_SOLVER_MODEL}"
echo "  image editor=${IMAGE_MODEL}"
echo "  image editing steps=${IMAGE_NUM_INFERENCE_STEPS} (augmentation and training reward)"
echo "  CLIP scorer=${QUESTION_CLIP_MODEL}"
echo "  local model root=${AUG_MODEL_ROOT}"
echo "  cache root=${AUG_CACHE_ROOT}"
echo "  temporary files=${TMPDIR}"
echo "  Ray short path=${RAY_TMPDIR}/ray -> ${AUG_RAY_STORAGE_DIR}"
echo "  Ray filesystem threshold=${AUG_RAY_LOCAL_FS_CAPACITY_THRESHOLD}"
echo "  input=${AUG_BASE_DATASET}"
echo "  output root=${AUG_OUTPUT_ROOT}"
echo "  work root=${AUG_WORK_ROOT}"
echo "  W&B mode=${WANDB_MODE}, local logs=${TRAINING_LOG_ROOT}"
