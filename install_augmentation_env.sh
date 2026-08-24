#!/usr/bin/env bash
set -euo pipefail

# Standalone one-click installer for augmentation and question-generator GRPO.
# It deliberately uses only augmentation_env.sh and data_augmentation/*;
# neither training framework's requirements/configuration is installed or
# inspected at install time.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash ./install_augmentation_env.sh" >&2
  exit 2
fi

# Always ignore a stale AUG_ENV_DIR inherited from another checkout/shell.
# This installer is intentionally fixed to the current repository.
export AUG_ENV_DIR="${SCRIPT_DIR}/.venv-augmentation"
export AUG_CREATE_LOCAL_ENV=1
export AUG_INSTALL_DEPS=1
export AUG_DOWNLOAD_MODELS=1
export AUG_FORCE_MODEL_DOWNLOAD=0
export QWEN_VL_REPO="Qwen/Qwen2.5-VL-7B-Instruct"
export QWEN_IMAGE_EDIT_REPO="Qwen/Qwen-Image-Edit-2511"
export QUESTION_CLIP_REPO="openai/clip-vit-large-patch14"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/augmentation_env.sh"

echo
echo "Standalone augmentation and question-generator installation completed."
echo "No package or configuration was loaded from the training framework directories."
echo "Activate it in a new shell with:"
echo "  cd ${SCRIPT_DIR}"
echo "  source ./augmentation_env.sh"
