#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LOG_ROOT="${1:-${TRAINING_LOG_ROOT:-${SCRIPT_DIR}/training_logs}}"
PORT="${2:-6006}"
HOST="${TENSORBOARD_HOST:-127.0.0.1}"
TENSORBOARD_BIN="${SCRIPT_DIR}/.venv-augmentation/bin/tensorboard"

if [[ ! -d "${LOG_ROOT}" ]]; then
  echo "Training log directory does not exist yet: ${LOG_ROOT}" >&2
  exit 1
fi
if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
  echo "Port must be an integer in [1, 65535]; got ${PORT}." >&2
  exit 2
fi
if [[ ! -x "${TENSORBOARD_BIN}" ]]; then
  echo "TensorBoard is not installed in the repository environment." >&2
  echo "Run: source ./augmentation_env.sh" >&2
  exit 1
fi

echo "W&B offline files and TensorBoard events: ${LOG_ROOT}"
echo "Local viewer: http://${HOST}:${PORT}"
echo "For a remote server, run this on your computer in another terminal:"
echo "  ssh -L ${PORT}:127.0.0.1:${PORT} <user>@<server>"
echo "Then open: http://127.0.0.1:${PORT}"
exec "${TENSORBOARD_BIN}" \
  --logdir "${LOG_ROOT}" \
  --host "${HOST}" \
  --port "${PORT}"
