#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Generic public entry point. Keep the original launcher name as an internal
# compatibility path for existing runs and checkpoints.
exec bash "${SCRIPT_DIR}/run_five_round_coevolution_8gpu.sh" "$@"
