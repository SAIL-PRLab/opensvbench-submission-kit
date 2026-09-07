#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${KIT_ROOT}/configs/full_core.env"
EXAMPLE_CONFIG="${KIT_ROOT}/configs/full_core.example.env"
CONFIG_PATH="${OPENSVBENCH_CONFIG:-${DEFAULT_CONFIG}}"

load_config() {
  if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[OpenSVBench] Config not found: ${CONFIG_PATH}"
    echo "  Create one with:"
    echo "    cp ${EXAMPLE_CONFIG} ${DEFAULT_CONFIG}"
    echo "  Then edit ${DEFAULT_CONFIG} for your local data/model paths."
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"

  DATA_ROOT="${DATA_ROOT:-${KIT_ROOT}/data}"
  OUTPUT_DIR="${OUTPUT_DIR:-${KIT_ROOT}/output}"
  LOG_DIR="${LOG_DIR:-${KIT_ROOT}/logs}"
  MAX_UTTS="${MAX_UTTS:-10000}"
  NJ="${NJ:-16}"
  GPU="${GPU:-0}"
  MODEL_NAME="${MODEL_NAME:-}"
  BACKEND="${BACKEND:-}"
  SUBMISSION_NAME="${SUBMISSION_NAME:-}"
  VOX1_ZIP_DIR="${VOX1_ZIP_DIR:-}"
  WHISPER40_ARCHIVE="${WHISPER40_ARCHIVE:-}"
  EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-0}"
  EXTRACT_WAVEFORM_CMVN="${EXTRACT_WAVEFORM_CMVN:-none}"
  EXTRACT_MAX_CHUNK_SECONDS="${EXTRACT_MAX_CHUNK_SECONDS:-20.0}"
  SPEECHBRAIN_NORMALIZE="${SPEECHBRAIN_NORMALIZE:-auto}"
  FORCE_EXTRACT="${FORCE_EXTRACT:-0}"

  mkdir -p "${DATA_ROOT}" "${OUTPUT_DIR}" "${LOG_DIR}"
}

run_step() {
  local name="$1"
  shift
  echo
  echo "=== ${name} ==="
  "$@"
}

cd_kit_root() {
  cd "${KIT_ROOT}" || exit 1
}
