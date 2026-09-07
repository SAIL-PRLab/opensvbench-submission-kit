#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
cd_kit_root

if [ -z "${MODEL_NAME}" ] || [ -z "${BACKEND}" ]; then
  echo "[OpenSVBench] MODEL_NAME and BACKEND must be set in ${CONFIG_PATH}."
  exit 2
fi

submission_args=(
  --data "${DATA_ROOT}"
  --model-name "${MODEL_NAME}"
  --backend "${BACKEND}"
  --gpu "${GPU}"
  --output-dir "${OUTPUT_DIR}"
  --extract-batch-size "${EXTRACT_BATCH_SIZE}"
  --extract-waveform-cmvn "${EXTRACT_WAVEFORM_CMVN}"
  --extract-max-chunk-seconds "${EXTRACT_MAX_CHUNK_SECONDS}"
  --speechbrain-normalize "${SPEECHBRAIN_NORMALIZE}"
)

if [ -n "${SUBMISSION_NAME}" ]; then
  submission_args+=(--submission-name "${SUBMISSION_NAME}")
fi

case "${FORCE_EXTRACT}" in
  1|true|TRUE|yes|YES)
    submission_args+=(--force-extract)
    ;;
esac

echo "=== OpenSVBench submission zip generation ==="
python3 tools/generate_submission.py "${submission_args[@]}"
