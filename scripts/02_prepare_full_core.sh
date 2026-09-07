#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
cd_kit_root

echo "=== OpenSVBench full-core data preparation ==="
echo "data_root: ${DATA_ROOT}"
echo "max_utts : ${MAX_UTTS}"
echo "nj       : ${NJ}"

prepare_args=(
  --data "${DATA_ROOT}" \
  --max_utts "${MAX_UTTS}" \
  --nj "${NJ}"
)

if [ -n "${VOX1_ZIP_DIR}" ]; then
  prepare_args+=(--vox1_zip_dir "${VOX1_ZIP_DIR}")
fi
if [ -n "${WHISPER40_ARCHIVE}" ]; then
  prepare_args+=(--whisper40_archive "${WHISPER40_ARCHIVE}")
fi

./prepare_data.sh "${prepare_args[@]}"
