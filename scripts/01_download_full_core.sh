#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
cd_kit_root

failures=0
run_download() {
  local name="$1"
  shift
  if run_step "${name}" "$@"; then
    echo "[OK] ${name}"
  else
    echo "[FAIL] ${name}"
    failures=$((failures + 1))
  fi
}

run_download "TidyVoice placement check" ./local/download_tidyvoice.sh --data "${DATA_ROOT}"
run_download "HI-MIA download" ./local/download_hi-mia.sh --data "${DATA_ROOT}"
run_download "Google Speech Commands download" ./local/download_gsc.sh --data "${DATA_ROOT}"
run_download "GLOBE download" ./local/download_globe.sh --data "${DATA_ROOT}"
run_download "CN-Celeb download" ./local/download_cnceleb.sh --data "${DATA_ROOT}"
if [ -n "${VOX1_ZIP_DIR}" ]; then
  run_download "VoxCeleb download / placement check" ./local/download_voxceleb.sh --data "${DATA_ROOT}" --vox1-zip-dir "${VOX1_ZIP_DIR}"
else
  run_download "VoxCeleb download / placement check" ./local/download_voxceleb.sh --data "${DATA_ROOT}"
fi
run_download "3D-Speaker download" ./local/download_3dspeaker.sh --data "${DATA_ROOT}"
run_download "ESD download" ./local/download_esd.sh --data "${DATA_ROOT}"
run_download "AliMeeting download" ./local/download_alimeeting.sh --data "${DATA_ROOT}"
run_download "VoxKnesset Hugging Face access check" ./local/download_voxknesset.sh --data "${DATA_ROOT}"
run_download "VoxPopuli Hugging Face access/download" ./local/download_voxpopuli.sh --data "${DATA_ROOT}"
run_download "FFSVC 2022 download" ./local/download_ffsvc2022.sh --data "${DATA_ROOT}"
if [ -n "${WHISPER40_ARCHIVE}" ]; then
  run_download "Whisper40 archive placement" ./local/download_whisper40.sh --data "${DATA_ROOT}" --archive "${WHISPER40_ARCHIVE}"
else
  run_download "Whisper40 archive placement" ./local/download_whisper40.sh --data "${DATA_ROOT}"
fi
run_download "Lombard Grid download" ./local/download_lombardgrid.sh --data "${DATA_ROOT}"
run_download "VOiCES download" ./local/download_voices.sh --data "${DATA_ROOT}"
run_download "CHiME-6 download" ./local/download_chime6.sh --data "${DATA_ROOT}"

if [ "${failures}" -ne 0 ]; then
  echo
  echo "[OpenSVBench] Download/placement step finished with ${failures} failure(s)."
  echo "  Check MANUAL_PLACEMENT.md for datasets that require manual access."
  exit 2
fi

echo
echo "[OpenSVBench] All download/placement checks passed."
