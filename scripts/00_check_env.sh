#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

if [ -f "${CONFIG_PATH}" ]; then
  load_config
else
  DATA_ROOT="${KIT_ROOT}/data"
  OUTPUT_DIR="${KIT_ROOT}/output"
  LOG_DIR="${KIT_ROOT}/logs"
  MODEL_NAME=""
  BACKEND=""
  GPU=0
fi

echo "=== OpenSVBench Submission Kit Environment Check ==="
echo "kit_root : ${KIT_ROOT}"
echo "config   : ${CONFIG_PATH}"
echo "data_root: ${DATA_ROOT}"
echo "output   : ${OUTPUT_DIR}"

missing=0
check_cmd() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "[OK] command: ${cmd}"
  else
    echo "[MISS] command: ${cmd}"
    missing=$((missing + 1))
  fi
}

for cmd in python3 bash curl sox ffmpeg unrar; do
  check_cmd "${cmd}"
done

if command -v aria2c >/dev/null 2>&1; then
  echo "[OK] command: aria2c"
else
  echo "[WARN] command: aria2c not found; download scripts will fall back to curl/wget where supported."
fi

python3 - <<'PY'
import importlib

required = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "soundfile",
    "torch",
    "torchaudio",
    "tqdm",
]
optional = [
    "datasets",
    "huggingface_hub",
    "pyarrow",
    "requests",
    "textgrid",
    "transformers",
    "wespeaker",
    "speechbrain",
    "modelscope",
    "redimnet2",
]

missing_required = []
for name in required:
    try:
        importlib.import_module(name)
        print(f"[OK] python module: {name}")
    except Exception as exc:
        print(f"[MISS] python module: {name} ({exc})")
        missing_required.append(name)

for name in optional:
    try:
        importlib.import_module(name)
        print(f"[OK] python module: {name}")
    except Exception as exc:
        print(f"[WARN] python module: {name} unavailable ({exc})")

if missing_required:
    raise SystemExit(2)
PY

python3 - <<'PY'
try:
    import torch
    print(f"[Info] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Info] CUDA device count: {torch.cuda.device_count()}")
except Exception:
    pass
PY

if [ "${missing}" -ne 0 ]; then
  echo "[OpenSVBench] Missing required system command(s): ${missing}"
  exit 2
fi

echo "[OpenSVBench] Environment check finished."
