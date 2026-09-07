#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
cd_kit_root

echo "=== OpenSVBench full-core precheck ==="
python3 tools/generate_submission.py \
  --data "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --precheck-only

python3 tools/verify_core_trials.py --data "${DATA_ROOT}"
