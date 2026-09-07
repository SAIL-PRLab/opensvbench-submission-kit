#!/bin/bash

set -eo pipefail

data=data
split=dev
. tools/parse_options.sh || exit 1
set -u

data=$(realpath "${data}")
raw_dir="${data}/ffsvc2022/raw_data"
download_dir="${data}/ffsvc2022/download_data"
prep_dir="${data}/ffsvc2022/prepared"
OPEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${prep_dir}"

echo "=== [FFSVC2022] prepare_ffsvc2022.sh ==="
echo "[FFSVC2022] raw_dir     : ${raw_dir}"
echo "[FFSVC2022] download_dir: ${download_dir}"
echo "[FFSVC2022] prep_dir    : ${prep_dir}"
echo "[FFSVC2022] split       : ${split}"

if [ ! -f "${download_dir}/trials_dev_keys" ]; then
  echo "[FFSVC2022] ERROR: missing ${download_dir}/trials_dev_keys"
  echo "  Please run local/download_ffsvc2022.sh --data ${data}"
  exit 1
fi

if [ ! -f "${download_dir}/dev_meta_list" ]; then
  echo "[FFSVC2022] ERROR: missing ${download_dir}/dev_meta_list"
  echo "  Please run local/download_ffsvc2022.sh --data ${data}"
  exit 1
fi

if [ ! -d "${raw_dir}/${split}" ]; then
  echo "[FFSVC2022] ERROR: missing wav directory ${raw_dir}/${split}"
  echo "  Please run local/download_ffsvc2022.sh --data ${data}"
  exit 1
fi

python3 "${OPEN_ROOT}/tools/prepare_ffsvc2022.py" \
  --raw-dir "${raw_dir}" \
  --download-dir "${download_dir}" \
  --prep-dir "${prep_dir}" \
  --split "${split}"
