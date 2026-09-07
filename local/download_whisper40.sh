#!/bin/bash

# Whisper40 is handled as a manual archive placement dataset.
# Expected workflow:
#   1. Obtain whisper40.rar from the upstream dataset source.
#   2. Place it at <DATA_ROOT>/whisper40/download_data/whisper40.rar,
#      or pass --archive /path/to/whisper40.rar.
#   3. Run this script to verify and extract it.

data=data
archive=
. tools/parse_options.sh || exit 1

set -euo pipefail

data=$(realpath "${data}")
dataset_root="${data}/whisper40"
download_dir="${dataset_root}/download_data"
raw_dir="${dataset_root}/raw_data"
dest_archive="${download_dir}/whisper40.rar"

mkdir -p "${download_dir}" "${raw_dir}"

has_extracted=false
if [ -d "${raw_dir}/whisper40/nature" ] && [ -d "${raw_dir}/whisper40/whsp" ]; then
  has_extracted=true
elif [ -d "${raw_dir}/nature" ] && [ -d "${raw_dir}/whsp" ]; then
  has_extracted=true
fi

if [ "${has_extracted}" = true ]; then
  echo "[Whisper40] raw_data already looks extracted. Skipping archive handling."
  exit 0
fi

if [ -n "${archive}" ]; then
  archive=$(realpath -m "${archive}")
fi

if [ -n "${archive}" ] && [ -f "${archive}" ] && [ "${archive}" != "${dest_archive}" ]; then
  echo "[Whisper40] Copying user-provided archive into dataset download_data ..."
  cp -p "${archive}" "${dest_archive}"
fi

if [ ! -f "${dest_archive}" ]; then
  echo "[Whisper40] ERROR: archive not found: ${dest_archive}"
  echo "  Obtain whisper40.rar from the upstream dataset source, then either:"
  echo "    1. Place it at ${dest_archive}"
  echo "    2. Or run: $0 --data ${data} --archive /path/to/whisper40.rar"
  exit 1
fi

echo "[Whisper40] Testing archive integrity ..."
unrar t -idq "${dest_archive}"

echo "[Whisper40] Extracting ${dest_archive} -> ${raw_dir} ..."
unrar x -o+ -idq "${dest_archive}" "${raw_dir}/"

if [ -d "${raw_dir}/whisper40/nature" ] && [ -d "${raw_dir}/whisper40/whsp" ]; then
  echo "[Whisper40] Extraction complete under ${raw_dir}/whisper40"
  exit 0
fi
if [ -d "${raw_dir}/nature" ] && [ -d "${raw_dir}/whsp" ]; then
  echo "[Whisper40] Extraction complete under ${raw_dir}"
  exit 0
fi

echo "[Whisper40] ERROR: expected nature/whsp directories not found after extraction."
find "${raw_dir}" -maxdepth 3 -type d | sort | head -50
exit 1
