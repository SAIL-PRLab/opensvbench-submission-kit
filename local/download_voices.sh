#!/bin/bash

# Download helper for the public VOiCES artifacts used by OpenSVBench.
#
# What this script does:
#   - downloads VOiCES_devkit.tar.gz
#   - optionally downloads recording_data.tar.gz
#   - records remote metadata (size / last-modified / ETag) in the log
#   - validates downloaded .tar.gz archives via size check + gzip -t
#
# What this script does NOT do:
#   - it does not extract archives
#   - it does not prepare manifests or trials
#
# Official sources:
#   https://iqtlabs.github.io/voices/downloads/
#   https://iqtlabs.github.io/voices/Lab41-SRI-VOiCES_README/

data=data
download_recording_data=false
devkit_url="https://lab41openaudiocorpus.s3.amazonaws.com/VOiCES_devkit.tar.gz"
recording_url="https://lab41openaudiocorpus.s3.amazonaws.com/recording_data.tar.gz"

. tools/parse_options.sh || exit 1
set -euo pipefail

data=$(realpath "${data}")
dataset_root="${data}/voices"
download_dir="${dataset_root}/download_data"
mkdir -p "${download_dir}"

DEVKIT_PATH="${download_dir}/VOiCES_devkit.tar.gz"
RECORDING_PATH="${download_dir}/recording_data.tar.gz"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

trim_cr() {
  tr -d '\r'
}

header_value() {
  local headers="$1"
  local key="$2"
  printf '%s\n' "${headers}" | awk -F': ' -v k="${key}" '
    BEGIN { IGNORECASE=1 }
    tolower($1) == tolower(k) { value = $2 }
    END {
      gsub("\r", "", value)
      print value
    }
  '
}

fetch_headers() {
  local url="$1"
  curl -fsSI -L --retry 3 --retry-delay 2 "${url}"
}

local_size() {
  local path="$1"
  stat -c '%s' "${path}"
}

verify_gzip() {
  local path="$1"
  echo "[VOiCES] Testing gzip integrity: ${path}"
  gzip -t "${path}"
}

download_with_aria2c() {
  local url="$1"
  local dest_path="$2"
  local dest_dir
  local dest_name
  dest_dir=$(dirname "${dest_path}")
  dest_name=$(basename "${dest_path}")

  aria2c \
    --continue=true \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=16M \
    --retry-wait=5 \
    --max-tries=0 \
    --summary-interval=30 \
    --file-allocation=none \
    --allow-overwrite=false \
    --auto-file-renaming=false \
    --dir="${dest_dir}" \
    --out="${dest_name}" \
    "${url}"
}

download_with_curl() {
  local url="$1"
  local dest_path="$2"
  curl -fL -C - --retry 5 --retry-delay 5 --output "${dest_path}" "${url}"
}

download_file() {
  local url="$1"
  local dest_path="$2"
  local label="$3"

  echo ""
  echo "[VOiCES] === ${label} ==="
  echo "[VOiCES] URL  : ${url}"
  echo "[VOiCES] Path : ${dest_path}"

  local headers
  headers=$(fetch_headers "${url}")

  local remote_size
  local remote_mtime
  local remote_etag
  remote_size=$(header_value "${headers}" "Content-Length" | trim_cr)
  remote_mtime=$(header_value "${headers}" "Last-Modified" | trim_cr)
  remote_etag=$(header_value "${headers}" "ETag" | trim_cr)

  if [ -z "${remote_size}" ]; then
    echo "[VOiCES] ERROR: failed to read remote Content-Length for ${label}"
    exit 1
  fi

  echo "[VOiCES] Remote size        : ${remote_size} bytes"
  echo "[VOiCES] Remote modified    : ${remote_mtime:-unknown}"
  echo "[VOiCES] Remote ETag        : ${remote_etag:-unknown}"

  if [ -f "${dest_path}" ]; then
    local existing_size
    existing_size=$(local_size "${dest_path}")
    echo "[VOiCES] Existing local size: ${existing_size} bytes"

    if [ "${existing_size}" = "${remote_size}" ]; then
      echo "[VOiCES] Local size matches remote. Verifying gzip and skipping download."
      if verify_gzip "${dest_path}"; then
        echo "[VOiCES] ${label} verified OK."
        return 0
      fi
      echo "[VOiCES] Gzip check failed despite matching size. Attempting resume/re-download."
    fi

    echo "[VOiCES] Local file is incomplete or mismatched. Attempting resume."
  fi

  if have_cmd aria2c; then
    echo "[VOiCES] Downloader         : aria2c"
    download_with_aria2c "${url}" "${dest_path}"
  else
    echo "[VOiCES] Downloader         : curl"
    download_with_curl "${url}" "${dest_path}"
  fi

  local final_size
  final_size=$(local_size "${dest_path}")
  echo "[VOiCES] Final local size   : ${final_size} bytes"

  if [ "${final_size}" != "${remote_size}" ]; then
    echo "[VOiCES] ERROR: size mismatch after download for ${label}"
    echo "  remote: ${remote_size}"
    echo "  local : ${final_size}"
    exit 1
  fi

  verify_gzip "${dest_path}"
  echo "[VOiCES] ${label} verified OK."
}

echo "=== [VOiCES] Download setup ==="
echo "data                    : ${data}"
echo "dataset_root            : ${dataset_root}"
echo "download_dir            : ${download_dir}"
echo "download_recording_data : ${download_recording_data}"
echo "devkit_url              : ${devkit_url}"
echo "recording_url           : ${recording_url}"

download_file "${devkit_url}" "${DEVKIT_PATH}" "VOiCES devkit"

if [ "${download_recording_data}" = true ]; then
  download_file "${recording_url}" "${RECORDING_PATH}" "VOiCES recording_data"
else
  echo "[VOiCES] Skipping recording_data.tar.gz by request."
fi

echo ""
echo "[VOiCES] Downloaded artifacts:"
echo "  ${DEVKIT_PATH}"
if [ "${download_recording_data}" = true ]; then
  echo "  ${RECORDING_PATH}"
fi
