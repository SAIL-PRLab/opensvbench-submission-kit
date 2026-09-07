#!/bin/bash

# Download helper for the public Lombard Grid artifacts used by OpenSVBench.
#
# Official source:
#   https://zenodo.org/records/3736465
#
# This script downloads:
#   - lombardgrid_audio.zip
#   - lombardgrid_json.zip
#
# It does not extract archives; run local/prepare_lombardgrid.sh after download.

data=data
download_json=true
no_proxy=false
audio_url="https://zenodo.org/records/3736465/files/lombardgrid_audio.zip?download=1"
json_url="https://zenodo.org/records/3736465/files/lombardgrid_json.zip?download=1"
audio_md5="fa0cfc739705323b53ba50e148b3a144"
json_md5="070bf2b1570d9381215e7e57d8f58a21"

. tools/parse_options.sh || exit 1
set -euo pipefail

data=$(realpath "${data}")
dataset_root="${data}/lombardgrid"
download_dir="${dataset_root}/download_data"
mkdir -p "${download_dir}"

if [ "${no_proxy}" = true ]; then
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
fi

AUDIO_PATH="${download_dir}/lombardgrid_audio.zip"
JSON_PATH="${download_dir}/lombardgrid_json.zip"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
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
  stat -c '%s' "$1"
}

verify_zip() {
  local path="$1"
  local label="$2"
  echo "[LombardGrid] Testing zip integrity: ${path}"
  if have_cmd unzip; then
    unzip -tq "${path}" >/dev/null
  else
    python3 - <<'PY' "${path}"
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    bad = zf.testzip()
if bad:
    raise SystemExit(f"bad zip member: {bad}")
PY
  fi
  echo "[LombardGrid] ${label} zip verified OK."
}

verify_md5() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [ -z "${expected}" ]; then
    return 0
  fi
  local actual
  actual=$(md5sum "${path}" | awk '{print $1}')
  echo "[LombardGrid] ${label} md5: ${actual}"
  if [ "${actual}" != "${expected}" ]; then
    echo "[LombardGrid] ERROR: md5 mismatch for ${label}"
    echo "  expected: ${expected}"
    echo "  actual  : ${actual}"
    exit 1
  fi
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
  local expected_md5="$4"

  echo ""
  echo "[LombardGrid] === ${label} ==="
  echo "[LombardGrid] URL  : ${url}"
  echo "[LombardGrid] Path : ${dest_path}"

  local headers
  headers=$(fetch_headers "${url}")

  local remote_size
  local remote_mtime
  local remote_etag
  remote_size=$(header_value "${headers}" "Content-Length")
  remote_mtime=$(header_value "${headers}" "Last-Modified")
  remote_etag=$(header_value "${headers}" "ETag")

  echo "[LombardGrid] Remote size     : ${remote_size:-unknown} bytes"
  echo "[LombardGrid] Remote modified : ${remote_mtime:-unknown}"
  echo "[LombardGrid] Remote ETag     : ${remote_etag:-unknown}"

  if [ -f "${dest_path}" ]; then
    local existing_size
    existing_size=$(local_size "${dest_path}")
    echo "[LombardGrid] Existing size   : ${existing_size} bytes"
    if [ -n "${remote_size}" ] && [ "${existing_size}" = "${remote_size}" ]; then
      echo "[LombardGrid] Local size matches remote. Verifying and skipping download."
      verify_zip "${dest_path}" "${label}"
      verify_md5 "${dest_path}" "${expected_md5}" "${label}"
      echo "[LombardGrid] ${label} verified OK."
      return 0
    fi
    echo "[LombardGrid] Local file is incomplete or mismatched. Attempting resume."
  fi

  if have_cmd aria2c; then
    echo "[LombardGrid] Downloader      : aria2c"
    download_with_aria2c "${url}" "${dest_path}"
  else
    echo "[LombardGrid] Downloader      : curl"
    download_with_curl "${url}" "${dest_path}"
  fi

  local final_size
  final_size=$(local_size "${dest_path}")
  echo "[LombardGrid] Final size      : ${final_size} bytes"
  if [ -n "${remote_size}" ] && [ "${final_size}" != "${remote_size}" ]; then
    echo "[LombardGrid] ERROR: size mismatch after download for ${label}"
    echo "  remote: ${remote_size}"
    echo "  local : ${final_size}"
    exit 1
  fi

  verify_zip "${dest_path}" "${label}"
  verify_md5 "${dest_path}" "${expected_md5}" "${label}"
  echo "[LombardGrid] ${label} verified OK."
}

echo "=== [LombardGrid] Download setup ==="
echo "data          : ${data}"
echo "dataset_root  : ${dataset_root}"
echo "download_dir  : ${download_dir}"
echo "download_json : ${download_json}"
echo "no_proxy      : ${no_proxy}"
echo "audio_url     : ${audio_url}"
echo "json_url      : ${json_url}"

download_file "${audio_url}" "${AUDIO_PATH}" "Lombard Grid audio" "${audio_md5}"

if [ "${download_json}" = true ]; then
  download_file "${json_url}" "${JSON_PATH}" "Lombard Grid JSON metadata" "${json_md5}"
else
  echo "[LombardGrid] Skipping lombardgrid_json.zip by request."
fi

echo ""
echo "[LombardGrid] Downloaded artifacts:"
echo "  ${AUDIO_PATH}"
if [ "${download_json}" = true ]; then
  echo "  ${JSON_PATH}"
fi
