#!/bin/bash

# Download helper for the public CHiME-6 dev-set artifacts used by OpenSVBench.
#
# Official source:
#   https://openslr.org/150/
#
# This script downloads:
#   - CHiME6_dev.tar.gz
#   - CHiME6_transcriptions.tar.gz
#
# It does not extract the archives; run local/prepare_chime6.sh after download.

data=data
base_url="https://openslr.trmal.net/resources/150"
dev_url=
transcriptions_url=

. tools/parse_options.sh || exit 1
set -euo pipefail

data=$(realpath "${data}")
dataset_root="${data}/chime6"
download_dir="${dataset_root}/download_data"
mkdir -p "${download_dir}"

dev_url="${dev_url:-${base_url}/CHiME6_dev.tar.gz}"
transcriptions_url="${transcriptions_url:-${base_url}/CHiME6_transcriptions.tar.gz}"

DEV_PATH="${download_dir}/CHiME6_dev.tar.gz"
TRANS_PATH="${download_dir}/CHiME6_transcriptions.tar.gz"

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

verify_archive() {
  local path="$1"
  local label="$2"
  local err_file
  err_file=$(mktemp)

  echo "[CHiME-6] Testing gzip integrity: ${path}"
  if gzip -t "${path}" 2>"${err_file}"; then
    rm -f "${err_file}"
    echo "[CHiME-6] ${label} gzip verified OK."
    return 0
  fi

  # Some OpenSLR tarballs can report "trailing garbage ignored" while still
  # being extractable. Accept that specific case only if tar can list entries.
  if grep -q "trailing garbage ignored" "${err_file}"; then
    echo "[CHiME-6] gzip reported trailing garbage for ${label}; checking tar listing."
    if tar --ignore-zeros -tzf "${path}" >/dev/null 2>"${err_file}"; then
      rm -f "${err_file}"
      echo "[CHiME-6] ${label} tar listing verified OK."
      return 0
    fi
    if grep -q "trailing garbage ignored" "${err_file}"; then
      local list_file
      local count
      local tar_status
      list_file=$(mktemp)
      set +e
      tar --ignore-zeros -tzf "${path}" >"${list_file}" 2>/dev/null
      tar_status=$?
      set -e
      count=$(wc -l < "${list_file}" | tr -d ' \t')
      rm -f "${list_file}"
      if [ "${count}" -gt 0 ]; then
        rm -f "${err_file}"
        echo "[CHiME-6] ${label} tar listing produced ${count} entries with status ${tar_status}; accepting trailing-garbage warning."
        return 0
      fi
    fi
  fi

  echo "[CHiME-6] ERROR: archive verification failed for ${label}"
  cat "${err_file}" || true
  rm -f "${err_file}"
  return 1
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
  echo "[CHiME-6] === ${label} ==="
  echo "[CHiME-6] URL  : ${url}"
  echo "[CHiME-6] Path : ${dest_path}"

  local headers
  headers=$(fetch_headers "${url}")

  local remote_size
  local remote_mtime
  local remote_etag
  remote_size=$(header_value "${headers}" "Content-Length")
  remote_mtime=$(header_value "${headers}" "Last-Modified")
  remote_etag=$(header_value "${headers}" "ETag")

  if [ -z "${remote_size}" ]; then
    echo "[CHiME-6] ERROR: failed to read remote Content-Length for ${label}"
    exit 1
  fi

  echo "[CHiME-6] Remote size        : ${remote_size} bytes"
  echo "[CHiME-6] Remote modified    : ${remote_mtime:-unknown}"
  echo "[CHiME-6] Remote ETag        : ${remote_etag:-unknown}"

  if [ -f "${dest_path}" ]; then
    local existing_size
    existing_size=$(local_size "${dest_path}")
    echo "[CHiME-6] Existing local size: ${existing_size} bytes"

    if [ "${existing_size}" = "${remote_size}" ]; then
      echo "[CHiME-6] Local size matches remote. Verifying archive and skipping download."
      verify_archive "${dest_path}" "${label}"
      echo "[CHiME-6] ${label} verified OK."
      return 0
    fi

    echo "[CHiME-6] Local file is incomplete or mismatched. Attempting resume."
  fi

  if have_cmd aria2c; then
    echo "[CHiME-6] Downloader         : aria2c"
    download_with_aria2c "${url}" "${dest_path}"
  else
    echo "[CHiME-6] Downloader         : curl"
    download_with_curl "${url}" "${dest_path}"
  fi

  local final_size
  final_size=$(local_size "${dest_path}")
  echo "[CHiME-6] Final local size   : ${final_size} bytes"

  if [ "${final_size}" != "${remote_size}" ]; then
    echo "[CHiME-6] ERROR: size mismatch after download for ${label}"
    echo "  remote: ${remote_size}"
    echo "  local : ${final_size}"
    exit 1
  fi

  verify_archive "${dest_path}" "${label}"
  echo "[CHiME-6] ${label} verified OK."
}

echo "=== [CHiME-6] Download setup ==="
echo "data         : ${data}"
echo "dataset_root : ${dataset_root}"
echo "download_dir : ${download_dir}"
echo "dev_url      : ${dev_url}"
echo "trans_url    : ${transcriptions_url}"

download_file "${dev_url}" "${DEV_PATH}" "CHiME-6 dev audio"
download_file "${transcriptions_url}" "${TRANS_PATH}" "CHiME-6 transcriptions"

echo ""
echo "[CHiME-6] Downloaded artifacts:"
echo "  ${DEV_PATH}"
echo "  ${TRANS_PATH}"
