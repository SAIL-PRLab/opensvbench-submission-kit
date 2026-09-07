#!/bin/bash

set -eo pipefail

# Download helper for the public FFSVC2022 artifacts used by OpenSVBench.
#
# What this script does automatically:
#   - downloads the public dev trial file with keys
#   - downloads the public dev meta file
#   - downloads the public dev wav archive and extracts it
#   - optionally downloads the public eval trial file
#
# What this script does NOT download by default:
#   - eval wav archive (~40+ GB)
#
# Rationale:
#   - OpenSVBench v0.1 should use the public dev set first
#   - the eval wav package is large and not needed for the public local benchmark
#
# Official sources:
#   https://ffsvc.github.io/dataset/
#   https://ffsvc.github.io/assets/pdf/ffsvc2022_plan_v2.pdf

data=data
with_eval=false
download_dev_wav=true
download_eval_wav=false
dev_wav_archive=
eval_wav_archive=

. tools/parse_options.sh || exit 1
set -u

data=$(realpath "${data}")
download_dir="${data}/ffsvc2022/download_data"
raw_dir="${data}/ffsvc2022/raw_data"
mkdir -p "${download_dir}" "${raw_dir}"

DEV_TRIAL_URL="https://ffsvc.github.io/assets/ffsvc2022/trials_dev_keys"
DEV_META_URL="https://ffsvc.github.io/assets/ffsvc2022/dev_meta_list"
EVAL_TRIAL_URL="https://ffsvc.github.io/assets/ffsvc2022/trials_eval"
DEV_WAV_URL="https://zenodo.org/records/6466068/files/dev.tar.gz?download=1"
EVAL_WAV_URL="https://zenodo.org/api/records/6461072/files-archive"

DEV_TRIAL_MD5="de0a22b157c3f5173a58b2bbc2ee0cab"
DEV_META_MD5="8ad1bdc99529e85be5ed453873a670e2"
EVAL_TRIAL_MD5="4a2af9dc8132ddd8826439cc3c1e440f"
DEV_WAV_MD5="e7fb015c5c6389da21d8966ce549e59f"
EVAL_WAV_MD5="e3cf9c525a64f53b9fe960cc1705b6f5"

DEV_WAV_RECORD_URL="https://zenodo.org/records/6466068"
EVAL_WAV_RECORD_URL="https://zenodo.org/records/6461072"

DEV_TRIAL_PATH="${download_dir}/trials_dev_keys"
DEV_META_PATH="${download_dir}/dev_meta_list"
EVAL_TRIAL_PATH="${download_dir}/trials_eval"
DEV_WAV_PATH="${download_dir}/dev.tar.gz"
EVAL_WAV_PATH="${download_dir}/ffsvc2022_eval.zip"

md5_of_file() {
  local path="$1"
  md5sum "${path}" | awk '{print $1}'
}

check_md5() {
  local path="$1"
  local expected="$2"
  if [ ! -f "${path}" ]; then
    return 1
  fi
  local actual
  actual=$(md5_of_file "${path}")
  if [ "${actual}" != "${expected}" ]; then
    echo "[FFSVC2022] ERROR: MD5 mismatch for ${path}"
    echo "  expected: ${expected}"
    echo "  actual  : ${actual}"
    return 1
  fi
  return 0
}

download_public_file() {
  local url="$1"
  local out_path="$2"
  local expected_md5="$3"
  local label="$4"

  if check_md5 "${out_path}" "${expected_md5}"; then
    echo "[FFSVC2022] Found cached ${label}: ${out_path}"
    return 0
  fi

  echo "[FFSVC2022] Downloading ${label} ..."
  echo "  URL: ${url}"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -C - -o "${out_path}" "${url}" || true
  fi

  if [ ! -f "${out_path}" ] || [ ! -s "${out_path}" ]; then
    if command -v wget >/dev/null 2>&1; then
      wget --no-check-certificate -c -O "${out_path}" "${url}" || true
    fi
  fi

  if ! check_md5 "${out_path}" "${expected_md5}"; then
    echo "[FFSVC2022] ERROR: failed to fetch ${label} from the official URL."
    echo "  Please manually download: ${url}"
    echo "  and place it at: ${out_path}"
    exit 1
  fi
}

maybe_match_archive_by_md5() {
  local expected_md5="$1"
  local found=""
  shopt -s nullglob
  for candidate in "${download_dir}"/*; do
    if [ ! -f "${candidate}" ]; then
      continue
    fi
    local actual
    actual=$(md5_of_file "${candidate}" 2>/dev/null || true)
    if [ "${actual}" = "${expected_md5}" ]; then
      found="${candidate}"
      break
    fi
  done
  shopt -u nullglob
  echo "${found}"
}

extract_archive() {
  local archive_path="$1"
  local dest_dir="$2"

  mkdir -p "${dest_dir}"
  echo "[FFSVC2022] Extracting ${archive_path} -> ${dest_dir}"

  case "${archive_path}" in
    *.tar.gz|*.tgz)
      tar -xzf "${archive_path}" -C "${dest_dir}" --no-same-owner
      ;;
    *.tar)
      tar -xf "${archive_path}" -C "${dest_dir}" --no-same-owner
      ;;
    *.zip)
      unzip -o "${archive_path}" -d "${dest_dir}" >/dev/null
      ;;
    *.7z)
      7z x -y "-o${dest_dir}" "${archive_path}" >/dev/null
      ;;
    *)
      echo "[FFSVC2022] ERROR: unsupported archive format: ${archive_path}"
      echo "  Supported: .tar.gz .tgz .tar .zip .7z"
      exit 1
      ;;
  esac
}

flatten_single_nested_dir() {
  local dest_dir="$1"
  local nested_dir="${dest_dir}/$(basename "${dest_dir}")"

  if [ ! -d "${nested_dir}" ]; then
    return 0
  fi

  if find "${dest_dir}" -mindepth 1 -maxdepth 1 ! -path "${nested_dir}" | read -r _; then
    return 0
  fi

  echo "[FFSVC2022] Flattening nested directory ${nested_dir} -> ${dest_dir}"
  find "${nested_dir}" -mindepth 1 -maxdepth 1 -exec mv {} "${dest_dir}/" \;
  rmdir "${nested_dir}"
}

handle_archive() {
  local user_archive="$1"
  local expected_md5="$2"
  local auto_download="$3"
  local url="$4"
  local default_download_path="$5"
  local record_url="$6"
  local label="$7"
  local extract_dir="$8"

  local archive_path="${user_archive}"
  if [ -z "${archive_path}" ]; then
    archive_path=$(maybe_match_archive_by_md5 "${expected_md5}")
  fi

  if [ -z "${archive_path}" ] && [ "${auto_download}" = true ]; then
    archive_path="${default_download_path}"
    download_public_file "${url}" "${archive_path}" "${expected_md5}" "${label}"
  fi

  if [ -z "${archive_path}" ]; then
    echo "[FFSVC2022] ${label} is not available locally and was not downloaded."
    echo "  Download page: ${record_url}"
    echo "  Direct link   : ${url}"
    echo "  You can place the archive under ${download_dir}"
    echo "  or pass it explicitly on the command line."
    return 0
  fi

  archive_path=$(realpath "${archive_path}")
  if ! check_md5 "${archive_path}" "${expected_md5}"; then
    echo "[FFSVC2022] ERROR: ${label} archive MD5 mismatch."
    echo "  archive : ${archive_path}"
    echo "  expected: ${expected_md5}"
    exit 1
  fi

  extract_archive "${archive_path}" "${extract_dir}"
  flatten_single_nested_dir "${extract_dir}"
  echo "[FFSVC2022] ${label} extracted."
}

echo "=== [FFSVC2022] Download setup ==="
echo "data          : ${data}"
echo "download_dir  : ${download_dir}"
echo "raw_dir       : ${raw_dir}"
echo "with_eval     : ${with_eval}"
echo "download_dev_wav  : ${download_dev_wav}"
echo "download_eval_wav : ${download_eval_wav}"

download_public_file "${DEV_TRIAL_URL}" "${DEV_TRIAL_PATH}" "${DEV_TRIAL_MD5}" "Dev Trial file"
download_public_file "${DEV_META_URL}" "${DEV_META_PATH}" "${DEV_META_MD5}" "Dev Meta file"

if [ "${with_eval}" = true ]; then
  download_public_file "${EVAL_TRIAL_URL}" "${EVAL_TRIAL_PATH}" "${EVAL_TRIAL_MD5}" "Eval Trial file"
fi

handle_archive \
  "${dev_wav_archive}" \
  "${DEV_WAV_MD5}" \
  "${download_dev_wav}" \
  "${DEV_WAV_URL}" \
  "${DEV_WAV_PATH}" \
  "${DEV_WAV_RECORD_URL}" \
  "Dev WAV files" \
  "${raw_dir}/dev"

if [ "${with_eval}" = true ]; then
  handle_archive \
    "${eval_wav_archive}" \
    "${EVAL_WAV_MD5}" \
    "${download_eval_wav}" \
    "${EVAL_WAV_URL}" \
    "${EVAL_WAV_PATH}" \
    "${EVAL_WAV_RECORD_URL}" \
    "Eval WAV files" \
    "${raw_dir}/eval"
fi

echo ""
echo "[FFSVC2022] Current local artifacts:"
echo "  Dev trial : ${DEV_TRIAL_PATH}"
echo "  Dev meta  : ${DEV_META_PATH}"
if [ "${with_eval}" = true ]; then
  echo "  Eval trial: ${EVAL_TRIAL_PATH}"
fi
echo ""
echo "[FFSVC2022] Notes:"
echo "  - OpenSVBench v0.1 should use the public dev set first."
echo "  - Dev wav is downloaded by default because the direct Zenodo file is public."
echo "  - Eval wav is NOT downloaded by default because the archive is very large and not needed for v0.1."
echo "  - After the wav package is available, run local/prepare_ffsvc2022.sh to build OpenSVBench manifests and trials."
