#!/bin/bash

# Download the CN-Celeb evaluation package (OpenSLR 82).
# Download cn-celeb_v2.tar.gz and extract it into raw_data/.
# The archive contains data/<spk_id>/<genre>-<session>-<idx>.flac and the
# official evaluation files under eval/, including trials.lst and enroll.lst.
# Obtain the complete CN-Celeb1/2 training set from OpenSLR 82 and place it in
# raw_data/ when the derived protocols require it.

data=data
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
download_dir=${data}/cnceleb/download_data
rawdata_dir=${data}/cnceleb/raw_data

mkdir -p "${download_dir}"
mkdir -p "${rawdata_dir}"

echo "=== [CN-Celeb] Downloading CN-Celeb Evaluation Set from OpenSLR ==="

# -----------------------------------------------------------------------
# CN-Celeb-E: publicly downloadable evaluation/trial package
# URL: https://www.openslr.org/resources/82/
# -----------------------------------------------------------------------
EVAL_URL="https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz"
EVAL_ARCHIVE="${download_dir}/cn-celeb_v2.tar.gz"

if [ ! -f "${EVAL_ARCHIVE}" ]; then
    echo "[CN-Celeb] Downloading cn-celeb_v2.tar.gz ..."
    wget --no-check-certificate -c -O "${EVAL_ARCHIVE}" "${EVAL_URL}" || {
        echo "[CN-Celeb] ERROR: Download failed."
        echo "[CN-Celeb] Please download manually from: ${EVAL_URL}"
        echo "[CN-Celeb] and place the file at: ${EVAL_ARCHIVE}"
        exit 1
    }
else
    echo "[CN-Celeb] Found existing archive: ${EVAL_ARCHIVE}, skipping download."
fi

# -----------------------------------------------------------------------
# Extract
# -----------------------------------------------------------------------
if [ ! -d "${rawdata_dir}/cn-celeb_v2" ] && [ ! -d "${rawdata_dir}/data" ] && [ ! -d "${rawdata_dir}/CN-Celeb_flac" ]; then
    echo "[CN-Celeb] Extracting ${EVAL_ARCHIVE} -> ${rawdata_dir}"
    tar -xzf "${EVAL_ARCHIVE}" -C "${rawdata_dir}" || {
        echo "[CN-Celeb] ERROR: Extraction failed."
        exit 1
    }
else
    echo "[CN-Celeb] Raw data already exists in ${rawdata_dir}, skipping extraction."
fi

echo "=== [CN-Celeb] Download complete. ==="
echo ""
echo "Expected layout for prepare_cnceleb.sh:"
echo "  ${rawdata_dir}/"
echo "  |-- CN-Celeb_flac/ or cn-celeb_v2/"
echo "  |   |-- data/<spk_id>/<genre>-<session>-<idx>.flac   # train/dev audio"
echo "  |   +-- eval/<spk_id>/<genre>-<session>-<idx>.flac   # evaluation audio"
echo "  |       +-- lists/trials.lst   # official trials (enroll_id test_path 0|1)"
echo "  |       +-- lists/enroll.lst   # enroll_id -> path"
echo ""
echo "Obtain the complete CN-Celeb1/2 training set from https://www.openslr.org/82/ and place it under ${rawdata_dir}/"
echo ""
echo "Next: local/prepare_cnceleb.sh --data ${data}"
