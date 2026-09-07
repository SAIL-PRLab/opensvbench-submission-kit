#!/bin/bash

# Download the 3D-Speaker test set, metadata, and trial files.
# Download test.tar.gz (~1.1G) and 3dspeaker_files.tar.gz into raw_data/.
# Expected layout after extraction:
#   raw_data/test/<spk_id>/<utt>.wav          # test audio
#   raw_data/files/test_wav.scp                # utt_id -> relative path
#   raw_data/files/test_utt2spk               # utt_id -> spk_id
#   raw_data/files/test_utt2info.csv           # metadata (device, distance, dialect, ...)
#   raw_data/files/trials_cross_device         # official cross-device trials
#   raw_data/files/trials_cross_distance       # official cross-distance trials
#   raw_data/files/trials_cross_dialect        # official cross-dialect trials
#
# Project page: https://3dspeaker.github.io/
# Dataset page: https://modelscope.cn/datasets/3D-Speaker/3D-Speaker

data=data
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
download_dir=${data}/3dspeaker/download_data
rawdata_dir=${data}/3dspeaker/raw_data

mkdir -p "${download_dir}"
mkdir -p "${rawdata_dir}"

echo "=== [3D-Speaker] Downloading 3D-Speaker Test Set ==="

# -----------------------------------------------------------------------
# 3D-Speaker test audio
# -----------------------------------------------------------------------
TEST_URL="https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/3D-Speaker/test.tar.gz"
TEST_ARCHIVE="${download_dir}/test.tar.gz"
TEST_MD5="45972606dd10d3f7c1c31f27acdfbed7"

if [ ! -f "${TEST_ARCHIVE}" ]; then
    echo "[3D-Speaker] Downloading test.tar.gz (~1.1G) ..."
    wget --no-check-certificate -c -O "${TEST_ARCHIVE}" "${TEST_URL}" || {
        echo "[3D-Speaker] ERROR: Download failed."
        echo "[3D-Speaker] Please download manually from: ${TEST_URL}"
        echo "[3D-Speaker] and place the file at: ${TEST_ARCHIVE}"
        exit 1
    }
else
    echo "[3D-Speaker] Found existing archive: ${TEST_ARCHIVE}, skipping download."
fi

# MD5 check
echo "[3D-Speaker] Verifying MD5 checksum for test.tar.gz ..."
actual_md5=$(md5sum "${TEST_ARCHIVE}" | awk '{print $1}')
if [ "${actual_md5}" != "${TEST_MD5}" ]; then
    echo "[3D-Speaker] WARNING: MD5 mismatch!"
    echo "  Expected: ${TEST_MD5}"
    echo "  Got:      ${actual_md5}"
    echo "  The file may be corrupted or incomplete. Consider re-downloading."
else
    echo "[3D-Speaker] MD5 OK."
fi

# -----------------------------------------------------------------------
# 3D-Speaker metadata + trial files
# -----------------------------------------------------------------------
FILES_URL="https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/3D-Speaker/3dspeaker_files.tar.gz"
FILES_ARCHIVE="${download_dir}/3dspeaker_files.tar.gz"

if [ ! -f "${FILES_ARCHIVE}" ]; then
    echo "[3D-Speaker] Downloading 3dspeaker_files.tar.gz ..."
    wget --no-check-certificate -c -O "${FILES_ARCHIVE}" "${FILES_URL}" || {
        echo "[3D-Speaker] ERROR: Download failed."
        echo "[3D-Speaker] Please download manually from: ${FILES_URL}"
        echo "[3D-Speaker] and place the file at: ${FILES_ARCHIVE}"
        exit 1
    }
else
    echo "[3D-Speaker] Found existing archive: ${FILES_ARCHIVE}, skipping download."
fi

# -----------------------------------------------------------------------
# Extract
# -----------------------------------------------------------------------
if [ ! -d "${rawdata_dir}/test" ]; then
    echo "[3D-Speaker] Extracting ${TEST_ARCHIVE} -> ${rawdata_dir}"
    tar -xzf "${TEST_ARCHIVE}" -C "${rawdata_dir}" --no-same-owner 2>/dev/null || {
        echo "[3D-Speaker] ERROR: Extraction of test.tar.gz failed."
        exit 1
    }
else
    echo "[3D-Speaker] test/ already exists in ${rawdata_dir}, skipping extraction."
fi

if [ ! -d "${rawdata_dir}/files" ]; then
    echo "[3D-Speaker] Extracting ${FILES_ARCHIVE} -> ${rawdata_dir}"
    # 3dspeaker_files.tar.gz contains files/ as top-level dir
    # We extract with --strip-components=1 to place contents directly into files/
    tar -xzf "${FILES_ARCHIVE}" -C "${rawdata_dir}" --strip-components=1 --no-same-owner 2>/dev/null || {
        echo "[3D-Speaker] ERROR: Extraction of 3dspeaker_files.tar.gz failed."
        exit 1
    }
else
    echo "[3D-Speaker] files/ already exists in ${rawdata_dir}, skipping extraction."
fi

echo "=== [3D-Speaker] Download complete. ==="
echo ""
echo "Expected layout for prepare_3dspeaker.sh:"
echo "  ${rawdata_dir}/"
echo "  |-- test/<spk_id>/<utt>.wav               # test audio"
echo "  |-- files/"
echo "      |-- test_wav.scp                        # utt_id -> relative path"
echo "      |-- test_utt2spk                        # utt_id -> spk_id"
echo "      |-- test_utt2info.csv                   # metadata"
echo "      |-- trials_cross_device                 # official cross-device trials"
echo "      |-- trials_cross_distance               # official cross-distance trials"
echo "      +-- trials_cross_dialect                # official cross-dialect trials"
echo ""
echo "Next: local/prepare_3dspeaker.sh --data ${data}"
