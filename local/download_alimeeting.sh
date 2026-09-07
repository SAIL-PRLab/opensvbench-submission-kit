#!/bin/bash

# Download the AliMeeting test set (M2MeT, ICASSP 2022).
#   Test_Ali.tar.gz (~8.9 GB) contains:
#     Test_Ali_far/  : 8-channel microphone-array recordings (far field)
#     Test_Ali_near/ : headset recordings (near field)
#   20 meeting sessions and 25 speakers.
#   Upstream release: CC BY-SA 4.0.
#
# Source: https://openslr.org/119/

data=data
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
download_dir="${data}/alimeeting/download_data"
raw_dir="${data}/alimeeting/raw_data"

mkdir -p "${download_dir}" "${raw_dir}"

BASE_URL="https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr"
ARCHIVE="Test_Ali.tar.gz"
archive_path="${download_dir}/${ARCHIVE}"

# Skip extraction when the data is already present.
if [ -d "${raw_dir}/Test_Ali" ] || [ -d "${raw_dir}/Test_Ali_far" ]; then
  echo "[AliMeeting] Already extracted. Skipping."
  exit 0
fi

# -----------------------------------------------------------------------
# Download.
# -----------------------------------------------------------------------
if [ ! -f "${archive_path}" ]; then
  echo "[AliMeeting] Downloading ${ARCHIVE} (~8.9 GB) from Alibaba OSS ..."
  echo "  URL: ${BASE_URL}/${ARCHIVE}"
  wget --no-check-certificate -c -O "${archive_path}" "${BASE_URL}/${ARCHIVE}" || {
    echo "[AliMeeting] ERROR: Download failed."
    echo "  Please manually download from: ${BASE_URL}/${ARCHIVE}"
    echo "  and place the file at: ${archive_path}"
    exit 1
  }
else
  echo "[AliMeeting] Found existing archive: ${archive_path}, skipping download."
fi

echo "[AliMeeting] Archive size: $(du -sh ${archive_path} | cut -f1)"

# -----------------------------------------------------------------------
# Extract.
# -----------------------------------------------------------------------
echo "[AliMeeting] Extracting ${archive_path} -> ${raw_dir} ..."
tar -xzf "${archive_path}" -C "${raw_dir}" --no-same-owner 2>/dev/null || {
  echo "[AliMeeting] ERROR: Extraction failed."
  exit 1
}

echo "[AliMeeting] Download & extraction complete."
echo "  raw_dir: ${raw_dir}"
echo ""
echo "Expected layout:"
echo "  ${raw_dir}/Test_Ali/"
echo "  |-- Test_Ali_far/"
echo "  |   |-- audio_dir/      # R0001_M0001_MS001.wav  (8ch array)"
echo "  |   +-- textgrid_dir/   # R0001_M0001_MS001.TextGrid"
echo "  +-- Test_Ali_near/"
echo "      |-- audio_dir/      # R0001_M0001_F_SPK0001.wav  (mono headset)"
echo "      +-- textgrid_dir/   # R0001_M0001_F_SPK0001.TextGrid"
echo ""
echo "Next: local/prepare_alimeeting.sh --data ${data}"
