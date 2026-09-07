#!/bin/bash

# Download VoxCeleb1 Dataset for Evaluation
#
# VoxCeleb1 is used for three standard SV evaluation protocols:
#   - Vox1-O (Original): 40 test speakers, ~37k pairs
#   - Vox1-E (Extended): all 1251 speakers, ~579k pairs
#   - Vox1-H (Hard):     all 1251 speakers (same gender+nationality), ~550k pairs
#
# Trial lists are publicly available and downloaded automatically.
# Audio data requires agreeing to the VoxCeleb license:
#   https://mm.kaist.ac.kr/datasets/voxceleb/#download
#
# If you already have VoxCeleb1 audio, symlink or copy it so that:
#   ${data}/voxceleb/raw_data/wav/<speaker_id>/<video_id>/<utterance>.wav
#
# Optional: if you have vox1_dev_wav.zip and vox1_test_wav.zip locally, run:
#   ./local/download_voxceleb.sh --data ${data} --vox1-zip-dir /path/to/dir
# The script will unzip and merge into raw_data/wav/.

data=data
vox1_zip_dir=
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
download_dir=${data}/voxceleb/download_data
rawdata_dir=${data}/voxceleb/raw_data

mkdir -p "${download_dir}"
mkdir -p "${rawdata_dir}"

echo "=== [VoxCeleb] Downloading VoxCeleb1 Evaluation Resources ==="

# -----------------------------------------------------------------------
# Step 1: Download official trial lists (publicly available)
# -----------------------------------------------------------------------
TRIAL_BASE_URL="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta"

trial_files=(
    "veri_test2.txt"       # Vox1-O (cleaned)
    "list_test_all2.txt"   # Vox1-E (cleaned)
    "list_test_hard2.txt"  # Vox1-H (cleaned)
)

echo "[VoxCeleb] Downloading trial lists..."
for fname in "${trial_files[@]}"; do
    target="${download_dir}/${fname}"
    if [ ! -f "${target}" ]; then
        echo "  Downloading ${fname} ..."
        wget --no-check-certificate -q -c -O "${target}" \
            "${TRIAL_BASE_URL}/${fname}" || {
            echo "  WARNING: Failed to download ${fname}."
            echo "  Please download manually from: ${TRIAL_BASE_URL}/${fname}"
            echo "  and place at: ${target}"
        }
    else
        echo "  Found existing ${fname}, skipping."
    fi
done

# Also grab metadata (optional, for reference)
for fname in vox1_meta.csv; do
    target="${download_dir}/${fname}"
    if [ ! -f "${target}" ]; then
        wget --no-check-certificate -q -c -O "${target}" \
            "${TRIAL_BASE_URL}/${fname}" 2>/dev/null || true
    fi
done

# -----------------------------------------------------------------------
# Step 2: VoxCeleb1 audio - extract from local zips when --vox1-zip-dir is set.
#
# When --vox1-zip-dir is set, look for vox1_test_wav.zip and vox1_dev_wav.zip
# in that directory, unzip and merge into raw_data/wav/.
# -----------------------------------------------------------------------
if [ -n "${vox1_zip_dir}" ]; then
    vox1_zip_dir=$(realpath "${vox1_zip_dir}")
    wav_merged="${rawdata_dir}/wav"
    if [ ! -d "${wav_merged}" ] || [ -z "$(find "${wav_merged}" -mindepth 1 -maxdepth 1 -type d -name 'id1*' 2>/dev/null | head -1)" ]; then
        echo "[VoxCeleb] Extracting from local zips in: ${vox1_zip_dir}"
        mkdir -p "${rawdata_dir}"
        for zipname in vox1_test_wav.zip vox1_dev_wav.zip; do
            zippath="${vox1_zip_dir}/${zipname}"
            if [ -f "${zippath}" ]; then
                echo "  Unzipping ${zipname} ..."
                unzip -o -q "${zippath}" -d "${rawdata_dir}" || true
            else
                echo "  [Skip] ${zipname} not found in ${vox1_zip_dir}"
            fi
        done
        # Merge: vox1_test_wav/wav/* and vox1_dev_wav/wav/* -> raw_data/wav/
        mkdir -p "${wav_merged}"
        for sub in vox1_test_wav vox1_dev_wav; do
            if [ -d "${rawdata_dir}/${sub}/wav" ]; then
                echo "  Merging ${sub}/wav into wav/ ..."
                cp -r "${rawdata_dir}/${sub}/wav"/id* "${wav_merged}/" 2>/dev/null || true
            fi
        done
        n_spk=$(find "${wav_merged}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        echo "  Merged ${n_spk} speaker(s) into ${wav_merged}"
    else
        echo "[VoxCeleb] wav/ already present, skipping zip extraction."
    fi
fi

# -----------------------------------------------------------------------
# Step 3: Check if audio data exists
#
# Vox1-O only needs the test set (40 speakers).
# Vox1-E and Vox1-H need the FULL VoxCeleb1 (dev + test, 1251 speakers).
# -----------------------------------------------------------------------

# Check if audio data already exists
wav_root=""
audio_ready=false
for candidate in \
    "${rawdata_dir}/wav" \
    "${rawdata_dir}/voxceleb1" \
    "${rawdata_dir}/VoxCeleb1" \
    "${rawdata_dir}/voxceleb1_test/wav" \
    "${rawdata_dir}/vox1_test_wav/wav"; do
    if [ -d "${candidate}" ]; then
        wav_root="${candidate}"
        break
    fi
done

if [ -n "${wav_root}" ]; then
    n_spks=$(find "${wav_root}" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "[VoxCeleb] Audio data found at: ${wav_root} (${n_spks} speaker dirs)"
    if [ "${n_spks}" -lt 1000 ]; then
        echo "  ERROR: Only ~${n_spks} speakers detected."
        echo "  Vox1-O may work with the test set, but OpenSVBench full-core also needs Vox1-E and Vox1-H."
        echo "  Download and place the full VoxCeleb1 dev + test audio."
    else
        audio_ready=true
    fi
else
    echo ""
    echo "======================================================================="
    echo "[VoxCeleb] Audio data NOT found."
    echo ""
    echo "Please download VoxCeleb1 audio and extract to:"
    echo "  ${rawdata_dir}/wav/<speaker_id>/<video_id>/<utterance>.wav"
    echo ""
    echo "Option 1: Official website (requires registration)"
    echo "  1. Visit: https://mm.kaist.ac.kr/datasets/voxceleb/#download"
    echo "  2. Download 'Vox1 test set' (vox1_test_wav.zip)"
    echo "  3. Download 'Vox1 dev set'  (vox1_dev_wav_parta{a,b,c,d})"
    echo "  4. Extract:"
    echo "     unzip vox1_test_wav.zip -d ${rawdata_dir}/"
    echo "     cat vox1_dev_wav_parta* > vox1_dev_wav.zip"
    echo "     unzip vox1_dev_wav.zip -d ${rawdata_dir}/"
    echo ""
    echo "Option 2: If you have VoxCeleb1 elsewhere, symlink it:"
    echo "  ln -s /path/to/voxceleb1/wav ${rawdata_dir}/wav"
    echo "======================================================================="
    echo ""
fi

echo "=== [VoxCeleb] Download step complete. ==="
echo ""
echo "Trial lists: ${download_dir}/"
for f in "${trial_files[@]}"; do
    if [ -f "${download_dir}/${f}" ]; then
        n=$(wc -l < "${download_dir}/${f}")
        echo "  ${f}: ${n} pairs"
    else
        echo "  ${f}: NOT FOUND"
    fi
done
echo ""
echo "You can now run: local/prepare_voxceleb.sh --data ${data}"

if [ "${audio_ready}" != true ]; then
    echo "[VoxCeleb] ERROR: full VoxCeleb1 audio is not ready for OpenSVBench full-core."
    exit 1
fi
