#!/bin/bash

# Prepare VoxCeleb1 Dataset for Evaluation
#
# 1. Scans raw_data for wav files: <speaker_id>/<video_id>/<utterance>.wav
# 2. Generates Kaldi-style lists: wav.scp, utt2spk, spk2utt
# 3. Converts official trial lists to OpenSVBench format:
#    - trials_vox1_O.txt  (Vox1-O Original, 40 test speakers)
#    - trials_vox1_E.txt  (Vox1-E Extended, all 1251 speakers)
#    - trials_vox1_H.txt  (Vox1-H Hard, same gender+nationality)
#
# utt_id convention: <spk_id>-<video_id>-<utt_name>
#   e.g. id10270-x6uYqmx31kE-00001

data=data
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
download_dir=${data}/voxceleb/download_data
rawdata_dir=${data}/voxceleb/raw_data
prep_dir=${data}/voxceleb/prepared

mkdir -p "${prep_dir}"

echo "=== [VoxCeleb] prepare_voxceleb.sh ==="
echo "[VoxCeleb] Raw data root: ${rawdata_dir}"

# -----------------------------------------------------------------------
# Step 1: Locate audio root
# VoxCeleb1 layouts after extraction:
#   raw_data/wav/<spk_id>/<video_id>/<utt>.wav          (standard)
#   raw_data/voxceleb1/wav/<spk_id>/...                  (nested)
#   raw_data/VoxCeleb1/wav/<spk_id>/...                  (case variant)
#   raw_data/<spk_id>/<video_id>/<utt>.wav               (direct)
# -----------------------------------------------------------------------
audio_root=""
for candidate in \
    "${rawdata_dir}/wav" \
    "${rawdata_dir}/voxceleb1/wav" \
    "${rawdata_dir}/VoxCeleb1/wav" \
    "${rawdata_dir}/voxceleb1" \
    "${rawdata_dir}/VoxCeleb1" \
    "${rawdata_dir}"; do
    if [ -d "${candidate}" ]; then
        # Verify it contains speaker directories (id1xxxx)
        n_id_dirs=$(find "${candidate}" -mindepth 1 -maxdepth 1 -type d -name "id1*" 2>/dev/null | wc -l)
        if [ "${n_id_dirs}" -gt 0 ]; then
            audio_root="${candidate}"
            break
        fi
    fi
done

if [ -z "${audio_root}" ]; then
    echo "[VoxCeleb] ERROR: Cannot locate audio root under ${rawdata_dir}."
    echo "  Expected: ${rawdata_dir}/wav/<speaker_id>/<video_id>/<utt>.wav"
    echo "  Please download VoxCeleb1 audio first (see local/download_voxceleb.sh)."
    exit 1
fi

echo "[VoxCeleb] Audio root: ${audio_root}"

# -----------------------------------------------------------------------
# Step 2: Generate Kaldi-style lists
# -----------------------------------------------------------------------
wav_scp="${prep_dir}/wav.scp"
utt2spk="${prep_dir}/utt2spk"

> "${wav_scp}"
> "${utt2spk}"

echo "[VoxCeleb] Scanning audio files..."

find "${audio_root}" -type f \( -name "*.wav" -o -name "*.flac" \) | sort | while read -r wav_path; do
    rel="${wav_path#${audio_root}/}"
    IFS='/' read -ra parts <<< "${rel}"

    if [ "${#parts[@]}" -ge 3 ]; then
        # <spk_id>/<video_id>/<utt>.wav
        spk_id="${parts[0]}"
        video_id="${parts[1]}"
        utt_name="${parts[2]%.*}"
    elif [ "${#parts[@]}" -eq 2 ]; then
        # <spk_id>/<utt>.wav (flat layout)
        spk_id="${parts[0]}"
        video_id="unknown"
        utt_name="${parts[1]%.*}"
    else
        continue
    fi

    utt_id="${spk_id}-${video_id}-${utt_name}"

    echo "${utt_id} ${wav_path}" >> "${wav_scp}"
    echo "${utt_id} ${spk_id}"  >> "${utt2spk}"
done

n_utts=$(wc -l < "${wav_scp}")
echo "[VoxCeleb] Processed ${n_utts} utterances."

if [ "${n_utts}" -eq 0 ]; then
    echo "[VoxCeleb] ERROR: No audio files found under ${audio_root}."
    exit 1
fi

# -----------------------------------------------------------------------
# Step 3: Generate spk2utt
# -----------------------------------------------------------------------
awk '{print $2, $1}' "${utt2spk}" | sort -k1 | \
awk '{
    if ($1 != prev) {
        if (prev != "") print line;
        prev=$1; line=$1 " " $2;
    } else {
        line=line " " $2;
    }
} END { if (prev != "") print line; }' > "${prep_dir}/spk2utt"

n_spks=$(wc -l < "${prep_dir}/spk2utt")
echo "[VoxCeleb] ${n_spks} speakers."

# -----------------------------------------------------------------------
# Step 4: Convert official trial lists to OpenSVBench format
# -----------------------------------------------------------------------
echo "[VoxCeleb] Converting official trial lists..."

# (source_file  output_name  condition_label)
trial_map=(
    "veri_test2.txt|trials_vox1_O.txt|vox1_O"
    "list_test_all2.txt|trials_vox1_E.txt|vox1_E"
    "list_test_hard2.txt|trials_vox1_H.txt|vox1_H"
)

for entry in "${trial_map[@]}"; do
    IFS='|' read -r src_file out_name cond_label <<< "$entry"
    src_path="${download_dir}/${src_file}"
    out_path="${prep_dir}/${out_name}"

    if [ ! -f "${src_path}" ]; then
        echo "  [Skip] ${src_file}: not found in ${download_dir}"
        echo "         Run local/download_voxceleb.sh first to get trial lists."
        continue
    fi

    echo "  Converting ${src_file} -> ${out_name} (condition: ${cond_label})"
    python3 tools/make_trials_voxceleb.py \
        --trial-src "${src_path}" \
        --wav-scp "${wav_scp}" \
        --condition "${cond_label}" \
        --output "${out_path}"
done

echo ""
echo "=== [VoxCeleb] Preparation Done ==="
echo "Kaldi Lists : ${prep_dir}/{wav.scp, utt2spk, spk2utt}"
echo "Trial Lists :"
for entry in "${trial_map[@]}"; do
    IFS='|' read -r _ out_name _ <<< "$entry"
    f="${prep_dir}/${out_name}"
    if [ -f "$f" ]; then
        n=$(wc -l < "$f")
        echo "  ${out_name}: ${n} trials"
    fi
done
