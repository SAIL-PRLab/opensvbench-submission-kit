#!/bin/bash

# Workflow summary:
#   1. Scan data/ and eval/ and write the Kaldi-style metadata files.
#   2. Convert trials.lst to trials_cnceleb.txt when the official list exists.
#   3. Otherwise generate trials_cnceleb.txt with make_trials_cnceleb.py.
#   4. Always generate the derived genre trial file with
#      make_trials_cnceleb.py.

data=data
max_utts=10000   # The official protocol below forces this to 0.
. tools/parse_options.sh || exit 1

# The official CN-Celeb protocol requires the complete utterance list.
max_utts=0

data=$(realpath "${data}")
rawdata_dir=${data}/cnceleb/raw_data
prep_dir=${data}/cnceleb/prepared

mkdir -p "${prep_dir}"

echo "=== [CN-Celeb] prepare_cnceleb.sh ==="
echo "[CN-Celeb] Raw data root: ${rawdata_dir}"

# -----------------------------------------------------------------------
# Step 1: locate the audio root. Files use
# <spk_id>/<genre>-<session>-<idx>.flac under data/.
# -----------------------------------------------------------------------
if [ -d "${rawdata_dir}/CN-Celeb_flac/data" ]; then
    audio_root="${rawdata_dir}/CN-Celeb_flac/data"
elif [ -d "${rawdata_dir}/cn-celeb_v2/data" ]; then
    audio_root="${rawdata_dir}/cn-celeb_v2/data"
elif [ -d "${rawdata_dir}/data" ]; then
    audio_root="${rawdata_dir}/data"
else
    echo "[CN-Celeb] ERROR: Cannot locate audio root under ${rawdata_dir}."
    echo "  Expected: ${rawdata_dir}/CN-Celeb_flac/data/<spk_id>/<genre>-<session>-<idx>.flac"
    exit 1
fi

echo "[CN-Celeb] Audio root: ${audio_root}"

# -----------------------------------------------------------------------
# Step 2: Generate Kaldi-style lists
# -----------------------------------------------------------------------
wav_scp="${prep_dir}/wav.scp"
utt2spk="${prep_dir}/utt2spk"
utt2genre="${prep_dir}/utt2genre"
utt2session="${prep_dir}/utt2session"

> "${wav_scp}"
> "${utt2spk}"
> "${utt2genre}"
> "${utt2session}"

echo "[CN-Celeb] Scanning audio files (data: <spk_id>/<genre>-<session>-<idx>.flac)..."

find "${audio_root}" -type f \( -name "*.wav" -o -name "*.flac" \) | sort | while read -r wav_path; do
    rel="${wav_path#${audio_root}/}"
    IFS='/' read -ra parts <<< "${rel}"
    [ "${#parts[@]}" -ne 2 ] && continue
    spk_id="${parts[0]}"
    stem="${parts[1]%.*}"
    IFS='-' read -ra fparts <<< "${stem}"
    genre="${fparts[0]}"
    session="${fparts[1]:-00}"
    utt_id="${spk_id}-${genre}-${session}-${stem}"
    echo "${utt_id} ${wav_path}"  >> "${wav_scp}"
    echo "${utt_id} ${spk_id}"   >> "${utt2spk}"
    echo "${utt_id} ${genre}"    >> "${utt2genre}"
    echo "${utt_id} ${session}"  >> "${utt2session}"
done

# -----------------------------------------------------------------------
# Step 2b: if eval/ exists, scan it with the same layout and append it.
# -----------------------------------------------------------------------
eval_root=""
for candidate in "${rawdata_dir}/CN-Celeb_flac/eval" "${rawdata_dir}/cn-celeb_v2/eval" "${rawdata_dir}/eval"; do
    [ -d "${candidate}" ] && eval_root="${candidate}" && break
done
if [ -n "${eval_root}" ]; then
    echo "[CN-Celeb] Scanning eval set: ${eval_root}"
    find "${eval_root}" -type f \( -name "*.wav" -o -name "*.flac" \) | sort | while read -r wav_path; do
        rel="${wav_path#${eval_root}/}"
        IFS='/' read -ra parts <<< "${rel}"
        [ "${#parts[@]}" -ne 2 ] && continue
        spk_id="${parts[0]}"
        stem="${parts[1]%.*}"
        IFS='-' read -ra fparts <<< "${stem}"
        genre="${fparts[0]:-eval}"
        session="${fparts[1]:-00}"
        utt_id="eval-${spk_id}-${genre}-${session}-${stem}"
        echo "${utt_id} ${wav_path}"  >> "${wav_scp}"
        echo "${utt_id} ${spk_id}"     >> "${utt2spk}"
        echo "${utt_id} ${genre}"      >> "${utt2genre}"
        echo "${utt_id} ${session}"    >> "${utt2session}"
    done
fi

n_utts=$(wc -l < "${wav_scp}")
echo "[CN-Celeb] Processed ${n_utts} utterances."

if [ "${n_utts}" -eq 0 ]; then
    echo "[CN-Celeb] ERROR: No wav files found. Check your raw data layout."
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
echo "[CN-Celeb] ${n_spks} speakers."

# -----------------------------------------------------------------------
# Step 4: build the main trial file. Convert the official list when available;
# otherwise generate it with make_trials_cnceleb.
# -----------------------------------------------------------------------
official_trials=""
for candidate in \
    "${rawdata_dir}/eval/lists/trials.lst" \
    "${rawdata_dir}/cn-celeb_v2/eval/lists/trials.lst" \
    "${rawdata_dir}/CN-Celeb_flac/eval/lists/trials.lst" \
    "${rawdata_dir}/lists/trials.lst"; do
    if [ -f "${candidate}" ]; then
        official_trials="${candidate}"
        break
    fi
done

if [ -n "${official_trials}" ]; then
    echo "[CN-Celeb] Using official eval trials for main score: ${official_trials}"
    enroll_lst=""
    trials_dir=$(dirname "${official_trials}")
    [ -f "${trials_dir}/enroll.lst" ] && enroll_lst="${trials_dir}/enroll.lst"
    [ -n "${enroll_lst}" ] && echo "[CN-Celeb] Using enroll.lst: ${enroll_lst}"
    python3 tools/convert_cnceleb_official_trials.py \
        --official-trials "${official_trials}" \
        --wav-scp "${wav_scp}" \
        ${enroll_lst:+--enroll-lst "${enroll_lst}"} \
        --output "${prep_dir}/trials_cnceleb.txt" \
        --condition default
    if [ $? -ne 0 ]; then
        echo "[CN-Celeb] WARNING: Official trial conversion failed, falling back to self-generated trials for main score."
        official_trials=""
    fi
fi

if [ -z "${official_trials}" ]; then
    echo "[CN-Celeb] No official trials: generating main trials_cnceleb.txt with make_trials_cnceleb.py"
    python3 tools/make_trials_cnceleb.py \
        --utt2spk   "${utt2spk}" \
        --utt2genre "${utt2genre}" \
        --output    "${prep_dir}/trials_cnceleb.txt" \
        --max-target-trials-per-condition 20000 \
        --nontarget-ratio 10 \
        --seed 777
    if [ $? -ne 0 ]; then
        echo "[CN-Celeb] ERROR: Trial generation failed."
        exit 1
    fi
fi

# -----------------------------------------------------------------------
# Step 5: build genre-condition trials from data/ only, excluding eval/.
# -----------------------------------------------------------------------
echo "[CN-Celeb] Generating custom genre-condition trials (data set only, no eval)..."
grep -v "^eval-" "${utt2spk}"   > "${prep_dir}/utt2spk_data_only"
grep -v "^eval-" "${utt2genre}" > "${prep_dir}/utt2genre_data_only"
python3 tools/make_trials_cnceleb.py \
    --utt2spk   "${prep_dir}/utt2spk_data_only" \
    --utt2genre "${prep_dir}/utt2genre_data_only" \
    --output    "${prep_dir}/trials_cnceleb_genre.txt" \
    --max-target-trials-per-condition 20000 \
    --nontarget-ratio 10 \
    --seed 777

echo "=== [CN-Celeb] Preparation Done ==="
echo "Kaldi Lists : ${prep_dir}/{wav.scp, utt2spk, utt2genre, utt2session}"
echo "Trial Lists : ${prep_dir}/trials_cnceleb.txt (main), ${prep_dir}/trials_cnceleb_genre.txt (derived genre conditions)"
