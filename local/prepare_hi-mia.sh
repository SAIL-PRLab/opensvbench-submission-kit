#!/bin/bash

# Prepare HI-MIA Dataset
# 1. Scans raw_data/train|dev|test
# 2. Generates standard Kaldi files: wav.scp, utt2spk, spk2utt
# 3. Generates metadata_{split}.csv (for backup/viewing)
# 4. Generates Short Duration Trial List (trials_short.txt) using the test set.

data=data
stage=1
stop_stage=1
max_utts=10000   # set to 0 to disable subsampling

. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
rawdata_dir=${data}/hi-mia/raw_data
prep_dir=${data}/hi-mia/prepared

mkdir -p "${prep_dir}"

echo "=== [HI-MIA] prepare_hi-mia.sh ==="
echo "[HI-MIA] Raw data root: ${rawdata_dir}"

# ---------------------------------------------------------
# Step 1: Scan folders and generate metadata (wav.scp, utt2spk)
# ---------------------------------------------------------
# HI-MIA filename format: SV0340_7_01_S0100.wav
# Speaker ID: SV0340 (First field separated by '_')

for split in test; do # train dev
  split_dir=${rawdata_dir}/${split}

  if [ ! -d "${split_dir}" ]; then
    echo "[HI-MIA] WARNING: ${split_dir} not found. Skipping."
    continue
  fi

  echo "[HI-MIA] Processing split: ${split} ..."

  # Define output files.
  my_prep_dir=${prep_dir}
  mkdir -p "${my_prep_dir}"

  wav_scp=${my_prep_dir}/wav.scp
  utt2spk=${my_prep_dir}/utt2spk
  meta_csv=${prep_dir}/metadata_${split}.csv

  # Clear previous output.
  > "${wav_scp}"
  > "${utt2spk}"
  echo "utt_id,speaker_id,wav_path" > "${meta_csv}"

  # Find WAV files recursively so both nested and flat layouts work.
  find "${split_dir}" -type f -name "*.wav" | sort | while read -r wav_path; do
    fname=$(basename "${wav_path}")

    # 1. Extract the speaker ID from the first underscore-delimited field.
    spk_id=$(echo "${fname}" | cut -d'_' -f1)

    # 2. Use the stem as the utterance ID; HI-MIA filenames are globally unique.
    utt_id=${fname%.*}

    # 3. Write the metadata rows.
    echo "${utt_id} ${wav_path}" >> "${wav_scp}"
    echo "${utt_id} ${spk_id}" >> "${utt2spk}"
    echo "${utt_id},${spk_id},${wav_path}" >> "${meta_csv}"
  done

  # Generate spk2utt for the standard Kaldi layout.
  if [ -s "${utt2spk}" ]; then
    # This is a small replacement for Kaldi's utt2spk_to_spk2utt.pl.
    awk '{print $2, $1}' "${utt2spk}" | sort -k1 | \
    awk '{
        if ($1 != prev) {
            if (prev != "") print line;
            prev=$1; line=$1 " " $2;
        } else {
            line=line " " $2;
        }
    } END { if (prev != "") print line; }' > "${my_prep_dir}/spk2utt"

    echo "[HI-MIA] ${split}: Processed $(wc -l < ${wav_scp}) utterances."
  else
    echo "[HI-MIA] WARNING: No wav files found in ${split}."
  fi
done

# ---------------------------------------------------------
# Step 1.5: Subsample to max_utts (speaker-balanced)
# ---------------------------------------------------------
if [ "${max_utts}" -gt 0 ] && [ "$(wc -l < ${prep_dir}/wav.scp)" -gt "${max_utts}" ]; then
  echo "[HI-MIA] Field-aware subsampling to ${max_utts} utterances..."
  python3 tools/subsample_hi_mia.py \
    --wav-scp  "${prep_dir}/wav.scp" \
    --utt2spk  "${prep_dir}/utt2spk" \
    --max-utts "${max_utts}" \
    --regen-spk2utt "${prep_dir}/spk2utt" \
    --seed 777
fi

# ---------------------------------------------------------
# Step 2: Generate Trial List for Short Duration Benchmark
# ---------------------------------------------------------
# Generate trials from the test split only.

test_utt2spk="${prep_dir}/utt2spk"
trials_out="${prep_dir}/trials_short.txt"

if [ -f "${test_utt2spk}" ]; then
  echo "=== [HI-MIA] Generating Short Duration Trials from test set ==="

  python3 tools/make_trials_hi-mia.py \
    --utt2spk              "${test_utt2spk}" \
    --wav-scp              "${prep_dir}/wav.scp" \
    --output               "${trials_out}" \
    --num-neg-target-ratio 10 \
    --max-target-trials-per-condition 20000 \
    --seed 777

  if [ -f "${trials_out}" ]; then
    echo "[HI-MIA] Trials generated: ${trials_out}"
    head -n 3 "${trials_out}"
    echo "..."
  else
    echo "[HI-MIA] ERROR: Failed to generate trials."
    exit 1
  fi

else
  echo "[HI-MIA] WARNING: utt2spk not found. Skipping trial generation."
fi

echo "=== [HI-MIA] Preparation All Done. ==="
