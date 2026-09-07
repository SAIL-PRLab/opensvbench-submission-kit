#!/bin/bash

data=data
max_utts=10000
. tools/parse_options.sh || exit 1

set -euo pipefail

data=$(realpath "${data}")
raw_dir="${data}/whisper40/raw_data"
prep_dir="${data}/whisper40/prepared"

mkdir -p "${prep_dir}"

echo "=== [Whisper40] prepare_whisper40.sh ==="

if [ ! -d "${raw_dir}" ]; then
  echo "[Whisper40] ERROR: raw_dir not found: ${raw_dir}"
  echo "  Please run local/download_whisper40.sh first."
  exit 1
fi

python3 tools/prepare_whisper40_data.py \
  --raw-dir "${raw_dir}" \
  --prep-dir "${prep_dir}" \
  --max-utts "${max_utts}"

python3 tools/make_trials_whisper40.py \
  --utt2spk "${prep_dir}/utt2spk" \
  --utt2style "${prep_dir}/utt2style" \
  --utt2text "${prep_dir}/utt2text" \
  --utt2split "${prep_dir}/utt2split" \
  --output "${prep_dir}/trials_whisper40_whisper.txt" \
  --output-pair-metadata "${prep_dir}/pair_metadata_whisper40.csv" \
  --output-stats "${prep_dir}/stats_whisper40.json" \
  --max-target-trials 5000 \
  --nontarget-ratio 10

echo "=== [Whisper40] Preparation Done ==="
echo "  wav.scp:                $(wc -l < "${prep_dir}/wav.scp" | tr -d ' \t') entries"
echo "  speakers:               $(wc -l < "${prep_dir}/spk2utt" | tr -d ' \t')"
echo "  trials_whisper40_whisper: $(wc -l < "${prep_dir}/trials_whisper40_whisper.txt" | tr -d ' \t') trials"
