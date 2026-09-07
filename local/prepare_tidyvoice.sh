#!/bin/bash

# Prepare TidyVoiceX2-ASV core benchmark manifests and trials.

data=data
max_utts=10000
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
raw_dir="${data}/tidyvoice/raw_data"
prep_dir="${data}/tidyvoice/prepared"
audio_root="${raw_dir}/TidyVoiceX2_ASV"
pair_file_a="${audio_root}/trials/tv26_eval-A.txt"
pair_file_u="${audio_root}/trials/tv26_eval-U.txt"

mkdir -p "${prep_dir}"

echo "=== [TidyVoice] prepare_tidyvoice.sh ==="

if [ ! -d "${audio_root}" ]; then
  echo "[TidyVoice] ERROR: audio root not found: ${audio_root}"
  echo "  Please place TidyVoiceX2_ASV under ${raw_dir} first."
  exit 1
fi
if [ ! -f "${pair_file_a}" ]; then
  echo "[TidyVoice] ERROR: pair/key file not found: ${pair_file_a}"
  exit 1
fi
if [ ! -f "${pair_file_u}" ]; then
  echo "[TidyVoice] ERROR: pair/key file not found: ${pair_file_u}"
  exit 1
fi

PYTHONUNBUFFERED=1 python3 tools/prepare_tidyvoice.py \
  --audio-root "${audio_root}" \
  --pair-files "${pair_file_a}" "${pair_file_u}" \
  --prep-dir "${prep_dir}" \
  --max-clean-utts "${max_utts}" \
  --max-target-trials-per-condition 10000 \
  --min-total-trials-per-condition 100000 \
  --candidate-pairs-per-bucket 60000 \
  --nontarget-ratio 4 \
  --target-selection speaker_density \
  --max-utts-per-speaker 120 \
  --seed 777 || exit 1

echo "=== [TidyVoice] Preparation Done ==="
echo "  Lists : ${prep_dir}/{wav.scp,utt2spk,utt2lang,spk2utt}"
echo "  Trials: ${prep_dir}/trials_cross_lingual.txt"
echo "  Stats : ${prep_dir}/stats_tidyvoice.json"
