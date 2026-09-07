#!/bin/bash

data=data
split=test
max_utts=10000
max_target_trials=5000
nontarget_ratio=10
min_duration=3.0
extract_recording_data=false
. tools/parse_options.sh || exit 1

set -euo pipefail

data=$(realpath "${data}")
dataset_root="${data}/voices"
download_dir="${dataset_root}/download_data"
raw_dir="${dataset_root}/raw_data"
prep_dir="${dataset_root}/prepared"
devkit_archive="${download_dir}/VOiCES_devkit.tar.gz"
recording_archive="${download_dir}/recording_data.tar.gz"

mkdir -p "${raw_dir}" "${prep_dir}"

echo "=== [VOiCES] prepare_voices.sh ==="

has_devkit=false
if {
  [ -f "${raw_dir}/VOiCES_devkit/references/${split}_index.csv" ] &&
  [ -d "${raw_dir}/VOiCES_devkit/distant-16k/speech/${split}" ] &&
  [ -d "${raw_dir}/VOiCES_devkit/source-16k/${split}" ];
} || {
  [ -f "${raw_dir}/references/${split}_index.csv" ] &&
  [ -d "${raw_dir}/distant-16k/speech/${split}" ] &&
  [ -d "${raw_dir}/source-16k/${split}" ];
}; then
  has_devkit=true
fi

if [ "${has_devkit}" = false ]; then
  if [ ! -f "${devkit_archive}" ]; then
    echo "[VOiCES] ERROR: missing archive ${devkit_archive}"
    echo "  Please run local/download_voices.sh first."
    exit 1
  fi
  echo "[VOiCES] Selectively extracting ${split} index/speech/source from ${devkit_archive} -> ${raw_dir} ..."
  tar -xzf "${devkit_archive}" -C "${raw_dir}" \
    "VOiCES_devkit/references/${split}_index.csv" \
    "VOiCES_devkit/distant-16k/speech/${split}" \
    "VOiCES_devkit/source-16k/${split}"
fi

if [ "${extract_recording_data}" = true ] && [ -f "${recording_archive}" ] && [ ! -d "${raw_dir}/recording_data" ] && [ ! -f "${raw_dir}/distances.csv" ]; then
  echo "[VOiCES] Extracting ${recording_archive} -> ${raw_dir} ..."
  tar -xzf "${recording_archive}" -C "${raw_dir}"
fi

python3 tools/prepare_voices_data.py \
  --raw-dir "${raw_dir}" \
  --prep-dir "${prep_dir}" \
  --split "${split}" \
  --max-utts "${max_utts}"

python3 tools/make_trials_voices.py \
  --manifest "${prep_dir}/voices_manifest.csv" \
  --output "${prep_dir}/trials_voices_noise_reverb.txt" \
  --output-pair-metadata "${prep_dir}/pair_metadata_voices.csv" \
  --output-stats "${prep_dir}/stats_voices_noise_reverb.json" \
  --max-target-trials "${max_target_trials}" \
  --nontarget-ratio "${nontarget_ratio}" \
  --min-duration "${min_duration}"

echo "=== [VOiCES] Preparation Done ==="
echo "  wav.scp:                 $(wc -l < "${prep_dir}/wav.scp" | tr -d ' \t') entries"
echo "  speakers:                $(wc -l < "${prep_dir}/spk2utt" | tr -d ' \t')"
echo "  trials_voices_noise_reverb: $(wc -l < "${prep_dir}/trials_voices_noise_reverb.txt" | tr -d ' \t') trials"
