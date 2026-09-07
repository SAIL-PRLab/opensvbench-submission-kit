#!/bin/bash

data=data
max_utts=10000
max_target_trials=0
nontarget_ratio=10

. tools/parse_options.sh || exit 1
set -euo pipefail

data=$(realpath "${data}")
dataset_root="${data}/lombardgrid"
download_dir="${dataset_root}/download_data"
raw_dir="${dataset_root}/raw_data"
prep_dir="${dataset_root}/prepared"
audio_archive="${download_dir}/lombardgrid_audio.zip"
json_archive="${download_dir}/lombardgrid_json.zip"

mkdir -p "${raw_dir}" "${prep_dir}"

echo "=== [LombardGrid] prepare_lombardgrid.sh ==="

has_audio=false
audio_probe=$(find "${raw_dir}" -type f -iname '*.wav' -print -quit)
if [ -n "${audio_probe}" ]; then
  has_audio=true
fi

if [ "${has_audio}" = false ]; then
  if [ ! -f "${audio_archive}" ]; then
    echo "[LombardGrid] ERROR: missing archive ${audio_archive}"
    echo "  Please run local/download_lombardgrid.sh first."
    exit 1
  fi
  echo "[LombardGrid] Extracting ${audio_archive} -> ${raw_dir} ..."
  unzip -q -o "${audio_archive}" -d "${raw_dir}"
fi

json_probe=$(find "${raw_dir}" -type f -iname '*.json' -print -quit)
if [ -f "${json_archive}" ] && [ -z "${json_probe}" ]; then
  echo "[LombardGrid] Extracting ${json_archive} -> ${raw_dir} ..."
  unzip -q -o "${json_archive}" -d "${raw_dir}"
fi

python3 tools/prepare_lombardgrid_data.py \
  --raw-dir "${raw_dir}" \
  --prep-dir "${prep_dir}" \
  --max-utts "${max_utts}"

python3 tools/make_trials_lombardgrid.py \
  --manifest "${prep_dir}/lombardgrid_manifest.csv" \
  --output "${prep_dir}/trials_lombardgrid_lombard.txt" \
  --output-pair-metadata "${prep_dir}/pair_metadata_lombardgrid.csv" \
  --output-stats "${prep_dir}/stats_lombardgrid_lombard.json" \
  --max-target-trials "${max_target_trials}" \
  --nontarget-ratio "${nontarget_ratio}"

echo "=== [LombardGrid] Preparation Done ==="
echo "  wav.scp:                    $(wc -l < "${prep_dir}/wav.scp" | tr -d ' \t') entries"
echo "  speakers:                   $(wc -l < "${prep_dir}/spk2utt" | tr -d ' \t')"
echo "  trials_lombardgrid_lombard: $(wc -l < "${prep_dir}/trials_lombardgrid_lombard.txt" | tr -d ' \t') trials"
