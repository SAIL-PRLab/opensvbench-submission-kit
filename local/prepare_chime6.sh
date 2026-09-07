#!/bin/bash

data=data
max_utts=10000
max_target_trials=5000
nontarget_ratio=10
min_duration=2.0
max_duration=20.0
overlap_threshold=0.20
array_channel=1
overlap_prepare_threshold=1.01
overlap_max_target_per_bucket=1200
overlap_nontarget_ratio=10
overlap_clean_max=0.05
overlap_light_max=0.20
overlap_mid_max=0.50

. tools/parse_options.sh || exit 1
set -euo pipefail

data=$(realpath "${data}")
dataset_root="${data}/chime6"
download_dir="${dataset_root}/download_data"
raw_dir="${dataset_root}/raw_data"
prep_dir="${dataset_root}/prepared"
overlap_prep_dir="${dataset_root}/prepared_overlap"
dev_archive="${download_dir}/CHiME6_dev.tar.gz"
trans_archive="${download_dir}/CHiME6_transcriptions.tar.gz"

mkdir -p "${raw_dir}" "${prep_dir}" "${overlap_prep_dir}"

echo "=== [CHiME-6] prepare_chime6.sh ==="

if ! command -v sox >/dev/null 2>&1; then
  echo "[CHiME-6] ERROR: sox not found. Please install sox first."
  exit 1
fi

has_audio=false
if find "${raw_dir}" -path "*/audio/dev" -type d | grep -q .; then
  has_audio=true
fi

if [ "${has_audio}" = false ]; then
  if [ ! -f "${dev_archive}" ]; then
    echo "[CHiME-6] ERROR: missing archive ${dev_archive}"
    echo "  Please run local/download_chime6.sh first."
    exit 1
  fi
  echo "[CHiME-6] Extracting ${dev_archive} -> ${raw_dir} ..."
  tar --ignore-zeros -xzf "${dev_archive}" -C "${raw_dir}" || {
    echo "[CHiME-6] ERROR: failed to extract ${dev_archive}"
    exit 1
  }
fi

has_trans=false
if find "${raw_dir}" -path "*/transcriptions/dev" -type d | grep -q .; then
  has_trans=true
fi

if [ "${has_trans}" = false ]; then
  if [ ! -f "${trans_archive}" ]; then
    echo "[CHiME-6] ERROR: missing archive ${trans_archive}"
    echo "  Please run local/download_chime6.sh first."
    exit 1
  fi
  echo "[CHiME-6] Extracting ${trans_archive} -> ${raw_dir} ..."
  set +e
  tar --ignore-zeros -xzf "${trans_archive}" -C "${raw_dir}"
  status=$?
  set -e
  if [ "${status}" -ne 0 ] && ! find "${raw_dir}" -path "*/transcriptions/dev" -type d | grep -q .; then
    echo "[CHiME-6] ERROR: failed to extract ${trans_archive}"
    exit 1
  fi
fi

max_segments=0
if [ "${max_utts}" -gt 0 ]; then
  max_segments=$((max_utts / 2))
fi

python3 tools/prepare_chime6_data.py \
  --raw-dir "${raw_dir}" \
  --prep-dir "${prep_dir}" \
  --split dev \
  --min-duration "${min_duration}" \
  --max-duration "${max_duration}" \
  --overlap-threshold "${overlap_threshold}" \
  --array-channel "${array_channel}" \
  --max-segments "${max_segments}"

python3 tools/make_trials_chime6.py \
  --manifest "${prep_dir}/chime6_manifest.csv" \
  --output "${prep_dir}/trials_chime6_domestic_farfield.txt" \
  --output-pair-metadata "${prep_dir}/pair_metadata_chime6.csv" \
  --output-stats "${prep_dir}/stats_chime6_domestic_farfield.json" \
  --max-target-trials "${max_target_trials}" \
  --nontarget-ratio "${nontarget_ratio}"

python3 tools/prepare_chime6_data.py \
  --raw-dir "${raw_dir}" \
  --prep-dir "${overlap_prep_dir}" \
  --split dev \
  --min-duration "${min_duration}" \
  --max-duration "${max_duration}" \
  --overlap-threshold "${overlap_prepare_threshold}" \
  --array-channel "${array_channel}" \
  --max-segments "${max_segments}"

python3 tools/make_trials_chime6_overlap.py \
  --manifest "${overlap_prep_dir}/chime6_manifest.csv" \
  --output "${overlap_prep_dir}/trials_chime6_overlap.txt" \
  --output-pair-metadata "${overlap_prep_dir}/pair_metadata_chime6_overlap.csv" \
  --output-stats "${overlap_prep_dir}/stats_chime6_overlap.json" \
  --clean-max-overlap "${overlap_clean_max}" \
  --light-max-overlap "${overlap_light_max}" \
  --mid-max-overlap "${overlap_mid_max}" \
  --max-target-per-bucket "${overlap_max_target_per_bucket}" \
  --nontarget-ratio "${overlap_nontarget_ratio}"

echo "=== [CHiME-6] Preparation Done ==="
echo "  wav.scp:                           $(wc -l < "${prep_dir}/wav.scp" | tr -d ' \t') entries"
echo "  speakers:                          $(wc -l < "${prep_dir}/spk2utt" | tr -d ' \t')"
echo "  trials_chime6_domestic_farfield:   $(wc -l < "${prep_dir}/trials_chime6_domestic_farfield.txt" | tr -d ' \t') trials"
echo "  overlap wav.scp:                   $(wc -l < "${overlap_prep_dir}/wav.scp" | tr -d ' \t') entries"
echo "  trials_chime6_overlap:             $(wc -l < "${overlap_prep_dir}/trials_chime6_overlap.txt" | tr -d ' \t') trials"
