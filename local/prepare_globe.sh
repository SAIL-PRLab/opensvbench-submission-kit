#!/bin/bash

# Prepare GLOBE Dataset
# The public kit uses the frozen OpenSVBench canonical manifest/trials. It only
# materializes the 10k utterances needed by the current core benchmark.

data=data
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
download_dir=${data}/globe/download_data
rawdata_dir=${data}/globe/raw_data
prep_dir=${data}/globe/prepared
OPEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
canonical_dir="${OPEN_ROOT}/resources/canonical/globe"

mkdir -p "${prep_dir}"
mkdir -p "${rawdata_dir}"

echo "=== [GLOBE] prepare_globe.sh ==="

if [ ! -f "${canonical_dir}/manifest.tsv" ] || [ ! -f "${canonical_dir}/trials_accent.txt" ]; then
  echo "[GLOBE] ERROR: Missing frozen OpenSVBench canonical manifest/trials in ${canonical_dir}"
  exit 1
fi

echo "[GLOBE] Extracting canonical WAVs and generating Kaldi lists..."
python3 tools/extract_globe.py \
  --download-dir "${download_dir}" \
  --wav-dir "${rawdata_dir}" \
  --out-prep "${prep_dir}" \
  --canonical-manifest "${canonical_dir}/manifest.tsv" \
  --shards-tsv "${canonical_dir}/shards.tsv"

if [ $? -ne 0 ]; then
    echo "[GLOBE] Extraction failed."
    exit 1
fi

echo "[GLOBE] Applying frozen OpenSVBench canonical manifest/trials ..."
python3 tools/apply_canonical_globe.py \
  --manifest "${canonical_dir}/manifest.tsv" \
  --trials "${canonical_dir}/trials_accent.txt" \
  --raw-dir "${rawdata_dir}" \
  --prep-dir "${prep_dir}"
echo "=== [GLOBE] Preparation Done ==="
echo "Kaldi Lists: ${prep_dir}/{wav.scp, wav_all.scp, utt2spk, utt2accent}"
echo "Trial List : ${prep_dir}/trials_accent.txt"
