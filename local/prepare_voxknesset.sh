#!/bin/bash

# Prepare VoxKnesset aging benchmark assets.

data=data
max_utts=10000
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
download_dir="${data}/voxknesset/download_data"
raw_dir="${data}/voxknesset/raw_data"
prep_dir="${data}/voxknesset/prepared"

mkdir -p "${download_dir}" "${raw_dir}" "${prep_dir}"

echo "=== [VoxKnesset] prepare_voxknesset.sh ==="

HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120 PYTHONUNBUFFERED=1 python3 tools/prepare_voxknesset.py \
  --download-dir "${download_dir}" \
  --raw-dir "${raw_dir}" \
  --prep-dir "${prep_dir}" \
  --max-clean-utts "${max_utts}" \
  --min-gap-years 1.0 \
  --max-target-trials-per-bucket 5000 \
  --nontarget-ratio 10 \
  --voiced-max-seconds 5.0 \
  --vad-frame-ms 30 \
  --vad-hop-ms 10 \
  --seed 777 || exit 1

echo "=== [VoxKnesset] Preparation Done ==="
echo "  Lists : ${prep_dir}/{wav.scp,utt2spk,utt2age,utt2gender,spk2utt}"
echo "  Trials: ${prep_dir}/trials_cross_time.txt"
echo "  Stats : ${prep_dir}/stats_voxknesset.json"
