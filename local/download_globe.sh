#!/bin/bash

# Download only the GLOBE parquet shards needed by the frozen OpenSVBench core
# manifest. The full upstream repository is much larger; the core protocol uses
# 10k utterances that live in a small subset of shards.

data=data
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
download_dir=${data}/globe/download_data
rawdata_dir=${data}/globe/raw_data
OPEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shards_tsv="${OPEN_ROOT}/resources/canonical/globe/shards.tsv"

mkdir -p "${download_dir}"
mkdir -p "${rawdata_dir}"

echo "=== [GLOBE] Checking for huggingface-cli ==="

if ! command -v huggingface-cli &> /dev/null; then
  echo "[GLOBE] ERROR: huggingface-cli not found."
  echo "  Install it in your active environment: pip install huggingface_hub"
  exit 1
fi

if [ ! -f "${shards_tsv}" ]; then
  echo "[GLOBE] ERROR: Missing canonical shard list: ${shards_tsv}"
  exit 1
fi

echo "=== [GLOBE] Downloading the pinned OpenSVBench shards ==="
echo "[GLOBE] Target directory: ${download_dir}"
echo "[GLOBE] Shard list: ${shards_tsv}"

include_args=(--include)
while IFS=$'\t' read -r parquet_relpath canonical_count; do
  if [ "${parquet_relpath}" = "parquet_relpath" ] || [ -z "${parquet_relpath}" ]; then
    continue
  fi
  include_args+=("${parquet_relpath}")
done < "${shards_tsv}"

echo "[GLOBE] Canonical shard count: $((${#include_args[@]} - 1))"

# Use the user's standard Hugging Face configuration. If direct access fails,
# configure a proxy, HF_ENDPOINT, or local cache outside this script.
# --local-dir-use-symlinks False: store real files instead of symlinks.
# --repo-type dataset: Hugging Face dataset repository.
huggingface-cli download MushanW/GLOBE_V2 \
  --repo-type dataset \
  --local-dir "${download_dir}" \
  --local-dir-use-symlinks False \
  "${include_args[@]}"

if [ $? -ne 0 ]; then
  echo "[GLOBE] Download failed! Please check your network or try running the command manually."
  exit 1
fi

missing=0
while IFS=$'\t' read -r parquet_relpath canonical_count; do
  if [ "${parquet_relpath}" = "parquet_relpath" ] || [ -z "${parquet_relpath}" ]; then
    continue
  fi
  if [ ! -f "${download_dir}/${parquet_relpath}" ]; then
    echo "[GLOBE] ERROR: Missing downloaded shard: ${download_dir}/${parquet_relpath}"
    missing=$((missing + 1))
  fi
done < "${shards_tsv}"

if [ "${missing}" -ne 0 ]; then
  echo "[GLOBE] Download incomplete: ${missing} canonical shard(s) missing."
  exit 1
fi

echo "=== [GLOBE] Download finished. Canonical shards are in: ${download_dir} ==="
echo "[GLOBE] You can now proceed to Stage 10."
