#!/bin/bash

# Validate access to the public VoxKnesset dataset hosted on Hugging Face.

data=data
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
download_dir="${data}/voxknesset/download_data"
mirror_dir="${download_dir}/mirror_repo"
mkdir -p "${download_dir}"

echo "=== [VoxKnesset] Access check ==="

if compgen -G "${mirror_dir}/data/test-*.parquet" > /dev/null; then
  echo "[VoxKnesset] Found cached local mirror parquet files under ${mirror_dir}/data"
  find "${mirror_dir}/data" -maxdepth 1 -name 'test-*.parquet' | sort | head -n 5
  echo "[VoxKnesset] Local mirror download is ready."
  exit 0
fi

HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120 python3 - << PYEOF
from datasets import load_dataset

dataset_id = "yanirmr/VoxKnesset"
split = "test"
parquet_files = [
    f"hf://datasets/{dataset_id}/data/{split}-{idx:05d}-of-00020.parquet"
    for idx in range(20)
]
try:
    ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)
    first = next(iter(ds))
    print(f"[VoxKnesset] Access OK via direct parquet: {dataset_id}:{split}")
    print(f"[VoxKnesset] Example speaker: {first.get('person_name')}")
    print(f"[VoxKnesset] Example age    : {first.get('age')}")
except Exception as exc:
    print("[VoxKnesset] ERROR: cannot access the dataset automatically.")
    print(f"  Reason: {exc}")
    print("  Suggested fixes:")
    print("    1. Configure Hugging Face network access, proxy, HF_ENDPOINT, or local cache.")
    print("    2. Or manually place test parquet files under ${mirror_dir}/data.")
    raise
PYEOF

echo "[VoxKnesset] The prepare step will materialize only the selected <=10000 clean utterances."
