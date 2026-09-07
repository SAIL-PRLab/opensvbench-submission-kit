#!/bin/bash

# Download a small local VoxPopuli English Parquet subset for OpenSVBench.
#
# VoxPopuli English train is large, so this helper intentionally downloads
# only the first few converted Parquet shards. The prepare script then
# materializes at most the selected benchmark wavs from these local shards.

data=data
dataset_id=facebook/voxpopuli
config=en
split=train
download_parquet=true
parquet_shards=2

. tools/parse_options.sh || exit 1
set -euo pipefail

data=$(realpath "${data}")
download_dir="${data}/voxpopuli/download_data"
mirror_dir="${download_dir}/mirror_repo"
manifest_path="${download_dir}/parquet_shards_${config}_${split}.tsv"
mkdir -p "${download_dir}"

echo "=== [VoxPopuli] Download/access setup ==="
echo "  data        : ${data}"
echo "  download_dir: ${download_dir}"
echo "  mirror_dir  : ${mirror_dir}"
echo "  dataset_id  : ${dataset_id}"
echo "  config      : ${config}"
echo "  split       : ${split}"
echo "  policy      : download ${parquet_shards} converted Parquet shard(s), not the full 61GB English release"

HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120 python3 - << PYEOF
from pathlib import Path
from huggingface_hub import HfApi

dataset_id = "${dataset_id}"
config = "${config}"
split = "${split}"
expected_prefix = f"{config}/{split}-"
manifest_path = Path("${manifest_path}")

try:
    info = HfApi().dataset_info(dataset_id, files_metadata=True)
except Exception as exc:
    print("[VoxPopuli] ERROR: cannot access Hugging Face dataset metadata.")
    print(f"  Dataset: {dataset_id}")
    print(f"  Reason : {exc}")
    raise

parquet_files = [
    sibling for sibling in info.siblings
    if sibling.rfilename.startswith(expected_prefix) and sibling.rfilename.endswith(".parquet")
]
if not parquet_files:
    raise RuntimeError(f"No converted Parquet shards found for prefix {expected_prefix!r}")

parquet_files = sorted(parquet_files, key=lambda item: item.rfilename)
first = parquet_files[0]
manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w") as f:
    for item in parquet_files:
        f.write(f"{item.rfilename}\t{getattr(item, 'size', '') or ''}\n")

print("[VoxPopuli] Access OK")
print(f"  parquet_shards : {len(parquet_files)}")
print(f"  first_shard    : {first.rfilename}")
print(f"  first_size     : {getattr(first, 'size', None)} bytes")
print(f"  manifest       : {manifest_path}")
print("  required fields are validated during prepare after local Parquet shards are available.")
PYEOF

if [ "${download_parquet}" != true ]; then
  echo "[VoxPopuli] download_parquet=false; access check complete."
  exit 0
fi

mapfile -t selected_shards < <(awk -v n="${parquet_shards}" 'NR <= n {print $1}' "${manifest_path}")
if [ "${#selected_shards[@]}" -eq 0 ]; then
  echo "[VoxPopuli] ERROR: no shards selected from ${manifest_path}"
  exit 1
fi

echo "[VoxPopuli] Selected Parquet shard(s):"
printf '  %s\n' "${selected_shards[@]}"

HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120 python3 - << PYEOF
from pathlib import Path
from huggingface_hub import snapshot_download

dataset_id = "${dataset_id}"
mirror_dir = Path("${mirror_dir}")
manifest_path = Path("${manifest_path}")
limit = int("${parquet_shards}")
selected = []
for line in manifest_path.read_text().splitlines()[:limit]:
    if line.strip():
        selected.append(line.split("\t", 1)[0])

mirror_dir.mkdir(parents=True, exist_ok=True)
print("[VoxPopuli] Downloading selected Parquet shard(s) with huggingface_hub.snapshot_download ...")
snapshot_download(
    repo_id=dataset_id,
    repo_type="dataset",
    allow_patterns=list(selected),
    local_dir=str(mirror_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"[VoxPopuli] Downloaded selected shard(s) under {mirror_dir}")
PYEOF

echo "[VoxPopuli] Verifying local Parquet shard sizes..."
awk -v n="${parquet_shards}" 'NR <= n {print $1 "\t" $2}' "${manifest_path}" | while IFS=$'\t' read -r rel_path expected_size; do
  local_path="${mirror_dir}/${rel_path}"
  if [ ! -f "${local_path}" ]; then
    echo "[VoxPopuli] ERROR: missing local shard: ${local_path}"
    exit 1
  fi
  actual_size=$(stat -c '%s' "${local_path}")
  echo "  ${rel_path}: ${actual_size} bytes"
  if [[ "${expected_size}" =~ ^[0-9]+$ ]] && [ "${actual_size}" != "${expected_size}" ]; then
    echo "[VoxPopuli] ERROR: size mismatch for ${rel_path}"
    echo "  expected: ${expected_size}"
    echo "  actual  : ${actual_size}"
    exit 1
  fi
done

echo "[VoxPopuli] Download/access check complete."
