#!/bin/bash

# Download ESD (Emotional Speech Dataset).
#   20 speakers (10 English and 10 Mandarin), five emotions, 350 utterances
#   per speaker and emotion.
#   Source: Hugging Face sonchuate/ESD_dataset.
#   Check the upstream repository for the current license terms.
#
# Expected layout after extraction:
#   raw_data/{0001..0020}/{Angry,Happy,Neutral,Sad,Surprise}/*.wav

data=data
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
raw_dir="${data}/esd/raw_data"
download_dir="${data}/esd/download_data"

mkdir -p "${raw_dir}" "${download_dir}"

# Skip when the data is already extracted.
if [ -d "${raw_dir}/0001" ]; then
  echo "[ESD] raw_data/0001/ already exists. Skipping download & extract."
  exit 0
fi

# -----------------------------------------------------------------------
# Accept a locally downloaded zip; the filename may vary by source.
# -----------------------------------------------------------------------
zip_archive=""
declare -a zip_candidates=(
  "${download_dir}/ESD.zip"
  "${download_dir}/Emotional Speech Dataset (ESD).zip"
  "${download_dir}/Emotion Speech Dataset.zip"
  "${download_dir}/Emotional Speech Dataset.zip"
)
for z in "${zip_candidates[@]}"; do
  if [ -f "${z}" ]; then
    zip_archive="${z}"
    echo "[ESD] Using existing archive: ${zip_archive}"
    break
  fi
done
# Any *.zip file supplied by the user.
if [ -z "${zip_archive}" ]; then
  shopt -s nullglob
  zips=( "${download_dir}"/*.zip )
  shopt -u nullglob
  if [ ${#zips[@]} -eq 1 ]; then
    zip_archive="${zips[0]}"
    echo "[ESD] Using only zip in download_data: ${zip_archive}"
  elif [ ${#zips[@]} -gt 1 ]; then
    echo "[ESD] ERROR: Multiple .zip files in ${download_dir}:"
    printf '  %s\n' "${zips[@]}"
    echo "  Please keep exactly one ESD archive, or rename one to ESD.zip"
    exit 1
  fi
fi

# -----------------------------------------------------------------------
# If no local zip is available, try downloading the public Hugging Face file.
# -----------------------------------------------------------------------
HF_URL="https://huggingface.co/datasets/sonchuate/ESD_dataset/resolve/main/Emotional%20Speech%20Dataset%20%28ESD%29.zip"

if [ -z "${zip_archive}" ]; then
  zip_archive="${download_dir}/ESD.zip"
  echo "[ESD] No local zip found. Downloading -> ${zip_archive} ..."
  wget --no-check-certificate -c -O "${zip_archive}" "${HF_URL}" || {
    echo "[ESD] Direct wget failed. Trying huggingface_hub Python API ..."
    python3 - << PYEOF
import os, sys
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="sonchuate/ESD_dataset",
        filename="Emotional Speech Dataset (ESD).zip",
        repo_type="dataset",
        local_dir="${download_dir}",
    )
    import shutil
    shutil.move(path, "${zip_archive}")
    print(f"[ESD] Downloaded to ${zip_archive}")
except Exception as e:
    print(f"[ESD] ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    if [ $? -ne 0 ]; then
      echo "[ESD] ERROR: Both download methods failed."
      echo "  Please manually download from: ${HF_URL}"
      echo "  and place a single .zip under: ${download_dir}"
      exit 1
    fi
  }
fi

# -----------------------------------------------------------------------
# Extract with the current Python environment.
# -----------------------------------------------------------------------
OPEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTRACT_PY="${OPEN_ROOT}/tools/extract_esd_zip.py"

echo "[ESD] Extracting ${zip_archive} -> ${raw_dir} ..."

if ! python3 "${EXTRACT_PY}" --zip "${zip_archive}" --out "${raw_dir}"; then
  echo "[ESD] ERROR: extract_esd_zip.py failed with the current python3 environment."
  exit 1
fi

if [ ! -d "${raw_dir}/0001" ]; then
  echo "[ESD] ERROR: ${raw_dir}/0001 not found after extraction."
  ls -la "${raw_dir}" 2>/dev/null | head -20 || true
  exit 1
fi

echo "[ESD] Download & extraction complete."
echo "  raw_data: ${raw_dir}"
echo "  Layout: 0001/{Angry,Happy,Neutral,Sad,Surprise}/*.wav ... 0020/"
