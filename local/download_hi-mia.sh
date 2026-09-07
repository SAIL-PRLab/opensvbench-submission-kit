#!/bin/bash

# Download and unpack the HI-MIA split used by OpenSVBench (OpenSLR 85).
# The current core protocol is built from the official test split only, so the
# public kit does not download train/dev by default.

data=data
download_train=false
download_dev=false
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
download_dir=${data}/hi-mia/download_data
rawdata_dir=${data}/hi-mia/raw_data

mkdir -p "${download_dir}"
mkdir -p "${rawdata_dir}"

echo "=== [HI-MIA] Downloading archives to: ${download_dir} ==="
echo "[HI-MIA] download_train=${download_train}"
echo "[HI-MIA] download_dev=${download_dev}"

base_url=https://openslr.magicdatatech.com/resources/85

train_tar=${download_dir}/train.tar.gz
dev_tar=${download_dir}/dev.tar.gz
test_tar=${download_dir}/test_v2.tar.gz
map_tar=${download_dir}/filename_mapping.tar.gz

# The upstream endpoint may require skipping certificate verification.
WGET="wget --no-check-certificate"

if [ "${download_train}" = true ]; then
  if [ ! -f "${train_tar}" ]; then
    echo "[HI-MIA] Downloading train.tar.gz ..."
    ${WGET} -O "${train_tar}" ${base_url}/train.tar.gz
  else
    echo "[HI-MIA] Found ${train_tar}, skip download."
  fi
else
  echo "[HI-MIA] Skipping train.tar.gz; OpenSVBench core uses test_v2 only."
fi

if [ "${download_dev}" = true ]; then
  if [ ! -f "${dev_tar}" ]; then
    echo "[HI-MIA] Downloading dev.tar.gz ..."
    ${WGET} -O "${dev_tar}" ${base_url}/dev.tar.gz
  else
    echo "[HI-MIA] Found ${dev_tar}, skip download."
  fi
else
  echo "[HI-MIA] Skipping dev.tar.gz; OpenSVBench core uses test_v2 only."
fi

if [ ! -f "${test_tar}" ]; then
  echo "[HI-MIA] Downloading test_v2.tar.gz ..."
  ${WGET} -O "${test_tar}" ${base_url}/test_v2.tar.gz
else
  echo "[HI-MIA] Found ${test_tar}, skip download."
fi

if [ ! -f "${map_tar}" ]; then
  echo "[HI-MIA] Downloading filename_mapping.tar.gz ..."
  ${WGET} -O "${map_tar}" ${base_url}/filename_mapping.tar.gz
else
  echo "[HI-MIA] Found ${map_tar}, skip download."
fi

echo "=== [HI-MIA] Extracting archives to: ${rawdata_dir} ==="

mkdir -p "${rawdata_dir}"

# Stop immediately if extraction fails.
if [ "${download_train}" = true ] && [ ! -d "${rawdata_dir}/train" ]; then
  echo "[HI-MIA] Extracting train.tar.gz ..."
  tar -xzf "${train_tar}" -C "${rawdata_dir}" || { echo "[HI-MIA] train.tar.gz extract failed"; exit 1; }
fi

if [ "${download_dev}" = true ] && [ ! -d "${rawdata_dir}/dev" ]; then
  echo "[HI-MIA] Extracting dev.tar.gz ..."
  tar -xzf "${dev_tar}" -C "${rawdata_dir}" || { echo "[HI-MIA] dev.tar.gz extract failed"; exit 1; }
fi

if [ ! -d "${rawdata_dir}/test" ]; then
  echo "[HI-MIA] Extracting test_v2.tar.gz ..."
  tar -xzf "${test_tar}" -C "${rawdata_dir}" || { echo "[HI-MIA] test_v2.tar.gz extract failed"; exit 1; }
fi

if [ ! -d "${rawdata_dir}/filename_mapping" ]; then
  echo "[HI-MIA] Extracting filename_mapping.tar.gz ..."
  tar -xzf "${map_tar}" -C "${rawdata_dir}" || { echo "[HI-MIA] filename_mapping.tar.gz extract failed"; exit 1; }
fi

echo "=== [HI-MIA] Download + unpack done. Raw data in: ${rawdata_dir} ==="
