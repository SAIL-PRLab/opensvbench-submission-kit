#!/bin/bash

# Workflow summary:
#   1. Read test_wav.scp and write prepared/wav.scp with absolute paths,
#      excluding Device09.
#   2. Copy speaker labels from test_utt2spk.
#   3. Extract device, distance, and dialect metadata from test_utt2info.csv.
#   4. Generate spk2utt.
#   5. Convert the three official trial files into the core trial files.

data=data
max_utts=10000
. tools/parse_options.sh || exit 1

max_utts=0  # Official trials require the complete utterance list.

data=$(realpath "${data}")
rawdata_dir=${data}/3dspeaker/raw_data
prep_dir=${data}/3dspeaker/prepared
files_dir=${rawdata_dir}/files

mkdir -p "${prep_dir}"

echo "=== [3D-Speaker] prepare_3dspeaker.sh ==="
echo "[3D-Speaker] Raw data root: ${rawdata_dir}"

# -----------------------------------------------------------------------
# Step 1: Locate metadata files
# -----------------------------------------------------------------------
test_wav_scp=""
for candidate in \
    "${files_dir}/test_wav.scp" \
    "${rawdata_dir}/test_wav.scp"; do
    if [ -f "${candidate}" ]; then
        test_wav_scp="${candidate}"
        break
    fi
done

if [ -z "${test_wav_scp}" ]; then
    echo "[3D-Speaker] ERROR: Cannot find test_wav.scp under ${rawdata_dir}"
    echo "  Searched: ${files_dir}/test_wav.scp, ${rawdata_dir}/test_wav.scp"
    exit 1
fi
echo "[3D-Speaker] Using test_wav.scp: ${test_wav_scp}"

test_utt2spk=""
for candidate in \
    "${files_dir}/test_utt2spk" \
    "${rawdata_dir}/test_utt2spk"; do
    if [ -f "${candidate}" ]; then
        test_utt2spk="${candidate}"
        break
    fi
done

if [ -z "${test_utt2spk}" ]; then
    echo "[3D-Speaker] ERROR: Cannot find test_utt2spk under ${rawdata_dir}"
    exit 1
fi
echo "[3D-Speaker] Using test_utt2spk: ${test_utt2spk}"

test_utt2info=""
for candidate in \
    "${files_dir}/test_utt2info.csv" \
    "${rawdata_dir}/test_utt2info.csv"; do
    if [ -f "${candidate}" ]; then
        test_utt2info="${candidate}"
        break
    fi
done

# -----------------------------------------------------------------------
# Step 2: Generate wav.scp (absolute paths, filter Device09)
# -----------------------------------------------------------------------
wav_scp="${prep_dir}/wav.scp"
utt2spk="${prep_dir}/utt2spk"
utt2device="${prep_dir}/utt2device"
utt2distance="${prep_dir}/utt2distance"
utt2dialect="${prep_dir}/utt2dialect"

echo "[3D-Speaker] Generating wav.scp (filtering Device09) ..."

# test_wav.scp uses paths such as test/<spk_id>/<utt>.wav relative to raw_data.
# Keep audio_root at raw_data; setting it to raw_data/test would duplicate test/.
audio_root="${rawdata_dir}"
if [ ! -d "${audio_root}" ]; then
    echo "[3D-Speaker] ERROR: raw_data not found: ${audio_root}"
    exit 1
fi

# Build wav.scp from test_wav.scp:
# Each line in test_wav.scp: utt_id relative_path
# We prepend audio_root to make absolute paths, and filter out Device09
> "${wav_scp}"
> "${utt2spk}"

# First, build a set of Device09 utt_ids to filter (from utt2info or utt_id pattern)
declare -A device09_utts
if [ -n "${test_utt2info}" ]; then
    # Extract Device09 utt_ids from utt2info.csv
    while IFS=',' read -r utt_id rest; do
        # Skip header
        [[ "${utt_id}" == "utt_id" ]] && continue
        [[ "${utt_id}" == "utterance_id" ]] && continue
        if echo "${rest}" | grep -qi "Device09"; then
            device09_utts["${utt_id}"]=1
        fi
    done < "${test_utt2info}"
fi

n_filtered=0
while read -r utt_id rel_path rest; do
    [ -z "${utt_id}" ] && continue
    [[ "${utt_id}" == "#"* ]] && continue

    # Filter Device09 (check utt2info set or utt_id/path pattern)
    if [ -n "${device09_utts[${utt_id}]+_}" ]; then
        n_filtered=$((n_filtered + 1))
        continue
    fi
    if echo "${utt_id} ${rel_path}" | grep -qi "Device09"; then
        n_filtered=$((n_filtered + 1))
        continue
    fi

    # Resolve absolute path
    if [[ "${rel_path}" == /* ]]; then
        abs_path="${rel_path}"
    else
        abs_path="${audio_root}/${rel_path}"
    fi

    echo "${utt_id} ${abs_path}" >> "${wav_scp}"
done < "${test_wav_scp}"

echo "[3D-Speaker] Filtered ${n_filtered} Device09 utterances."

# -----------------------------------------------------------------------
# Step 3: Generate utt2spk (filtering Device09)
# -----------------------------------------------------------------------
echo "[3D-Speaker] Generating utt2spk ..."

# Build a valid utt set from wav.scp
declare -A valid_utts
while read -r utt_id _; do
    valid_utts["${utt_id}"]=1
done < "${wav_scp}"

while read -r utt_id spk_id rest; do
    [ -z "${utt_id}" ] && continue
    [[ "${utt_id}" == "#"* ]] && continue
    if [ -n "${valid_utts[${utt_id}]+_}" ]; then
        echo "${utt_id} ${spk_id}" >> "${utt2spk}"
    fi
done < "${test_utt2spk}"

# -----------------------------------------------------------------------
# Step 4: Extract metadata from test_utt2info.csv
# -----------------------------------------------------------------------
> "${utt2device}"
> "${utt2distance}"
> "${utt2dialect}"

if [ -n "${test_utt2info}" ]; then
    echo "[3D-Speaker] Extracting metadata from test_utt2info.csv ..."
    python3 -c "
import csv, sys

wav_utts = set()
with open('${wav_scp}') as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if parts:
            wav_utts.add(parts[0])

device_f = open('${utt2device}', 'w')
distance_f = open('${utt2distance}', 'w')
dialect_f = open('${utt2dialect}', 'w')

with open('${test_utt2info}') as f:
    reader = csv.DictReader(f)
    headers_lower = {h.lower().strip(): h for h in reader.fieldnames} if reader.fieldnames else {}

    # Auto-detect column names (case-insensitive)
    utt_col = headers_lower.get('utt_id') or headers_lower.get('utterance_id') or headers_lower.get('uttid')
    device_col = headers_lower.get('device') or headers_lower.get('device_id')
    distance_col = headers_lower.get('distance') or headers_lower.get('dist')
    dialect_col = headers_lower.get('dialect') or headers_lower.get('accent')

    if utt_col is None:
        # Fallback: use first column as utt_id
        utt_col = reader.fieldnames[0] if reader.fieldnames else None

    if utt_col is None:
        print('[3D-Speaker] WARNING: Cannot detect utt_id column in utt2info.csv', file=sys.stderr)
        sys.exit(0)

    for row in reader:
        utt_id = row.get(utt_col, '').strip()
        if not utt_id or utt_id not in wav_utts:
            continue
        if device_col and row.get(device_col, '').strip():
            device_f.write(f'{utt_id} {row[device_col].strip()}\n')
        if distance_col and row.get(distance_col, '').strip():
            distance_f.write(f'{utt_id} {row[distance_col].strip()}\n')
        if dialect_col and row.get(dialect_col, '').strip():
            dialect_f.write(f'{utt_id} {row[dialect_col].strip()}\n')

device_f.close()
distance_f.close()
dialect_f.close()
"
    for meta in "${utt2device}" "${utt2distance}" "${utt2dialect}"; do
        n=$(wc -l < "${meta}")
        echo "  $(basename ${meta}): ${n} entries"
    done
else
    echo "[3D-Speaker] WARNING: test_utt2info.csv not found, skipping metadata extraction."
fi

n_utts=$(wc -l < "${wav_scp}")
echo "[3D-Speaker] Processed ${n_utts} utterances."

if [ "${n_utts}" -eq 0 ]; then
    echo "[3D-Speaker] ERROR: No utterances in wav.scp. Check your raw data layout."
    exit 1
fi

# -----------------------------------------------------------------------
# Step 5: Generate spk2utt
# -----------------------------------------------------------------------
awk '{print $2, $1}' "${utt2spk}" | sort -k1 | \
awk '{
    if ($1 != prev) {
        if (prev != "") print line;
        prev=$1; line=$1 " " $2;
    } else {
        line=line " " $2;
    }
} END { if (prev != "") print line; }' > "${prep_dir}/spk2utt"

n_spks=$(wc -l < "${prep_dir}/spk2utt")
echo "[3D-Speaker] ${n_spks} speakers."

# -----------------------------------------------------------------------
# Step 6: Convert official trials -> trials_3dspeaker.txt
# -----------------------------------------------------------------------
echo "[3D-Speaker] Converting official trial files ..."

trial_files=()
conditions=()

for trial_name in trials_cross_device trials_cross_distance trials_cross_dialect; do
    trial_path=""
    for candidate in "${files_dir}/${trial_name}" "${rawdata_dir}/${trial_name}"; do
        if [ -f "${candidate}" ]; then
            trial_path="${candidate}"
            break
        fi
    done
    if [ -n "${trial_path}" ]; then
        trial_files+=("${trial_path}")
        cond_name="${trial_name#trials_}"   # cross_device / cross_distance / cross_dialect
        conditions+=("${cond_name}")
        echo "  Found: ${trial_path} -> condition=${cond_name}"
    else
        echo "  WARNING: ${trial_name} not found, skipping."
    fi
done

if [ ${#trial_files[@]} -eq 0 ]; then
    echo "[3D-Speaker] ERROR: No official trial files found."
    exit 1
fi

python3 tools/convert_3dspeaker_trials.py \
    --trial-files "${trial_files[@]}" \
    --conditions "${conditions[@]}" \
    --wav-scp "${wav_scp}" \
    --output "${prep_dir}/trials_3dspeaker.txt"

if [ $? -ne 0 ]; then
    echo "[3D-Speaker] ERROR: Trial conversion failed."
    exit 1
fi

# -----------------------------------------------------------------------
# Step 7: Split merged trial list by official condition
# -----------------------------------------------------------------------
echo "[3D-Speaker] Splitting merged trial list by condition ..."

python3 tools/filter_trials_by_condition.py \
    --input "${prep_dir}/trials_3dspeaker.txt" \
    --output "${prep_dir}/trials_speaker3d_device.txt" \
    --conditions cross_device \
    --output-stats "${prep_dir}/stats_speaker3d_device.json" || exit 1

python3 tools/filter_trials_by_condition.py \
    --input "${prep_dir}/trials_3dspeaker.txt" \
    --output "${prep_dir}/trials_speaker3d_distance.txt" \
    --conditions cross_distance \
    --output-stats "${prep_dir}/stats_speaker3d_distance.json" || exit 1

python3 tools/filter_trials_by_condition.py \
    --input "${prep_dir}/trials_3dspeaker.txt" \
    --output "${prep_dir}/trials_speaker3d_dialect.txt" \
    --conditions cross_dialect \
    --output-stats "${prep_dir}/stats_speaker3d_dialect.json" || exit 1

echo "=== [3D-Speaker] Preparation Done ==="
echo "Kaldi Lists : ${prep_dir}/{wav.scp, utt2spk, spk2utt}"
echo "Metadata    : ${prep_dir}/{utt2device, utt2distance, utt2dialect}"
echo "Trial List  : ${prep_dir}/trials_3dspeaker.txt (merged official trial list)"
echo "Split Trials: ${prep_dir}/trials_speaker3d_device.txt"
echo "              ${prep_dir}/trials_speaker3d_distance.txt"
echo "              ${prep_dir}/trials_speaker3d_dialect.txt"
