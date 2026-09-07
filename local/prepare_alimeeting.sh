#!/bin/bash

# Prepare the AliMeeting test set for meeting speaker verification.
#
# Steps:
#   1. Parse TextGrid annotations and select headset and array-channel segments.
#   2. Generate wav.scp, utt2spk, spk2utt, and utt2channel.
#   3. Generate near-near and near-far meeting trials.
#   4. Generate light, medium, and heavy overlap trials.
#
# Dependencies: sox and the Python textgrid package.

data=data
max_utts=10000   # Match prepare_data.sh; sample before cutting to save sox time.
min_dur=3.0      # Minimum segment duration in seconds.
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
raw_dir="${data}/alimeeting/raw_data"
prep_dir="${data}/alimeeting/prepared"
OPEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${prep_dir}"

echo "=== [AliMeeting] prepare_alimeeting.sh ==="

if [ ! -d "${raw_dir}" ]; then
  echo "[AliMeeting] ERROR: raw_dir not found: ${raw_dir}"
  echo "  Please run local/download_alimeeting.sh first."
  exit 1
fi

# Check for sox.
if ! command -v sox &> /dev/null; then
  echo "[AliMeeting] ERROR: sox not found. Please install: sudo apt install sox"
  exit 1
fi

# Check textgrid dependency without mutating the user's Python environment.
if ! python3 -c "import textgrid" >/dev/null 2>&1; then
  echo "[AliMeeting] ERROR: Python package 'textgrid' is required."
  echo "  Install it in your active environment, for example: pip install textgrid"
  exit 1
fi

# -----------------------------------------------------------------------
# Step 1: scan TextGrids, sample max_utts, and cut selected segments with sox.
# -----------------------------------------------------------------------
echo "[AliMeeting] max_utts=${max_utts}, min_dur=${min_dur}s (sample before cutting)"
python3 "${OPEN_ROOT}/tools/prepare_alimeeting_data.py" \
  --raw-dir  "${raw_dir}" \
  --prep-dir "${prep_dir}" \
  --max-utts "${max_utts}" \
  --min-dur  "${min_dur}" || {
    echo "[AliMeeting] ERROR: prepare_alimeeting_data.py failed."
    exit 1
  }

n_utts=$(wc -l < "${prep_dir}/wav.scp")
n_spks=$(wc -l < "${prep_dir}/spk2utt")
echo "[AliMeeting] ${n_utts} utterances, ${n_spks} speakers."

if [ "${n_utts}" -eq 0 ]; then
  echo "[AliMeeting] ERROR: wav.scp is empty."
  exit 1
fi

# -----------------------------------------------------------------------
# Step 2: generate meeting trials.
# -----------------------------------------------------------------------
trial_out="${prep_dir}/trials_meeting.txt"
trial_near_far="${prep_dir}/trials_alimeeting_near_far.txt"
trial_near_near="${prep_dir}/trials_alimeeting_near_near.txt"

if [ ! -f "${trial_out}" ] || [ "${prep_dir}/wav.scp" -nt "${trial_out}" ]; then
  echo "[AliMeeting] Generating meeting trials ..."
  python3 "${OPEN_ROOT}/tools/make_trials_alimeeting.py" \
    --utt2spk     "${prep_dir}/utt2spk" \
    --utt2channel "${prep_dir}/utt2channel" \
    --output      "${trial_out}" \
    --max-target-trials 20000 \
    --nontarget-ratio   10 || exit 1
else
  echo "[AliMeeting] ${trial_out} up-to-date, skipping."
fi

echo "[AliMeeting] Splitting meeting trials by condition ..."
python3 "${OPEN_ROOT}/tools/filter_trials_by_condition.py" \
  --input "${trial_out}" \
  --output "${trial_near_far}" \
  --conditions near_far \
  --output-stats "${prep_dir}/stats_alimeeting_near_far.json" || exit 1

python3 "${OPEN_ROOT}/tools/filter_trials_by_condition.py" \
  --input "${trial_out}" \
  --output "${trial_near_near}" \
  --conditions near_near \
  --output-stats "${prep_dir}/stats_alimeeting_near_near.json" || exit 1

# -----------------------------------------------------------------------
# Step 3: generate overlap trials.
# -----------------------------------------------------------------------
trial_overlap_light="${prep_dir}/trials_overlap_light.txt"
trial_overlap_mid="${prep_dir}/trials_overlap_mid.txt"
trial_overlap_heavy="${prep_dir}/trials_overlap_heavy.txt"
trial_overlap="${prep_dir}/trials_alimeeting_overlap.txt"
pair_meta_out="${prep_dir}/pair_metadata_alimeeting_overlap.csv"
utt2session_out="${prep_dir}/utt2session"
overlap_stats_out="${prep_dir}/stats_alimeeting_overlap.json"

if [ ! -f "${trial_overlap_light}" ] || [ ! -f "${trial_overlap_mid}" ] || [ ! -f "${trial_overlap_heavy}" ] || [ "${prep_dir}/wav.scp" -nt "${trial_overlap_light}" ]; then
  echo "[AliMeeting] Generating overlap trials ..."
  python3 "${OPEN_ROOT}/tools/make_trials_alimeeting_overlap.py" \
    --raw-dir "${raw_dir}" \
    --utt2spk "${prep_dir}/utt2spk" \
    --utt2channel "${prep_dir}/utt2channel" \
    --output-light "${trial_overlap_light}" \
    --output-mid "${trial_overlap_mid}" \
    --output-heavy "${trial_overlap_heavy}" \
    --output-pair-metadata "${pair_meta_out}" \
    --output-utt2session "${utt2session_out}" \
    --output-stats "${overlap_stats_out}" \
    --min-dur "${min_dur}" \
    --max-target-trials 5000 \
    --nontarget-ratio 10 || exit 1
else
  echo "[AliMeeting] overlap trial files are up-to-date, skipping."
fi

echo "[AliMeeting] Combining overlap trials into one ranked trial set ..."
cat "${trial_overlap_light}" "${trial_overlap_mid}" "${trial_overlap_heavy}" > "${trial_overlap}"

echo "=== [AliMeeting] Preparation Done ==="
echo "  wav.scp:        ${n_utts} entries"
echo "  spk2utt:        ${n_spks} speakers"
echo "  trials_meeting: $(wc -l < ${trial_out}) trials"
echo "  Conditions:     near_near (headset-to-headset), near_far (headset-to-array)"
echo "  near_far:       $(wc -l < ${trial_near_far}) trials"
echo "  near_near:      $(wc -l < ${trial_near_near}) trials"
echo "  overlap:        $(wc -l < ${trial_overlap}) trials"
echo "  overlap_light:  $(wc -l < ${trial_overlap_light}) trials"
echo "  overlap_mid:    $(wc -l < ${trial_overlap_mid}) trials"
echo "  overlap_heavy:  $(wc -l < ${trial_overlap_heavy}) trials"
