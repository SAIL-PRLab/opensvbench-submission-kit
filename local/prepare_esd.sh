#!/bin/bash

# Prepare Kaldi-style files and cross-emotion trials for ESD.
#
# Output under prepared/:
#   wav.scp       : utt_id  /abs/path/to/wav
#   utt2spk       : utt_id  spk_id
#   spk2utt       : spk_id  utt_id1 utt_id2 ...
#   utt2emotion   : utt_id  emotion  (angry/happy/neutral/sad/surprise)
#   utt2lang      : utt_id  en|zh    (0001-0010=English, 0011-0020=Mandarin)
#   trials_emotion.txt  : cross-emotion and same-emotion speaker trials

data=data
max_utts=10000
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
raw_dir="${data}/esd/raw_data"
prep_dir="${data}/esd/prepared"

mkdir -p "${prep_dir}"

echo "=== [ESD] prepare_esd.sh ==="

OPEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check the input data.
if [ ! -d "${raw_dir}" ]; then
  echo "[ESD] ERROR: raw_dir not found: ${raw_dir}"
  echo "  Please run local/download_esd.sh first."
  exit 1
fi

# -----------------------------------------------------------------------
# Step 1: scan the directory structure and write wav.scp, utt2spk,
# utt2emotion, and utt2lang.
# -----------------------------------------------------------------------
echo "[ESD] Scanning directory structure and building Kaldi files ..."

# Use python3 from the active environment.
# Keep validation in a temporary Python file so shell stdin handling is stable.
python3 << PYEOF
import os, sys, random

raw_dir   = "${raw_dir}"
prep_dir  = "${prep_dir}"
max_utts  = int("${max_utts}")
random.seed(777)

EMOTIONS = ["angry", "happy", "neutral", "sad", "surprise"]
# ESD speakers: 0001-0010 are English and 0011-0020 are Mandarin.
EN_SPKS = {f"{i:04d}" for i in range(1, 11)}


def find_esd_speaker_root(base):
    """Accept raw_data/0001/... or raw_data/Emotion Speech Dataset/0001/."""
    if os.path.isdir(os.path.join(base, "0001")):
        return base
    try:
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if not os.path.isdir(p):
                continue
            if os.path.isdir(os.path.join(p, "0001")):
                return p
    except FileNotFoundError:
        pass
    return None


data_root = find_esd_speaker_root(raw_dir)
if data_root is None:
    print(
        f"[ESD] ERROR: Cannot find 0001 speaker folder under {raw_dir}. "
        "Re-run Stage 15: ./prepare_data.sh --stage 15 --stop_stage 15 (or local/download_esd.sh).",
        file=sys.stderr,
    )
    sys.exit(1)
if data_root != raw_dir:
    print(f"[ESD] Using nested data root: {data_root}")

# Find speaker directories (0001-0020) below data_root.
speaker_dirs = {}
for name in sorted(os.listdir(data_root)):
    path = os.path.join(data_root, name)
    if os.path.isdir(path) and len(name) == 4 and name.isdigit():
        speaker_dirs[name] = path

if not speaker_dirs:
    print(f"[ESD] ERROR: No speaker directories found in {data_root}", file=sys.stderr)
    sys.exit(1)

print(f"[ESD] Found {len(speaker_dirs)} speakers: {sorted(speaker_dirs.keys())[:3]} ...")

records = []   # (utt_id, spk_id, emotion, lang, wav_path)

for spk, spk_dir in sorted(speaker_dirs.items()):
    lang = "en" if spk in EN_SPKS else "zh"
    # Enumerate emotion directories case-insensitively.
    emo_map = {}
    for entry in sorted(os.listdir(spk_dir)):
        elo = entry.lower()
        if elo in EMOTIONS:
            emo_map[elo] = os.path.join(spk_dir, entry)

    for emo, emo_dir in sorted(emo_map.items()):
        wavs = sorted([
            f for f in os.listdir(emo_dir) if f.lower().endswith(".wav")
        ])
        for wav_name in wavs:
            stem = os.path.splitext(wav_name)[0]
            utt_id = f"esd_{spk}_{emo}_{stem}"
            wav_path = os.path.join(emo_dir, wav_name)
            records.append((utt_id, spk, emo, lang, wav_path))

print(f"[ESD] Total utterances found: {len(records)}")

# Sample max_utts while keeping speakers balanced.
if max_utts > 0 and len(records) > max_utts:
    # Sample evenly across speakers.
    from collections import defaultdict
    spk_recs = defaultdict(list)
    for r in records:
        spk_recs[r[1]].append(r)
    sampled = []
    per_spk = max_utts // len(spk_recs)
    for spk, recs in sorted(spk_recs.items()):
        if len(recs) <= per_spk:
            sampled.extend(recs)
        else:
            sampled.extend(random.sample(recs, per_spk))
    records = sampled
    print(f"[ESD] Sampled {len(records)} utterances (max_utts={max_utts})")

# Write metadata files.
wav_scp     = open(os.path.join(prep_dir, "wav.scp"),    "w")
utt2spk     = open(os.path.join(prep_dir, "utt2spk"),    "w")
utt2emotion = open(os.path.join(prep_dir, "utt2emotion"), "w")
utt2lang    = open(os.path.join(prep_dir, "utt2lang"),   "w")

for utt_id, spk, emo, lang, wav_path in records:
    wav_scp.write(f"{utt_id} {wav_path}\n")
    utt2spk.write(f"{utt_id} {spk}\n")
    utt2emotion.write(f"{utt_id} {emo}\n")
    utt2lang.write(f"{utt_id} {lang}\n")

wav_scp.close(); utt2spk.close(); utt2emotion.close(); utt2lang.close()

print(f"[ESD] Wrote {len(records)} entries.")
PYEOF

if [ $? -ne 0 ]; then
  echo "[ESD] ERROR: Failed to scan ESD files."
  exit 1
fi

if [ ! -f "${prep_dir}/wav.scp" ]; then
  echo "[ESD] ERROR: ${prep_dir}/wav.scp was not created."
  exit 1
fi

n_utts=$(wc -l < "${prep_dir}/wav.scp" | tr -d ' \t')
if [ "${n_utts}" -eq 0 ]; then
  echo "[ESD] ERROR: wav.scp is empty."
  exit 1
fi

# -----------------------------------------------------------------------
# Step 2: generate spk2utt.
# -----------------------------------------------------------------------
awk '{print $2, $1}' "${prep_dir}/utt2spk" | sort -k1 | \
awk '{
    if ($1 != prev) {
        if (prev != "") print line;
        prev=$1; line=$1 " " $2;
    } else {
        line=line " " $2;
    }
} END { if (prev != "") print line; }' > "${prep_dir}/spk2utt"

n_spks=$(wc -l < "${prep_dir}/spk2utt")
echo "[ESD] ${n_utts} utterances, ${n_spks} speakers."

# -----------------------------------------------------------------------
# Step 3: generate cross-emotion trials.
# -----------------------------------------------------------------------
trial_out="${prep_dir}/trials_emotion.txt"

if [ ! -f "${trial_out}" ]; then
  echo "[ESD] Generating emotion trials ..."
  python3 "${OPEN_ROOT}/tools/make_trials_esd.py" \
    --utt2spk     "${prep_dir}/utt2spk" \
    --utt2emotion "${prep_dir}/utt2emotion" \
    --output      "${trial_out}" \
    --max-target-trials 20000 \
    --nontarget-ratio 10 || exit 1
else
  echo "[ESD] ${trial_out} already exists, skipping."
fi

echo "=== [ESD] Preparation Done ==="
echo "  wav.scp:          $(wc -l < "${prep_dir}/wav.scp" | tr -d ' \t') entries"
echo "  utt2spk:          ${n_spks} speakers"
if [ -f "${trial_out}" ]; then
  echo "  trials_emotion:   $(wc -l < "${trial_out}" | tr -d ' \t') trials"
else
  echo "  trials_emotion:   (missing)"
fi
