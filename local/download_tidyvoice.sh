#!/bin/bash

# Validate or describe manual placement for TidyVoiceX2-ASV.

data=data
. tools/parse_options.sh || exit 1

data=$(realpath "${data}")
raw_dir="${data}/tidyvoice/raw_data"
audio_root="${raw_dir}/TidyVoiceX2_ASV"
pair_file_a="${audio_root}/trials/tv26_eval-A.txt"
pair_file_u="${audio_root}/trials/tv26_eval-U.txt"

mkdir -p "${raw_dir}"

echo "=== [TidyVoice] Download / placement check ==="

if [ -d "${audio_root}" ] && [ -f "${pair_file_a}" ] && [ -f "${pair_file_u}" ]; then
  echo "[TidyVoice] Found existing local copy."
  echo "  audio_root : ${audio_root}"
  echo "  pair_file A: ${pair_file_a}"
  echo "  pair_file U: ${pair_file_u}"
  exit 0
fi

echo "[TidyVoice] Automatic download is not configured in this repo."
echo "[TidyVoice] Please manually place the official TidyVoiceX2-ASV evaluation set here:"
echo "  ${audio_root}"
echo "[TidyVoice] And place the released labeled eval pair/key files here:"
echo "  ${pair_file_a}"
echo "  ${pair_file_u}"
echo ""
echo "[TidyVoice] Expected layout:"
echo "  ${audio_root}/<speaker_id>/<language>/<file>.wav"
echo "  ${audio_root}/trials/tv26_eval-A.txt   # lines: <rel_path_1> <rel_path_2> <target|nontarget>"
echo "  ${audio_root}/trials/tv26_eval-U.txt   # lines: <rel_path_1> <rel_path_2> <target|nontarget>"
echo ""
echo "[TidyVoice] The downstream prepare script will work directly from those paths."
exit 1
