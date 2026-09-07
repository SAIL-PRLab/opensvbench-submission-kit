#!/bin/bash

stage=1
stop_stage=20
data=data                                                     # Users can override with --data=...
nj=16                                                         # Number of preparation workers.
max_utts=10000                                                # Clean-audio cap; 0 disables the cap.
vox1_zip_dir=                                                 # Optional directory containing VoxCeleb1 zip archives.
whisper40_archive=                                            # Optional: manually downloaded whisper40.rar

KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${KIT_ROOT}" || exit 1

. tools/parse_options.sh || exit 1

data=$(realpath "${data}")

########################################
# Stage 1-2: TidyVoice
########################################

# Stage 1: download / placement check for TidyVoice
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "=== Stage 1: TidyVoice download / placement check ==="
  ./local/download_tidyvoice.sh --data "${data}"
fi

# Stage 2: prepare TidyVoice
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "=== Stage 2: TidyVoice prepare ==="
  ./local/prepare_tidyvoice.sh --data "${data}" --max-utts "${max_utts}"
fi

########################################
# Stage 3-4: HI-MIA
########################################

# Stage 3: HI-MIA download + unpack
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "=== Stage 3: HI-MIA download + unpack ==="
  ./local/download_hi-mia.sh --data "${data}"
fi

# Stage 4: HI-MIA prepare
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "=== Stage 4: HI-MIA prepare ==="
  ./local/prepare_hi-mia.sh --data "${data}" --max-utts "${max_utts}"
fi

########################################
# Stage 5-6: Google Speech Commands (GSC)
########################################

# Stage 5: GSC download + unpack
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "=== Stage 5: GSC download + unpack ==="
  ./local/download_gsc.sh --data "${data}"
fi

# Stage 6: GSC prepare
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "=== Stage 6: GSC prepare ==="
  ./local/prepare_gsc.sh --data "${data}" --max-utts "${max_utts}"
fi

########################################
# Stage 7-8: GLOBE (Common Voice Accent)
########################################

# Stage 7: GLOBE download (Git Clone)
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "=== Stage 7: GLOBE download ==="
  ./local/download_globe.sh --data "${data}"
fi

# Stage 8: GLOBE prepare (Parquet -> Wav + CSV)
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "=== Stage 8: GLOBE prepare ==="
  ./local/prepare_globe.sh --data "${data}" --max-utts "${max_utts}"
fi

########################################
# Stage 9-10: CN-Celeb (device, environment, and session conditions)
########################################

# Stage 9: CN-Celeb download
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "=== Stage 9: CN-Celeb download ==="
  ./local/download_cnceleb.sh --data "${data}"
fi

# Stage 10: CN-Celeb prepare
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  echo "=== Stage 10: CN-Celeb prepare ==="
  ./local/prepare_cnceleb.sh --data "${data}" --max-utts "${max_utts}"
fi

########################################
# Stage 11-12: VoxCeleb (official evaluation protocols)
########################################

# Stage 11: VoxCeleb download
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  echo "=== Stage 11: VoxCeleb download ==="
  ./local/download_voxceleb.sh --data "${data}" ${vox1_zip_dir:+--vox1-zip-dir "${vox1_zip_dir}"}
fi

# Stage 12: VoxCeleb prepare
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  echo "=== Stage 12: VoxCeleb prepare ==="
  ./local/prepare_voxceleb.sh --data "${data}"
fi

########################################
# Stage 13-14: 3D-Speaker (device, distance, and dialect conditions)
########################################

# Stage 13: 3D-Speaker download
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
  echo "=== Stage 13: 3D-Speaker download ==="
  ./local/download_3dspeaker.sh --data "${data}"
fi

# Stage 14: 3D-Speaker prepare
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
  echo "=== Stage 14: 3D-Speaker prepare ==="
  ./local/prepare_3dspeaker.sh --data "${data}" --max-utts "${max_utts}"
fi

########################################
# Stage 15-16: ESD (cross-emotion condition)
########################################

# Stage 15: ESD download
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
  echo "=== Stage 15: ESD download ==="
  ./local/download_esd.sh --data "${data}"
fi

# Stage 16: ESD prepare
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
  echo "=== Stage 16: ESD prepare ==="
  ./local/prepare_esd.sh --data "${data}" --max-utts "${max_utts}"
fi

########################################
# Stage 17-18: AliMeeting + VoxKnesset + FFSVC 2022 + Whisper40 + Lombard Grid + VOiCES + CHiME-6
########################################

# Stage 17: AliMeeting download + VoxKnesset access check + VoxPopuli access check + FFSVC 2022 download + Whisper40 archive handling + Lombard Grid download + VOiCES download + CHiME-6 download
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
  echo "=== Stage 17: AliMeeting download ==="
  ./local/download_alimeeting.sh --data "${data}"
  echo "=== Stage 17: VoxKnesset access check ==="
  ./local/download_voxknesset.sh --data "${data}"
  echo "=== Stage 17: VoxPopuli access check ==="
  ./local/download_voxpopuli.sh --data "${data}"
  echo "=== Stage 17: FFSVC 2022 download ==="
  ./local/download_ffsvc2022.sh --data "${data}"
  echo "=== Stage 17: Whisper40 archive handling ==="
  if [ -n "${whisper40_archive}" ]; then
    ./local/download_whisper40.sh --data "${data}" --archive "${whisper40_archive}"
  else
    ./local/download_whisper40.sh --data "${data}"
  fi
  echo "=== Stage 17: Lombard Grid download ==="
  ./local/download_lombardgrid.sh --data "${data}"
  echo "=== Stage 17: VOiCES download ==="
  ./local/download_voices.sh --data "${data}"
  echo "=== Stage 17: CHiME-6 download ==="
  ./local/download_chime6.sh --data "${data}"
fi

# Stage 18: AliMeeting prepare + VoxKnesset prepare + VoxPopuli prepare + FFSVC 2022 prepare + Whisper40 prepare + Lombard Grid prepare + VOiCES prepare + CHiME-6 prepare
if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ]; then
  echo "=== Stage 18: AliMeeting prepare ==="
  ./local/prepare_alimeeting.sh --data "${data}" --max-utts "${max_utts}"
  echo "=== Stage 18: VoxKnesset prepare ==="
  ./local/prepare_voxknesset.sh --data "${data}" --max-utts "${max_utts}"
  echo "=== Stage 18: VoxPopuli prepare ==="
  ./local/prepare_voxpopuli.sh --data "${data}" --max-utts "${max_utts}"
  echo "=== Stage 18: FFSVC 2022 prepare ==="
  ./local/prepare_ffsvc2022.sh --data "${data}"
  echo "=== Stage 18: Whisper40 prepare ==="
  ./local/prepare_whisper40.sh --data "${data}" --max-utts "${max_utts}"
  echo "=== Stage 18: Lombard Grid prepare ==="
  ./local/prepare_lombardgrid.sh --data "${data}" --max-utts "${max_utts}"
  echo "=== Stage 18: VOiCES prepare ==="
  ./local/prepare_voices.sh --data "${data}" --max-utts "${max_utts}"
  echo "=== Stage 18: CHiME-6 prepare ==="
  ./local/prepare_chime6.sh --data "${data}" --max-utts "${max_utts}"
fi

########################################
# Stage 19: Cut Short Utterances (VoxCeleb + CN-Celeb)
########################################

# Stage 19: VAD + duration-bucketed cutting for VoxCeleb and CN-Celeb
if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ]; then
  echo "=== Stage 19: Short Utterance Cutting (VAD + 0.5s/1s/2s/3s buckets) ==="

  for ds_tag in voxceleb cnceleb; do
    prep_dir=${data}/${ds_tag}/prepared

    if [ ! -f "${prep_dir}/wav.scp" ]; then
      echo "  [Skip] ${ds_tag}: ${prep_dir}/wav.scp not found"
      continue
    fi

    echo "  Processing ${ds_tag} ..."
    mkdir -p "${prep_dir}/short_utts"

    short_scps=()
    for dur in 0.5 1 2 3; do
      # python one-liner to get the same dur_str as the Python script
      dur_str=$(python3 -c "d=float('${dur}'); print(f'{int(d)}s' if d==int(d) else f'{d}s')")
      out_dir="${prep_dir}/short_utts/dur_${dur_str}"
      out_scp="${prep_dir}/wav_short_dur_${dur_str}.scp"

      if [ ! -f "${out_scp}" ]; then
        echo "    Cutting dur=${dur_str} for ${ds_tag} ..."
        python3 tools/cut_short_utterances.py \
          --wav-scp    "${prep_dir}/wav.scp" \
          --utt2spk    "${prep_dir}/utt2spk" \
          --output-dir "${out_dir}" \
          --target-dur  ${dur} \
          --output-scp  "${out_scp}" \
          --max-utts    5000 \
          --nj ${nj}
      else
        echo "    [Skip] ${out_scp} already exists"
      fi
      short_scps+=("${out_scp}")
    done

    # Merge all per-duration scps into wav_short.scp
    cat "${short_scps[@]}" > "${prep_dir}/wav_short.scp"
    echo "  ${ds_tag}: wav_short.scp created ($(wc -l < "${prep_dir}/wav_short.scp") entries)"

    cp "${prep_dir}/wav.scp" "${prep_dir}/wav_all.scp"

    # Only append if short utts not already present
    if ! grep -q "_dur_" "${prep_dir}/wav_all.scp" 2>/dev/null; then
      cat "${prep_dir}/wav_short.scp" >> "${prep_dir}/wav_all.scp"
      echo "  ${ds_tag}: short utts appended to wav_all.scp" \
           "($(wc -l < "${prep_dir}/wav_all.scp") total entries)"
    else
      echo "  ${ds_tag}: short utts already in wav_all.scp, skipping"
    fi
  done
fi

########################################
# Clean-only wav_all.scp refresh
########################################

if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ]; then
  echo "=== Refreshing clean-only wav_all.scp files ==="

  merge_targets=(
    "hi-mia|${data}/hi-mia/prepared"
    "tidyvoice|${data}/tidyvoice/prepared"
    "gsc|${data}/gsc/prepared"
    "globe|${data}/globe/prepared"
    "cnceleb|${data}/cnceleb/prepared"
    "voxceleb|${data}/voxceleb/prepared"
    "3dspeaker|${data}/3dspeaker/prepared"
    "ffsvc2022|${data}/ffsvc2022/prepared"
    "whisper40|${data}/whisper40/prepared"
    "lombardgrid|${data}/lombardgrid/prepared"
    "voices|${data}/voices/prepared"
    "chime6|${data}/chime6/prepared"
    "chime6_overlap|${data}/chime6/prepared_overlap"
    "esd|${data}/esd/prepared"
    "alimeeting|${data}/alimeeting/prepared"
    "voxknesset|${data}/voxknesset/prepared"
    "voxpopuli|${data}/voxpopuli/prepared"
  )

  for entry in "${merge_targets[@]}"; do
    IFS='|' read -r ds_name prep <<< "$entry"

    if [ ! -f "${prep}/wav.scp" ]; then
      continue
    fi

    cp "${prep}/wav.scp" "${prep}/wav_all.scp"
    if [ -f "${prep}/wav_short.scp" ]; then
      cat "${prep}/wav_short.scp" >> "${prep}/wav_all.scp"
    fi

    n_orig=$(wc -l < "${prep}/wav.scp")
    n_all=$(wc -l < "${prep}/wav_all.scp")
    echo "  [${ds_name}] wav_all.scp: ${n_all} entries (clean benchmark, base: ${n_orig})"
  done
fi

########################################
# Stage 20: Generate Short Utterance Trials (VoxCeleb + CN-Celeb)
########################################

# Stage 20: generate duration-bucketed trial files
if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ]; then
  echo "=== Stage 20: Short Utterance Trial Generation ==="

  for ds_tag in voxceleb cnceleb; do
    prep_dir=${data}/${ds_tag}/prepared

    if [ ! -f "${prep_dir}/wav_short.scp" ]; then
      echo "  [Skip] ${ds_tag}: wav_short.scp not found (run Stage 22 first)"
      continue
    fi

    if [ ! -f "${prep_dir}/utt2spk" ]; then
      echo "  [Skip] ${ds_tag}: utt2spk not found"
      continue
    fi

    trial_out="${prep_dir}/trials_short_dur.txt"

    python3 tools/make_trials_short_dur.py \
      --wav-scp           "${prep_dir}/wav_short.scp" \
      --utt2spk           "${prep_dir}/utt2spk" \
      --output            "${trial_out}" \
      --max-target-trials 20000 \
      --nontarget-ratio   10

    echo "  ${ds_tag}: $(wc -l < "${trial_out}") trials -> ${trial_out}"
  done
fi

echo "=== Data Preparation Complete ==="
