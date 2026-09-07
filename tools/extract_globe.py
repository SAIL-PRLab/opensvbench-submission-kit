#!/usr/bin/env python3
# OpenSVBench GLOBE extraction helper.

import os
import glob
import argparse
import pandas as pd
import soundfile as sf
import io
import re
import sys

def get_args():
    parser = argparse.ArgumentParser(description="Extract GLOBE parquet and generate Kaldi files")
    parser.add_argument("--download-dir", required=True, help="Root directory containing .parquet files")
    parser.add_argument("--wav-dir", required=True, help="Output directory for WAV files")
    parser.add_argument("--out-prep", required=True, help="Directory to output wav.scp, utt2spk, metadata.csv")
    # Limit the number of speakers; the default is 5000.
    parser.add_argument("--max-spks", type=int, default=5000, help="Max number of speakers to extract (default: 5000)")
    parser.add_argument(
        "--canonical-manifest",
        default=None,
        help="Optional OpenSVBench TSV manifest. If set, only these utterances are materialized.",
    )
    parser.add_argument(
        "--shards-tsv",
        default=None,
        help="Optional TSV with parquet_relpath. Restricts scanning to listed shards.",
    )
    return parser.parse_args()

def clean_name(name):
    # Replace non-alphanumeric characters with underscores for directory names.
    return re.sub(r'[^a-zA-Z0-9]', '_', str(name))

def load_canonical_manifest(path):
    canonical = {}
    with open(path, "r") as f:
        header = f.readline().strip().split("\t")
        expected = ["utt_id", "rel_path", "speaker_id", "accent"]
        if header != expected:
            raise ValueError(f"Unexpected canonical manifest header in {path}: {header}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"Malformed canonical manifest row: {line!r}")
            utt_id, rel_path, speaker_id, accent = parts
            canonical[utt_id] = {
                "rel_path": rel_path,
                "speaker_id": speaker_id,
                "accent": accent,
            }
    return canonical

def load_shard_relpaths(path):
    relpaths = []
    with open(path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or parts[0] in ("", "parquet_relpath"):
                continue
            relpaths.append(parts[0])
    return relpaths

def main():
    args = get_args()
    canonical = load_canonical_manifest(args.canonical_manifest) if args.canonical_manifest else None

    # 1. Find parquet files. If no shard list is supplied, scan the directory.
    # Sorting keeps subset selection reproducible.
    if args.shards_tsv:
        shard_relpaths = load_shard_relpaths(args.shards_tsv)
        parquet_files = [os.path.join(args.download_dir, relpath) for relpath in shard_relpaths]
        missing_shards = [path for path in parquet_files if not os.path.isfile(path)]
        if missing_shards:
            print("Error: Missing canonical GLOBE parquet shards:")
            for path in missing_shards[:20]:
                print(f"  {path}")
            if len(missing_shards) > 20:
                print(f"  ... and {len(missing_shards) - 20} more")
            sys.exit(2)
    else:
        parquet_files = sorted(glob.glob(os.path.join(args.download_dir, "**/*.parquet"), recursive=True))

    if not parquet_files:
        print(f"Error: No .parquet files found in {args.download_dir}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet files. Starting processing...")
    print(f"Processing target: {args.wav_dir}")
    if canonical:
        print(f"Canonical manifest: {args.canonical_manifest}")
        print(f"Canonical utterances requested: {len(canonical)}")
    else:
        print(f"Subset limit: {args.max_spks} speakers.")

    os.makedirs(args.wav_dir, exist_ok=True)
    os.makedirs(args.out_prep, exist_ok=True)

    metadata_list = []

    # Prepare Kaldi-style output files.
    wav_scp_path = os.path.join(args.out_prep, "wav.scp")
    utt2spk_path = os.path.join(args.out_prep, "utt2spk")
    utt2accent_path = os.path.join(args.out_prep, "utt2accent")

    f_wav = open(wav_scp_path, 'w')
    f_u2s = open(utt2spk_path, 'w')
    f_u2a = open(utt2accent_path, 'w')

    count = 0
    accepted_spks = set()  # Speakers admitted to the sampled subset.
    found_canonical = set()

    for p_file in parquet_files:
        print(f"Processing: {os.path.basename(p_file)}...")
        try:
            df = pd.read_parquet(p_file)
            for idx, row in df.iterrows():
                spk_id = str(row.get('speaker_id', 'unknown'))
                utt_id = f"{spk_id}_{idx:06d}"

                if canonical is not None:
                    if utt_id not in canonical or utt_id in found_canonical:
                        continue
                    expected = canonical[utt_id]
                    if expected["speaker_id"] != spk_id:
                        raise ValueError(
                            f"Canonical speaker mismatch for {utt_id}: "
                            f"manifest={expected['speaker_id']} parquet={spk_id}"
                        )
                else:
                    expected = None

                # === Subsampling logic ===
                if canonical is None and spk_id not in accepted_spks:
                    # Check the speaker cap when a new speaker appears.
                    if len(accepted_spks) >= args.max_spks:
                        continue  # Skip rows for speakers beyond the cap.
                    else:
                        accepted_spks.add(spk_id)
                # Keep all rows for speakers already admitted.
                # =================

                raw_accent = str(row.get('accent', 'unknown'))
                accent_folder = clean_name(raw_accent)

                # Decode the audio payload.
                audio_data = row['audio']
                if isinstance(audio_data, dict) and 'bytes' in audio_data:
                    audio_bytes = audio_data['bytes']
                    # Use SPEAKER_INDEX.wav as the output filename.
                    filename = f"{utt_id}.wav"
                else:
                    continue

                # Default path: raw_data/ACCENT/SPEAKER/filename.
                if expected is not None:
                    out_path = os.path.join(args.wav_dir, expected["rel_path"])
                    target_dir = os.path.dirname(out_path)
                else:
                    target_dir = os.path.join(args.wav_dir, accent_folder, spk_id)
                    out_path = os.path.join(target_dir, filename)
                os.makedirs(target_dir, exist_ok=True)

                # Write the WAV once; existing files can be reused.
                if not os.path.exists(out_path):
                    try:
                        data_array, samplerate = sf.read(io.BytesIO(audio_bytes))
                        sf.write(out_path, data_array, samplerate)
                    except Exception as e:
                        print(f"  Warning: Failed to save {filename}: {e}")
                        continue

                # The utterance ID is already extension-free.
                abs_path = os.path.abspath(out_path)

                # 1. Write Kaldi-style files as rows are processed.
                f_wav.write(f"{utt_id} {abs_path}\n")
                f_u2s.write(f"{utt_id} {spk_id}\n")
                f_u2a.write(f"{utt_id} {accent_folder}\n")

                # 2. Collect metadata for the backup CSV.
                metadata_list.append({
                    'utt_id': utt_id,
                    'wav_path': abs_path,
                    'speaker_id': spk_id,
                    'accent': raw_accent,
                    'clean_accent': accent_folder
                })
                count += 1
                if canonical is not None:
                    found_canonical.add(utt_id)

        except Exception as e:
            print(f"Error reading {p_file}: {e}")

    f_wav.close()
    f_u2s.close()
    f_u2a.close()

    if canonical is not None and len(found_canonical) != len(canonical):
        missing = sorted(set(canonical) - found_canonical)
        print(
            f"Error: Missing {len(missing)} canonical GLOBE utterances "
            f"after scanning {len(parquet_files)} parquet files."
        )
        print("Sample missing utterances:")
        for utt_id in missing[:20]:
            print(f"  {utt_id}\t{canonical[utt_id]['rel_path']}")
        sys.exit(2)

    # Save metadata.csv as a convenient backup.
    if metadata_list:
        pd.DataFrame(metadata_list).to_csv(os.path.join(args.out_prep, "metadata.csv"), index=False)
        speaker_count = len({item["speaker_id"] for item in metadata_list})
        print(f"Done. Processed {count} utterances from {speaker_count} speakers.")
        print(f"Generated: {wav_scp_path}, {utt2spk_path}")
    else:
        print("No data extracted.")
        sys.exit(1)

if __name__ == "__main__":
    main()
