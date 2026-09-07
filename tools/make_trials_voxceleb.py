#!/usr/bin/env python3
"""Convert official VoxCeleb trial lists to the OpenSVBench format.

Official VoxCeleb format (one trial per line):
    <1|0> <spk_id/video_id/utt.wav> <spk_id/video_id/utt.wav>

OpenSVBench format:
    <utt1> <utt2> <target|nontarget> <condition>

Path-to-uttid mapping:
    id10270/x6uYqmx31kE/00001.wav  ->  id10270-x6uYqmx31kE-00001

Three trial lists are supported:
    veri_test2.txt      -> condition = vox1_O
    list_test_all2.txt  -> condition = vox1_E
    list_test_hard2.txt -> condition = vox1_H
"""

import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert VoxCeleb official trial lists to OpenSVBench format"
    )
    parser.add_argument(
        "--trial-src", required=True,
        help="Path to official VoxCeleb trial file (e.g. veri_test2.txt)"
    )
    parser.add_argument(
        "--wav-scp", default=None,
        help="Path to wav.scp; if provided, only include trials whose "
             "utterances exist in wav.scp"
    )
    parser.add_argument(
        "--condition", required=True,
        help="Condition label for this trial list (e.g. vox1_O, vox1_E, vox1_H)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output trials file"
    )
    return parser.parse_args()


def path_to_uttid(path_str):
    """Convert VoxCeleb path 'id10270/x6uYqmx31kE/00001.wav' to utt_id."""
    stem = path_str.replace(".wav", "").replace(".flac", "")
    return stem.replace("/", "-")


def main():
    args = get_args()

    available_utts = None
    if args.wav_scp:
        available_utts = set()
        with open(args.wav_scp) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if parts:
                    available_utts.add(parts[0])
        print(f"Loaded {len(available_utts)} utterances from wav.scp")

    trials = []
    skipped = 0

    with open(args.trial_src) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            label_int, path1, path2 = parts[0], parts[1], parts[2]

            utt1 = path_to_uttid(path1)
            utt2 = path_to_uttid(path2)

            if available_utts is not None:
                if utt1 not in available_utts or utt2 not in available_utts:
                    skipped += 1
                    continue

            label = "target" if label_int == "1" else "nontarget"
            trials.append((utt1, utt2, label, args.condition))

    if skipped > 0:
        print(f"Skipped {skipped} trials (utterances not in wav.scp)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for u1, u2, label, cond in trials:
            f.write(f"{u1} {u2} {label} {cond}\n")

    n_tar = sum(1 for t in trials if t[2] == "target")
    n_non = sum(1 for t in trials if t[2] == "nontarget")
    print(f"Saved {len(trials)} trials to {args.output} "
          f"(target: {n_tar}, nontarget: {n_non}, condition: {args.condition})")


if __name__ == "__main__":
    main()
