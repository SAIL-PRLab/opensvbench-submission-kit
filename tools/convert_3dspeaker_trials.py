#!/usr/bin/env python3
"""Convert 3D-Speaker official trial lists to OpenSVBench format.

Official trial format (trials_cross_device, trials_cross_distance, trials_cross_dialect):
  <utt1> <utt2> <0|1>

OpenSVBench format:
  <utt1> <utt2> <target|nontarget> <condition>

Multiple trial files are merged into a single output, each tagged with its
corresponding condition label.
"""

import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert 3D-Speaker official trials to OpenSVBench format"
    )
    parser.add_argument(
        "--trial-files", nargs="+", required=True,
        help="Paths to official trial files (space-separated)"
    )
    parser.add_argument(
        "--conditions", nargs="+", required=True,
        help="Condition labels corresponding to each trial file (same order)"
    )
    parser.add_argument(
        "--wav-scp", required=True,
        help="Path to wav.scp (utt_id path) for validation"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output merged trial file path"
    )
    return parser.parse_args()


def main():
    args = get_args()

    if len(args.trial_files) != len(args.conditions):
        print(
            f"ERROR: Number of trial files ({len(args.trial_files)}) "
            f"!= number of conditions ({len(args.conditions)})",
            file=sys.stderr,
        )
        sys.exit(1)

    utt_set = set()
    with open(args.wav_scp) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if parts:
                utt_set.add(parts[0])

    all_trials = []
    total_skipped = 0

    for trial_path, condition in zip(args.trial_files, args.conditions):
        n_kept = 0
        n_skip = 0
        with open(trial_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                utt1, utt2, label_int = parts[0], parts[1], parts[2]
                if utt1 not in utt_set or utt2 not in utt_set:
                    n_skip += 1
                    continue
                label = "target" if label_int in ("1", "target") else "nontarget"
                all_trials.append((utt1, utt2, label, condition))
                n_kept += 1
        total_skipped += n_skip
        n_tar = sum(1 for t in all_trials[-n_kept:] if t[2] == "target")
        n_non = n_kept - n_tar
        print(f"  [{condition}] {trial_path}: {n_kept} trials "
              f"(target: {n_tar}, nontarget: {n_non}, skipped: {n_skip})")

    if total_skipped > 0:
        print(f"Total skipped: {total_skipped} (utt_id not in wav.scp)")

    with open(args.output, "w") as f:
        for utt1, utt2, label, cond in all_trials:
            f.write(f"{utt1} {utt2} {label} {cond}\n")

    n_tar = sum(1 for t in all_trials if t[2] == "target")
    n_non = len(all_trials) - n_tar
    print(f"Saved {len(all_trials)} trials to {args.output} "
          f"(target: {n_tar}, nontarget: {n_non})")


if __name__ == "__main__":
    main()
