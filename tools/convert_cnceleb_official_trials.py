#!/usr/bin/env python3
"""Convert CN-Celeb official trial list to OpenSVBench format.

Official trials.lst format: 3 columns
  - Column 1: enroll_id (index into enroll.lst to get enroll path) or path
  - Column 2: test_path (path to test utterance, often under eval/)
  - Column 3: 0 (nontarget) or 1 (target)

If --enroll-lst is provided, column 1 is interpreted as enroll_id and looked up
in enroll.lst (two columns: enroll_id path) to get the enroll path.

OpenSVBench format:
  <utt1> <utt2> <target|nontarget> [condition]
"""

import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Convert CN-Celeb official trials to OpenSVBench format")
    parser.add_argument("--official-trials", required=True, help="Path to official trials.lst")
    parser.add_argument("--wav-scp", required=True, help="Path to wav.scp (utt_id path)")
    parser.add_argument("--enroll-lst", default=None, help="Path to enroll.lst (enroll_id path); if set, trial col1 is enroll_id")
    parser.add_argument("--output", required=True, help="Output trial file path")
    parser.add_argument("--condition", default="default", help="Condition label (default: default)")
    return parser.parse_args()


def main():
    args = get_args()

    # Build path/suffix -> utt_id from wav.scp (many key variants for O(1) lookup)
    path2utt = {}
    utt_set = set()
    with open(args.wav_scp) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            utt_id, path = parts[0], parts[1].strip()
            utt_set.add(utt_id)
            path_n = path.replace("\\", "/")
            path2utt[path] = utt_id
            path2utt[path_n] = utt_id
            # Suffixes: data/... and eval/... (with and without extension)
            for sep in ["/data/", "data/"]:
                if sep in path_n:
                    suffix = path_n.split(sep, 1)[-1].strip("/")
                    if suffix:
                        path2utt.setdefault(suffix, utt_id)
                        path2utt.setdefault("data/" + suffix, utt_id)
                        if "." in suffix:
                            path2utt.setdefault(suffix.rsplit(".", 1)[0], utt_id)
                    break
            for sep in ["/eval/", "eval/"]:
                if sep in path_n:
                    suffix = path_n.split(sep, 1)[-1].strip("/")
                    if suffix:
                        path2utt.setdefault(suffix, utt_id)
                        path2utt.setdefault("eval/" + suffix, utt_id)
                        if "." in suffix:
                            path2utt.setdefault(suffix.rsplit(".", 1)[0], utt_id)
                    break
            base = path_n.split("/")[-1]
            if base:
                path2utt.setdefault(base, utt_id)
                if "." in base:
                    path2utt.setdefault(base.rsplit(".", 1)[0], utt_id)

    def resolve(key):
        if key is None or not key:
            return None
        key = key.strip().replace("\\", "/")
        if key in utt_set:
            return key
        out = path2utt.get(key)
        if out is not None:
            return out
        for ext in (".flac", ".wav"):
            if key.endswith(ext):
                out = path2utt.get(key[:-len(ext)])
                if out is not None:
                    return out
        out = path2utt.get(key + ".flac") or path2utt.get(key + ".wav")
        return out

    # enroll_id -> path (from enroll.lst); trial col1 is enroll_id when this is set
    enroll_id2path = {}
    if args.enroll_lst and os.path.isfile(args.enroll_lst):
        with open(args.enroll_lst) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    enroll_id2path[parts[0]] = parts[1].strip().replace("\\", "/")
        print(f"Loaded {len(enroll_id2path)} enroll IDs from {args.enroll_lst}")

    trials = []
    skipped = 0
    with open(args.official_trials) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            a, b, label_int = parts[0], parts[1], parts[2]
            path_a = enroll_id2path.get(a, a) if enroll_id2path else a
            u1 = resolve(path_a)
            u2 = resolve(b)
            if u1 is None or u2 is None:
                skipped += 1
                continue
            label = "target" if label_int == "1" else "nontarget"
            trials.append((u1, u2, label, args.condition))

    if skipped > 0:
        print(f"Skipped {skipped} trials (could not resolve to wav.scp utt_id)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for u1, u2, label, cond in trials:
            f.write(f"{u1} {u2} {label} {cond}\n")

    n_tar = sum(1 for t in trials if t[2] == "target")
    n_non = sum(1 for t in trials if t[2] == "nontarget")
    print(f"Saved {len(trials)} trials to {args.output} (target: {n_tar}, nontarget: {n_non})")


if __name__ == "__main__":
    main()
