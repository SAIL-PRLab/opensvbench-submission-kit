#!/usr/bin/env python3
"""Generate cross-emotion speaker verification trials for ESD.

Trial format (4 columns):
  utt1  utt2  target|nontarget  condition

Conditions:
  same_emotion   : enroll & test utterances from same emotion category
  cross_emotion  : enroll & test utterances from different emotion categories

Both target (same speaker) and nontarget (different speaker) pairs are
generated for each condition.

Usage:
  python3 tools/make_trials_esd.py \
      --utt2spk prepared/utt2spk \
      --utt2emotion prepared/utt2emotion \
      --output prepared/trials_emotion.txt \
      --max-target-trials 20000 \
      --nontarget-ratio 10
"""

import argparse
import random
from collections import defaultdict
from itertools import combinations


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--utt2spk",    required=True)
    p.add_argument("--utt2emotion", required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--max-target-trials", type=int, default=20000,
                   help="Cap on target trials per condition (0 = keep all)")
    p.add_argument("--nontarget-ratio", type=int, default=10,
                   help="Nontarget count = actual target count multiplied by this ratio")
    p.add_argument("--seed", type=int, default=777)
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)

    # Load metadata.
    utt2spk = {}
    with open(args.utt2spk) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2spk[parts[0]] = parts[1]

    utt2emo = {}
    with open(args.utt2emotion) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2emo[parts[0]] = parts[1]

    # only keep utts that have both spk and emotion
    valid = {u for u in utt2spk if u in utt2emo}
    print(f"[ESD trials] {len(valid)} valid utterances, "
          f"{len(set(utt2spk[u] for u in valid))} speakers, "
          f"{len(set(utt2emo[u] for u in valid))} emotions")

    spk2utts  = defaultdict(list)
    emo2utts  = defaultdict(list)
    for u in sorted(valid):
        spk2utts[utt2spk[u]].append(u)
        emo2utts[utt2emo[u]].append(u)

    speakers = sorted(spk2utts.keys())
    emotions = sorted(emo2utts.keys())

    all_trials = []
    seen = set()

    def add_target(u1, u2, cond):
        p = tuple(sorted((u1, u2)))
        if p not in seen and u1 != u2:
            seen.add(p)
            all_trials.append((p[0], p[1], "target", cond))

    def add_nontarget(u1, u2, cond):
        p = tuple(sorted((u1, u2)))
        if p not in seen:
            seen.add(p)
            all_trials.append((p[0], p[1], "nontarget", cond))

    # Same-emotion target pairs: same speaker and emotion.
    same_emo_targets = []
    for spk in speakers:
        utts = spk2utts[spk]
        # group by emotion
        spk_emo2utts = defaultdict(list)
        for u in utts:
            spk_emo2utts[utt2emo[u]].append(u)
        for emo, eu in spk_emo2utts.items():
            for u1, u2 in combinations(sorted(eu), 2):
                same_emo_targets.append((u1, u2))

    if args.max_target_trials > 0 and len(same_emo_targets) > args.max_target_trials:
        same_emo_targets = random.sample(same_emo_targets, args.max_target_trials)
    print(f"[ESD trials] same_emotion target pairs: {len(same_emo_targets)}")
    for u1, u2 in same_emo_targets:
        add_target(u1, u2, "same_emotion")

    # Cross-emotion target pairs: same speaker, different emotion.
    cross_emo_targets = []
    for spk in speakers:
        utts = spk2utts[spk]
        spk_emo2utts = defaultdict(list)
        for u in utts:
            spk_emo2utts[utt2emo[u]].append(u)
        emo_list = sorted(spk_emo2utts.keys())
        for e1, e2 in combinations(emo_list, 2):
            for u1 in spk_emo2utts[e1]:
                for u2 in spk_emo2utts[e2]:
                    cross_emo_targets.append((u1, u2))

    if args.max_target_trials > 0 and len(cross_emo_targets) > args.max_target_trials:
        cross_emo_targets = random.sample(cross_emo_targets, args.max_target_trials)
    print(f"[ESD trials] cross_emotion target pairs: {len(cross_emo_targets)}")
    for u1, u2 in cross_emo_targets:
        add_target(u1, u2, "cross_emotion")

    # Non-target pairs.
    # Same-emotion non-targets use different speakers with the same emotion.
    n_same_nt = len(same_emo_targets) * args.nontarget_ratio
    cnt, attempts = 0, 0
    while cnt < n_same_nt and attempts < n_same_nt * 30:
        attempts += 1
        emo = random.choice(emotions)
        eu = emo2utts[emo]
        if len(eu) < 2:
            continue
        u1, u2 = random.sample(eu, 2)
        if utt2spk[u1] == utt2spk[u2]:
            continue
        add_nontarget(u1, u2, "same_emotion")
        cnt += 1
    print(f"[ESD trials] same_emotion nontarget pairs: {cnt}")

    # cross_emotion nontarget: different speaker, different emotion
    n_cross_nt = len(cross_emo_targets) * args.nontarget_ratio
    cnt, attempts = 0, 0
    while cnt < n_cross_nt and attempts < n_cross_nt * 30:
        attempts += 1
        e1, e2 = random.sample(emotions, 2)
        u1 = random.choice(emo2utts[e1])
        u2 = random.choice(emo2utts[e2])
        if utt2spk[u1] == utt2spk[u2]:
            continue
        add_nontarget(u1, u2, "cross_emotion")
        cnt += 1
    print(f"[ESD trials] cross_emotion nontarget pairs: {cnt}")

    # Write the trial file.
    all_trials.sort()
    stats = defaultdict(int)
    for t in all_trials:
        stats[(t[2], t[3])] += 1
    print("Final stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k[0]:12s}  {k[1]:16s}  {v}")

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for t in all_trials:
            f.write(f"{t[0]} {t[1]} {t[2]} {t[3]}\n")
    print(f"Saved {len(all_trials)} trials to {args.output}")


if __name__ == "__main__":
    main()
