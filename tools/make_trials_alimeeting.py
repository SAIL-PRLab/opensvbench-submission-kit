#!/usr/bin/env python3
"""Generate speaker verification trials for AliMeeting (meeting scenario).

Conditions:
  near_near  : enrollment and test both from headset microphone (clean baseline)
  near_far   : enrollment from headset, test from far-field array (realistic meeting SV)

Usage:
  python3 tools/make_trials_alimeeting.py \
      --utt2spk    prepared/utt2spk \
      --utt2channel prepared/utt2channel \
      --output      prepared/trials_meeting.txt \
      --max-target-trials 20000 \
      --nontarget-ratio   10
"""

import argparse
import random
from collections import defaultdict
from itertools import combinations


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--utt2spk",     required=True)
    p.add_argument("--utt2channel", required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--max-target-trials", type=int, default=20000)
    p.add_argument("--nontarget-ratio",   type=int, default=10)
    p.add_argument("--seed", type=int, default=777)
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)

    utt2spk = {}
    with open(args.utt2spk) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2spk[parts[0]] = parts[1]

    utt2ch = {}
    with open(args.utt2channel) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2ch[parts[0]] = parts[1]

    valid = {u for u in utt2spk if u in utt2ch}
    print(f"[AliMeeting trials] {len(valid)} valid utterances, "
          f"{len(set(utt2spk[u] for u in valid))} speakers")

    # Separate near and far utterances per speaker
    spk_near = defaultdict(list)
    spk_far  = defaultdict(list)
    for u in sorted(valid):
        spk = utt2spk[u]
        ch  = utt2ch[u]
        if ch == "near":
            spk_near[spk].append(u)
        elif ch == "far":
            spk_far[spk].append(u)

    # speakers that have BOTH near and far utterances (needed for near_far trials)
    both_spks = sorted(set(spk_near) & set(spk_far))
    near_spks = sorted(set(spk_near))
    print(f"  near-only speakers: {len(spk_near)}, "
          f"far-only speakers: {len(spk_far)}, "
          f"both: {len(both_spks)}")

    speakers = sorted(set(utt2spk.values()))
    all_near = [u for spk in near_spks for u in spk_near[spk]]
    all_far  = [u for spk in sorted(set(spk_far)) for u in spk_far[spk]]

    all_trials = []
    seen = set()

    def add(u1, u2, label, cond):
        p = tuple(sorted((u1, u2)))
        if p not in seen and u1 != u2:
            seen.add(p)
            all_trials.append((p[0], p[1], label, cond))
            return True
        return False

    # Near-near target pairs.
    nn_targets = []
    for spk in near_spks:
        utts = spk_near[spk]
        for u1, u2 in combinations(sorted(utts), 2):
            nn_targets.append((u1, u2))
    if args.max_target_trials > 0 and len(nn_targets) > args.max_target_trials:
        nn_targets = random.sample(nn_targets, args.max_target_trials)
    for u1, u2 in nn_targets:
        add(u1, u2, "target", "near_near")
    print(f"  near_near target: {len(nn_targets)}")

    # Near-far target pairs.
    nf_targets = []
    for spk in both_spks:
        for u_near in spk_near[spk]:
            for u_far in spk_far[spk]:
                nf_targets.append((u_near, u_far))
    if args.max_target_trials > 0 and len(nf_targets) > args.max_target_trials:
        nf_targets = random.sample(nf_targets, args.max_target_trials)
    for u1, u2 in nf_targets:
        # Ensure near is u1
        if utt2ch.get(u1) == "far":
            u1, u2 = u2, u1
        add(u1, u2, "target", "near_far")
    print(f"  near_far target:  {len(nf_targets)}")

    # Near-near non-target pairs.
    n_nn_nt = len(nn_targets) * args.nontarget_ratio
    cnt, attempts = 0, 0
    while cnt < n_nn_nt and attempts < n_nn_nt * 30:
        attempts += 1
        if len(all_near) < 2:
            break
        u1, u2 = random.sample(all_near, 2)
        if utt2spk[u1] == utt2spk[u2]:
            continue
        if add(u1, u2, "nontarget", "near_near"):
            cnt += 1
    print(f"  near_near nontarget: {cnt}")

    # Near-far non-target pairs.
    n_nf_nt = len(nf_targets) * args.nontarget_ratio
    cnt, attempts = 0, 0
    while cnt < n_nf_nt and attempts < n_nf_nt * 30:
        attempts += 1
        if not all_near or not all_far:
            break
        u_near = random.choice(all_near)
        u_far  = random.choice(all_far)
        if utt2spk[u_near] == utt2spk[u_far]:
            continue
        p = tuple(sorted((u_near, u_far)))
        if p not in seen:
            seen.add(p)
            all_trials.append((u_near, u_far, "nontarget", "near_far"))
            cnt += 1
    print(f"  near_far nontarget: {cnt}")

    # Write the trial file.
    all_trials.sort()
    stats = defaultdict(int)
    for t in all_trials:
        stats[(t[2], t[3])] += 1
    print("Final stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k[0]:12s}  {k[1]:12s}  {v}")

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for t in all_trials:
            f.write(f"{t[0]} {t[1]} {t[2]} {t[3]}\n")
    print(f"Saved {len(all_trials)} trials -> {args.output}")


if __name__ == "__main__":
    main()
