#!/usr/bin/env python3
"""Generate GSC speaker verification trials.

The active core uses:

trials_short.txt   - text-independent; target pairs are all C(n,2) pairs per
                     speaker, capped at --max-target-trials; non-target count is
                     target count multiplied by the ratio.

The archived text-dependent branch can also request:

trials_text_dependent.txt - text-dependent protocol:
                       target   : same speaker, same word
                       nontarget: same speaker different word, or
                                  different speaker same word
                     Target pairs are enumerated exhaustively, capped by
                     --max-target-trials, then matched with nontarget trials
                     at target count multiplied by --nontarget-ratio.
"""

import os
import random
import argparse
import json
import tempfile
from collections import defaultdict
from itertools import combinations


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate GSC trials with balanced target/nontarget ratios"
    )
    parser.add_argument("--utt2spk", required=True)
    parser.add_argument("--utt2text", required=True,
                        help="Path to utt2text (utt_id word)")
    parser.add_argument("--out-short", required=True,
                        help="Output for text-independent (short) trials")
    parser.add_argument("--out-text-dependent", default=None,
                        help="Output for text-dependent trials")
    parser.add_argument("--out-text-dependent-stats", default=None,
                        help="Optional JSON stats output for text-dependent trials")
    parser.add_argument("--out-content-shift", default=None,
                        help="Deprecated alias for --out-text-dependent")
    parser.add_argument(
        "--preserve-existing-short",
        action="store_true",
        help="Do not overwrite --out-short when it already exists.",
    )
    parser.add_argument("--max-target-trials", type=int, default=20000,
                        help="Max target trials (total for short; per sub-condition "
                             "for text-dependent). All C(n,2) pairs are enumerated first; "
                             "subsampled if count exceeds this limit. (default: 20000)")
    parser.add_argument("--nontarget-ratio", type=int, default=10,
                        help="Nontarget count = actual target count multiplied by this ratio "
                             "(default: 10)")
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def load_data(utt2spk_path, utt2text_path):
    spk2utt = defaultdict(list)
    utt2info = {}

    with open(utt2spk_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, s = parts[0], parts[1]
            spk2utt[s].append(u)
            utt2info.setdefault(u, {})["spk"] = s

    with open(utt2text_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, t = parts[0], parts[1]
            if u in utt2info:
                utt2info[u]["text"] = t

    return spk2utt, utt2info


def get_pair_key(u1, u2):
    return tuple(sorted((u1, u2)))


# ---------------------------------------------------------------------------
# Short (text-independent) trials
# ---------------------------------------------------------------------------

def generate_short_trials(spk2utt, max_target, nontarget_ratio, rng, out_path):
    """Enumerate same-speaker pairs and add sampled non-target pairs."""
    print(f"[Short] Enumerating all same-speaker target pairs "
          f"(cap={max_target}) ...")

    seen = set()
    all_target_pairs = []

    for spk, utts in spk2utt.items():
        for u1, u2 in combinations(sorted(utts), 2):
            p = get_pair_key(u1, u2)
            if p not in seen:
                seen.add(p)
                all_target_pairs.append(p)

    if max_target > 0 and len(all_target_pairs) > max_target:
        all_target_pairs = rng.sample(all_target_pairs, max_target)

    n_target = len(all_target_pairs)
    n_nontarget = n_target * nontarget_ratio

    print(f"[Short] {n_target} target pairs. Generating {n_nontarget} nontarget ...")
    speakers = list(spk2utt.keys())
    nontarget_pairs = []
    attempts = 0
    while len(nontarget_pairs) < n_nontarget and attempts < n_nontarget * 20:
        attempts += 1
        if len(speakers) < 2:
            break
        s1, s2 = rng.sample(speakers, 2)
        u1 = rng.choice(spk2utt[s1])
        u2 = rng.choice(spk2utt[s2])
        p = get_pair_key(u1, u2)
        if p not in seen:
            seen.add(p)
            nontarget_pairs.append(p)

    trials = (
        [(p[0], p[1], "target") for p in all_target_pairs] +
        [(p[0], p[1], "nontarget") for p in nontarget_pairs]
    )
    trials.sort()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for u1, u2, label in trials:
            f.write(f"{u1} {u2} {label}\n")

    print(f"[Short] Saved {len(trials)} trials "
          f"({n_target} target + {len(nontarget_pairs)} nontarget) -> {out_path}")


# ---------------------------------------------------------------------------
# Text-dependent trials
# ---------------------------------------------------------------------------

def round_robin_limit(grouped, limit, rng):
    prepared = {}
    active = []
    for key, items in grouped.items():
        copied = list(items)
        rng.shuffle(copied)
        if copied:
            prepared[key] = copied
            active.append(key)

    if limit <= 0:
        selected = []
        for key in sorted(prepared):
            selected.extend(prepared[key])
        return selected

    selected = []
    while active and len(selected) < limit:
        next_active = []
        for key in active:
            if len(selected) >= limit:
                break
            items = prepared[key]
            if items:
                selected.append(items.pop())
            if items:
                next_active.append(key)
        active = next_active
    return selected


def generate_text_dependent_trials(spk2utt, utt2info, max_target, nontarget_ratio,
                                   rng, out_path):
    """Generate a text-dependent SV condition.

    In text-dependent SV, a trial is accepted only when both speaker identity
    and lexical content match. Same-speaker different-word pairs are therefore
    nontarget trials, not target trials.
    """
    print(f"[TextDependent] Enumerating target pairs (cap={max_target}) ...")

    seen = set()

    # Index by speaker and word so targets require both speaker and text match.
    speaker_word_utts = defaultdict(lambda: defaultdict(list))
    word_all_utts = defaultdict(list)
    for u, info in utt2info.items():
        s = info.get("spk")
        t = info.get("text", "unknown")
        if s:
            speaker_word_utts[s][t].append(u)
            word_all_utts[t].append(u)

    # --- Target: same speaker, same word ---
    grouped_targets = defaultdict(list)
    for spk, word_map in speaker_word_utts.items():
        for word, utts in word_map.items():
            for u1, u2 in combinations(sorted(utts), 2):
                p = get_pair_key(u1, u2)
                if p not in seen:
                    grouped_targets[(spk, word)].append(p)

    target_pairs = sorted(round_robin_limit(grouped_targets, max_target, rng))
    for p in target_pairs:
        seen.add(p)
    print(f"  target_same_speaker_same_text: {len(target_pairs)} pairs")

    # --- Nontarget A: same speaker, different word ---
    n_nontarget_total = len(target_pairs) * nontarget_ratio
    n_same_spk_diff_text = n_nontarget_total // 2
    n_diff_spk_same_text = n_nontarget_total - n_same_spk_diff_text

    eligible_speakers = [
        spk for spk, word_map in speaker_word_utts.items()
        if sum(1 for utts in word_map.values() if utts) >= 2
    ]
    nontarget_same_spk_diff_text = []
    attempts = 0
    while (
        len(nontarget_same_spk_diff_text) < n_same_spk_diff_text
        and attempts < max(1, n_same_spk_diff_text * 100)
    ):
        attempts += 1
        spk = rng.choice(eligible_speakers)
        words_for_spk = [word for word, utts in speaker_word_utts[spk].items() if utts]
        w1, w2 = rng.sample(words_for_spk, 2)
        u1 = rng.choice(speaker_word_utts[spk][w1])
        u2 = rng.choice(speaker_word_utts[spk][w2])
        p = get_pair_key(u1, u2)
        if p not in seen:
            seen.add(p)
            nontarget_same_spk_diff_text.append(p)
    print(f"  nontarget_same_speaker_diff_text: {len(nontarget_same_spk_diff_text)} pairs")

    # --- Nontarget B: different speaker, same word ---
    words = list(word_all_utts.keys())
    nontarget_diff_spk_same_text = []
    attempts = 0
    while (
        len(nontarget_diff_spk_same_text) < n_diff_spk_same_text
        and attempts < max(1, n_diff_spk_same_text * 100)
    ):
        attempts += 1
        word = rng.choice(words)
        cands = word_all_utts[word]
        if len(cands) < 2:
            continue
        u1, u2 = rng.sample(cands, 2)
        if utt2info[u1].get("spk") == utt2info[u2].get("spk"):
            continue
        p = get_pair_key(u1, u2)
        if p not in seen:
            seen.add(p)
            nontarget_diff_spk_same_text.append(p)
    print(f"  nontarget_diff_speaker_same_text: {len(nontarget_diff_spk_same_text)} pairs")

    if len(nontarget_same_spk_diff_text) < n_same_spk_diff_text:
        raise RuntimeError(
            f"Could only generate {len(nontarget_same_spk_diff_text)} same-speaker different-text "
            f"nontarget pairs; expected {n_same_spk_diff_text}"
        )
    if len(nontarget_diff_spk_same_text) < n_diff_spk_same_text:
        raise RuntimeError(
            f"Could only generate {len(nontarget_diff_spk_same_text)} different-speaker same-text "
            f"nontarget pairs; expected {n_diff_spk_same_text}"
        )

    # --- Write ---
    trials = (
        [(p[0], p[1], "target", "text_dependent") for p in target_pairs] +
        [(p[0], p[1], "nontarget", "text_dependent") for p in nontarget_same_spk_diff_text] +
        [(p[0], p[1], "nontarget", "text_dependent") for p in nontarget_diff_spk_same_text]
    )
    trials.sort()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for u1, u2, lbl, cond in trials:
            f.write(f"{u1} {u2} {lbl} {cond}\n")

    print(f"[TextDependent] Saved {len(trials)} trials -> {out_path}")
    return {
        "num_trials": len(trials),
        "num_target_trials": len(target_pairs),
        "num_nontarget_trials": len(nontarget_same_spk_diff_text) + len(nontarget_diff_spk_same_text),
        "num_trial_utterances": len({u for trial in trials for u in trial[:2]}),
        "condition_counts": {"text_dependent": len(trials)},
        "nontarget_source_counts": {
            "same_speaker_different_text_nontarget": len(nontarget_same_spk_diff_text),
            "different_speaker_same_text_nontarget": len(nontarget_diff_spk_same_text),
        },
        "max_target_trials": max_target,
        "nontarget_ratio": nontarget_ratio,
        "eligible_speakers": len(eligible_speakers),
        "target_candidates": sum(len(items) for items in grouped_targets.values()),
    }


def main():
    args = get_args()
    rng = random.Random(args.seed)

    spk2utt, utt2info = load_data(args.utt2spk, args.utt2text)
    print(f"Loaded {len(spk2utt)} speakers, {len(utt2info)} utterances.")

    if args.preserve_existing_short and os.path.isfile(args.out_short):
        # Keep the text-dependent sample stable relative to the original
        # generator shape, where short-trial sampling consumes RNG first.
        with tempfile.NamedTemporaryFile(prefix="gsc_short_preserve_", suffix=".txt", delete=False) as tmp:
            tmp_short_path = tmp.name
        try:
            generate_short_trials(spk2utt, args.max_target_trials,
                                  args.nontarget_ratio, rng, tmp_short_path)
        finally:
            if os.path.exists(tmp_short_path):
                os.unlink(tmp_short_path)
        print(f"[Short] Preserved existing trial file -> {args.out_short}")
    else:
        generate_short_trials(spk2utt, args.max_target_trials,
                              args.nontarget_ratio, rng, args.out_short)
    text_dependent_out = args.out_text_dependent or args.out_content_shift
    if not text_dependent_out:
        print("[TextDependent] No --out-text-dependent provided; skipping archived text-dependent protocol.")
        return
    stats = generate_text_dependent_trials(spk2utt, utt2info, args.max_target_trials,
                                           args.nontarget_ratio, rng, text_dependent_out)
    if args.out_text_dependent_stats:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_text_dependent_stats)) or ".", exist_ok=True)
        with open(args.out_text_dependent_stats, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"[TextDependent] Saved stats -> {args.out_text_dependent_stats}")


if __name__ == "__main__":
    main()
