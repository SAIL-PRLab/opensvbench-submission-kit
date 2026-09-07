#!/usr/bin/env python3
"""Generate Whisper40 neutral-to-whisper verification trials for OpenSVBench."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CONDITION = "nature_to_whsp_same_text"
TARGET_SPLIT_PRIORITY = {
    ("enroll", "test"): 0,
    ("test", "test"): 1,
    ("enroll", "enroll"): 2,
    ("test", "enroll"): 3,
}


@dataclass(frozen=True)
class PairRecord:
    enroll_utt: str
    test_utt: str
    speaker_id: str
    text_id: str
    split_pair: tuple[str, str]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Whisper40 same-text neutral-to-whisper trials")
    parser.add_argument("--utt2spk", required=True)
    parser.add_argument("--utt2style", required=True)
    parser.add_argument("--utt2text", required=True)
    parser.add_argument("--utt2split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-pair-metadata", default=None)
    parser.add_argument("--output-stats", default=None)
    parser.add_argument("--max-target-trials", type=int, default=5000)
    parser.add_argument("--nontarget-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def read_mapping(path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def round_robin_limit(grouped: dict[str, list[PairRecord]], limit: int, rng: random.Random) -> list[PairRecord]:
    prepared: dict[str, list[PairRecord]] = {}
    active: list[str] = []
    for key, items in grouped.items():
        copied = list(items)
        rng.shuffle(copied)
        if copied:
            prepared[key] = copied
            active.append(key)

    if limit <= 0:
        limit = sum(len(items) for items in prepared.values())

    total = sum(len(items) for items in prepared.values())
    if total <= limit:
        selected: list[PairRecord] = []
        for key in sorted(prepared):
            selected.extend(prepared[key])
        return selected

    selected: list[PairRecord] = []
    while active and len(selected) < limit:
        next_active: list[str] = []
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


def canonical_pair(utt1: str, utt2: str) -> tuple[str, str]:
    return (utt1, utt2) if utt1 <= utt2 else (utt2, utt1)


def write_trials(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    with path.open("w") as f:
        for utt1, utt2, label, condition in rows:
            f.write(f"{utt1} {utt2} {label} {condition}\n")


def write_pair_metadata(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "utt1",
        "utt2",
        "label",
        "condition",
        "speaker1",
        "speaker2",
        "style1",
        "style2",
        "text1",
        "text2",
        "split1",
        "split2",
        "source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = get_args()
    rng = random.Random(args.seed)

    utt2spk = read_mapping(args.utt2spk)
    utt2style = read_mapping(args.utt2style)
    utt2text = read_mapping(args.utt2text)
    utt2split = read_mapping(args.utt2split)

    valid_utts = sorted(set(utt2spk) & set(utt2style) & set(utt2text) & set(utt2split))
    if not valid_utts:
        raise ValueError("No Whisper40 utterances have speaker/style/text/split metadata simultaneously.")

    by_speaker_style_text: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_style_text: dict[tuple[str, str], list[str]] = defaultdict(list)
    for utt in valid_utts:
        spk = utt2spk[utt]
        style = utt2style[utt]
        text_id = utt2text[utt]
        by_speaker_style_text[spk][style][text_id].append(utt)
        by_style_text[(style, text_id)].append(utt)

    target_groups: dict[str, list[PairRecord]] = defaultdict(list)
    split_pair_counts = Counter()
    for speaker_id, style_map in by_speaker_style_text.items():
        nature_texts = style_map.get("nature", {})
        whsp_texts = style_map.get("whsp", {})
        shared_texts = sorted(set(nature_texts) & set(whsp_texts))
        for text_id in shared_texts:
            candidates: list[PairRecord] = []
            for enroll_utt in sorted(nature_texts[text_id]):
                for test_utt in sorted(whsp_texts[text_id]):
                    split_pair = (utt2split[enroll_utt], utt2split[test_utt])
                    candidates.append(
                        PairRecord(
                            enroll_utt=enroll_utt,
                            test_utt=test_utt,
                            speaker_id=speaker_id,
                            text_id=text_id,
                            split_pair=split_pair,
                        )
                    )
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (
                    TARGET_SPLIT_PRIORITY.get(item.split_pair, 99),
                    item.enroll_utt,
                    item.test_utt,
                )
            )
            chosen = candidates[0]
            target_groups[speaker_id].append(chosen)
            split_pair_counts[f"{chosen.split_pair[0]}_to_{chosen.split_pair[1]}"] += 1

    selected_targets = round_robin_limit(target_groups, args.max_target_trials, rng)

    all_rows: list[tuple[str, str, str, str]] = []
    pair_meta_rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    nontargets_requested = 0
    nontargets_added = 0

    def add_row(utt1: str, utt2: str, label: str, source: str) -> bool:
        key = canonical_pair(utt1, utt2)
        if utt1 == utt2 or key in seen_pairs:
            return False
        seen_pairs.add(key)
        all_rows.append((utt1, utt2, label, CONDITION))
        pair_meta_rows.append(
            {
                "utt1": utt1,
                "utt2": utt2,
                "label": label,
                "condition": CONDITION,
                "speaker1": utt2spk[utt1],
                "speaker2": utt2spk[utt2],
                "style1": utt2style[utt1],
                "style2": utt2style[utt2],
                "text1": utt2text[utt1],
                "text2": utt2text[utt2],
                "split1": utt2split[utt1],
                "split2": utt2split[utt2],
                "source": source,
            }
        )
        return True

    for target in selected_targets:
        add_row(target.enroll_utt, target.test_utt, "target", "same_text_target")
        same_text_whsp = [
            utt
            for utt in by_style_text[("whsp", target.text_id)]
            if utt2spk[utt] != target.speaker_id
        ]
        same_split_first = [
            utt for utt in same_text_whsp if utt2split[utt] == target.split_pair[1]
        ]
        fallback = [utt for utt in same_text_whsp if utt not in same_split_first]
        rng.shuffle(same_split_first)
        rng.shuffle(fallback)
        ordered = same_split_first + fallback

        nontargets_requested += args.nontarget_ratio
        added_for_target = 0
        for impostor_utt in ordered:
            if add_row(target.enroll_utt, impostor_utt, "nontarget", "same_text_nontarget"):
                added_for_target += 1
                nontargets_added += 1
            if added_for_target >= args.nontarget_ratio:
                break

    all_rows.sort()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trials(output_path, all_rows)

    if args.output_pair_metadata:
        pair_metadata_path = Path(args.output_pair_metadata).resolve()
        pair_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        write_pair_metadata(pair_metadata_path, pair_meta_rows)

    stats = {
        "num_trials": len(all_rows),
        "labels": dict(sorted(Counter(row[2] for row in all_rows).items())),
        "conditions": dict(sorted(Counter(row[3] for row in all_rows).items())),
        "target_count": len(selected_targets),
        "nontarget_ratio_requested": args.nontarget_ratio,
        "nontargets_requested": nontargets_requested,
        "nontargets_added": nontargets_added,
        "num_speakers": len({utt2spk[utt] for utt in valid_utts}),
        "num_utterances": len(valid_utts),
        "split_pair_counts": dict(sorted(split_pair_counts.items())),
    }
    if args.output_stats:
        stats_path = Path(args.output_stats).resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("=== [Whisper40] Trial generation done ===")
    print(f"  valid utterances : {stats['num_utterances']}")
    print(f"  speakers         : {stats['num_speakers']}")
    print(f"  targets          : {stats['target_count']}")
    print(f"  labels           : {stats['labels']}")
    print(f"  split pairs      : {stats['split_pair_counts']}")
    if nontargets_added < nontargets_requested:
        print(
            f"  note             : only generated {nontargets_added} / {nontargets_requested} "
            "requested nontarget pairs because some same-text impostor pools were limited"
        )


if __name__ == "__main__":
    main()
