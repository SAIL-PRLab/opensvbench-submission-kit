#!/usr/bin/env python3
"""Prepare the TidyVoiceX2-ASV set for the OpenSVBench core benchmark."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scoring_utils import canonical_pair_key


CONDITIONS = ("same_language", "cross_language")


@dataclass(frozen=True)
class PairRecord:
    label: str
    rel1: str
    rel2: str
    spk1: str
    spk2: str
    lang1: str
    lang2: str
    condition: str
    source: str


def get_args():
    parser = argparse.ArgumentParser(description="Prepare TidyVoice core benchmark manifests and trials")
    parser.add_argument("--audio-root", required=True, help="Root directory of TidyVoiceX2_ASV")
    parser.add_argument(
        "--pair-file",
        action="append",
        default=[],
        help="Official TidyVoice pair/key file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pair-files",
        nargs="+",
        default=[],
        help="Official TidyVoice pair/key files to pool before sampling.",
    )
    parser.add_argument("--prep-dir", required=True, help="Output prepared directory")
    parser.add_argument("--max-clean-utts", type=int, default=10000)
    parser.add_argument("--max-target-trials-per-condition", type=int, default=5000)
    parser.add_argument(
        "--min-total-trials-per-condition",
        type=int,
        default=100000,
        help="Minimum total trials (target + nontarget) to keep per condition.",
    )
    parser.add_argument("--candidate-pairs-per-bucket", type=int, default=60000)
    parser.add_argument("--nontarget-ratio", type=int, default=10)
    parser.add_argument(
        "--target-selection",
        choices=("random", "speaker_freq", "speaker_density"),
        default="random",
        help="Target-pair ordering strategy before applying the utterance cap.",
    )
    parser.add_argument(
        "--max-utts-per-speaker",
        type=int,
        default=0,
        help="Optional per-speaker utterance cap. Use 0 for the automatic cap.",
    )
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def parse_rel_path(rel_path: str) -> tuple[str, str]:
    parts = PurePosixPath(rel_path).parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected TidyVoice path: {rel_path}")
    return parts[0], parts[1]


def make_utt_id(spk: str, lang: str, rel_path: str) -> str:
    stem = Path(rel_path).stem.replace("/", "-")
    return f"tidyvoice-{spk}-{lang}-{stem}"


def pair_source_name(pair_file: Path) -> str:
    name = pair_file.name.lower()
    if "eval-a" in name or "trials_a" in name or name.endswith("_a.kaldi"):
        return "official_A"
    if "eval-u" in name or "trials_u" in name or name.endswith("_u.kaldi"):
        return "official_U"
    return "official"


def parse_pair_line(line: str, source: str) -> PairRecord | None:
    parts = line.strip().split()
    if len(parts) != 3:
        return None
    if parts[0] in {"0", "1"}:
        label_raw, rel1, rel2 = parts
        label = "target" if label_raw == "1" else "nontarget"
    elif parts[2].lower() in {"target", "nontarget"}:
        rel1, rel2, label_raw = parts
        label = label_raw.lower()
    else:
        return None
    spk1, lang1 = parse_rel_path(rel1)
    spk2, lang2 = parse_rel_path(rel2)
    if label == "target" and spk1 != spk2:
        raise ValueError(f"Target trial has different speakers: {rel1} {rel2}")
    if label == "nontarget" and spk1 == spk2:
        raise ValueError(f"Nontarget trial has same speaker: {rel1} {rel2}")
    condition = "same_language" if lang1 == lang2 else "cross_language"
    return PairRecord(label, rel1, rel2, spk1, spk2, lang1, lang2, condition, source)


def reservoir_sample_pairs(
    pair_files: list[Path],
    per_bucket: int,
    seed: int,
) -> tuple[dict[tuple[str, str], list[PairRecord]], Counter, Counter]:
    rng = random.Random(seed)
    samples: dict[tuple[str, str], list[PairRecord]] = defaultdict(list)
    seen = Counter()
    source_seen = Counter()
    total_scanned = 0
    for pair_file in pair_files:
        source = pair_source_name(pair_file)
        print(f"[TidyVoice] Scanning pair/key file ({source}): {pair_file}")
        with pair_file.open("r") as f:
            for line_idx, line in enumerate(f, 1):
                rec = parse_pair_line(line, source)
                if rec is None:
                    continue
                total_scanned += 1
                source_seen[source] += 1
                bucket = (rec.label, rec.condition)
                seen[bucket] += 1
                bucket_samples = samples[bucket]
                if len(bucket_samples) < per_bucket:
                    bucket_samples.append(rec)
                else:
                    replace_idx = rng.randint(0, seen[bucket] - 1)
                    if replace_idx < per_bucket:
                        bucket_samples[replace_idx] = rec
                if total_scanned % 1_000_000 == 0:
                    print(
                        f"[TidyVoice] scanned {total_scanned:,} pairs "
                        f"(target/same={seen[('target', 'same_language')]:,}, "
                        f"target/cross={seen[('target', 'cross_language')]:,}, "
                        f"nontarget/same={seen[('nontarget', 'same_language')]:,}, "
                        f"nontarget/cross={seen[('nontarget', 'cross_language')]:,})"
                    )
    return samples, seen, source_seen


def speaker_cap(candidate_buckets: dict[tuple[str, str], list[PairRecord]], max_clean_utts: int) -> int:
    speakers = set()
    for bucket, pairs in candidate_buckets.items():
        if bucket[0] != "target":
            continue
        for rec in pairs:
            speakers.add(rec.spk1)
            speakers.add(rec.spk2)
    if not speakers:
        return max_clean_utts
    quota = math.ceil(max_clean_utts / len(speakers) * 1.6)
    return max(12, min(40, quota))


def order_target_pairs(pairs: list[PairRecord], strategy: str, rng: random.Random) -> list[PairRecord]:
    ordered = list(pairs)
    if strategy == "random":
        rng.shuffle(ordered)
        return ordered

    by_speaker: dict[str, list[PairRecord]] = defaultdict(list)
    for rec in ordered:
        by_speaker[rec.spk1].append(rec)

    if strategy == "speaker_freq":
        counts = {spk: len(items) for spk, items in by_speaker.items()}
        ordered.sort(key=lambda rec: (-counts[rec.spk1], rec.spk1, rec.rel1, rec.rel2))
        return ordered

    if strategy == "speaker_density":
        density = {}
        counts = {}
        for spk, items in by_speaker.items():
            utts = {rel for rec in items for rel in (rec.rel1, rec.rel2)}
            counts[spk] = len(items)
            density[spk] = len(items) / max(1, len(utts))
        ordered.sort(key=lambda rec: (-density[rec.spk1], -counts[rec.spk1], rec.spk1, rec.rel1, rec.rel2))
        return ordered

    raise ValueError(f"Unsupported target selection strategy: {strategy}")


def build_spk2utt(utt2spk_path: Path, out_path: Path) -> None:
    spk2utts: dict[str, list[str]] = defaultdict(list)
    with utt2spk_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                spk2utts[parts[1]].append(parts[0])
    with out_path.open("w") as f:
        for spk in sorted(spk2utts):
            f.write(f"{spk} {' '.join(sorted(spk2utts[spk]))}\n")


def main():
    args = get_args()
    rng = random.Random(args.seed)
    audio_root = Path(args.audio_root).resolve()
    pair_files = [Path(item).resolve() for item in [*args.pair_file, *args.pair_files]]
    prep_dir = Path(args.prep_dir).resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)

    if not audio_root.is_dir():
        raise FileNotFoundError(f"TidyVoice audio root not found: {audio_root}")
    if not pair_files:
        raise ValueError("At least one --pair-file or --pair-files entry is required")
    for pair_file in pair_files:
        if not pair_file.is_file():
            raise FileNotFoundError(f"TidyVoice pair/key file not found: {pair_file}")

    print(f"[TidyVoice] Audio root: {audio_root}")
    for pair_file in pair_files:
        print(f"[TidyVoice] Pair/key file: {pair_file}")
    print("[TidyVoice] Reservoir sampling official pair/key files ...")
    candidate_pairs, bucket_seen, source_seen = reservoir_sample_pairs(
        pair_files=pair_files,
        per_bucket=args.candidate_pairs_per_bucket,
        seed=args.seed,
    )
    for bucket in sorted(candidate_pairs):
        print(f"  bucket={bucket} sampled={len(candidate_pairs[bucket])} seen={bucket_seen[bucket]}")
    print(f"[TidyVoice] Official pair/key source counts: {dict(sorted(source_seen.items()))}")

    per_speaker_cap = args.max_utts_per_speaker or speaker_cap(candidate_pairs, args.max_clean_utts)
    print(f"[TidyVoice] Per-speaker clean-utterance cap: {per_speaker_cap}")
    print(f"[TidyVoice] Target selection strategy: {args.target_selection}")

    for bucket, pairs in list(candidate_pairs.items()):
        if bucket[0] == "target":
            candidate_pairs[bucket] = order_target_pairs(pairs, args.target_selection, rng)
        else:
            rng.shuffle(pairs)

    selected_utts: dict[str, dict[str, str]] = {}
    speaker_utt_counts = Counter()
    selected_trials: list[dict[str, str]] = []
    selected_pair_keys: set[tuple[str, str]] = set()

    def try_register_utt(rel_path: str, spk: str, lang: str) -> dict[str, str] | None:
        if rel_path in selected_utts:
            return selected_utts[rel_path]
        abs_path = audio_root / rel_path
        utt_info = {
            "utt_id": make_utt_id(spk, lang, rel_path),
            "spk": spk,
            "lang": lang,
            "abs_path": str(abs_path),
            "rel_path": rel_path,
        }
        return utt_info

    def add_pair(rec: PairRecord) -> bool:
        utt1 = try_register_utt(rec.rel1, rec.spk1, rec.lang1)
        utt2 = try_register_utt(rec.rel2, rec.spk2, rec.lang2)
        if utt1 is None or utt2 is None:
            return False
        pair_key = canonical_pair_key(utt1["utt_id"], utt2["utt_id"])
        if pair_key in selected_pair_keys:
            return False

        new_items = []
        new_counts = Counter()
        for rel_path, info in ((rec.rel1, utt1), (rec.rel2, utt2)):
            if rel_path not in selected_utts:
                new_items.append((rel_path, info))
                new_counts[info["spk"]] += 1

        if len(selected_utts) + len(new_items) > args.max_clean_utts:
            return False
        for spk, extra in new_counts.items():
            if speaker_utt_counts[spk] + extra > per_speaker_cap:
                return False

        for rel_path, info in new_items:
            selected_utts[rel_path] = info
            speaker_utt_counts[info["spk"]] += 1

        selected_pair_keys.add(pair_key)
        selected_trials.append(
            {
                "utt1": utt1["utt_id"],
                "utt2": utt2["utt_id"],
                "label": rec.label,
                "condition": rec.condition,
                "source": rec.source,
                "speaker1": rec.spk1,
                "speaker2": rec.spk2,
                "lang1": rec.lang1,
                "lang2": rec.lang2,
            }
        )
        return True

    target_counts = Counter()
    target_offsets = {cond: 0 for cond in CONDITIONS}
    progress = True
    while progress:
        progress = False
        for cond in CONDITIONS:
            if target_counts[cond] >= args.max_target_trials_per_condition:
                continue
            bucket = candidate_pairs.get(("target", cond), [])
            while target_offsets[cond] < len(bucket):
                rec = bucket[target_offsets[cond]]
                target_offsets[cond] += 1
                if add_pair(rec):
                    target_counts[cond] += 1
                    if target_counts[cond] % 1000 == 0:
                        print(
                            f"[TidyVoice] target progress {cond}: {target_counts[cond]} "
                            f"pairs, selected_utts={len(selected_utts)}"
                        )
                    progress = True
                    break

    print("[TidyVoice] Selected target pairs:")
    for cond in CONDITIONS:
        print(f"  {cond}: {target_counts[cond]}")

    nontarget_goal = {
        cond: max(
            target_counts[cond] * args.nontarget_ratio,
            max(0, args.min_total_trials_per_condition - target_counts[cond]),
        )
        for cond in CONDITIONS
    }
    nontarget_counts = Counter()

    for cond in CONDITIONS:
        for rec in candidate_pairs.get(("nontarget", cond), []):
            if nontarget_counts[cond] >= nontarget_goal[cond]:
                break
            if rec.rel1 not in selected_utts or rec.rel2 not in selected_utts:
                continue
            if add_pair(rec):
                nontarget_counts[cond] += 1
                if nontarget_counts[cond] % 5000 == 0:
                    print(
                        f"[TidyVoice] official nontarget progress {cond}: {nontarget_counts[cond]} / "
                        f"{nontarget_goal[cond]}"
                    )

    selected_by_lang: dict[str, list[dict[str, str]]] = defaultdict(list)
    selected_by_lang_spk: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for info in selected_utts.values():
        selected_by_lang[info["lang"]].append(info)
        selected_by_lang_spk[info["lang"]][info["spk"]].append(info)
    langs = sorted(selected_by_lang)

    synthetic_attempts = Counter()
    valid_same_langs = [lang for lang, spks in selected_by_lang_spk.items() if len(spks) >= 2]
    if valid_same_langs:
        print(f"[TidyVoice] same-language synthetic pool: {len(valid_same_langs)} languages")
    if len(langs) >= 2:
        print(f"[TidyVoice] cross-language synthetic pool: {len(langs)} languages")

    for cond in CONDITIONS:
        max_attempts = max(nontarget_goal[cond] * 80, 200000)
        while nontarget_counts[cond] < nontarget_goal[cond] and synthetic_attempts[cond] < max_attempts:
            synthetic_attempts[cond] += 1
            if cond == "same_language":
                if not valid_same_langs:
                    break
                lang = rng.choice(valid_same_langs)
                speakers = list(selected_by_lang_spk[lang])
                spk1, spk2 = rng.sample(speakers, 2)
                info1 = rng.choice(selected_by_lang_spk[lang][spk1])
                info2 = rng.choice(selected_by_lang_spk[lang][spk2])
            else:
                if len(langs) < 2:
                    break
                lang1, lang2 = rng.sample(langs, 2)
                spk1 = rng.choice(list(selected_by_lang_spk[lang1]))
                spk2 = rng.choice(list(selected_by_lang_spk[lang2]))
                info1 = rng.choice(selected_by_lang_spk[lang1][spk1])
                info2 = rng.choice(selected_by_lang_spk[lang2][spk2])
            if info1["spk"] == info2["spk"]:
                continue
            pair_key = canonical_pair_key(info1["utt_id"], info2["utt_id"])
            if pair_key in selected_pair_keys:
                continue
            selected_pair_keys.add(pair_key)
            selected_trials.append(
                {
                    "utt1": info1["utt_id"],
                    "utt2": info2["utt_id"],
                    "label": "nontarget",
                    "condition": cond,
                    "source": "synthetic_matched",
                    "speaker1": info1["spk"],
                    "speaker2": info2["spk"],
                    "lang1": info1["lang"],
                    "lang2": info2["lang"],
                }
            )
            nontarget_counts[cond] += 1
            if nontarget_counts[cond] % 5000 == 0 or nontarget_counts[cond] == nontarget_goal[cond]:
                print(
                    f"[TidyVoice] synthetic nontarget progress {cond}: {nontarget_counts[cond]} / "
                    f"{nontarget_goal[cond]} (attempts={synthetic_attempts[cond]})"
                )

    print("[TidyVoice] Final nontarget pairs:")
    for cond in CONDITIONS:
        print(f"  {cond}: {nontarget_counts[cond]} / goal {nontarget_goal[cond]}")

    utt_rows = sorted(selected_utts.values(), key=lambda item: item["utt_id"])
    missing_rows = [row for row in utt_rows if not Path(row["abs_path"]).is_file()]
    if missing_rows:
        preview = ", ".join(row["rel_path"] for row in missing_rows[:5])
        raise FileNotFoundError(
            f"[TidyVoice] Selected {len(missing_rows)} missing audio paths. "
            f"Examples: {preview}"
        )
    trial_rows = sorted(
        selected_trials,
        key=lambda item: (item["utt1"], item["utt2"], item["label"], item["condition"]),
    )

    wav_scp = prep_dir / "wav.scp"
    wav_all_scp = prep_dir / "wav_all.scp"
    utt2spk = prep_dir / "utt2spk"
    utt2lang = prep_dir / "utt2lang"
    spk2utt = prep_dir / "spk2utt"
    trials_path = prep_dir / "trials_cross_lingual.txt"
    pair_meta = prep_dir / "pair_metadata_tidyvoice.csv"
    stats_path = prep_dir / "stats_tidyvoice.json"

    with (
        wav_scp.open("w") as f_wav,
        wav_all_scp.open("w") as f_wav_all,
        utt2spk.open("w") as f_spk,
        utt2lang.open("w") as f_lang,
    ):
        for info in utt_rows:
            wav_line = f"{info['utt_id']} {info['abs_path']}\n"
            f_wav.write(wav_line)
            f_wav_all.write(wav_line)
            f_spk.write(f"{info['utt_id']} {info['spk']}\n")
            f_lang.write(f"{info['utt_id']} {info['lang']}\n")
    build_spk2utt(utt2spk, spk2utt)

    with trials_path.open("w") as f_trial:
        for row in trial_rows:
            f_trial.write(f"{row['utt1']} {row['utt2']} {row['label']} {row['condition']}\n")

    with pair_meta.open("w") as f_meta:
        f_meta.write("utt1,utt2,label,condition,source,speaker1,speaker2,lang1,lang2\n")
        for row in trial_rows:
            f_meta.write(
                f"{row['utt1']},{row['utt2']},{row['label']},{row['condition']},"
                f"{row['source']},{row['speaker1']},{row['speaker2']},{row['lang1']},{row['lang2']}\n"
            )

    lang_distribution = Counter(info["lang"] for info in utt_rows)
    speaker_distribution = Counter(info["spk"] for info in utt_rows)
    trial_distribution = Counter((row["label"], row["condition"]) for row in trial_rows)
    source_distribution = Counter(row["source"] for row in trial_rows)
    stats = {
        "dataset": "tidyvoice",
        "source_name": "TidyVoiceX2-ASV",
        "official_pair_files": [str(path) for path in pair_files],
        "official_pair_source_counts": dict(sorted(source_seen.items())),
        "seed": args.seed,
        "max_clean_utts": args.max_clean_utts,
        "clean_source_utterances": len(utt_rows),
        "speakers": len(speaker_distribution),
        "languages": len(lang_distribution),
        "language_distribution": dict(sorted(lang_distribution.items())),
        "speaker_distribution_top10": speaker_distribution.most_common(10),
        "pair_distribution": {f"{label}|{cond}": count for (label, cond), count in sorted(trial_distribution.items())},
        "pair_sources": dict(sorted(source_distribution.items())),
        "same_language_pairs": sum(
            count for (label, cond), count in trial_distribution.items() if cond == "same_language"
        ),
        "cross_language_pairs": sum(
            count for (label, cond), count in trial_distribution.items() if cond == "cross_language"
        ),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("[TidyVoice] Statistics")
    print(f"  clean_source_utterances: {len(utt_rows)}")
    print(f"  speakers               : {len(speaker_distribution)}")
    print(f"  languages              : {len(lang_distribution)}")
    for (label, cond), count in sorted(trial_distribution.items()):
        print(f"  {label:10s} {cond:16s}: {count}")
    print(f"  pair_sources           : {dict(sorted(source_distribution.items()))}")
    print(
        f"[TidyVoice] Wrote: {wav_scp}, {wav_all_scp}, {utt2spk}, {utt2lang}, "
        f"{spk2utt}, {trials_path}, {stats_path}"
    )


if __name__ == "__main__":
    main()
