#!/usr/bin/env python3
"""Prepare the VoxKnesset aging benchmark subset for OpenSVBench."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scoring_utils import canonical_pair_key


GAP_BUCKETS = ("gap_1_2y", "gap_2_5y", "gap_5plus")


def get_args():
    parser = argparse.ArgumentParser(description="Prepare VoxKnesset aging benchmark assets")
    parser.add_argument("--dataset-id", default="yanirmr/VoxKnesset")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--prep-dir", required=True)
    parser.add_argument("--parquet-shards", type=int, default=20)
    parser.add_argument("--max-clean-utts", type=int, default=10000)
    parser.add_argument("--min-gap-years", type=float, default=1.0)
    parser.add_argument("--max-target-trials-per-bucket", type=int, default=5000)
    parser.add_argument("--nontarget-ratio", type=int, default=10)
    parser.add_argument("--voiced-max-seconds", type=float, default=5.0)
    parser.add_argument("--vad-frame-ms", type=float, default=30.0)
    parser.add_argument("--vad-hop-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def evenly_spaced_take(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return list(items)
    if limit <= 1:
        return [items[0]]
    chosen = []
    used = set()
    last = len(items) - 1
    for idx in range(limit):
        pos = round(idx * last / (limit - 1))
        if pos in used:
            continue
        used.add(pos)
        chosen.append(items[pos])
    if len(chosen) < limit:
        for pos, item in enumerate(items):
            if pos not in used:
                chosen.append(item)
                if len(chosen) >= limit:
                    break
    return chosen[:limit]


def gap_bucket(age_gap: float, min_gap_years: float) -> str | None:
    if age_gap < min_gap_years:
        return None
    if age_gap < 2.0:
        return "gap_1_2y"
    if age_gap < 5.0:
        return "gap_2_5y"
    return "gap_5plus"


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


def parquet_paths(dataset_id: str, split: str, shards: int) -> list[str]:
    dataset_repo = dataset_id.strip("/")
    return [
        f"hf://datasets/{dataset_repo}/data/{split}-{idx:05d}-of-{shards:05d}.parquet"
        for idx in range(shards)
    ]


def resolve_parquet_sources(download_dir: Path, dataset_id: str, split: str, shards: int) -> tuple[list[str], str]:
    local_candidates = sorted((download_dir / "mirror_repo" / "data").glob(f"{split}-*.parquet"))
    if local_candidates:
        return [str(path.resolve()) for path in local_candidates], "local mirror parquet files"
    return parquet_paths(dataset_id, split, shards), "remote parquet shards"


def local_mirror_parquet_files(download_dir: Path, split: str) -> list[Path]:
    return sorted(path.resolve() for path in (download_dir / "mirror_repo" / "data").glob(f"{split}-*.parquet"))


def discover_local_columns(local_files: list[Path]) -> tuple[str, str, str]:
    schema_names = set(pq.ParquetFile(local_files[0]).schema_arrow.names)
    speaker_key = next((name for name in ("person_name", "speaker", "speaker_id", "person_id") if name in schema_names), None)
    age_key = "age" if "age" in schema_names else None
    gender_key = next((name for name in ("gender", "speaker_gender") if name in schema_names), None)
    if speaker_key is None or age_key is None or gender_key is None:
        raise RuntimeError(f"Unexpected VoxKnesset parquet schema: {sorted(schema_names)}")
    return speaker_key, age_key, gender_key


def scan_local_metadata(
    local_files: list[Path],
    speaker_key: str,
    age_key: str,
    gender_key: str,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    global_index = 0
    for file_idx, parquet_path in enumerate(local_files, 1):
        table = pq.read_table(parquet_path, columns=[speaker_key, age_key, gender_key], memory_map=True)
        speakers = table.column(speaker_key).to_pylist()
        ages = table.column(age_key).to_pylist()
        genders = table.column(gender_key).to_pylist()
        valid_rows = 0
        for local_row, (speaker_raw, age_raw, gender_raw) in enumerate(zip(speakers, ages, genders)):
            speaker = str(speaker_raw or "").strip()
            if not speaker:
                global_index += 1
                continue
            try:
                age = float(age_raw)
            except (TypeError, ValueError):
                global_index += 1
                continue
            gender = str(gender_raw).strip() if gender_raw is not None else "unknown"
            grouped[speaker].append(
                {
                    "index": global_index,
                    "speaker": speaker,
                    "age": age,
                    "gender": gender or "unknown",
                    "file_path": parquet_path,
                    "local_row": local_row,
                }
            )
            global_index += 1
            valid_rows += 1
        print(
            f"[VoxKnesset] scanned local parquet {file_idx}/{len(local_files)}: "
            f"{parquet_path.name} rows={table.num_rows:,}, valid={valid_rows:,}, speakers_so_far={len(grouped):,}"
        )
    return grouped


def decode_audio_field(audio_field) -> tuple[object, int]:
    audio_bytes = None
    audio_path = None
    if isinstance(audio_field, dict):
        audio_bytes = audio_field.get("bytes")
        audio_path = audio_field.get("path")
    elif hasattr(audio_field, "as_py"):
        decoded = audio_field.as_py()
        if isinstance(decoded, dict):
            audio_bytes = decoded.get("bytes")
            audio_path = decoded.get("path")
    if isinstance(audio_bytes, memoryview):
        audio_bytes = audio_bytes.tobytes()
    if audio_bytes:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    elif audio_path and Path(audio_path).is_file():
        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    else:
        raise FileNotFoundError(f"Unable to decode VoxKnesset audio field: {audio_field}")
    if getattr(data, "ndim", 1) == 2:
        data = data.mean(axis=1)
    return data, sr


def voiced_crop_config(args) -> dict[str, float | str]:
    return {
        "clip_strategy": "max_voiced_window",
        "clip_target_seconds": float(args.voiced_max_seconds),
        "clip_frame_ms": float(args.vad_frame_ms),
        "clip_hop_ms": float(args.vad_hop_ms),
    }


def frame_rms(audio: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size <= frame_len:
        return np.array([float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))], dtype=np.float32)
    n_frames = 1 + (audio.size - frame_len) // hop
    framed = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, frame_len),
        strides=(audio.strides[0] * hop, audio.strides[0]),
        writeable=False,
    )
    energy = np.einsum("ij,ij->i", framed, framed, dtype=np.float64)
    return np.sqrt(np.maximum(energy / frame_len, 1e-12)).astype(np.float32)


def voiced_frame_mask(rms: np.ndarray) -> np.ndarray:
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    peak = float(np.percentile(rms, 95))
    floor = float(np.percentile(rms, 20))
    if peak <= 1e-6:
        return np.zeros(rms.shape, dtype=bool)
    threshold = max(floor * 2.0, peak * 0.10, 1e-4)
    mask = rms >= threshold
    if not mask.any():
        threshold = max(peak * 0.05, 5e-5)
        mask = rms >= threshold
    return mask


def select_voiced_excerpt(
    audio: np.ndarray,
    sr: int,
    target_seconds: float,
    frame_ms: float,
    hop_ms: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    source_duration_sec = float(audio.shape[0] / sr) if sr > 0 else 0.0
    target_samples = int(round(target_seconds * sr)) if sr > 0 and target_seconds > 0 else audio.shape[0]
    if audio.size == 0 or target_samples <= 0:
        return audio, {
            "clip_source_duration_sec": source_duration_sec,
            "clip_selected_duration_sec": source_duration_sec,
            "clip_start_sec": 0.0,
            "clip_end_sec": source_duration_sec,
        }

    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))

    rms = frame_rms(audio, frame_len, hop)
    mask = voiced_frame_mask(rms)
    trim_start = 0
    trim_end = audio.shape[0]
    if mask.any():
        voiced_idx = np.flatnonzero(mask)
        trim_start = int(voiced_idx[0] * hop)
        trim_end = min(audio.shape[0], int(voiced_idx[-1] * hop + frame_len))
    trimmed = audio[trim_start:trim_end]
    if trimmed.size == 0:
        trimmed = audio
        trim_start = 0
        trim_end = audio.shape[0]

    if trimmed.shape[0] <= target_samples:
        selected = trimmed
        start_sample = trim_start
        end_sample = trim_start + trimmed.shape[0]
    else:
        rms = frame_rms(trimmed, frame_len, hop)
        mask = voiced_frame_mask(rms)
        window_frames = max(1, int(np.ceil(max(target_samples - frame_len, 0) / hop)) + 1)
        if rms.size <= window_frames:
            start_local = 0
        else:
            peak = float(max(rms.max(), 1e-6))
            weights = mask.astype(np.float32) + 0.05 * (rms / peak)
            prefix = np.concatenate(([0.0], np.cumsum(weights, dtype=np.float64)))
            scores = prefix[window_frames:] - prefix[:-window_frames]
            best_frame = int(np.argmax(scores))
            start_local = best_frame * hop
        end_local = min(trimmed.shape[0], start_local + target_samples)
        start_local = max(0, end_local - target_samples)
        selected = trimmed[start_local:end_local]
        start_sample = trim_start + start_local
        end_sample = trim_start + end_local

    return selected, {
        "clip_source_duration_sec": source_duration_sec,
        "clip_selected_duration_sec": float(selected.shape[0] / sr) if sr > 0 else 0.0,
        "clip_start_sec": float(start_sample / sr) if sr > 0 else 0.0,
        "clip_end_sec": float(end_sample / sr) if sr > 0 else 0.0,
    }


def duration_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(item) for item in values)
    def percentile(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return ordered[idx]
    return {
        "min": round(ordered[0], 6),
        "median": round(percentile(0.5), 6),
        "mean": round(sum(ordered) / len(ordered), 6),
        "p95": round(percentile(0.95), 6),
        "max": round(ordered[-1], 6),
    }


def materialize_local_selection(
    local_files: list[Path],
    selected_records: list[dict],
    speaker_key: str,
    age_key: str,
    gender_key: str,
    audio_root: Path,
    manifest_path: Path,
    crop_cfg: dict[str, float | str],
) -> dict[int, dict]:
    selected_lookup = {(item["file_path"], item["local_row"]): item for item in selected_records}
    selected_by_file: dict[Path, set[int]] = defaultdict(set)
    for item in selected_records:
        selected_by_file[item["file_path"]].add(item["local_row"])

    materialized: dict[int, dict] = {}
    with manifest_path.open("w") as f_manifest:
        written = 0
        for file_idx, parquet_path in enumerate(local_files, 1):
            wanted_rows = selected_by_file.get(parquet_path, set())
            if not wanted_rows:
                continue
            parquet_file = pq.ParquetFile(parquet_path)
            row_cursor = 0
            for batch in parquet_file.iter_batches(
                batch_size=64,
                columns=[speaker_key, age_key, gender_key, "audio"],
                use_threads=True,
            ):
                batch_start = row_cursor
                batch_end = row_cursor + batch.num_rows
                hit_offsets = [row - batch_start for row in wanted_rows if batch_start <= row < batch_end]
                if hit_offsets:
                    rows = batch.to_pylist()
                    for offset in sorted(hit_offsets):
                        row = rows[offset]
                        local_row = batch_start + offset
                        selected = selected_lookup[(parquet_path, local_row)]
                        audio, sr = decode_audio_field(row["audio"])
                        audio, clip_meta = select_voiced_excerpt(
                            audio,
                            sr,
                            target_seconds=float(crop_cfg["clip_target_seconds"]),
                            frame_ms=float(crop_cfg["clip_frame_ms"]),
                            hop_ms=float(crop_cfg["clip_hop_ms"]),
                        )
                        speaker = str(row.get(speaker_key) or selected["speaker"])
                        gender = str(row.get(gender_key) if row.get(gender_key) is not None else selected["gender"]).strip() or "unknown"
                        age = float(row.get(age_key))
                        utt_id = f"voxknesset-{speaker}-{selected['index']:06d}"
                        out_dir = audio_root / speaker
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{utt_id}.wav"
                        sf.write(out_path, audio, sr, subtype="PCM_16")
                        item = {
                            "source_index": selected["index"],
                            "utt_id": utt_id,
                            "speaker": speaker,
                            "gender": gender,
                            "age": age,
                            "audio_path": str(out_path.resolve()),
                        }
                        item.update(crop_cfg)
                        item.update(clip_meta)
                        materialized[selected["index"]] = item
                        f_manifest.write(json.dumps(item, ensure_ascii=True) + "\n")
                        written += 1
                        if written % 250 == 0:
                            print(f"[VoxKnesset] materialized {written:,} / {len(selected_records):,} selected utterances")
                row_cursor = batch_end
            print(f"[VoxKnesset] processed audio parquet {file_idx}/{len(local_files)}: {parquet_path.name}")
    return materialized


def load_voxknesset_split(
    dataset_id: str,
    split: str,
    download_dir: Path,
    cache_dir: Path,
    shards: int,
    streaming: bool,
    decode_audio: bool,
):
    direct_files, direct_source = resolve_parquet_sources(download_dir, dataset_id, split, shards)
    last_error = None
    for mode in ("direct_parquet", "dataset_builder"):
        try:
            if mode == "direct_parquet":
                print(f"[VoxKnesset] Loading {split} split from {direct_source} ({len(direct_files)} files)")
                dataset = load_dataset(
                    "parquet",
                    data_files=direct_files,
                    split="train",
                    cache_dir=str(cache_dir),
                    streaming=streaming,
                )
            else:
                print(f"[VoxKnesset] Falling back to the official dataset builder for split={split}")
                dataset = load_dataset(
                    dataset_id,
                    split=split,
                    cache_dir=str(cache_dir),
                    streaming=streaming,
                )
            if "audio" in dataset.features:
                if decode_audio:
                    # Keep the raw Audio feature payload and decode with soundfile
                    # below. Newer datasets versions may require torchcodec for
                    # Audio(decode=True), which is unnecessary for this workflow.
                    dataset = dataset.cast_column("audio", Audio(decode=False))
                else:
                    dataset = dataset.remove_columns(["audio"])
            return dataset
        except Exception as exc:  # pragma: no cover - fallback path depends on remote availability.
            last_error = exc
            print(f"[VoxKnesset] {mode} load failed: {exc}")
    raise RuntimeError(f"Unable to load VoxKnesset split={split}: {last_error}") from last_error


def main():
    args = get_args()
    rng = random.Random(args.seed)

    download_dir = Path(args.download_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    prep_dir = Path(args.prep_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = download_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_files = local_mirror_parquet_files(download_dir, args.dataset_split)

    print(f"[VoxKnesset] Loading metadata from {args.dataset_id}:{args.dataset_split}")
    if local_files:
        speaker_key, age_key, gender_key = discover_local_columns(local_files)
        print(f"[VoxKnesset] Metadata columns: speaker={speaker_key}, age={age_key}, gender={gender_key}")
        grouped = scan_local_metadata(local_files, speaker_key, age_key, gender_key)
    else:
        ds_meta = load_voxknesset_split(
            dataset_id=args.dataset_id,
            split=args.dataset_split,
            download_dir=download_dir,
            cache_dir=cache_dir,
            shards=args.parquet_shards,
            streaming=True,
            decode_audio=False,
        )

        grouped: dict[str, list[dict]] = defaultdict(list)
        for idx, row in enumerate(ds_meta):
            speaker = str(row.get("person_name") or row.get("speaker") or row.get("speaker_id") or row.get("person_id") or "")
            if not speaker:
                continue
            try:
                age = float(row.get("age"))
            except (TypeError, ValueError):
                continue
            gender = str(row.get("gender", "unknown")).strip() or "unknown"
            grouped[speaker].append({"index": idx, "speaker": speaker, "age": age, "gender": gender})
            if idx and idx % 5_000 == 0:
                print(f"[VoxKnesset] scanned {idx:,} metadata rows; eligible speakers so far: {len(grouped):,}")

    eligible = {}
    for speaker, rows in grouped.items():
        rows = sorted(rows, key=lambda item: (item["age"], item["index"]))
        if len(rows) < 4:
            continue
        if rows[-1]["age"] - rows[0]["age"] < args.min_gap_years:
            continue
        eligible[speaker] = rows

    if not eligible:
        raise RuntimeError("No VoxKnesset speakers satisfy the minimum aging requirement.")

    quota = max(4, args.max_clean_utts // len(eligible))
    selected_by_speaker: dict[str, list[dict]] = {}
    leftovers: dict[str, list[dict]] = {}
    for speaker, rows in sorted(eligible.items()):
        chosen = evenly_spaced_take(rows, quota)
        selected_ids = {item["index"] for item in chosen}
        selected_by_speaker[speaker] = chosen
        leftovers[speaker] = [item for item in rows if item["index"] not in selected_ids]

    selected_records = [item for rows in selected_by_speaker.values() for item in rows]
    speaker_cycle = sorted(leftovers)
    while len(selected_records) < args.max_clean_utts:
        made_progress = False
        for speaker in speaker_cycle:
            if not leftovers[speaker]:
                continue
            selected_records.append(leftovers[speaker].pop(0))
            made_progress = True
            if len(selected_records) >= args.max_clean_utts:
                break
        if not made_progress:
            break

    selected_records = sorted(selected_records, key=lambda item: item["index"])
    selected_indices = [item["index"] for item in selected_records]
    selected_index_set = set(selected_indices)

    print(
        f"[VoxKnesset] Selected {len(selected_records)} clean utterances from "
        f"{len(eligible)} eligible speakers (quota={quota}, max={args.max_clean_utts})"
    )

    manifest_path = raw_dir / "selected_manifest.jsonl"
    audio_root = raw_dir / "audio"
    crop_cfg = voiced_crop_config(args)
    need_materialize = True
    existing_manifest: dict[int, dict] = {}
    if manifest_path.is_file():
        for line in manifest_path.open("r"):
            item = json.loads(line)
            existing_manifest[int(item["source_index"])] = item
        if (
            set(existing_manifest) == selected_index_set
            and all(
                Path(item["audio_path"]).is_file()
                and all(item.get(key) == value for key, value in crop_cfg.items())
                for item in existing_manifest.values()
            )
        ):
            need_materialize = False

    materialized = {}
    if need_materialize:
        if local_files:
            materialized = materialize_local_selection(
                local_files=local_files,
                selected_records=selected_records,
                speaker_key=speaker_key,
                age_key=age_key,
                gender_key=gender_key,
                audio_root=audio_root,
                manifest_path=manifest_path,
                crop_cfg=crop_cfg,
            )
        else:
            ds_audio = load_voxknesset_split(
                dataset_id=args.dataset_id,
                split=args.dataset_split,
                download_dir=download_dir,
                cache_dir=cache_dir,
                shards=args.parquet_shards,
                streaming=True,
                decode_audio=True,
            )
            selected_set = set(selected_indices)
            with manifest_path.open("w") as f_manifest:
                written = 0
                for idx, row in enumerate(ds_audio):
                    if idx not in selected_set:
                        continue
                    audio = row["audio"]
                    clipped_audio, clip_meta = select_voiced_excerpt(
                        audio["array"],
                        audio["sampling_rate"],
                        target_seconds=float(crop_cfg["clip_target_seconds"]),
                        frame_ms=float(crop_cfg["clip_frame_ms"]),
                        hop_ms=float(crop_cfg["clip_hop_ms"]),
                    )
                    speaker = str(row.get("person_name") or row.get("speaker") or row.get("speaker_id") or row.get("person_id") or "")
                    gender = str(row.get("gender", "unknown")).strip() or "unknown"
                    age = float(row.get("age"))
                    utt_id = f"voxknesset-{speaker}-{idx:06d}"
                    out_dir = audio_root / speaker
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{utt_id}.wav"
                    sf.write(out_path, clipped_audio, audio["sampling_rate"], subtype="PCM_16")
                    item = {
                        "source_index": idx,
                        "utt_id": utt_id,
                        "speaker": speaker,
                        "gender": gender,
                        "age": age,
                        "audio_path": str(out_path.resolve()),
                    }
                    item.update(crop_cfg)
                    item.update(clip_meta)
                    materialized[idx] = item
                    f_manifest.write(json.dumps(item, ensure_ascii=True) + "\n")
                    written += 1
                    if written % 250 == 0:
                        print(f"[VoxKnesset] materialized {written:,} / {len(selected_set):,} selected utterances")
    else:
        materialized = existing_manifest
        print("[VoxKnesset] Reusing previously materialized selected audio subset.")

    enriched = []
    for item in selected_records:
        meta = dict(materialized[item["index"]])
        enriched.append(meta)

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for item in enriched:
        by_speaker[item["speaker"]].append(item)
    for rows in by_speaker.values():
        rows.sort(key=lambda item: (item["age"], item["source_index"]))

    target_pairs: dict[str, list[dict]] = {bucket: [] for bucket in GAP_BUCKETS}
    for speaker, rows in sorted(by_speaker.items()):
        for left, right in combinations(rows, 2):
            bucket = gap_bucket(abs(left["age"] - right["age"]), args.min_gap_years)
            if bucket is None:
                continue
            target_pairs[bucket].append(
                {
                    "utt1": left["utt_id"],
                    "utt2": right["utt_id"],
                    "label": "target",
                    "condition": bucket,
                    "age1": left["age"],
                    "age2": right["age"],
                    "age_gap": abs(left["age"] - right["age"]),
                    "gender1": left["gender"],
                    "gender2": right["gender"],
                    "speaker1": left["speaker"],
                    "speaker2": right["speaker"],
                    "source": "self_constructed",
                }
            )

    for bucket in GAP_BUCKETS:
        rng.shuffle(target_pairs[bucket])
        if len(target_pairs[bucket]) > args.max_target_trials_per_bucket:
            target_pairs[bucket] = target_pairs[bucket][: args.max_target_trials_per_bucket]

    all_records = []
    seen_pairs = set()
    for bucket in GAP_BUCKETS:
        for row in target_pairs[bucket]:
            pair_key = canonical_pair_key(row["utt1"], row["utt2"])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            all_records.append(row)

    gender_groups: dict[str, list[dict]] = defaultdict(list)
    for item in enriched:
        gender_groups[item["gender"]].append(item)
    all_pool = list(enriched)

    nontarget_counts = Counter()
    nontarget_goal = {bucket: len(target_pairs[bucket]) * args.nontarget_ratio for bucket in GAP_BUCKETS}
    attempts = Counter()

    def sample_negative(bucket: str) -> dict | None:
        prefer_gender_pool = [g for g, rows in gender_groups.items() if len({row["speaker"] for row in rows}) >= 2]
        if prefer_gender_pool:
            pool = gender_groups[rng.choice(prefer_gender_pool)]
        else:
            pool = all_pool
        if len(pool) < 2:
            return None
        for _ in range(50):
            left, right = rng.sample(pool, 2)
            if left["speaker"] == right["speaker"]:
                continue
            age_gap = abs(left["age"] - right["age"])
            if gap_bucket(age_gap, args.min_gap_years) != bucket:
                continue
            pair_key = canonical_pair_key(left["utt_id"], right["utt_id"])
            if pair_key in seen_pairs:
                continue
            return {
                "utt1": left["utt_id"],
                "utt2": right["utt_id"],
                "label": "nontarget",
                "condition": bucket,
                "age1": left["age"],
                "age2": right["age"],
                "age_gap": age_gap,
                "gender1": left["gender"],
                "gender2": right["gender"],
                "speaker1": left["speaker"],
                "speaker2": right["speaker"],
                "source": "matched_gender_age_gap",
            }
        return None

    while any(nontarget_counts[bucket] < nontarget_goal[bucket] for bucket in GAP_BUCKETS):
        made_progress = False
        for bucket in GAP_BUCKETS:
            if nontarget_counts[bucket] >= nontarget_goal[bucket]:
                continue
            attempts[bucket] += 1
            if attempts[bucket] > nontarget_goal[bucket] * 80 + 2000:
                continue
            row = sample_negative(bucket)
            if row is None:
                continue
            pair_key = canonical_pair_key(row["utt1"], row["utt2"])
            seen_pairs.add(pair_key)
            all_records.append(row)
            nontarget_counts[bucket] += 1
            made_progress = True
        if not made_progress:
            break

    utt_rows = sorted(enriched, key=lambda item: item["utt_id"])
    trial_rows = sorted(all_records, key=lambda item: (item["utt1"], item["utt2"], item["label"], item["condition"]))

    wav_scp = prep_dir / "wav.scp"
    utt2spk = prep_dir / "utt2spk"
    utt2age = prep_dir / "utt2age"
    utt2gender = prep_dir / "utt2gender"
    spk2utt = prep_dir / "spk2utt"
    trials_path = prep_dir / "trials_cross_time.txt"
    pair_meta = prep_dir / "pair_metadata_voxknesset.csv"
    stats_path = prep_dir / "stats_voxknesset.json"

    with wav_scp.open("w") as f_wav, utt2spk.open("w") as f_spk, utt2age.open("w") as f_age, utt2gender.open("w") as f_gender:
        for row in utt_rows:
            f_wav.write(f"{row['utt_id']} {row['audio_path']}\n")
            f_spk.write(f"{row['utt_id']} {row['speaker']}\n")
            f_age.write(f"{row['utt_id']} {row['age']:.6f}\n")
            f_gender.write(f"{row['utt_id']} {row['gender']}\n")
    build_spk2utt(utt2spk, spk2utt)

    with trials_path.open("w") as f_trial:
        for row in trial_rows:
            f_trial.write(f"{row['utt1']} {row['utt2']} {row['label']} {row['condition']}\n")

    with pair_meta.open("w") as f_meta:
        f_meta.write("utt1,utt2,label,condition,source,speaker1,speaker2,gender1,gender2,age1,age2,age_gap\n")
        for row in trial_rows:
            f_meta.write(
                f"{row['utt1']},{row['utt2']},{row['label']},{row['condition']},{row['source']},"
                f"{row['speaker1']},{row['speaker2']},{row['gender1']},{row['gender2']},"
                f"{row['age1']:.6f},{row['age2']:.6f},{row['age_gap']:.6f}\n"
            )

    pair_distribution = Counter((row["label"], row["condition"]) for row in trial_rows)
    gender_distribution = Counter(row["gender"] for row in utt_rows)
    speaker_distribution = Counter(row["speaker"] for row in utt_rows)
    stats = {
        "dataset": "voxknesset",
        "seed": args.seed,
        "dataset_id": args.dataset_id,
        "dataset_split": args.dataset_split,
        "clean_source_utterances": len(utt_rows),
        "speakers": len(speaker_distribution),
        "age_range": {
            "min": min(row["age"] for row in utt_rows),
            "max": max(row["age"] for row in utt_rows),
        },
        "gender_distribution": dict(sorted(gender_distribution.items())),
        "pair_distribution": {f"{label}|{bucket}": count for (label, bucket), count in sorted(pair_distribution.items())},
        "source_limit": args.max_clean_utts,
        "audio_clip": crop_cfg,
        "source_duration_seconds": duration_summary([row["clip_source_duration_sec"] for row in utt_rows]),
        "selected_duration_seconds": duration_summary([row["clip_selected_duration_sec"] for row in utt_rows]),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("[VoxKnesset] Statistics")
    print(f"  clean_source_utterances: {len(utt_rows)}")
    print(f"  speakers               : {len(speaker_distribution)}")
    print(f"  genders                : {dict(sorted(gender_distribution.items()))}")
    for (label, bucket), count in sorted(pair_distribution.items()):
        print(f"  {label:10s} {bucket:10s}: {count}")
    print(f"[VoxKnesset] Wrote: {wav_scp}, {utt2spk}, {utt2age}, {utt2gender}, {spk2utt}, {trials_path}, {stats_path}")


if __name__ == "__main__":
    main()
