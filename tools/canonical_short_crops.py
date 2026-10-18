#!/usr/bin/env python3
"""Materialize and verify frozen short-utterance crops."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "resources" / "canonical" / "short_crops"
MANIFEST_HEADER = [
    "short_utt_id",
    "source_utt_id",
    "duration_label",
    "sample_rate",
    "start_sample",
    "num_samples",
    "pcm_sha256",
]
SHORT_SCENARIOS = {
    "cnceleb_short": "cnceleb",
    "voxceleb_short": "voxceleb",
}


class ShortCropProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class CropRecord:
    short_utt_id: str
    source_utt_id: str
    duration_label: str
    sample_rate: int
    start_sample: int
    num_samples: int
    pcm_sha256: str


def manifest_path(dataset_id: str) -> Path:
    path = MANIFEST_DIR / f"{dataset_id}.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing canonical short-crop manifest: {path}")
    return path


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_sha256(samples: np.ndarray) -> str:
    canonical = np.asarray(samples, dtype=np.dtype("<i2"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def load_manifest(path: Path) -> list[CropRecord]:
    records = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header != MANIFEST_HEADER:
            raise ValueError(f"Unexpected short-crop manifest header: {header}")
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(MANIFEST_HEADER):
                raise ValueError(f"Malformed short-crop row at {path}:{line_number}")
            short_utt_id, source_utt_id, duration_label = fields[:3]
            if short_utt_id in seen:
                raise ValueError(f"Duplicate short utterance in {path}: {short_utt_id}")
            seen.add(short_utt_id)
            record = CropRecord(
                short_utt_id=short_utt_id,
                source_utt_id=source_utt_id,
                duration_label=duration_label,
                sample_rate=int(fields[3]),
                start_sample=int(fields[4]),
                num_samples=int(fields[5]),
                pcm_sha256=fields[6],
            )
            if record.sample_rate <= 0 or record.start_sample < 0 or record.num_samples <= 0:
                raise ValueError(f"Invalid short-crop dimensions at {path}:{line_number}")
            if len(record.pcm_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in record.pcm_sha256
            ):
                raise ValueError(f"Invalid PCM SHA-256 at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Empty short-crop manifest: {path}")
    return records


def read_wav_scp(path: Path) -> tuple[list[str], dict[str, str]]:
    order = []
    mapping = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed wav.scp row at {path}:{line_number}")
            utt_id, value = parts
            if utt_id in mapping:
                raise ValueError(f"Duplicate utterance in {path}: {utt_id}")
            order.append(utt_id)
            mapping[utt_id] = value
    return order, mapping


def resolve_audio_path(value: str, base_dir: Path) -> Path:
    if value.endswith("|"):
        raise ShortCropProtocolError("Canonical short crops must reference audio files, not pipelines")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def crop_output_path(prepared_dir: Path, record: CropRecord) -> Path:
    safe_name = record.source_utt_id.replace("/", "_") + ".wav"
    return prepared_dir / "short_utts" / f"dur_{record.duration_label}" / safe_name


def inspect_crop(record: CropRecord, path: Path) -> str | None:
    if not path.is_file():
        return "missing file"
    try:
        samples, sample_rate = sf.read(path, dtype="int16", always_2d=True)
    except Exception as exc:
        return f"decode failed: {exc}"
    if sample_rate != record.sample_rate:
        return f"sample rate {sample_rate} != {record.sample_rate}"
    if samples.shape[1] != 1:
        return f"channels {samples.shape[1]} != 1"
    mono = samples[:, 0]
    if len(mono) != record.num_samples:
        return f"samples {len(mono)} != {record.num_samples}"
    digest = pcm_sha256(mono)
    if digest != record.pcm_sha256:
        return f"PCM SHA-256 {digest[:12]} != {record.pcm_sha256[:12]}"
    return None


def _verify_item(item: tuple[CropRecord, str]) -> tuple[str, str | None]:
    record, path_text = item
    return record.short_utt_id, inspect_crop(record, Path(path_text))


def verify_prepared(
    prepared_dir: Path,
    manifest: Path,
    workers: int = 8,
) -> dict:
    records = load_manifest(manifest)
    wav_scp = prepared_dir / "wav_short.scp"
    if not wav_scp.is_file():
        raise ShortCropProtocolError(f"Missing canonical short wav.scp: {wav_scp}")
    actual_order, mapping = read_wav_scp(wav_scp)
    expected_order = [record.short_utt_id for record in records]
    if actual_order != expected_order:
        expected_set = set(expected_order)
        actual_set = set(actual_order)
        raise ShortCropProtocolError(
            "Canonical short wav.scp differs from the frozen manifest "
            f"(missing={len(expected_set - actual_set)}, extra={len(actual_set - expected_set)}, "
            f"order_match={actual_order == expected_order})"
        )

    items = [
        (record, str(resolve_audio_path(mapping[record.short_utt_id], prepared_dir)))
        for record in records
    ]
    failures = []
    with Pool(max(1, workers)) as pool:
        for short_utt_id, issue in pool.imap(_verify_item, items, chunksize=32):
            if issue:
                failures.append(f"{short_utt_id}: {issue}")
    if failures:
        preview = "\n".join(f"  - {item}" for item in failures[:20])
        suffix = "" if len(failures) <= 20 else f"\n  ... {len(failures) - 20} more"
        raise ShortCropProtocolError(
            f"Canonical short-crop verification failed ({len(failures)} files):\n{preview}{suffix}"
        )
    return {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256(manifest),
        "utterance_count": len(records),
    }


def _write_crop(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.wav")
    try:
        sf.write(tmp_path, samples, sample_rate, format="WAV", subtype="PCM_16")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _materialize_source(
    item: tuple[str, list[CropRecord], str, str, bool],
) -> tuple[dict[str, str], list[str]]:
    source_utt_id, records, source_path_text, prepared_dir_text, force = item
    prepared_dir = Path(prepared_dir_text)
    output_paths = {record.short_utt_id: crop_output_path(prepared_dir, record) for record in records}
    failures = []
    pending = []
    for record in records:
        output_path = output_paths[record.short_utt_id]
        issue = inspect_crop(record, output_path) if output_path.exists() else "missing file"
        if issue is None and not force:
            continue
        if output_path.exists() and issue is not None and not force:
            failures.append(f"{record.short_utt_id}: {issue}; use --force to rebuild")
        else:
            pending.append(record)
    if failures or not pending:
        return {key: str(value.resolve()) for key, value in output_paths.items()}, failures

    try:
        source, sample_rate = sf.read(source_path_text, dtype="int16", always_2d=True)
    except Exception as exc:
        return {}, [f"{source_utt_id}: source decode failed: {exc}"]
    if source.shape[1] != 1:
        return {}, [f"{source_utt_id}: source channels {source.shape[1]} != 1"]
    mono = source[:, 0]
    for record in pending:
        if sample_rate != record.sample_rate:
            failures.append(
                f"{record.short_utt_id}: source sample rate {sample_rate} != {record.sample_rate}"
            )
            continue
        end_sample = record.start_sample + record.num_samples
        if end_sample > len(mono):
            failures.append(f"{record.short_utt_id}: source is shorter than the frozen crop")
            continue
        crop = mono[record.start_sample:end_sample]
        digest = pcm_sha256(crop)
        if digest != record.pcm_sha256:
            failures.append(
                f"{record.short_utt_id}: source crop PCM SHA-256 "
                f"{digest[:12]} != {record.pcm_sha256[:12]}"
            )
            continue
        output_path = output_paths[record.short_utt_id]
        _write_crop(output_path, crop, sample_rate)
        issue = inspect_crop(record, output_path)
        if issue:
            failures.append(f"{record.short_utt_id}: written crop failed verification: {issue}")
    return {key: str(value.resolve()) for key, value in output_paths.items()}, failures


def write_scp(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for utt_id, audio_path in rows:
            handle.write(f"{utt_id} {audio_path}\n")
    os.replace(tmp_path, path)


def materialize(
    prepared_dir: Path,
    source_wav_scp: Path,
    manifest: Path,
    workers: int = 8,
    force: bool = False,
) -> dict:
    records = load_manifest(manifest)
    _, source_mapping = read_wav_scp(source_wav_scp)
    grouped = defaultdict(list)
    for record in records:
        grouped[record.source_utt_id].append(record)

    missing_sources = sorted(set(grouped) - set(source_mapping))
    if missing_sources:
        preview = ", ".join(missing_sources[:10])
        raise ShortCropProtocolError(
            f"Missing {len(missing_sources)} canonical source utterances in {source_wav_scp}: {preview}"
        )

    work = [
        (
            source_utt_id,
            source_records,
            str(resolve_audio_path(source_mapping[source_utt_id], source_wav_scp.parent)),
            str(prepared_dir.resolve()),
            force,
        )
        for source_utt_id, source_records in grouped.items()
    ]
    output_mapping = {}
    failures = []
    with Pool(max(1, workers)) as pool:
        for paths, source_failures in pool.imap(_materialize_source, work, chunksize=4):
            output_mapping.update(paths)
            failures.extend(source_failures)
    if failures:
        preview = "\n".join(f"  - {item}" for item in failures[:20])
        suffix = "" if len(failures) <= 20 else f"\n  ... {len(failures) - 20} more"
        raise ShortCropProtocolError(
            f"Canonical short-crop materialization failed ({len(failures)} files):\n{preview}{suffix}"
        )

    duration_rows = defaultdict(list)
    merged_rows = []
    for record in records:
        output_path = output_mapping[record.short_utt_id]
        row = (record.short_utt_id, output_path)
        duration_rows[record.duration_label].append(row)
        merged_rows.append(row)
    for duration_label, rows in duration_rows.items():
        write_scp(prepared_dir / f"wav_short_dur_{duration_label}.scp", rows)
    write_scp(prepared_dir / "wav_short.scp", merged_rows)
    return verify_prepared(prepared_dir, manifest, workers=workers)


def protocol_for_dataset(dataset_id: str) -> tuple[Path, str, set[str]]:
    path = manifest_path(dataset_id)
    records = load_manifest(path)
    return path, manifest_sha256(path), {record.short_utt_id for record in records}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("materialize", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--data", required=True, help="OpenSVBench data root")
        subparser.add_argument("--dataset", required=True, choices=["cnceleb", "voxceleb"])
        subparser.add_argument("--manifest", default=None)
        subparser.add_argument("--nj", type=int, default=min(8, os.cpu_count() or 1))
        if command == "materialize":
            subparser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    data_root = Path(args.data).expanduser().resolve()
    prepared_dir = data_root / args.dataset / "prepared"
    manifest = Path(args.manifest).resolve() if args.manifest else manifest_path(args.dataset)
    try:
        if args.command == "materialize":
            report = materialize(
                prepared_dir=prepared_dir,
                source_wav_scp=prepared_dir / "wav.scp",
                manifest=manifest,
                workers=args.nj,
                force=args.force,
            )
        else:
            report = verify_prepared(prepared_dir, manifest, workers=args.nj)
    except (FileNotFoundError, ValueError, ShortCropProtocolError) as exc:
        print(f"[OpenSVBench] Canonical short crops failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(
        f"[OpenSVBench] Canonical short crops passed: dataset={args.dataset} "
        f"utterances={report['utterance_count']} manifest_sha256={report['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
