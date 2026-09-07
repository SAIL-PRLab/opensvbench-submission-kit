#!/usr/bin/env python3
"""Extract ESD zip to raw_data with correct layout.

Handles:
  - Top folder like "Emotion Speech Dataset/0001/Angry/*.wav"
  - Skips __MACOSX and AppleDouble (._*)
  - Strips root prefix before first 4-digit speaker dir
  - Hoists speaker folders to output root if needed
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys
import zipfile

SPK_DIR = re.compile(r"^\d{4}$")


def is_junk_path(parts: tuple[str, ...]) -> bool:
    if any(p == "__MACOSX" for p in parts):
        return True
    if any(p.startswith("._") for p in parts):
        return True
    return False


def safe_destination(out_dir: pathlib.Path, relative_path: pathlib.PurePosixPath) -> pathlib.Path:
    """Return a destination inside out_dir or reject the archive member."""
    if relative_path.is_absolute() or any(part in ("", ".", "..") for part in relative_path.parts):
        raise ValueError(f"Unsafe archive path: {relative_path}")

    root = out_dir.resolve()
    destination = (root / pathlib.Path(*relative_path.parts)).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Archive path escapes output directory: {relative_path}") from exc
    return destination


def extract_zip(zip_path: pathlib.Path, out_dir: pathlib.Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        root_prefix: str | None = None
        for n in names:
            parts = pathlib.PurePosixPath(n).parts
            if is_junk_path(parts):
                continue
            for i, p in enumerate(parts):
                if SPK_DIR.match(p):
                    root_prefix = "/".join(parts[:i]) if i > 0 else ""
                    break
            if root_prefix is not None:
                break

        print(f"[ESD] Detected archive root prefix to strip: {root_prefix!r}")

        if root_prefix:
            prefix_parts = len(pathlib.PurePosixPath(root_prefix).parts)
        else:
            prefix_parts = 0

        n_written = 0
        for member in z.infolist():
            member_path = pathlib.PurePosixPath(member.filename)
            parts = member_path.parts
            if is_junk_path(parts):
                continue

            if prefix_parts > 0 and len(parts) >= prefix_parts:
                head = "/".join(parts[:prefix_parts])
                if head == root_prefix and len(parts) > prefix_parts:
                    rel = pathlib.PurePosixPath(*parts[prefix_parts:])
                else:
                    rel = member_path
            else:
                rel = member_path

            if rel is None or str(rel) in ("", "."):
                continue

            dest = safe_destination(out_dir, rel)
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n_written += 1

        print(f"[ESD] Extracted {n_written} files.")

    return n_written


def hoist_speaker_dirs(out_dir: pathlib.Path) -> None:
    if (out_dir / "0001").is_dir():
        return
    for p in sorted(out_dir.iterdir()):
        if not p.is_dir():
            continue
        has_spk = any(
            SPK_DIR.match(c.name) and (p / c.name).is_dir() for c in p.iterdir()
        )
        if has_spk and (p / "0001").is_dir():
            print(f"[ESD] Hoisting speaker folders from {p.name}/ -> {out_dir.name}/")
            for child in list(p.iterdir()):
                target = out_dir / child.name
                if target.exists():
                    continue
                shutil.move(str(child), str(target))
            try:
                p.rmdir()
            except OSError:
                pass
            break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    if not args.zip.is_file():
        print(f"[ESD] ERROR: zip not found: {args.zip}", file=sys.stderr)
        return 1

    try:
        n = extract_zip(args.zip, args.out)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"[ESD] ERROR: Extraction failed: {exc}", file=sys.stderr)
        return 1
    if n == 0:
        print("[ESD] ERROR: No files extracted (zip empty or unreadable?)", file=sys.stderr)
        return 1

    hoist_speaker_dirs(args.out)

    if not (args.out / "0001").is_dir():
        print(
            "[ESD] ERROR: After extract/hoist, output/0001 still missing. "
            "Check zip integrity, disk space, and write permissions.",
            file=sys.stderr,
        )
        print("[ESD] Contents of output dir:", file=sys.stderr)
        try:
            for c in sorted(args.out.iterdir())[:30]:
                print(" ", c.name, file=sys.stderr)
        except OSError as e:
            print(" ", e, file=sys.stderr)
        return 1

    print("[ESD] Extraction OK: found", args.out / "0001")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
