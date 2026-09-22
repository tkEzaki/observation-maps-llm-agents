#!/usr/bin/env python3
"""Compare the working analysis tree against the frozen results.

Re-running an analysis script overwrites files under ``analysis/``.  This tool
reports how the result changed relative to ``results_frozen/``, which no
pipeline script writes.  Use it after re-running anything, before trusting a
number.

    python tools/verify_frozen_results.py                 # every frozen file
    python tools/verify_frozen_results.py --glob '*decision.json'
    python tools/verify_frozen_results.py --traces        # clone completeness

``--traces`` checks the acquisition traces and the large analysis artefacts
against the sizes and SHA-256 values recorded in ``DATA_ARCHIVE_MANIFEST.json``
instead of comparing the analysis tree.  Use it after cloning to confirm that
nothing was truncated in transit.

Exit status is 0 when every compared file is byte-identical, 1 otherwise.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results_frozen"


# The published copies have personal absolute paths replaced by placeholders
# (see tools/scrub_paths.py).  A local re-run writes the real path back, which
# is not a scientific difference, so both sides are normalised before hashing.
import re

_PATH_RULES = [
    (re.compile(r"[A-Za-z]:\\\\Dropbox\\\\research_current\\\\llm_emergent\\\\pilot4_kuramoto"), "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:\\Dropbox\\research_current\\llm_emergent\\pilot4_kuramoto"), "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9_.\-]+\\\\Dropbox\\\\research_current\\\\llm_emergent\\\\pilot4_kuramoto"), "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.\-]+\\Dropbox\\research_current\\llm_emergent\\pilot4_kuramoto"), "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9_.\-]+"), "<HOME>"),
    (re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.\-]+"), "<HOME>"),
]
_TEXT_SUFFIXES = {".py", ".md", ".json", ".txt", ".csv", ".bat", ".cfg", ".toml",
                  ".yml", ".yaml"}


def _normalised_bytes(path: Path) -> bytes:
    """File bytes with personal absolute paths and line endings normalised."""
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        return path.read_bytes()
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return path.read_bytes()
    for pattern, repl in _PATH_RULES:
        text = pattern.sub(repl, text)
    return text.replace("\r\n", "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(_normalised_bytes(path)).hexdigest()


def flatten(obj, prefix=""):
    """Flatten JSON into path -> scalar, so differences can be located."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}"))
    elif isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) for x in obj) and len(obj) <= 4:
            out[prefix] = tuple(obj)
        else:
            out[prefix] = f"list[{len(obj)}]"
    else:
        out[prefix] = obj
    return out


def json_differences(a: Path, b: Path, limit: int = 12) -> list[str]:
    try:
        fa = flatten(json.loads(a.read_text(encoding="utf-8")))
        fb = flatten(json.loads(b.read_text(encoding="utf-8")))
    except Exception as exc:                      # not JSON, or malformed
        return [f"could not compare as JSON: {exc}"]
    lines = []
    for key in sorted(set(fa) | set(fb)):
        if fa.get(key) != fb.get(key):
            lines.append(f"      {key}: frozen={fa.get(key)!r} current={fb.get(key)!r}")
            if len(lines) >= limit:
                lines.append("      ...")
                break
    return lines


def raw_sha256(path: Path) -> str:
    """Unnormalised hash, for the manifest check."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(quiet: bool) -> int:
    """Check every file listed in DATA_ARCHIVE_MANIFEST.json."""
    manifest = ROOT / "DATA_ARCHIVE_MANIFEST.json"
    if not manifest.exists():
        print("DATA_ARCHIVE_MANIFEST.json not found")
        return 1
    entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    ok = bad = absent = 0
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.exists():
            absent += 1
            print(f"MISSING   {entry['path']}")
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            bad += 1
            print(f"SIZE      {entry['path']}  "
                  f"expected {entry['bytes']} got {size}")
            continue
        if raw_sha256(path) != entry["sha256"]:
            bad += 1
            print(f"SHA256    {entry['path']}")
            continue
        ok += 1
        if not quiet:
            print(f"ok        {entry['path']}")
    total = sum(e["bytes"] for e in entries)
    print(f"\n{ok} ok, {bad} corrupt, {absent} missing "
          f"({len(entries)} files, {total / 1048576:.0f} MB)")
    return 0 if (bad == 0 and absent == 0) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="*", help="only compare paths matching this pattern")
    ap.add_argument("--quiet", action="store_true", help="only report problems")
    ap.add_argument("--traces", action="store_true",
                    help="check traces and large artefacts against "
                         "DATA_ARCHIVE_MANIFEST.json instead")
    args = ap.parse_args()

    if args.traces:
        return verify_manifest(args.quiet)

    identical = differing = missing = 0
    for src in sorted((FROZEN / "analysis").rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(FROZEN)
        if not fnmatch.fnmatch(rel.as_posix(), args.glob) and \
           not fnmatch.fnmatch(rel.name, args.glob):
            continue
        live = ROOT / rel
        if not live.exists():
            missing += 1
            print(f"MISSING   {rel}  (run tools/restore_frozen_results.py)")
            continue
        if sha256(src) == sha256(live):
            identical += 1
            if not args.quiet:
                print(f"identical {rel}")
        else:
            differing += 1
            print(f"DIFFERS   {rel}")
            if rel.suffix.lower() == ".json":
                for line in json_differences(src, live):
                    print(line)

    print(f"\n{identical} identical, {differing} differing, {missing} missing")
    return 0 if (differing == 0 and missing == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
