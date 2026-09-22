#!/usr/bin/env python3
"""Pre-publication checks for this repository.

Refuses to pass if the tree contains an API key, a personal absolute path, a
file too large for a normal git repository, or a stray cache directory.  Run
before pushing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "OpenAI-style API key"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "Anthropic-style API key"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "Google-style API key"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"'][^\"'\s]{16,}"),
     "inline credential"),
]
PATH_PATTERNS = [
    (re.compile(r"[A-Za-z]:\\+Users\\+[A-Za-z0-9_.\-]+"), "absolute Windows user path"),
    (re.compile(r"[A-Za-z]:\\+Dropbox"), "absolute Dropbox drive path"),
    (re.compile(r"/(?:home|Users)/(?!runner\b)[A-Za-z0-9_.\-]+/"), "absolute POSIX home path"),
]
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".txt", ".cfg", ".toml", ".yml", ".yaml",
                 ".bat", ".sh", ".csv", ".cff"}
MAX_FILE_BYTES = 50_000_000

def main() -> int:
    problems: list[str] = []
    total = 0
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = p.relative_to(ROOT).as_posix()
        total += p.stat().st_size
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            problems.append(f"cache file committed: {rel}")
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"file too large ({p.stat().st_size/1e6:.0f} MB): {rel}")
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, label in SECRET_PATTERNS:
            if pattern.search(text):
                problems.append(f"{label} in {rel}")
        for pattern, label in PATH_PATTERNS:
            m = pattern.search(text)
            if m:
                problems.append(f"{label} in {rel}: {m.group(0)}")

    print(f"scanned tree, {total/1e6:.1f} MB total")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems[:60]:
            print("  " + line)
        if len(problems) > 60:
            print(f"  ... and {len(problems)-60} more")
        return 1
    print("no secrets, personal paths, oversized files or caches found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
