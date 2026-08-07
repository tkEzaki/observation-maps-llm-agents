"""Replace personal absolute paths in the release tree with placeholders.

The transformation is purely textual and touches provenance strings only.  It
is applied identically by tools/verify_frozen_results.py before comparing, so
re-running an analysis on any machine still diffs cleanly against the frozen
copy.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

# escaped (JSON) form first, then the raw form; longest match first
RULES = [
    (re.compile(r"[A-Za-z]:\\\\Dropbox\\\\research_current\\\\llm_emergent\\\\pilot4_kuramoto"),
     "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:\\Dropbox\\research_current\\llm_emergent\\pilot4_kuramoto"),
     "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9_.\-]+\\\\Dropbox\\\\research_current\\\\llm_emergent\\\\pilot4_kuramoto"),
     "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.\-]+\\Dropbox\\research_current\\llm_emergent\\pilot4_kuramoto"),
     "<PROJECT_ROOT>"),
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9_.\-]+"), "<HOME>"),
    (re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.\-]+"), "<HOME>"),
]
SUFFIXES = {".py", ".md", ".json", ".jsonl", ".txt", ".csv", ".bat", ".cfg", ".toml", ".yml", ".yaml"}

def scrub_text(text: str) -> tuple[str, int]:
    n = 0
    for pattern, repl in RULES:
        text, k = pattern.subn(repl, text)
        n += k
    return text, n

def main(root: Path) -> int:
    touched = total = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUFFIXES:
            continue
        try:
            original = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        cleaned, n = scrub_text(original)
        if n:
            p.write_text(cleaned, encoding="utf-8", newline="")
            touched += 1
            total += n
    print(f"scrubbed {total} occurrence(s) in {touched} file(s)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
