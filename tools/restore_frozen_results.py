#!/usr/bin/env python3
"""Copy the frozen results into the working analysis tree.

The analysis and figure scripts read their inputs from ``analysis/``.  This
repository ships those artefacts under ``results_frozen/`` instead, because
``results_frozen/`` is never written by any pipeline script and therefore stays
byte-identical to the state the manuscript reports.  Run this once after
cloning, and the figure scripts will work without re-running any acquisition.

    python tools/restore_frozen_results.py            # skip files already there
    python tools/restore_frozen_results.py --force    # overwrite them
    python tools/restore_frozen_results.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results_frozen"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="overwrite files that already exist in analysis/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not FROZEN.is_dir():
        print("results_frozen/ not found", file=sys.stderr)
        return 2

    copied = skipped = 0
    # Only the mirrored analysis tree is restored.  Files that sit directly in
    # results_frozen/ (its README, its manifest) describe the directory itself
    # and must never be copied over a same-named file at the repository root.
    for src in sorted((FROZEN / "analysis").rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(FROZEN)
        dst = ROOT / rel
        if dst.exists() and not args.force:
            skipped += 1
            continue
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {copied} file(s); skipped {skipped} already present"
          f"{' (use --force to overwrite)' if skipped and not args.force else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
