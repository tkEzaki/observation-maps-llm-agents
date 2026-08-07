"""Delete only what the build can regenerate.

The figure directories accumulate scratch renders under `_to_delete/` during
iteration. This removes those, and optionally the built pages themselves —
never anything that is not reproducible from `build_all`.

    py -m analysis.figures_si.cleanup_generated            # scratch only
    py -m analysis.figures_si.cleanup_generated --pages    # + built pages
    py -m analysis.figures_si.cleanup_generated --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from analysis.figures_main.style import ROOT

FIG_DIRS = [
    ROOT / "figures" / "main",
    ROOT / "figures" / "si",
]
# Regenerable page stems. Anything else in these directories is left alone.
PAGE_GLOBS = ("fig[0-9]*.png", "fig[0-9]*.pdf", "figS[0-9]*.png", "figS[0-9]*.pdf")
KEEP = {"README.md", "manifest.json", "CAPTION_NOTES.md", "OMITTED_AND_CAVEATS.md"}


def _size(paths) -> str:
    n = sum(p.stat().st_size for p in paths if p.is_file())
    return f"{n / 1e6:.1f} MB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", action="store_true", help="also delete built pages")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    for d in FIG_DIRS:
        if not d.exists():
            continue
        scratch = d / "_to_delete"
        if scratch.exists():
            files = [p for p in scratch.rglob("*") if p.is_file()]
            print(f"{scratch}: {len(files)} scratch file(s), {_size(files)}")
            if not a.dry_run:
                shutil.rmtree(scratch)
        if a.pages:
            pages = [p for g in PAGE_GLOBS for p in d.glob(g) if p.name not in KEEP]
            print(f"{d}: {len(pages)} built page file(s), {_size(pages)}")
            if not a.dry_run:
                for p in pages:
                    p.unlink()
    if a.dry_run:
        print("\ndry run - nothing deleted")
    else:
        print("\nrebuild with: py -m analysis.figures_main.build_all"
              " && py -m analysis.figures_si.build_all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
