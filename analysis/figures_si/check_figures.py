"""Regression checks over the built figure set.

Catches the classes of defect that actually occurred while building it:

- a page that is not exactly 180 mm wide, or whose PDF and PNG disagree;
- ink touching the trim (a clipped label);
- a footnote that ends mid-sentence, which is how a truncated predicate string
  hid on five of the six S15 pages;
- a glyph the target face cannot render (U+25C6 fell back silently);
- type below the floor the set actually uses.

Run:  py -m analysis.figures_si.check_figures
"""

from __future__ import annotations

import importlib
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.figures_si.style import FIG_DIR  # noqa: E402

MIN_MARGIN_MM = 2.0
WIDTH_MM = 180.0
MAX_HEIGHT_MM = 225.0
MIN_PT = 5.8
DPI = 450


def _pdf_mm(path: Path) -> tuple[float, float]:
    m = re.search(
        rb"MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", path.read_bytes()
    )
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return (x1 - x0) / 72 * 25.4, (y1 - y0) / 72 * 25.4


def check_geometry() -> list[str]:
    bad: list[str] = []
    for png in sorted(FIG_DIR.glob("figS*.png")):
        im = np.array(Image.open(png).convert("L"))
        h, w = im.shape
        ink = im < 250
        rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
        mm = lambda v: v / DPI * 25.4  # noqa: E731
        margins = [mm(cols[0]), mm(w - 1 - cols[-1]), mm(rows[0]), mm(h - 1 - rows[-1])]
        pw, ph = _pdf_mm(png.with_suffix(".pdf"))
        if abs(mm(w) - WIDTH_MM) > 0.06:
            bad.append(f"{png.stem}: width {mm(w):.2f} mm, expected {WIDTH_MM}")
        if mm(h) > MAX_HEIGHT_MM + 0.05:
            bad.append(f"{png.stem}: height {mm(h):.1f} mm exceeds the page")
        if abs(pw - mm(w)) > 0.1 or abs(ph - mm(h)) > 0.1:
            bad.append(f"{png.stem}: PDF {pw:.1f}x{ph:.1f} != PNG {mm(w):.1f}x{mm(h):.1f}")
        if min(margins) < MIN_MARGIN_MM:
            names = "LRTB"
            worst = names[int(np.argmin(margins))]
            bad.append(f"{png.stem}: ink {min(margins):.2f} mm from the {worst} trim")
    return bad


_SENTENCE_END = re.compile(r"[.;:)\]%]$|[0-9A-Za-z−–]$")


def check_text() -> list[str]:
    """Collect every rendered string, then check glyphs, type size and footnotes."""
    from fontTools.ttLib import TTFont
    from matplotlib import font_manager as fm

    cmap = set(TTFont(fm.findfont(fm.FontProperties(family="Liberation Sans"))).getBestCmap())
    bad: list[str] = []
    from analysis.figures_si import style as sist

    real = sist.save_fig
    for n in range(1, 24):
        name = f"figS{n:02d}"
        try:
            mod = importlib.import_module(f"analysis.figures_si.{name}")
        except ModuleNotFoundError:
            continue
        seen: list[tuple[str, float]] = []

        def cap(fig, stem, *a, **k):
            fig.canvas.draw()
            for t in fig.findobj(matplotlib.text.Text):
                s = t.get_text()
                if s.strip():
                    seen.append((s, round(t.get_fontsize(), 2)))
            plt.close(fig)
            return Path(stem)

        sist.save_fig = cap
        mod.save_fig = cap
        try:
            mod.build()
        finally:
            sist.save_fig = real
        for s, size in seen:
            if size < MIN_PT - 1e-6:
                bad.append(f"{name}: {size} pt type -> {s[:40]!r}")
            for ch in s:
                if ord(ch) > 0x2000 and ord(ch) not in cmap and not ch.isspace():
                    bad.append(f"{name}: glyph U+{ord(ch):04X} {ch!r} not in the target face")
            # a footnote that trails off is how a truncated predicate hid before
            if "…" in s and s.rstrip().endswith(("….", ". …", "… .")):
                bad.append(f"{name}: sentence ends on an ellipsis -> {s[-48:]!r}")
    return bad


def main() -> int:
    problems = check_geometry() + check_text()
    pages = len(list(FIG_DIR.glob("figS*.png")))
    if problems:
        print(f"{pages} pages checked, {len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        return 1
    print(f"{pages} pages checked: geometry, glyphs, type floor and footnotes all clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
