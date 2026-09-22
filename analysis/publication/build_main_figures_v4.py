"""Figs. 3-5 in the publication house style, from the plot_main builders.

Run from the repository root:  python -m analysis.publication.build_main_figures_v4

Only typography, panel-letter and colour constants are patched, before the
builders are imported; no data, statistics or panel content change. Output goes
to figures/rendered_main so the earlier exports stay untouched.

Changes relative to figures/source_main:
  * bold capital panel letters at 9 pt, generated natively (no PDF glyph patching);
  * no text below 6 pt (the previous floor was 5.8 pt);
  * single-letter encoding abbreviations M/C/I, as in Figs. 1-2 and Fig. 3D;
  * Gemini drawn in neutral grey instead of gold, which was close to the centers orange.
Bug fixes made in the builders themselves (see _backup_20260922 for the originals):
  * Fig. 5A printed "ildez" because "\\t" in a non-raw string was read as a tab;
  * the permutation nulls in Figs. 3F and 5D were invisible (50 narrow bins with
    white edges over a wide x range); bins now span the null and have no edges;
  * the Fig. 3E title ran past the right edge of the canvas.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analysis.plot_main.style as S  # noqa: E402

S.FS_LETTER = 9.0
S.FS_TINY = 6.0
S.FS_SMALL = max(S.FS_SMALL, 6.2)
S.REP_ABBR = {"moments_m1_m3": "M", "centers_24_standard": "C", "intervals_24_decimal6": "I"}
S.FAMILY_EDGE = {"gpt": "#1A1A1A", "claude": "#7059A8", "gemini": "#8C8C8C"}
S.FIG_DIR = ROOT / "figures" / "rendered_main"


def panel_label(ax, letter, title="", dx_mm=S.LETTER_DX_MM, dy_mm=S.LETTER_DY_MM,
                title_color="#2B2B2B", fig=None):
    fig = fig or ax.figure
    W, H = S._fig_mm(fig)
    pos = ax.get_position()
    x = pos.x0 + dx_mm / W
    y = pos.y1 + dy_mm / H
    fig.text(x, y, letter.upper(), fontsize=S.FS_LETTER, fontweight="bold",
             ha="left", va="baseline", color=S.INK)
    if title:
        fig.text(x + S.LETTER_ADVANCE_MM / W, y, title, fontsize=S.FS_TITLE,
                 ha="left", va="baseline", color=title_color)
    return fig


def panel_label_at(fig, x_mm, y_mm, letter, title="", title_color="#2B2B2B"):
    W, H = S._fig_mm(fig)
    x, y = x_mm / W, (H - y_mm) / H
    fig.text(x, y, letter.upper(), fontsize=S.FS_LETTER, fontweight="bold",
             ha="left", va="baseline", color=S.INK)
    if title:
        fig.text(x + S.LETTER_ADVANCE_MM / W, y, title, fontsize=S.FS_TITLE,
                 ha="left", va="baseline", color=title_color)
    return fig


S.panel_label = panel_label
S.panel_label_at = panel_label_at

from analysis.plot_main import fig3_replay, fig4_cross_family, fig5_omap  # noqa: E402

if __name__ == "__main__":
    S.apply_style()
    for build in (fig3_replay.build, fig4_cross_family.build, fig5_omap.build):
        print(build())
