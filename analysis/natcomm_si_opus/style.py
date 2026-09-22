"""SI style — re-exports the main-figure style so the two sets stay identical.

Only three things differ for the SI:

1. ``FIG_DIR`` points at ``figures/natcomm_si_opus``.
2. SI figures may run to the full page height (``H_MAX_MM`` = 225 mm) rather
   than the 100-165 mm the main figures use, because an SI figure is allowed a
   page of its own.
3. ``page_label`` is retired: multi-page figures are identified by their caption
   corner, which the main set never needs.

Everything else — palette, type scale, mm grid, mark metrics, save path
discipline — is imported unchanged from ``natcomm_figures_opus.style``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from analysis.natcomm_figures_opus.style import *  # noqa: F401,F403
from analysis.natcomm_figures_opus.style import (
    FS_LETTER,
    FS_SMALL,
    INK,
    MUTED,
    MM,
    ROOT,
    W_FULL_MM,
    H_MAX_MM,
    _fig_mm,
    apply_style,
)
from analysis.natcomm_figures_opus.style import save_fig as _save_fig_main

FIG_DIR = ROOT / "figures" / "natcomm_si_opus"
DATA_DIR = ROOT / "analysis" / "natcomm_figures" / "data"

# Raw acquisition roots the SI figures reach into that the main set does not.
RUNS = ROOT / "runs"
R2_CLAUDE = (
    RUNS / "stage_c" / "r2_claude_macro"
    / "matched-rep-collective-r2-claude-v0.1_ff2ef27f0dfe" / "20260725T050234Z"
)

# An SI figure may take a full page; the main-set ceiling does not apply.
H_PAGE_MM = 225.0


def save_fig(fig, stem: str, out_dir: Path | None = None) -> Path:
    """Save into the SI figure directory at the exact canvas size."""
    return _save_fig_main(fig, stem, out_dir or FIG_DIR)


def page_label(fig, text: str) -> None:  # retired, do not call
    """Retired. Multi-page figures must not stamp their own page marker.

    This used to print a corner tag such as 'S8a', built from the SCRIPT number.
    The script series and the Supplementary Information numbering are
    independent, so the tag drifted out of step with the document as soon as the
    figures were reordered, and it duplicated information the caption and the
    page footer already carry. Page identity now comes from the caption label
    alone.
    """
    raise RuntimeError(
        "page_label is retired: the caption label identifies the page, and a "
        "stamp built from the script number drifts from the document numbering"
    )


def si_note(fig, text: str, y_mm: float | None = None) -> None:
    """A muted footnote across the foot of an SI figure."""
    W, H = _fig_mm(fig)
    y = (H - (y_mm if y_mm is not None else H - 2.6))
    fig.text(4.0 / W, y / H, text, ha="left", va="baseline",
             fontsize=FS_SMALL, color=MUTED)
