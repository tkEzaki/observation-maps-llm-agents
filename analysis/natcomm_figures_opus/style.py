"""Shared journal figure style - current redesign.

Design contract (differs from the earlier figure layout in three structural ways):

1. **True print geometry is enforced, not approximated.** Figures are built at
   exactly 180 mm and saved with a fixed canvas (no ``bbox_inches="tight"``),
   so the delivered PDF/PNG is the size the journal receives. The earlier
   variant cropped to the tight bbox, which silently produced figures between
   169 mm and 206 mm wide - the last of which exceeds the NC maximum.
2. **Panels are placed in millimetres from the top-left of the canvas** via
   :func:`mm_axes`, instead of nested GridSpecs with generous ``hspace``.
   This removes the large dead bands between panel rows and makes panel
   letters align on shared optical columns across every figure.
3. **Panel letters live in figure coordinates**, anchored a fixed distance
   outside each panel's axes box, so every letter in the set sits on the same
   baseline grid regardless of how wide that panel's tick labels are.

Colours are unchanged from the earlier layout (Okabe-Ito, CVD-validated) so the two
variants remain directly comparable; only geometry, typography metrics and
mark weights are redesigned.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures" / "natcomm_opus"
DATA_DIR = ROOT / "analysis" / "natcomm_figures" / "data"

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
MM = 1.0 / 25.4  # mm -> inch

W_FULL_MM = 180.0     # journal double column
W_SINGLE_MM = 88.0    # single column
W_THREEQ_MM = 136.0   # 1.5 column
H_MAX_MM = 225.0      # maximum page height

W_FULL = W_FULL_MM * MM   # 7.087 in
W_SINGLE = W_SINGLE_MM * MM
H_MAX = H_MAX_MM * MM

# Canvas margins (mm).
M_LEFT = 4.0
M_RIGHT = 3.0
M_TOP = 5.0
M_BOTTOM = 4.0

# Panel-letter placement relative to a panel's axes box (mm).
LETTER_DY_MM = 1.6
LETTER_DX_MM = -8.6
LETTER_ADVANCE_MM = 3.4

# ---------------------------------------------------------------------------
# Colour (unchanged from the earlier layout: Okabe-Ito, CVD-validated)
# ---------------------------------------------------------------------------
REP = {
    "moments_m1_m3": "#0072B2",
    "centers_24_standard": "#E69F00",
    "intervals_24_decimal6": "#009E73",
}
REP_LIGHT = {
    "moments_m1_m3": "#7FB8D9",
    "centers_24_standard": "#F3CF80",
    "intervals_24_decimal6": "#7FCFB9",
}
REP_SHORT = {
    "moments_m1_m3": "moments",
    "centers_24_standard": "centers",
    "intervals_24_decimal6": "intervals",
}
REP_ABBR = {
    "moments_m1_m3": "mom",
    "centers_24_standard": "cen",
    "intervals_24_decimal6": "int",
}
REP_ORDER = list(REP.keys())

ACTION = {
    "p_minus": "#AA3377",
    "p_zero": "#E3E3E3",
    "p_plus": "#004488",
}
ACTION_LABELS = (r"$p_-$", r"$p_0$", r"$p_+$")

OMAP = {
    "moments_original": "#4D4D4D",
    "moments_reformatted": "#9E91CC",
    "moments_length_matched": "#5E3C99",
}
OMAP_MID = "#8060B5"

FAMILY_EDGE = {"gpt": "#3A3A3A", "claude": "#7059A8", "gemini": "#B8860B"}
FAMILY_MARKER = {"gpt": "o", "claude": "s", "gemini": "D"}
FAMILY_LABEL = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini"}

PHENOTYPE = {
    "polar_locked": "#08519C",
    "partial_polar_order": "#9ECAE1",
    "high_r2_nonpolar": "#C0392B",
    "low_polar_active": "#D9D9D9",
}
PHENOTYPE_LABEL = {
    "polar_locked": "polar locked",
    "partial_polar_order": "partial polar order",
    "high_r2_nonpolar": "high $r_2$ non-polar",
    "low_polar_active": "low-polar active",
}

VERDICT = {
    "supported": "#1A7F37",
    "suggested": "#8A6D00",
    "not_established": "#8C8C8C",
}

PASS_GREEN = "#1A7F37"
STOP_RED = "#B0201F"
ACCENT_RED = "#B0201F"

GRAY_BOX = dict(facecolor="#F5F5F5", edgecolor="#8C8C8C", lw=0.5)
INK = "#1A1A1A"
MUTED = "#6E6E6E"
RULE = "#C8C8C8"

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FS_LETTER = 8.5   # bold panel letter
FS_TITLE = 7.4    # running panel title next to the letter
FS_BODY = 7.0     # axis labels, in-panel text
FS_TICK = 6.6
FS_SMALL = 6.2    # dense annotation, heatmap cell text
FS_TINY = 5.8     # last resort; NC floor is 5 pt

# Shared mark metrics - use these instead of per-call literals so line
# weights and marker areas are identical across all six figures.
LW_TRACE = 1.15   # emphasised trajectory / group mean line
LW_LINE = 0.9     # standard data line
LW_THIN = 0.6     # individual replicate / background line
LW_HAIR = 0.45    # reference rules, zero lines
LW_FAINT = 0.35   # dense replicate bundles (tens of overlapping lines)
MS_POINT = 2.2    # individual observation
MS_SERIES = 2.8   # marker on a data line
MS_MEAN = 3.4     # mean / summary marker
CAPSIZE = 1.6

# Arial is the target face. Liberation Sans / Nimbus Sans are metric-compatible
# substitutes so a Linux render previews the Windows output faithfully instead
# of silently falling back to DejaVu (which is wider).
SANS_STACK = [
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "Nimbus Sans",
    "TeX Gyre Heros",
    "DejaVu Sans",
]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": SANS_STACK,
            # STIX Sans: a sans maths companion that ships WITH matplotlib, so
            # every symbol resolves identically on Windows and Linux. The
            # earlier "dejavusans" set maths noticeably wider and heavier than
            # the Arial running text; a "custom" set mapped \mathcal to the
            # cursive family, which falls back to Comic Sans MS on Windows.
            "mathtext.fontset": "stixsans",
            "mathtext.default": "it",
            "font.size": FS_BODY,
            "axes.titlesize": FS_TITLE,
            "axes.labelsize": FS_BODY,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_SMALL,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.minor.size": 1.2,
            "ytick.minor.size": 1.2,
            "xtick.major.pad": 1.8,
            "ytick.major.pad": 1.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.labelpad": 1.8,
            "axes.titlepad": 3.0,
            "lines.linewidth": 0.9,
            "lines.markersize": 2.6,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.5,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 0.8,
            "legend.labelspacing": 0.28,
            "legend.borderaxespad": 0.15,
            "legend.borderpad": 0.25,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.axisbelow": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# ---------------------------------------------------------------------------
# Canvas + absolute panel placement
# ---------------------------------------------------------------------------
def new_figure(height_mm: float, width_mm: float = W_FULL_MM):
    """Create a fixed-size canvas in millimetres."""
    if height_mm > H_MAX_MM:
        raise ValueError(f"height {height_mm} mm exceeds the {H_MAX_MM} mm page limit")
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    fig._mm = (width_mm, height_mm)  # type: ignore[attr-defined]
    return fig


def _fig_mm(fig) -> tuple[float, float]:
    mm = getattr(fig, "_mm", None)
    if mm is None:
        w, h = fig.get_size_inches()
        mm = (w / MM, h / MM)
    return mm


def mm_axes(fig, left: float, top: float, width: float, height: float, **kw):
    """Add an axes positioned in mm from the **top-left** of the canvas."""
    W, H = _fig_mm(fig)
    return fig.add_axes([left / W, (H - top - height) / H, width / W, height / H], **kw)


def blank(ax) -> None:
    """Strip an axes down to a bare drawing surface for schematics (0-1 coords)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.patch.set_visible(False)


def mm_panel(fig, left: float, top: float, width: float, height: float, **kw):
    """A blank schematic panel (no frame, no ticks) placed in mm."""
    ax = mm_axes(fig, left, top, width, height, **kw)
    blank(ax)
    return ax


def despine(ax, top=True, right=True, left=False, bottom=False):
    for name, off in (("top", top), ("right", right), ("left", left), ("bottom", bottom)):
        ax.spines[name].set_visible(not off)
    return ax


def trim_spines(ax, x: bool = True, y: bool = True):
    """Clip left/bottom spines to the tick extent (Tufte range frame)."""
    if y and len(ax.get_yticks()):
        lo, hi = ax.get_ylim()
        t = [v for v in ax.get_yticks() if lo <= v <= hi]
        if t:
            ax.spines["left"].set_bounds(min(t), max(t))
    if x and len(ax.get_xticks()):
        lo, hi = ax.get_xlim()
        t = [v for v in ax.get_xticks() if lo <= v <= hi]
        if t:
            ax.spines["bottom"].set_bounds(min(t), max(t))
    return ax


# ---------------------------------------------------------------------------
# Panel letters
# ---------------------------------------------------------------------------
def panel_label(ax, letter: str, title: str = "", dx_mm: float = LETTER_DX_MM,
                dy_mm: float = LETTER_DY_MM, title_color: str = "#2B2B2B", fig=None):
    """Bold lowercase panel letter + regular running title, in figure space.

    Anchoring in figure coordinates (rather than axes-fraction offsets) keeps
    every letter on a shared optical column even when neighbouring panels have
    very different tick-label widths.
    """
    fig = fig or ax.figure
    W, H = _fig_mm(fig)
    pos = ax.get_position()
    x = pos.x0 + dx_mm / W
    y = pos.y1 + dy_mm / H
    fig.text(x, y, letter.lower(), fontsize=FS_LETTER, fontweight="bold",
             ha="left", va="baseline", color=INK)
    if title:
        fig.text(x + LETTER_ADVANCE_MM / W, y, title, fontsize=FS_TITLE,
                 ha="left", va="baseline", color=title_color)
    return fig


def panel_label_at(fig, x_mm: float, y_mm: float, letter: str, title: str = "",
                   title_color: str = "#2B2B2B"):
    """Panel letter at an absolute mm position (top-left origin).

    Use for schematic panels whose axes box does not describe where the reader
    perceives the panel to begin.
    """
    W, H = _fig_mm(fig)
    x, y = x_mm / W, (H - y_mm) / H
    fig.text(x, y, letter.lower(), fontsize=FS_LETTER, fontweight="bold",
             ha="left", va="baseline", color=INK)
    if title:
        fig.text(x + LETTER_ADVANCE_MM / W, y, title, fontsize=FS_TITLE,
                 ha="left", va="baseline", color=title_color)
    return fig


def fig_text_mm(fig, x_mm: float, y_mm: float, s: str, **kw):
    """Figure-level text placed in mm from the top-left."""
    W, H = _fig_mm(fig)
    kw.setdefault("fontsize", FS_SMALL)
    kw.setdefault("color", INK)
    return fig.text(x_mm / W, (H - y_mm) / H, s, **kw)


def rule_mm(fig, x0_mm: float, x1_mm: float, y_mm: float, color: str = RULE, lw: float = 0.5):
    """Hairline separator across the canvas, in mm (top-left origin)."""
    W, H = _fig_mm(fig)
    line = mpl.lines.Line2D([x0_mm / W, x1_mm / W], [(H - y_mm) / H, (H - y_mm) / H],
                            transform=fig.transFigure, color=color, lw=lw, zorder=0)
    fig.add_artist(line)
    return line


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _copy_with_retry(src: Path, dst: Path, attempts: int = 6, delay: float = 1.5) -> None:
    """Copy onto a synced path, tolerating a transient lock on the destination."""
    last = None
    for i in range(attempts):
        try:
            shutil.copyfile(src, dst)
            return
        except PermissionError as exc:  # Dropbox holding the old file open
            last = exc
            time.sleep(delay * (i + 1))
    raise PermissionError(
        f"could not write {dst} after {attempts} attempts; is it open elsewhere?"
    ) from last


def save_fig(fig, stem: str, out_dir: Path | None = None) -> Path:
    """Save at the exact canvas size - no tight-bbox cropping.

    ``bbox_inches="tight"`` makes the delivered width depend on whatever
    happens to stick out furthest, which is how the earlier set drifted to
    169-206 mm. Saving the fixed canvas guarantees the submitted figure is
    the width declared in the design spec.
    """
    d = out_dir or FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    png = d / f"{stem}.png"
    pdf = d / f"{stem}.pdf"

    # Render to a scratch directory outside the synced folder first, then copy
    # in. Writing matplotlib output straight onto a Dropbox path intermittently
    # fails with PermissionError while the client holds the previous PDF open,
    # which would leave a half-written file where a figure used to be. Rendering
    # elsewhere means a failure can only ever cost a retry, never a deliverable.
    with tempfile.TemporaryDirectory(prefix="figbuild_") as tmp:
        t = Path(tmp)
        tmp_png, tmp_pdf = t / png.name, t / pdf.name
        fig.savefig(tmp_png, facecolor="white")
        fig.savefig(tmp_pdf, facecolor="white")
        plt.close(fig)
        for src, dst in ((tmp_png, png), (tmp_pdf, pdf)):
            _copy_with_retry(src, dst)
    return png


# ---------------------------------------------------------------------------
# Statistics + drawing helpers
# ---------------------------------------------------------------------------
def paired_ci(values: np.ndarray, n_boot: int = 5000,
              seed: int = 20260724) -> tuple[float, float, float]:
    """Mean and a percentile bootstrap 95% interval over the sampled units.

    The units here are physical seeds (n = 6), so a normal approximation on
    the standard error would not be the quantity inference actually rests on.
    Resampling the seeds themselves matches the seed-block bootstrap used for
    the paired contrasts, and the draw is seeded so the interval is
    reproducible.
    """
    v = np.asarray(values, dtype=float)
    m = float(np.mean(v))
    if len(v) < 2:
        return m, m, m
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return m, float(lo), float(hi)


def beeswarm_x(n: int, center: float, width: float = 0.12,
               rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return center + rng.uniform(-width, width, size=n)


def swarm_x(values, center: float, width: float = 0.13, bins: int = 22) -> np.ndarray:
    """Deterministic beeswarm: symmetric spread within value bins, no jitter noise.

    Reads cleaner than uniform jitter at print size because equal values line
    up symmetrically instead of forming random clumps.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.zeros(0)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    span = (hi - lo) or 1.0
    idx = np.clip(((v - lo) / span * (bins - 1)).round().astype(int), 0, bins - 1)
    out = np.zeros_like(v)
    for b in np.unique(idx):
        sel = np.flatnonzero(idx == b)
        k = sel.size
        if k == 1:
            out[sel] = center
            continue
        out[sel] = center + np.linspace(-1.0, 1.0, k) * width * min(1.0, k / 6.0)
    return out


def errorbar_mean(ax, x, values, color=INK, marker="D", ms=MS_MEAN, lw=LW_LINE,
                  capsize=CAPSIZE, zorder=5):
    """Mean with a seed-bootstrap 95% interval, in the house style."""
    m, lo, hi = paired_ci(np.asarray(values, dtype=float))
    ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt=marker, color=color,
                markersize=ms, lw=lw, capsize=capsize, capthick=lw,
                zorder=zorder, clip_on=False)
    return m, lo, hi


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i: i + 2], 16) / 255 for i in (0, 2, 4)])


def text_on(color) -> str:
    """Pick black or white text for legibility on a filled cell."""
    r, g, b = mpl.colors.to_rgb(color)

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#FFFFFF" if lum < 0.42 else INK


def annotate_cells(ax, data, fmt="{:.2f}", cmap=None, norm=None,
                   fontsize=FS_SMALL, colors=None):
    """Write values into heatmap cells with automatic contrast."""
    arr = np.asarray(data, dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if colors is not None:
                c = colors[i][j]
            elif cmap is not None and norm is not None:
                c = text_on(cmap(norm(arr[i, j])))
            else:
                c = INK
            # U+2212 MINUS SIGN, not ASCII hyphen-minus: heatmap cell values
            # sit next to maths-set axis labels and must match their minus
            txt = fmt.format(arr[i, j]).replace("-", "\u2212")
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=fontsize, color=c)


def cbar_mm(fig, mappable, left, top, width, height, label="", ticks=None,
            orientation="vertical"):
    """Colourbar as an mm-placed axes so it never steals width from its panel."""
    cax = mm_axes(fig, left, top, width, height)
    cb = fig.colorbar(mappable, cax=cax, orientation=orientation, ticks=ticks)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("#666666")
    cax.tick_params(labelsize=FS_TICK, width=0.45, length=1.6, pad=1.4)
    if label:
        cb.set_label(label, fontsize=FS_BODY, labelpad=2.0)
    return cb


def rep_tick_colors(ax, order=None) -> None:
    """Colour x tick labels by representation."""
    order = order or REP_ORDER
    for tick, r in zip(ax.get_xticklabels(), order):
        tick.set_color(REP[r])


def direct_label(ax, x, y, text, color, fontsize=FS_BODY, weight="bold", **kw):
    """Label a series next to its own mark instead of in a legend box."""
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    return ax.text(x, y, text, color=color, fontsize=fontsize,
                   fontweight=weight, clip_on=False, **kw)


def stacked_trinomial(ax, probs_by_rep, show_legend: bool = True,
                      annotate: bool = True, bar_width: float = 0.58) -> None:
    """Trinomial (p-, p0, p+) stack per representation with A / a0 annotations."""
    bottoms = np.zeros(3)
    cols = [ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"]]
    for k, lab, col in zip(range(3), ACTION_LABELS, cols):
        vals = [probs_by_rep[r][k] for r in REP_ORDER]
        ax.bar([0, 1, 2], vals, bottom=bottoms, color=col, edgecolor="white",
               linewidth=0.7, width=bar_width, label=lab if show_legend else None)
        bottoms = bottoms + np.array(vals)
    for i, r in enumerate(REP_ORDER):
        p = probs_by_rep[r]
        a = p[0] + p[2]
        a0 = p[2] - p[0]
        if annotate:
            ax.text(i, 1.035, f"$A$={a:.2f}\n$a_0$={a0:+.2f}", ha="center",
                    va="bottom", fontsize=FS_TINY, linespacing=1.2, color=INK)
        ax.plot([i - 0.29, i + 0.29], [-0.04, -0.04], color=REP[r], lw=1.8,
                solid_capstyle="butt", clip_on=False)
    ax.set_ylim(0, 1.26)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([REP_SHORT[r] for r in REP_ORDER])
    rep_tick_colors(ax)
    if show_legend:
        ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=FS_TINY,
                  borderaxespad=0.0, handlelength=0.9, columnspacing=0.6,
                  handletextpad=0.35)
