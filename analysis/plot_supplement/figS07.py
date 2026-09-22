"""Supplementary Figure S7 - sparse finite-peer fields and variance decomposition.

Stage B Step D ran one nested campaign on a *sparse* antipodal relative-phase
field: peer count N = 16, concentration kappa = 6, partner-paired draws, three
count imbalances (8:8, 9:7, 10:6), six independent physical realizations of
each, three observation representations, and two independent seed blocks. The
figure documents (i) what those discrete fields actually look like, (ii) the
trinomial response they elicit, (iii) how the response depends on peer count,
(iv) how large the physical-field and LLM-resampling variation is next to the
representation effect, (v) that exact balance produces a stable moments
abstention rather than a one-draw fluke, and (vi) that a one-peer majority is
enough to restore a strong signed polar channel.

Every number is read from the frozen artifacts:

* ``analysis/complex_kernel_sparse_peer/complex_endpoints.csv``
* ``analysis/complex_kernel_sparse_peer/realization_means.csv``
* ``analysis/complex_kernel_sparse_peer/within_realization_variance.csv``
* ``analysis/complex_kernel_sparse_peer/sparse_peer_analysis.json``
* ``analysis/stage_b_offline_p3_p4/phenotype_atlas.csv`` (panels c, e only)

and the stimulus geometry in panel **a** is rebuilt from the frozen catalog in
``circlemap/stimuli.py``.

Layout
------
Every free-floating text block has an explicit width budget and is wrapped to
it with the renderer (``_wrap`` measures each trial line), so no note can run
off the 180 mm canvas or across a neighbouring label:

* each panel note sits *below* its own panel, never in the panel-title band
  (panel c) and never across the group labels it explains (panel b);
* panel c reserves a right-hand gutter for its three direct series labels;
* panels d and e share **one** key row under the second panel row instead of
  two overlapping in-panel legends;
* the closing note wraps to the canvas width minus the outer margins.

``_overflow`` re-measures every text artist after the figure is assembled and
warns on stderr if any of them leaves the ink area, so a future edit cannot
silently reintroduce the overflow.

Claim boundary, panel c
-----------------------
The plan asks for a peer-count sweep over N = 8, 16 and 240. The sparse-peer
campaign contains **N = 16 only**. The N = 240 dense antipodal field and the
deterministic integer-count N = 16 antipodal field come from a *different*
acquisition (``analysis/complex_kernel_stimulus_manifold``), and there is no
antipodal N = 8 field at all - the only peer-8 acquisition is a *unimodal*
field, whose dense counterpart is kappa = 9, so peer count and concentration
would be confounded. Panel c therefore shows the two peer counts that exist
for a matched field family (antipodal, kappa = 6), keeps the two acquisitions
visually distinct (open vs filled markers, tinted band), states under the panel
that it is not a controlled sweep, and omits N = 8 entirely.

Trinomial probabilities are recovered from the frozen activity and $a_0$
exactly as main Fig. 2 does, so the shared profiles carry identical values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plot_supplement.style import (
    ACTION,
    ACTION_LABELS,
    CAPSIZE,
    FS_BODY,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    LW_TRACE,
    MS_MEAN,
    MS_POINT,
    MS_SERIES,
    MUTED,
    REP,
    REP_ABBR,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    apply_style,
    despine,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label_at,
    paired_ci,
    save_fig,
    swarm_x,
    trim_spines,
)

from circlemap.stimuli import STIMULUS_CATALOG, build_stimulus_histogram

# --------------------------------------------------------------------------
# Layout board (mm from the top-left of a 180 mm canvas)
# --------------------------------------------------------------------------
H_MM = 174.0

PT_MM = 25.4 / 72.0
LS_PARA = 1.42          # leading of the wrapped prose notes
LS_KEY = 1.75           # leading of the shared key rows

X_LEFT = 4.8            # outer ink margin == panel-letter column
X_RIGHT = 176.0         # right-hand ink limit

COL_X = (13.4, 61.0, 125.0)    # shared left edge of each panel column
R1_W = (42.0, 52.0, 37.0)      # row-1 widths (c is narrow: it needs a label gutter)
R2_W = (42.0, 52.0, 50.0)      # row-2 widths
LETTER_Y = (8.4, 85.5)         # shared letter baseline per row

# row 1 -- panel a discs
A_DISC = 16.0
A_X = (14.4, 34.8)
A_CAP = (17.6, 39.2)           # baseline of the caption above each disc row
A_TOP = (18.4, 40.0)
A_NOTE_Y = 61.4                # first baseline of panel a's note

# row 1 -- panels b and c
B_TOP, B_H = 14.0, 39.0
B_GROUP_Y = -0.235             # group label position, axes data units
B_NOTE_Y = 68.8                # below the 8:8 / 9:7 / 10:6 group labels
C_TOP = (14.0, 29.6, 45.2)
C_H = 12.0
C_LAB_X = 1.045                # series-label gutter, in axes-width fractions
C_NOTE_Y = 68.8                # below panel c's x label

# row 2 -- every panel is a pair of stacked mini-axes on one shared grid
R2_TOP = (91.0, 119.0)
R2_H = 22.0

KEY_Y = 154.4                  # first baseline of the shared d/e key
FOOT_Y = 165.6                 # first baseline of the closing note

IMB = ("8_8", "9_7", "10_6")
IMB_LABEL = {"8_8": "8:8", "9_7": "9:7", "10_6": "10:6"}
ATLAS_SHORT = {"moments_m1_m3": "moments",
               "centers_24_standard": "centers",
               "intervals_24_decimal6": "intervals"}

SPARSE_DIR = ROOT / "analysis" / "complex_kernel_sparse_peer"
ATLAS = ROOT / "analysis" / "stage_b_offline_p3_p4" / "phenotype_atlas.csv"


# --------------------------------------------------------------------------
# Measured text: every free-floating block gets a width budget
# --------------------------------------------------------------------------
def _mm(fig) -> tuple[float, float]:
    w, h = fig.get_size_inches()
    return w * 25.4, h * 25.4


def _pitch(fontsize: float, lead: float = LS_PARA) -> float:
    return fontsize * lead * PT_MM


def _text_w_mm(fig, s: str, fontsize: float, **kw) -> float:
    """Rendered width of ``s`` in mm - measured, never estimated."""
    art = fig.text(0.0, 0.0, s, fontsize=fontsize, **kw)
    bb = art.get_window_extent(renderer=fig.canvas.get_renderer())
    art.remove()
    return float(bb.width) / float(fig.dpi) * 25.4


def _wrap(fig, text: str, max_mm: float, fontsize: float, **kw) -> list[str]:
    """Word-wrap to a measured width; display-only, no characters added."""
    out: list[str] = []
    cur = ""
    for word in text.split(" "):
        trial = word if not cur else f"{cur} {word}"
        if not cur or _text_w_mm(fig, trial, fontsize, **kw) <= max_mm:
            cur = trial
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def _para(fig, x_mm: float, y_mm: float, text: str, max_mm: float,
          fontsize: float = FS_TINY, color: str = MUTED) -> float:
    """Draw a wrapped prose note; return the baseline just past its last line."""
    pitch = _pitch(fontsize)
    lines = _wrap(fig, text, max_mm, fontsize)
    for i, line in enumerate(lines):
        fig_text_mm(fig, x_mm, y_mm + i * pitch, line, ha="left", va="baseline",
                    fontsize=fontsize, color=color)
    return y_mm + len(lines) * pitch


def _key_row(fig, x_mm: float, y_mm: float, tag: str, items, max_mm: float,
             fontsize: float = FS_TINY) -> float:
    """One horizontal key line: bold panel tag, then marker + label entries.

    Entries are placed by measured width and wrap onto a further line rather
    than running past ``x_mm + max_mm``.
    """
    W, H = _mm(fig)
    pitch = _pitch(fontsize, LS_KEY)
    fig_text_mm(fig, x_mm, y_mm, tag, ha="left", va="baseline", fontsize=fontsize,
                fontweight="bold", color=INK)
    x0 = x_mm + _text_w_mm(fig, tag, fontsize, fontweight="bold") + 2.6
    x, y = x0, y_mm
    for marker, colour, label, mkw in items:
        w = _text_w_mm(fig, label, fontsize) + 2.6
        if x > x0 and (x - x_mm) + w > max_mm:
            x, y = x0, y + pitch
        mark = dict(ms=MS_SERIES)
        mark.update(mkw)
        fig.add_artist(mlines.Line2D(
            [x / W], [(H - (y - 0.62)) / H], transform=fig.transFigure,
            marker=marker, color=colour, linestyle="none",
            clip_on=False, **mark))
        fig_text_mm(fig, x + 2.6, y, label, ha="left", va="baseline",
                    fontsize=fontsize, color=colour)
        x += w + 4.4
    return y + pitch


def _overflow(fig) -> None:
    """Warn on stderr if any text artist leaves the ink area of the canvas."""
    W, H = _mm(fig)
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    for ax in fig.axes:
        artists.extend(ax.texts)
        artists.extend(ax.get_xticklabels())
        artists.extend(ax.get_yticklabels())
    for art in artists:
        if not art.get_text():
            continue
        try:
            bb = art.get_window_extent(renderer=renderer)
        except Exception:      # pragma: no cover - unrenderable artist
            continue
        x0 = bb.x0 / fig.dpi * 25.4
        x1 = bb.x1 / fig.dpi * 25.4
        y_top = H - bb.y1 / fig.dpi * 25.4
        y_bot = H - bb.y0 / fig.dpi * 25.4
        label = art.get_text()[:52].replace("\n", " ")
        if x1 > X_RIGHT or x0 < 2.2 or y_bot > H - 2.2 or y_top < 2.2:
            print(f"[figS07] overflow x=({x0:.1f},{x1:.1f}) y=({y_top:.1f},"
                  f"{y_bot:.1f}) limits=(2.2,{X_RIGHT:.1f},{H - 2.2:.1f}): "
                  f"{label!r}", file=sys.stderr)


# --------------------------------------------------------------------------
# Small shared computations
# --------------------------------------------------------------------------
def _trinomial(activity: float, a0: float) -> np.ndarray:
    """Recover $(p_-, p_0, p_+)$ from activity and $a_0$ - main Fig. 2's rule."""
    p_plus = float(np.clip(0.5 * (activity + a0), 0.0, 1.0))
    p_minus = float(np.clip(0.5 * (activity - a0), 0.0, 1.0))
    p_zero = float(np.clip(1.0 - activity, 0.0, 1.0))
    s = p_minus + p_zero + p_plus
    return np.array([p_minus / s, p_zero / s, p_plus / s])


def _pair_tv(probs_by_rep: dict) -> tuple[list[float], float]:
    """Pairwise total-variation distances between the three representations."""
    vals = []
    for i in range(len(REP_ORDER)):
        for j in range(i + 1, len(REP_ORDER)):
            vals.append(
                0.5 * float(np.abs(probs_by_rep[REP_ORDER[i]] - probs_by_rep[REP_ORDER[j]]).sum())
            )
    return vals, float(np.mean(vals))


def _sd(values) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.std(v, ddof=1)) if v.size > 1 else 0.0


def _peer_counts(profile: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Bin centres, integer peer counts and N for a frozen catalog profile."""
    spec = STIMULUS_CATALOG[profile]
    hist = build_stimulus_histogram(profile, 0.0)
    edges = np.asarray(hist.edges, dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = np.rint(np.asarray(hist.fractions, dtype=float) * spec.peer_count).astype(int)
    return centres, counts, int(spec.peer_count)


# --------------------------------------------------------------------------
# Panel drawing helpers
# --------------------------------------------------------------------------
def _disc(fig, left: float, top: float, profile: str, caption: str, cap_y: float) -> None:
    """One sparse field as peers on the phase circle (dot = one peer)."""
    ax = mm_axes(fig, left, top, A_DISC, A_DISC)
    ax.set_xlim(-1.34, 1.34)
    ax.set_ylim(-1.34, 1.34)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color=RULE, lw=LW_LINE))
    centres, counts, _ = _peer_counts(profile)
    for centre, n in zip(centres, counts):
        for k in range(int(n)):
            r = 1.0 - 0.145 * k
            ax.plot([r * np.cos(centre)], [r * np.sin(centre)], "o",
                    color=INK, ms=MS_POINT, mew=0.0, clip_on=False, zorder=4)
    # focal reference at relative phase 0
    ax.plot([1.30], [0.0], marker=(3, 0, -90), color=MUTED, ms=MS_SERIES,
            mew=0.0, linestyle="none", clip_on=False, zorder=5)
    fig_text_mm(fig, left + A_DISC / 2.0, cap_y, caption, ha="center",
                va="baseline", fontsize=FS_SMALL, color=INK)


# --------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()

    endpoints = pd.read_csv(SPARSE_DIR / "complex_endpoints.csv")
    realization = pd.read_csv(SPARSE_DIR / "realization_means.csv")
    within = pd.read_csv(SPARSE_DIR / "within_realization_variance.csv")
    meta = json.loads((SPARSE_DIR / "sparse_peer_analysis.json").read_text(encoding="utf-8"))
    atlas = pd.read_csv(ATLAS)

    n_real = int(meta["n_realizations"])
    n_blocks = int(endpoints["block"].nunique())
    peer_n = int(endpoints["count_a"].iloc[0] + endpoints["count_b"].iloc[0])

    fig = new_figure(H_MM)

    # ======================================================================
    # a  sparse finite-peer stimulus inventory
    # ======================================================================
    panel_label_at(fig, COL_X[0] - 8.6, LETTER_Y[0], "a", "Sparse finite-peer fields")
    discs = (
        (A_X[0], A_TOP[0], f"sparse_peer{peer_n}_antipodal_8_8_r00_k6",
         f"8:8 · $N$ = {peer_n}", A_CAP[0]),
        (A_X[1], A_TOP[0], f"sparse_peer{peer_n}_antipodal_9_7_r00_k6",
         f"9:7 · $N$ = {peer_n}", A_CAP[0]),
        (A_X[0], A_TOP[1], f"sparse_peer{peer_n}_antipodal_10_6_r00_k6",
         f"10:6 · $N$ = {peer_n}", A_CAP[1]),
        (A_X[1], A_TOP[1], "sparse_N8_unimodal_k6", "unimodal · $N$ = 8", A_CAP[1]),
    )
    for left, top, profile, caption, cap_y in discs:
        _disc(fig, left, top, profile, caption, cap_y)
    _para(
        fig, COL_X[0], A_NOTE_Y,
        "dot = one peer, stacked inwards where a 24-bin cell holds several; "
        "grey triangle = focal reference at relative phase 0. Realization r00 "
        "shown; the antipodal fields are $\\kappa$ = 6 partner-paired draws.",
        R1_W[0] + 3.0,
    )

    # ======================================================================
    # b  trinomial response by sparse field, per independent seed block
    # ======================================================================
    axB = mm_axes(fig, COL_X[1], B_TOP, R1_W[1], B_H)
    panel_label_at(fig, COL_X[1] - 8.6, LETTER_Y[0], "b",
                   "Trinomial response by sparse field")
    blocks = sorted(endpoints["block"].unique())
    bar_w = 0.40
    group_step = 4.0
    tallest = (0.0, 0.0, 0.5, 2)
    for g, imb in enumerate(IMB):
        for i, rep in enumerate(REP_ORDER):
            for k, block in enumerate(blocks):
                sub = endpoints[(endpoints["imbalance"] == imb)
                                & (endpoints["representation"] == rep)
                                & (endpoints["block"] == block)]
                p = _trinomial(float(sub["activity"].mean()), float(sub["a0"].mean()))
                x = g * group_step + i * 1.0 + (k - 0.5) * (bar_w + 0.06)
                bottom = 0.0
                for c, colour in zip(range(3), (ACTION["p_minus"], ACTION["p_zero"],
                                                ACTION["p_plus"])):
                    axB.bar([x], [p[c]], bottom=[bottom], width=bar_w, color=colour,
                            edgecolor="white", linewidth=LW_HAIR, zorder=3)
                    if p[c] > tallest[0]:
                        tallest = (p[c], x, bottom + p[c] / 2.0, c)
                    bottom += p[c]
            # representation identity as a coloured rule under its bar pair
            xc = g * group_step + i * 1.0
            axB.plot([xc - 0.46, xc + 0.46], [-0.035, -0.035], color=REP[rep],
                     lw=LW_TRACE, solid_capstyle="butt", clip_on=False, zorder=4)
    axB.set_xlim(-0.85, 2 * group_step + 2.85)
    axB.set_ylim(0.0, 1.0)
    axB.set_yticks([0.0, 0.5, 1.0])
    axB.set_ylabel("probability")
    axB.set_xticks([g * group_step + i for g in range(3) for i in range(3)])
    axB.set_xticklabels([REP_ABBR[r] for _ in range(3) for r in REP_ORDER],
                        fontsize=FS_TINY)
    for tick, rep in zip(axB.get_xticklabels(), [r for _ in range(3) for r in REP_ORDER]):
        tick.set_color(REP[rep])
    axB.tick_params(axis="x", length=0.0, pad=3.4)
    for g, imb in enumerate(IMB):
        axB.text(g * group_step + 1.0, B_GROUP_Y, IMB_LABEL[imb], ha="center",
                 va="baseline", fontsize=FS_BODY, color=INK, clip_on=False)
    despine(axB)
    trim_spines(axB, x=False)
    axB.spines["bottom"].set_visible(False)
    # direct segment labels instead of a legend box
    _, x_t, y_t, c_t = tallest
    axB.text(x_t, y_t, ACTION_LABELS[c_t], ha="center", va="center",
             color="white" if c_t != 1 else INK, fontsize=FS_TINY, zorder=6)
    key_x = 2 * group_step + 2.0 + 0.75
    # each label sits against its own band: p_+ is the navy top of the stack,
    # p_0 the grey middle, p_- the magenta foot (same grammar as Fig. S6b).
    for lab, colour, yy in zip(ACTION_LABELS,
                               (ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"]),
                               (0.14, 0.50, 0.86)):
        axB.text(key_x, yy, lab, ha="left", va="center", fontsize=FS_TINY,
                 color=INK if colour == ACTION["p_zero"] else colour, clip_on=False)
    _para(fig, COL_X[1], B_NOTE_Y,
          f"left / right bar = seed block 1 / 2 (independent estimates); "
          f"each bar averages {n_real} realizations",
          R1_W[1])

    # ======================================================================
    # c  peer-count dependence within one field family (spliced acquisitions)
    # ======================================================================
    panel_label_at(fig, COL_X[2] - 8.6, LETTER_Y[0], "c", "Peer-count dependence")

    def _atlas_row(profile: str, rep: str) -> tuple[float, float]:
        row = atlas[(atlas["representation"] == ATLAS_SHORT[rep])
                    & (atlas["profile"] == profile)].iloc[0]
        return float(row["activity"]), float(row["a0"])

    # condition 0: this campaign (N = 16 partner-paired 8:8, 6 realizations)
    # conditions 1-2: the stimulus-manifold campaign, same antipodal kappa = 6 family
    cond_labels = (f"{peer_n}\n8:8 paired", f"{peer_n}\ninteger", "240\ndense")
    cond_open = (True, False, False)
    series_A: dict = {}
    series_a1: dict = {}
    tv_by_cond: list = []
    for rep in REP_ORDER:
        sub = realization[(realization["imbalance"] == "8_8")
                          & (realization["representation"] == rep)]
        series_A[rep] = [(float(sub["activity"].mean()), list(sub["activity"]))]
        series_a1[rep] = [(float(sub["a1"].mean()), list(sub["a1"]))]
    for profile in ("sparse_N16_antipodal_k6", "antipodal_equal_k6"):
        for rep in REP_ORDER:
            act, _a0 = _atlas_row(profile, rep)
            row = atlas[(atlas["representation"] == ATLAS_SHORT[rep])
                        & (atlas["profile"] == profile)].iloc[0]
            series_A[rep].append((act, [act]))
            series_a1[rep].append((float(row["a1"]), [float(row["a1"])]))
    probs0 = {}
    for rep in REP_ORDER:
        sub = endpoints[(endpoints["imbalance"] == "8_8")
                        & (endpoints["representation"] == rep)]
        probs0[rep] = _trinomial(float(sub["activity"].mean()), float(sub["a0"].mean()))
    tv_by_cond.append(_pair_tv(probs0))
    for profile in ("sparse_N16_antipodal_k6", "antipodal_equal_k6"):
        probs = {rep: _trinomial(*_atlas_row(profile, rep)) for rep in REP_ORDER}
        tv_by_cond.append(_pair_tv(probs))

    axC = [mm_axes(fig, COL_X[2], t, R1_W[2], C_H) for t in C_TOP]
    for ax in axC:
        ax.add_patch(mpatches.Rectangle((-0.55, 0.0), 1.05, 1.0, transform=
                                        ax.get_xaxis_transform(), facecolor="#F2F2F2",
                                        edgecolor="none", zorder=0))
        ax.set_xlim(-0.55, 2.55)
        despine(ax)
    for ax, store, lo, hi, ylab, ticks in (
        (axC[0], series_A, 0.0, 1.06, "$A$", [0.0, 0.5, 1.0]),
        (axC[1], series_a1, -0.17, 0.17, "$a_1$", [-0.1, 0.0, 0.1]),
    ):
        for rep in REP_ORDER:
            xs, ys = [], []
            for j, (mean, pts) in enumerate(store[rep]):
                x = j + (REP_ORDER.index(rep) - 1) * 0.16
                xs.append(x)
                ys.append(mean)
                if len(pts) > 1:
                    ax.plot(swarm_x(pts, x, width=0.055), pts, "o", color=REP[rep],
                            ms=MS_POINT, mew=0.0, alpha=0.45, zorder=3)
                ax.plot([x], [mean], "o" if not cond_open[j] else "s",
                        color=REP[rep] if not cond_open[j] else "white",
                        markeredgecolor=REP[rep], markeredgewidth=LW_LINE,
                        ms=MS_SERIES, zorder=5)
        if lo < 0.0:
            ax.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=1)
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks)
        ax.set_ylabel(ylab)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0.0)
        trim_spines(ax, x=False)
    axC[1].set_yticklabels(["−0.1", "0", "0.1"], fontsize=FS_TICK)
    # third mini: pairwise representation total variation
    for j, (pairs, mean) in enumerate(tv_by_cond):
        axC[2].plot([j] * len(pairs), pairs, "o", color=MUTED, ms=MS_POINT,
                    mew=0.0, alpha=0.6, zorder=3)
        axC[2].plot([j], [mean], "D", color=INK, ms=MS_SERIES, mew=0.0, zorder=5)
    axC[2].set_ylim(0.0, 0.68)
    axC[2].set_yticks([0.0, 0.3, 0.6])
    axC[2].set_ylabel("$d_{\\mathrm{TV}}$")
    axC[2].set_xticks([0, 1, 2])
    axC[2].set_xticklabels(cond_labels, fontsize=FS_TINY, linespacing=1.25)
    axC[2].tick_params(axis="x", length=0.0, pad=1.6)
    axC[2].set_xlabel("peer count $N$", labelpad=1.0)
    trim_spines(axC[2], x=False)
    # direct series labels in the reserved right-hand gutter of panel c
    for rep, yy in (("moments_m1_m3", 0.10),
                    ("centers_24_standard", 0.44),
                    ("intervals_24_decimal6", 0.74)):
        axC[0].text(C_LAB_X, yy, REP_SHORT[rep], color=REP[rep], fontsize=FS_TINY,
                    fontweight="bold", ha="left", va="center", clip_on=False,
                    transform=axC[0].get_yaxis_transform())
    _para(fig, COL_X[2], C_NOTE_Y,
          "open = this campaign (tinted) · filled = stimulus-manifold campaign; "
          "spliced sources, not a controlled sweep. Antipodal $\\kappa$ = 6 only; "
          "no peer-8 antipodal field exists, so $N$ = 8 is not shown.",
          X_RIGHT - COL_X[2])

    # ======================================================================
    # d  variance decomposition
    # ======================================================================
    panel_label_at(fig, COL_X[0] - 8.6, LETTER_Y[1], "d", "Variance decomposition")
    comp_style = (
        ("representation effect", INK, "s"),
        ("physical realization", MUTED, "o"),
        ("seed block (LLM re-sampling)", "#B4B4B4", "^"),
    )
    axD = [mm_axes(fig, COL_X[0], t, R2_W[0], R2_H) for t in R2_TOP]
    for ax, metric, ylab in ((axD[0], "activity", "SD of $A$"),
                             (axD[1], "a1", "SD of $a_1$")):
        for g, imb in enumerate(IMB):
            per_rep_block = endpoints[endpoints["imbalance"] == imb].groupby(
                ["representation", "realization"])[metric].agg(_sd)
            block_sd = float(np.sqrt(np.mean(np.asarray(per_rep_block.values) ** 2)))
            per_rep_real = realization[realization["imbalance"] == imb].groupby(
                "representation")[metric].agg(_sd)
            real_sd = float(np.sqrt(np.mean(np.asarray(per_rep_real.values) ** 2)))
            rep_means = realization[realization["imbalance"] == imb].groupby(
                "representation")[metric].mean()
            rep_sd = _sd(rep_means.values)
            for value, (_lab, colour, marker) in zip((rep_sd, real_sd, block_sd),
                                                     comp_style):
                ax.plot([g, g], [1e-3, value], "-", color=colour, lw=LW_THIN, zorder=2)
                ax.plot([g], [value], marker, color=colour, ms=MS_SERIES,
                        mew=0.0, zorder=4)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 2.0)
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
        ax.set_yticklabels(["0.001", "0.01", "0.1", "1"], fontsize=FS_TICK)
        # decade subticks only inside the drawn spine (1e-3 to 1): the default
        # log minor locator also dashes 2.0, above the top of the spine.
        ax.yaxis.set_minor_locator(mticker.FixedLocator(
            [d * 10.0 ** e for e in (-3, -2, -1) for d in range(2, 10)]))
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([IMB_LABEL[i] for i in IMB], fontsize=FS_TICK)
        despine(ax)
        trim_spines(ax, x=False)
    axD[1].set_xlabel("count imbalance", labelpad=1.0)
    axD[0].set_xticklabels([])

    # ======================================================================
    # e  exact-balance abstention, realization by realization
    # ======================================================================
    panel_label_at(fig, COL_X[1] - 8.6, LETTER_Y[1], "e",
                   "Exact balance (8:8): abstention")
    axE = [mm_axes(fig, COL_X[1], t, R2_W[1], R2_H) for t in R2_TOP]
    ref_profile = "sparse_N16_antipodal_k6"
    for ax, metric, ylab, hi, yticks in (
            (axE[0], "activity", "$A$", 0.86, [0.0, 0.2, 0.4, 0.6, 0.8]),
            (axE[1], "R1", "$R_1$", 0.30, [0.0, 0.1, 0.2])):
        for i, rep in enumerate(REP_ORDER):
            sub = realization[(realization["imbalance"] == "8_8")
                              & (realization["representation"] == rep)]
            vals = np.asarray(sub[metric], dtype=float)
            ax.plot(swarm_x(vals, i - 0.16, width=0.075), vals, "o", color=REP[rep],
                    ms=MS_POINT, mew=0.0, alpha=0.6, zorder=3)
            mean, lo, hi_ci = paired_ci(vals)
            ax.errorbar([i - 0.16], [mean], yerr=[[mean - lo], [hi_ci - mean]],
                        fmt="D", color=REP[rep], markersize=MS_MEAN, lw=LW_LINE,
                        capsize=CAPSIZE, capthick=LW_LINE, zorder=5)
            row = atlas[(atlas["representation"] == ATLAS_SHORT[rep])
                        & (atlas["profile"] == ref_profile)].iloc[0]
            ref = float(row["activity"]) if metric == "activity" else float(row["R1"])
            ax.plot([i + 0.24], [ref], "s", color="white", markeredgecolor=REP[rep],
                    markeredgewidth=LW_LINE, ms=MS_SERIES, zorder=5)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(0.0, hi)
        # explicit ticks: the auto locator emits 0.30000000000000004 on the
        # R1 axis, which falls outside the limit and orphans its dash above
        # the top of the spine.
        ax.set_yticks(yticks)
        ax.set_ylabel(ylab)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([REP_SHORT[r] for r in REP_ORDER], fontsize=FS_TICK)
        for tick, rep in zip(ax.get_xticklabels(), REP_ORDER):
            tick.set_color(REP[rep])
        ax.tick_params(axis="x", length=0.0)
        despine(ax)
        trim_spines(ax, x=False)
    axE[0].set_xticklabels([])

    # ======================================================================
    # f  one-peer majority activation
    # ======================================================================
    panel_label_at(fig, COL_X[2] - 8.6, LETTER_Y[1], "f",
                   "One-peer majority activation")
    axF = [mm_axes(fig, COL_X[2], t, R2_W[2], R2_H) for t in R2_TOP]
    for ax, metric, ylab, lo, hi, ticks in (
        (axF[0], "activity", "$A$", 0.0, 1.12, [0.0, 0.5, 1.0]),
        (axF[1], "a1", "$a_1$", -0.45, 1.35, [0.0, 0.5, 1.0]),
    ):
        for rep in REP_ORDER:
            means = []
            for g, imb in enumerate(IMB):
                sub = realization[(realization["imbalance"] == imb)
                                  & (realization["representation"] == rep)]
                vals = np.asarray(sub[metric], dtype=float)
                x = g + (REP_ORDER.index(rep) - 1) * 0.17
                ax.plot(swarm_x(vals, x, width=0.055), vals, "o", color=REP[rep],
                        ms=MS_POINT, mew=0.0, alpha=0.45, zorder=3)
                means.append((x, float(vals.mean())))
            ax.plot([m[0] for m in means], [m[1] for m in means], "-o", color=REP[rep],
                    lw=LW_LINE, ms=MS_SERIES, mew=0.0, zorder=5)
        if lo < 0.0:
            ax.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=1)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks)
        ax.set_ylabel(ylab)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([IMB_LABEL[i] for i in IMB], fontsize=FS_TICK)
        despine(ax)
        trim_spines(ax, x=False)
    axF[0].set_xticklabels([])
    axF[1].set_xlabel("count imbalance", labelpad=1.0)
    for rep, yy in (("moments_m1_m3", 0.05),
                    ("centers_24_standard", 0.72),
                    ("intervals_24_decimal6", 0.52)):
        axF[0].text(-0.42, yy, REP_SHORT[rep], color=REP[rep], fontsize=FS_TINY,
                    fontweight="bold", ha="left", va="center")

    # ======================================================================
    # shared key for the second row: one block, no overlapping legends
    # ======================================================================
    y = _key_row(fig, X_LEFT, KEY_Y, "d",
                 [(m, c, lab, {}) for lab, c, m in comp_style],
                 X_RIGHT - X_LEFT)
    _key_row(fig, X_LEFT, y, "e",
             [("D", INK, f"mean ± 95% CI over {n_real} realizations", {}),
              ("o", MUTED, "small = one realization",
               {"ms": MS_POINT, "alpha": 0.6}),
              ("s", MUTED, f"open = integer-count $N$ = {peer_n} antipodal field "
                           "(other acquisition)",
               {"markerfacecolor": "white", "markeredgecolor": MUTED,
                "markeredgewidth": LW_LINE})],
             X_RIGHT - X_LEFT)

    # ======================================================================
    m88 = realization[(realization["imbalance"] == "8_8")
                      & (realization["representation"] == "moments_m1_m3")]
    m97 = realization[(realization["imbalance"] == "9_7")
                      & (realization["representation"] == "moments_m1_m3")]
    rms88 = float(within[(within["imbalance"] == "8_8")
                         & (within["representation"] == "moments_m1_m3")]["activity_rms"].iloc[0])
    y = FOOT_Y
    for para in (
        f"Exact balance is stable, not a one-draw fluke: moments $A$ = "
        f"{m88['activity'].min():.3f}–{m88['activity'].max():.3f} across "
        f"realizations (SD {rms88:.3f}), against $A$ = {m97['activity'].mean():.3f} "
        f"at 9:7. Realizations are independent draws at each imbalance, so lines "
        f"in f join means, not matched fields.",
    ):
        y = _para(fig, X_LEFT, y, para, X_RIGHT - X_LEFT, fontsize=FS_SMALL)

    _overflow(fig)
    return save_fig(fig, "figS07_sparse_finite_peer")
