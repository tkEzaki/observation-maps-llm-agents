"""Supplementary Figure S9 - Robustness of the GPT collective phenotype classification.

The main-text headline (Fig. 1c) is an *operational* statement: with the frozen
rule ``final r1 > 0.9`` the moments representation locks on 6/6 seeds at every
positive coupling while centers and intervals lock on 0/6, giving an exact
paired sign-test p = 2/2^6. This figure asks whether that statement survives
the choices buried in the rule.

Panels

* a  polar-lock threshold sweep - lock fraction against the final-``r1``
  threshold, over a prespecified band around the frozen 0.9;
* b  terminal-lock window sweep - lock fraction against the length ``W`` of
  the trailing window over which ``r1`` must stay above threshold;
* c  first passage versus terminal-lock onset - whether a crossing of 0.9 is
  permanent or transient, run by run;
* d  the frozen harmonic-relative summary ``Q2 = r2 - r1`` beside the one
  alternative the artifacts already carry, ``Q2|r1 = r2 - r1^2``;
* e  frozen phenotype counts for every representation and coupling;
* f  every seed-level endpoint contrast, one row per endpoint and one column
  per representation contrast.

Data honesty
------------
Panels a-d are *re-classifications* of the frozen trajectories
(``_tmp_session_{0,1,2}/trajectory_harmonics_timeseries.csv``), not new
analyses. The classification functions are imported from the frozen analysis
module ``analysis.matched_rep_collective.analyze_paired_endpoints`` rather than
re-implemented here, and the module asserts at build time that the
re-classification at the frozen setting reproduces ``cluster_phenotype``,
``locking`` and ``sustained_lock`` in ``endpoints_long.csv`` cell for cell, and
that the recomputed exact sign-test p reproduces ``exact_sign_locking.csv``.
A mismatch raises; the threshold is never tuned to make it agree.

Panels e and f read the frozen columns directly. Every printed number is
computed from an artifact.

This is a sensitivity analysis around a prespecified rule, not a search for a
better threshold - no panel reports an optimum and none is claimed.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator

from analysis.figures_si.style import (
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
    MUTED,
    PHENOTYPE,
    PHENOTYPE_LABEL,
    REP,
    REP_ABBR,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    W_FULL_MM,
    apply_style,
    direct_label,
    errorbar_mean,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label,
    panel_label_at,
    rep_tick_colors,
    save_fig,
    swarm_x,
    trim_spines,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The frozen classifiers themselves, not a local copy of them.
from analysis.matched_rep_collective.analyze_paired_endpoints import (  # noqa: E402
    _cluster_phenotype,
    _sustained_lock_time,
)

PAIRED = ROOT / "analysis" / "matched_rep_collective" / "paired_analysis"

# The frozen operational rule. These are the *rule*, not statistics: they are
# the numbers this figure sweeps around, and they are checked against the
# frozen artifact before anything is drawn.
THR_FROZEN = 0.9
W_FROZEN = 1          # the frozen suffix rule is satisfied by a single step
K_POS = (0.08, 0.15)
K_ALL = (-0.15, 0.0, 0.08, 0.15)
N_SEEDS = 6

# Scatter takes an area; derive it from the shared marker diameter.
S_POINT = MS_POINT ** 2
S_SMALL = (MS_POINT * 0.82) ** 2

CONTRASTS = [
    ("centers_24_standard", "moments_m1_m3"),
    ("intervals_24_decimal6", "moments_m1_m3"),
    ("intervals_24_decimal6", "centers_24_standard"),
]
ENDPOINTS = [
    ("delta_final_r1", "final $r_1$"),
    ("delta_mean_r1", "mean $r_1$"),
    ("delta_final_Q2", "final $Q_2$"),
    ("delta_sustained_lock", "terminal\nlock"),
    ("delta_mean_activity", "mean\nactivity"),
    ("delta_mean_tau_social", "mean social\ntorque"),
]

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm page)
# ---------------------------------------------------------------------------
# Page margins the layout must respect on all four sides. The foot notes are
# placed against M_FOOT rather than against the canvas edge, and every free
# text block below is given an explicit width budget derived from M_SIDE.
M_SIDE = 4.0
M_FOOT = 4.8

H_MM = 175.5

R1_TOP, R1_H = 9.0, 29.0
R1_LEFT = (14.0, 72.0, 130.0)
R1_W = 44.0
LETTER_DX = -9.0

R2_TOP, R2_H = 52.0, 27.0
D_LEFT, D_W = 14.0, 60.0
E_LEFT, E_W = 92.0, 82.0
E_K_Y = 82.3
E_REP_Y = 88.3
E_LEG_TOP, E_LEG_H = 89.9, 4.2

F_LETTER_Y = 95.9
F_TITLE_Y = 99.3
F_TOP = 101.0
F_COL_LEFT = (26.0, 78.0, 130.0)
F_COL_W = 44.0
F_ROW_H, F_ROW_GAP = 7.4, 2.3
F_LABEL_X = 20.0

# Foot notes: an explicit width budget rather than "start at 4 mm and hope".
NOTE_BUDGET_MM = W_FULL_MM - 2 * M_SIDE
NOTE_LS = 1.42


def _minus(x: float) -> str:
    """Number as a tick label with a true U+2212 minus, not a hyphen."""
    return f"{x:g}".replace("-", "−")


# ---------------------------------------------------------------------------
# Text metrics — measure with the renderer instead of guessing at widths
# ---------------------------------------------------------------------------
def _renderer(fig):
    canvas = fig.canvas
    if not hasattr(canvas, "get_renderer"):
        canvas = FigureCanvasAgg(fig)
    return canvas.get_renderer()


def _wrap_mm(fig, text: str, width_mm: float, fontsize: float, **kw) -> str:
    """Greedy word wrap to an explicit width budget, measured in millimetres.

    Assumes no spaces inside ``$...$`` spans, which holds for every string
    this figure wraps.
    """
    rend = _renderer(fig)
    probe = fig.text(0.0, -1.0, "", fontsize=fontsize, **kw)
    # An em dash never opens a line: bind it to the word in front of it.
    words: list[str] = []
    for w in text.split():
        if w == "—" and words:
            words[-1] = f"{words[-1]} —"
        else:
            words.append(w)
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        probe.set_text(trial)
        wide = probe.get_window_extent(renderer=rend).width / fig.dpi * 25.4
        if cur and wide > width_mm:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    probe.remove()
    return "\n".join(lines)


def _line_mm(fontsize: float, linespacing: float = 1.2) -> float:
    """Baseline-to-baseline advance of wrapped text, in millimetres."""
    return fontsize * linespacing / 72.0 * 25.4


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_runs() -> dict[tuple[str, float, int], dict[str, np.ndarray]]:
    """r1/r2 time series for all 72 runs, keyed by (representation, K, seed)."""
    runs: dict[tuple[str, float, int], dict[str, np.ndarray]] = {}
    for i, rep in enumerate(REP_ORDER):
        ts = pd.read_csv(
            PAIRED / f"_tmp_session_{i}" / "trajectory_harmonics_timeseries.csv"
        )
        for (k, s), g in ts.groupby(["coupling", "seed_index"]):
            g = g.sort_values("t")
            runs[(rep, float(k), int(s))] = {
                "r1": g["r1"].to_numpy(dtype=float),
                "r2": g["r2"].to_numpy(dtype=float),
            }
    return runs


def _lock(runs, rep, k, s, thr: float, window: int = W_FROZEN) -> bool:
    """Generalised lock rule: r1 > thr over the trailing ``window`` steps.

    ``window = 1`` is the frozen rule: ``_sustained_lock_time`` returns a finite
    onset exactly when the final sample is above threshold.
    """
    r1 = runs[(rep, k, s)]["r1"]
    return bool(np.all(r1[-window:] > thr))


def _lock_fraction(runs, rep, thr: float, window: int = W_FROZEN,
                   ks=K_POS) -> float:
    vals = [_lock(runs, rep, k, s, thr, window) for k in ks for s in range(N_SEEDS)]
    return float(np.mean(vals))


def _exact_sign_p(ref_locks: np.ndarray, other_locks: np.ndarray) -> float:
    """Two-sided exact sign test on the paired lock indicators (frozen form)."""
    n_pos = int(np.sum((ref_locks == 1) & (other_locks == 0)))
    n_neg = int(np.sum((ref_locks == 0) & (other_locks == 1)))
    n_nz = n_pos + n_neg
    if n_nz == 0:
        return 1.0
    k = max(n_pos, n_neg)
    tail = sum(comb(n_nz, i) for i in range(k, n_nz + 1)) / (2 ** n_nz)
    return min(1.0, 2.0 * tail)


def _verify_frozen(runs, ep: pd.DataFrame) -> dict:
    """Re-classify at the frozen setting and check it against the artifact.

    Raises rather than adjusting anything. Returns the frozen headline numbers
    it verified, so the figure can print them from the recomputation.
    """
    bad: list[str] = []
    for (rep, k, s), tr in runs.items():
        row = ep[
            (ep["representation"] == rep)
            & (np.isclose(ep["coupling"], k))
            & (ep["seed_index"] == s)
        ].iloc[0]
        pheno = _cluster_phenotype(
            {"final_r1": tr["r1"][-1], "final_r2": tr["r2"][-1]}
        )
        lock = int(tr["r1"][-1] > THR_FROZEN)
        sustained = int(np.isfinite(_sustained_lock_time(tr["r1"], THR_FROZEN)))
        if pheno != row["cluster_phenotype"]:
            bad.append(f"{rep} K={k} s{s}: phenotype {pheno} != {row['cluster_phenotype']}")
        if lock != int(row["locking"]):
            bad.append(f"{rep} K={k} s{s}: locking {lock} != {int(row['locking'])}")
        if sustained != int(row["sustained_lock"]):
            bad.append(
                f"{rep} K={k} s{s}: sustained {sustained} != {int(row['sustained_lock'])}"
            )
    if bad:
        raise RuntimeError(
            "S9: re-classification at the frozen threshold does NOT reproduce the "
            "frozen phenotype table; refusing to draw.\n" + "\n".join(bad[:12])
        )

    # exact sign test, recomputed from the re-classification
    sign = pd.read_csv(PAIRED / "exact_sign_locking.csv")
    p_pos: list[float] = []
    for k in K_POS:
        ref = np.array(
            [int(_lock(runs, REP_ORDER[0], k, s, THR_FROZEN)) for s in range(N_SEEDS)]
        )
        for other in REP_ORDER[1:]:
            oth = np.array(
                [int(_lock(runs, other, k, s, THR_FROZEN)) for s in range(N_SEEDS)]
            )
            p = _exact_sign_p(ref, oth)
            frozen_p = float(
                sign[
                    (np.isclose(sign["coupling"], k))
                    & (sign["contrast"] == f"{other}__minus__{REP_ORDER[0]}")
                ]["exact_sign_p_two_sided"].iloc[0]
            )
            if not np.isclose(p, frozen_p):
                raise RuntimeError(
                    f"S9: recomputed exact sign p={p} != frozen {frozen_p} "
                    f"(K={k}, {other}); refusing to draw."
                )
            p_pos.append(p)

    n_lock = {
        rep: int(sum(_lock(runs, rep, k, s, THR_FROZEN)
                     for k in K_POS for s in range(N_SEEDS)))
        for rep in REP_ORDER
    }
    return {
        "p_exact": float(np.unique(np.round(p_pos, 12))[0]),
        "n_lock_pos": n_lock,
        "n_pos_runs": len(K_POS) * N_SEEDS,
    }


# ---------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()

    runs = _load_runs()
    ep = pd.read_csv(PAIRED / "endpoints_long.csv")
    pc = pd.read_csv(PAIRED / "paired_contrasts.csv")
    frozen = _verify_frozen(runs, ep)

    fig = new_figure(H_MM)

    # =====================================================================
    # a  polar-lock threshold sweep
    # =====================================================================
    axA = mm_axes(fig, R1_LEFT[0], R1_TOP, R1_W, R1_H)
    panel_label(axA, "a", "Polar-lock threshold sweep", dx_mm=LETTER_DX)

    thr_lo, thr_hi = 0.70, 1.00
    thr_grid = np.linspace(thr_lo, thr_hi, 301)

    finals_pos = {
        rep: np.array([runs[(rep, k, s)]["r1"][-1]
                       for k in K_POS for s in range(N_SEEDS)])
        for rep in REP_ORDER
    }
    # the band of thresholds over which the 6/6-versus-0/6 split is unchanged
    band_lo = float(max(finals_pos[r].max() for r in REP_ORDER[1:]))
    band_hi = float(finals_pos[REP_ORDER[0]].min())
    axA.axvspan(band_lo, band_hi, ymax=0.885, color="#EDEDED", lw=0, zorder=0)
    axA.text((band_lo + band_hi) / 2.0, 0.62,
             "6/6 vs 0/6 at\n"
             r"each $K>0$ for" "\n"
             f"{band_lo:.3f}–{band_hi:.3f}",
             ha="center", va="center", fontsize=FS_TINY, color=MUTED,
             linespacing=1.35, zorder=5,
             bbox=dict(facecolor="#EDEDED", edgecolor="none", pad=1.0))

    for rep in REP_ORDER:
        frac = [_lock_fraction(runs, rep, t) for t in thr_grid]
        axA.plot(thr_grid, frac, color=REP[rep], lw=LW_TRACE,
                 solid_joinstyle="miter", zorder=3)
    axA.axvline(THR_FROZEN, color=INK, lw=LW_HAIR, ls=(0, (2.4, 1.6)), zorder=4)
    axA.text(THR_FROZEN - 0.006, 1.10, f"prespecified {THR_FROZEN:g}", ha="right",
             va="center", fontsize=FS_TINY, color=INK)

    direct_label(axA, thr_lo + 0.006, 1.055, REP_SHORT[REP_ORDER[0]],
                 REP[REP_ORDER[0]], fontsize=FS_SMALL)
    direct_label(axA, thr_lo + 0.006, 0.925, REP_SHORT[REP_ORDER[1]],
                 REP[REP_ORDER[1]], fontsize=FS_SMALL)
    direct_label(axA, thr_lo + 0.006, 0.295, REP_SHORT[REP_ORDER[2]],
                 REP[REP_ORDER[2]], fontsize=FS_SMALL)

    axA.set_xlim(thr_lo, thr_hi)
    axA.set_ylim(-0.04, 1.16)
    axA.set_xticks([0.70, 0.80, 0.90, 1.00])
    axA.set_yticks([0, 0.5, 1.0])
    axA.set_xlabel(r"final-$r_1$ lock threshold")
    axA.set_ylabel(r"lock fraction, $K>0$")
    trim_spines(axA)

    # =====================================================================
    # b  terminal-lock window sweep
    # =====================================================================
    axB = mm_axes(fig, R1_LEFT[1], R1_TOP, R1_W, R1_H)
    panel_label(axB, "b", "Minimum-duration sweep", dx_mm=LETTER_DX)

    windows = np.arange(1, 101)
    frac_w = {
        rep: np.array([_lock_fraction(runs, rep, THR_FROZEN, int(w))
                       for w in windows])
        for rep in REP_ORDER
    }
    for rep in REP_ORDER:
        axB.plot(windows, frac_w[rep], color=REP[rep], lw=LW_TRACE, zorder=3)
    w_all = int(windows[frac_w[REP_ORDER[0]] == 1.0].max())
    w_only = int(
        windows[
            (frac_w[REP_ORDER[0]] > 0)
            & (frac_w[REP_ORDER[1]] == 0)
            & (frac_w[REP_ORDER[2]] == 0)
        ].max()
    )
    axB.axvline(W_FROZEN, color=INK, lw=LW_HAIR, ls=(0, (2.4, 1.6)), zorder=4)
    axB.text(W_FROZEN + 2.5, 1.12, f"prespecified $W={W_FROZEN}$", ha="left",
             va="center", fontsize=FS_TINY, color=INK)
    axB.axvline(w_all, color=MUTED, lw=LW_HAIR, ls=(0, (1.2, 1.4)), zorder=2)
    axB.text(0.36, 0.58,
             f"all {frozen['n_pos_runs']} moments runs locked\n"
             rf"for $W\leq{w_all}$; moments-only" "\n"
             rf"locking for every $W\leq{w_only}$",
             transform=axB.transAxes, ha="center", va="center",
             fontsize=FS_TINY, color=MUTED, linespacing=1.35)

    axB.set_xlim(0, 101)
    axB.set_ylim(-0.04, 1.16)
    axB.set_xticks([1, 25, 50, 75, 100])
    axB.set_yticks([0, 0.5, 1.0])
    axB.set_xlabel(r"trailing window $W$ (steps) with $r_1>0.9$")
    axB.set_ylabel(r"lock fraction, $K>0$")
    trim_spines(axB)

    # =====================================================================
    # c  first passage versus terminal-lock onset
    # =====================================================================
    axC = mm_axes(fig, R1_LEFT[2], R1_TOP, R1_W, R1_H)
    panel_label(axC, "c", "Transient or permanent crossing", dx_mm=LETTER_DX)

    y_never = 118.0
    markers = {K_POS[0]: "o", K_POS[1]: "s"}
    n_cross = n_perm = 0
    for rep in REP_ORDER:
        for k in K_POS:
            for s in range(N_SEEDS):
                r1 = runs[(rep, k, s)]["r1"]
                hits = np.flatnonzero(r1 > THR_FROZEN)
                if hits.size == 0:
                    continue
                n_cross += 1
                fp = float(hits[0])
                onset = _sustained_lock_time(r1, THR_FROZEN)
                if np.isfinite(onset):
                    n_perm += 1
                    axC.scatter([fp], [onset], marker=markers[k], s=S_POINT,
                                c=REP[rep], linewidths=0, zorder=4, clip_on=False)
                else:
                    axC.scatter([fp], [y_never], marker=markers[k], s=S_POINT,
                                facecolors="none", edgecolors=REP[rep],
                                linewidths=LW_THIN, zorder=4, clip_on=False)
    axC.plot([0, 100], [0, 100], color="#BBBBBB", lw=LW_HAIR,
             ls=(0, (2.4, 1.8)), zorder=1)
    axC.axhline(108.0, color=RULE, lw=LW_HAIR, zorder=1)
    # Right-aligned inside the data box: set left-aligned at x=70 this label
    # ran past the canvas edge, since panel c is the last column on the page.
    axC.text(97, 46, "onset = first passage", ha="right", va="center",
             fontsize=FS_TINY, color=MUTED)
    axC.text(2, 128.0, "transient crossing, never terminal",
             ha="left", va="center", fontsize=FS_TINY, color=MUTED)

    n_total = len(K_ALL) * N_SEEDS * len(REP_ORDER)
    n_runs_pos = len(K_POS) * N_SEEDS * len(REP_ORDER)
    axC.text(3, 92,
             f"{n_perm}/{n_cross} crossings permanent\n"
             f"{n_runs_pos - n_cross}/{n_runs_pos} runs never cross",
             ha="left", va="top", fontsize=FS_TINY, color=INK,
             linespacing=1.35)

    axC.set_xlim(0, 100)
    axC.set_ylim(0, 134)
    axC.set_xticks([0, 50, 100])
    axC.set_yticks([0, 50, 100, y_never])
    axC.set_yticklabels(["0", "50", "100", "never"])
    axC.set_xlabel(r"first passage of $r_1>0.9$  ($t$)")
    axC.set_ylabel(r"terminal-lock onset ($t$)")
    axC.spines["left"].set_bounds(0, 100)
    axC.spines["bottom"].set_bounds(0, 100)
    # The "never" strip is a break-out above the y spine (which stops at 100),
    # so its tick mark had nothing to sit on and floated in white paper. Keep
    # the row label, drop the orphan tick.
    for tick, loc in zip(axC.yaxis.get_major_ticks(), axC.get_yticks()):
        if loc > 100:
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
    for k, m in markers.items():
        axC.scatter([], [], marker=m, s=S_POINT, c=MUTED, linewidths=0,
                    label=rf"$K={k}$")
    axC.legend(loc="lower right", frameon=False, fontsize=FS_TINY,
               handletextpad=0.3, borderaxespad=0.0, labelspacing=0.15)

    # =====================================================================
    # d  alternative harmonic-relative summary
    # =====================================================================
    axD = mm_axes(fig, D_LEFT, R2_TOP, D_W, R2_H)
    panel_label(axD, "d", "Harmonic-relative summary definition", dx_mm=LETTER_DX)

    defs = [
        ("final_Q2", r"$Q_2=r_2-r_1$"),
        ("final_Q2_given_r1", r"$Q_2|r_1=r_2-r_1^{2}$"),
    ]
    pos = ep[ep["coupling"] > 0]
    block_dx = 4.0
    order_txt = []
    for b, (col, cap) in enumerate(defs):
        means = {}
        for i, rep in enumerate(REP_ORDER):
            vals = pos[pos["representation"] == rep][col].to_numpy(dtype=float)
            x0 = b * block_dx + i
            axD.scatter(swarm_x(vals, x0, width=0.16), vals, c=REP[rep],
                        s=S_SMALL, linewidths=0, zorder=3)
            m, _, _ = errorbar_mean(axD, x0, vals, color="k", ms=MS_MEAN,
                                    lw=LW_LINE, capsize=CAPSIZE, zorder=4)
            means[rep] = m
        axD.text(b * block_dx + 1, 0.99, cap, ha="center", va="bottom",
                 fontsize=FS_SMALL, color=INK)
        order_txt.append(" < ".join(REP_ABBR[r] for r in
                                    sorted(REP_ORDER, key=lambda r: means[r])))
    axD.axhline(0, color=RULE, lw=LW_HAIR)
    axD.axvline(2.55, color="#DDDDDD", lw=LW_HAIR, ls=(0, (3, 2)))
    for b, txt in enumerate(order_txt):
        axD.text(b * block_dx + 1, -0.42, txt, ha="center", va="center",
                 fontsize=FS_TINY, color=MUTED)
    axD.set_xticks([0, 1, 2, 4, 5, 6])
    axD.set_xticklabels([REP_ABBR[r] for r in REP_ORDER] * 2)
    rep_tick_colors(axD, REP_ORDER * 2)
    axD.set_xlim(-0.6, 6.6)
    axD.set_ylim(-0.55, 1.12)
    axD.set_yticks([-0.25, 0, 0.5, 1.0])
    axD.set_ylabel("value at $t=100$")
    axD.text(0.5, -0.20,
             r"$K>0$ only, $n=12$ runs per encoding" "\n"
             "intervals highest under both; the moments/centers order swaps",
             transform=axD.transAxes, ha="center", va="top",
             fontsize=FS_TINY, color=MUTED, linespacing=1.4)

    # =====================================================================
    # e  frozen phenotype counts by K
    # =====================================================================
    axE = mm_axes(fig, E_LEFT, R2_TOP, E_W, R2_H)
    panel_label(axE, "e", "Prespecified phenotype counts", dx_mm=LETTER_DX)

    ph_order = list(PHENOTYPE.keys())
    xs, xticklabels = [], []
    for j, rep in enumerate(REP_ORDER):
        bottoms = np.zeros(len(K_ALL))
        col_x = np.array([j * 5.0 + kk for kk in range(len(K_ALL))])
        for ph in ph_order:
            counts = np.array([
                int(((ep["representation"] == rep)
                     & (np.isclose(ep["coupling"], k))
                     & (ep["cluster_phenotype"] == ph)).sum())
                for k in K_ALL
            ], dtype=float)
            axE.bar(col_x, counts, bottom=bottoms, width=0.78,
                    color=PHENOTYPE[ph], edgecolor="white", linewidth=LW_HAIR)
            bottoms += counts
        xs.extend(col_x.tolist())
        xticklabels.extend([_minus(k) for k in K_ALL])
    axE.set_xticks(xs)
    axE.set_xticklabels(xticklabels, fontsize=FS_TINY)
    axE.set_xlim(-0.8, 13.8)
    axE.set_ylim(0, N_SEEDS)
    axE.set_yticks([0, 3, 6])
    axE.set_ylabel("runs per cell")
    axE.tick_params(axis="x", length=0, pad=1.2)
    fig_text_mm(fig, E_LEFT - 2.5, E_K_Y, r"$K$", ha="right", va="baseline",
                fontsize=FS_TINY, color=MUTED)
    cell_w = E_W / 14.6
    for j, rep in enumerate(REP_ORDER):
        fig_text_mm(fig, E_LEFT + cell_w * (j * 5.0 + 2.3), E_REP_Y,
                    REP_SHORT[rep], ha="center", va="baseline",
                    color=REP[rep], fontweight="bold", fontsize=FS_SMALL)

    axL = mm_panel(fig, E_LEFT, E_LEG_TOP, E_W, E_LEG_H)
    axL.legend(
        handles=[mpatches.Patch(facecolor=PHENOTYPE[p], edgecolor="none",
                                label=PHENOTYPE_LABEL[p]) for p in ph_order],
        loc="center", ncol=4, frameon=False, fontsize=FS_TINY,
        handlelength=1.0, handleheight=1.0, columnspacing=1.1,
        handletextpad=0.4, borderpad=0.0, borderaxespad=0.0,
    )

    # =====================================================================
    # f  every seed-level endpoint contrast
    # =====================================================================
    panel_label_at(fig, R1_LEFT[0] + LETTER_DX, F_LETTER_Y, "f",
                   "All seed-level endpoint contrasts (six seeds per cell)")
    for c, (other, ref) in enumerate(CONTRASTS):
        cx = F_COL_LEFT[c] + F_COL_W / 2.0
        fig_text_mm(fig, cx, F_TITLE_Y, "$-$", ha="center", va="baseline",
                    fontsize=FS_SMALL, color=INK)
        fig_text_mm(fig, cx - 1.5, F_TITLE_Y, REP_SHORT[other], ha="right",
                    va="baseline", fontsize=FS_SMALL, fontweight="bold",
                    color=REP[other])
        fig_text_mm(fig, cx + 1.5, F_TITLE_Y, REP_SHORT[ref], ha="left",
                    va="baseline", fontsize=FS_SMALL, fontweight="bold",
                    color=REP[ref])

    for r, (col, row_lab) in enumerate(ENDPOINTS):
        top = F_TOP + r * (F_ROW_H + F_ROW_GAP)
        lo = float(pc[col].min())
        hi = float(pc[col].max())
        pad = 0.12 * (hi - lo or 1.0)
        for c, (other, ref) in enumerate(CONTRASTS):
            ax = mm_axes(fig, F_COL_LEFT[c], top, F_COL_W, F_ROW_H)
            sub = pc[pc["contrast"] == f"{other}__minus__{ref}"]
            for ki, k in enumerate(K_ALL):
                vals = sub[np.isclose(sub["coupling"], k)].sort_values(
                    "seed_index")[col].to_numpy(dtype=float)
                ax.scatter(swarm_x(vals, float(ki), width=0.17), vals,
                           c=REP[other], s=S_SMALL, linewidths=0, zorder=3)
                errorbar_mean(ax, float(ki), vals, color="k", ms=MS_POINT,
                              lw=LW_HAIR, capsize=CAPSIZE * 0.7, zorder=4)
            ax.axhline(0, color=RULE, lw=LW_HAIR, zorder=1)
            ax.set_xlim(-0.55, len(K_ALL) - 0.45)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks(range(len(K_ALL)))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(axis="y", labelsize=FS_TINY, pad=1.0)
            ax.tick_params(axis="x", labelsize=FS_TINY, pad=1.2)
            if c > 0:
                ax.tick_params(labelleft=False)
            if r == len(ENDPOINTS) - 1:
                ax.set_xticklabels([_minus(k) for k in K_ALL])
                ax.set_xlabel(r"coupling $K$", fontsize=FS_SMALL, labelpad=1.2)
            else:
                ax.set_xticklabels([])
        fig_text_mm(fig, F_LABEL_X, top + F_ROW_H / 2.0, row_lab, ha="right",
                    va="center", fontsize=FS_TINY, color=INK, linespacing=1.25)

    # ---------------------------------------------------------------------
    # foot notes — wrapped to an explicit width budget and stacked upward
    # from a fixed last baseline, so the block cannot reach either edge
    # whatever the rendered text metrics turn out to be.
    # ---------------------------------------------------------------------
    p_txt = f"{frozen['p_exact']:.5f}"
    n_pos = frozen["n_pos_runs"]
    stat_txt = _wrap_mm(
        fig,
        rf"At the prespecified setting (final $r_1>{THR_FROZEN:g}$, $W={W_FROZEN}$) the "
        "re-classification reproduces the stored phenotype table cell for cell: "
        f"moments {frozen['n_lock_pos'][REP_ORDER[0]]}/{n_pos} locked at "
        rf"positive $K$, centers {frozen['n_lock_pos'][REP_ORDER[1]]}/{n_pos}, "
        f"intervals {frozen['n_lock_pos'][REP_ORDER[2]]}/{n_pos}; "
        rf"exact paired sign $p={p_txt}$ per positive $K$ (6 paired seeds).",
        NOTE_BUDGET_MM, FS_SMALL,
    )
    y_stat = H_MM - M_FOOT
    fig_text_mm(fig, M_SIDE, y_stat, stat_txt, ha="left", va="baseline",
                fontsize=FS_SMALL, color=INK, linespacing=NOTE_LS)

    return save_fig(fig, "figS09_phenotype_robustness")


if __name__ == "__main__":
    print(build())
