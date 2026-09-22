"""Supplementary Figure S20 - Claude R2 controls, exact inference, prespecified gates.

The complete confirmatory analysis behind main-text Figure 4, on the Claude
macroscopic R2 acquisition (``style.R2_CLAUDE``).

Panels

* a  the exact ``K = 0`` engine/matching control: the three observation maps
  drive *identical* physical trajectories (max |dr_m| = 0 for m = 1, 2, 3 on
  every core seed) while the actions they elicit differ;
* b  the primary endpoint, sustained-lock score ``L``, seed by seed;
* c  the secondary endpoint, final polar order ``Y``, seed by seed;
* d  the exact permutation null for ``T_L`` - all ``(3!)^6 = 46 656``
  within-seed relabellings enumerated, not sampled;
* e  the same for ``T_Y``;
* f  the complete phenotype matrix, core seeds above the separator, held-out
  below, with the lock counts of both groups;
* g  the three pairwise seed-level contrasts for both endpoints, with the
  frozen bootstrap CIs and exact sign-flip p-values;
* h  core versus held-out direction, lock fraction and continuous endpoint
  shown separately and with no pooled confirmatory test;
* i  the prespecified gate verdicts, including the Gate C **FAIL**.

Data honesty
------------
Every printed statistic is read from ``r2_inference/primary_inference.json``,
``r2_inference/trajectory_rows.json`` or ``r2_inference/k0_control.json``.
Nothing is hard-coded.

The permutation nulls of panels d/e are not stored in the artifact, but the
null is *exactly enumerable*: 6 relabellings of the three representations
within each of the 6 core seeds. The module re-enumerates all 46 656 of them
and asserts that the recomputed ``T_obs`` and ``p_perm`` reproduce the frozen
values before anything is drawn; a mismatch raises rather than silently
plotting a locally computed number. The ``K = 0`` control of panel a is
likewise recomputed from the stored ``phases.npy`` and checked against the
frozen ``max_abs_diff_rm``.

Panels b/c show the representation mean with a 95% CI over the six core
seeds, exactly as main-text Fig. 4d does for the same quantity; the bootstrap
intervals the artifact froze are for the *contrasts*, and those are the ones
drawn in panel g.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

from analysis.natcomm_si_opus.style import (
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
    PASS_GREEN,
    PHENOTYPE,
    PHENOTYPE_LABEL,
    REP,
    REP_ABBR,
    REP_LIGHT,
    REP_ORDER,
    REP_SHORT,
    R2_CLAUDE,
    RULE,
    STOP_RED,
    apply_style,
    errorbar_mean,
    fig_text_mm,
    hex_rgb,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label_at,
    save_fig,
    trim_spines,
)

INF_DIR = R2_CLAUDE / "r2_inference"

K_POS = (0.08, 0.15)
M_ORDERS = (1, 2, 3)
CONTRASTS = ("M-C", "M-I", "I-C")

# Scatter takes an area; derive it from the shared marker diameters so dot
# sizes stay on the MS_* scale the rest of the set uses.
S_POINT = MS_POINT ** 2
S_SERIES = MS_SERIES ** 2

_DASH = (0, (2.4, 1.5))
_DOT = (0, (0.6, 2.6))
_LW_BLOCK_SEP = 2.4    # white mask between representation blocks in panel f
_LW_GROUP_SEP = 1.6    # core / held-out separator in panel f
_NULL_GREY = "#B9C6D2"
_HEADROOM = 20.0       # log-axis headroom above the tallest null bar

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm-wide page)
# ---------------------------------------------------------------------------
# 174 mm, not 192: the figure used to reserve a 13 mm band at the foot for a
# "Locked reading" summary box, which has moved into the caption. The panels
# themselves are placed in mm from the TOP of the canvas, so dropping the band
# is a pure shortening of the page — nothing above it moves.
H_MM = 174.0

COL_L = (13.0, 71.0, 129.0)
COL_W = 46.0
LETTER_DX = -8.6                      # matches style.LETTER_DX_MM
LETTER_X = tuple(x + LETTER_DX for x in COL_L)

R1_TOP, R2_TOP, R3_TOP = 10.5, 61.0, 109.0
LETTER_Y = (R1_TOP - 1.6, R2_TOP - 1.6, R3_TOP - 1.6)

# row 1
A1_H = 20.0
A_BADGE_Y = 38.4
A_HEAD_Y = 43.6
# The action bars carry their fraction inside the band in white type, so a
# band shorter than the digits does not merely crowd them — the part of each
# glyph that overshoots the band is white ink on white paper and simply
# vanishes, which read as the numbers being cut off along their top edge. The
# band is sized from the type: 0.82 of a 1.0 row over 8.6 mm gives a 2.35 mm
# band against a 5.8 pt digit whose cap height is 1.5 mm, so the glyph clears
# the band edge (and its 0.6 pt white stroke) at both ends.
A2_TOP, A2_H = 44.4, 8.6
A_BAR_H = 0.82
A_KEY_Y = 55.2
A_KEY_X0 = 6.4          # first swatch, relative to the panel-a left edge
A_KEY_DX = 13.4         # advance between action-key entries
BC_H = 30.0
BC_KEY_TOP = 49.6

# row 2
DE_H = 30.0
F_W, F_H = 36.0, 30.0
LEGEND_TOP = 98.5

# row 3
G_TOP, G_H = 112.0, 16.0
G2_TOP = 138.0
G_NOTE_Y = 162.0
H_TOP, H_H = 112.0, 16.0
H2_TOP, H2_H = 138.0, 15.0
H_NOTE_Y = 160.5
I_TOP, I_H = 109.0, 56.0



# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load(name: str):
    return json.loads((INF_DIR / name).read_text(encoding="utf-8"))


def _cell_id(k: float, seed: int) -> str:
    return f"N17_K+{k:g}_s{seed}"


def _seeds(rows, panel: str) -> list[int]:
    return sorted({r["seed_index"] for r in rows if r["panel"] == panel})


def _endpoint_matrix(rows, panel: str, field: str) -> np.ndarray:
    """(n_seeds, 3) seed x representation endpoint, averaged over K > 0."""
    return np.array([
        [
            float(np.mean([
                r[field] for r in rows
                if r["panel"] == panel and r["seed_index"] == s
                and r["representation"] == rep and r["coupling"] > 0
            ]))
            for rep in REP_ORDER
        ]
        for s in _seeds(rows, panel)
    ])


def _order_parameter(phases: np.ndarray, m: int) -> np.ndarray:
    return np.abs(np.mean(np.exp(1j * m * np.asarray(phases, float)), axis=-1))


def _pfmt(p: float) -> str:
    return f"{p:.3g}".replace("-", "−")


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _sci(x: float) -> str:
    """Format a tolerance such as 1e-12 as set maths."""
    s = f"{x:e}"
    mant, exp = s.split("e")
    e = int(exp)
    m = float(mant)
    lead = "" if abs(m - 1.0) < 1e-12 else f"{m:g}" + r"\times "
    return rf"${lead}10^{{{e}}}$".replace("-", "−")


# ---------------------------------------------------------------------------
# Exact enumeration of the within-seed relabelling null
# ---------------------------------------------------------------------------
def _T(matrix: np.ndarray) -> float:
    """Between-representation dispersion: sum_R (mean_R - grand mean)^2."""
    m = matrix.mean(axis=0)
    return float(((m - m.mean()) ** 2).sum())


def _exact_null(matrix: np.ndarray) -> np.ndarray:
    """All (3!)^n_seeds within-seed representation relabellings of ``matrix``."""
    n = matrix.shape[0]
    perms = np.array(list(itertools.permutations(range(3))))
    grid = np.array(list(itertools.product(range(len(perms)), repeat=n)))
    relabelled = matrix[np.arange(n)[:, None, None], perms[None, :, :]]
    picked = relabelled[
        np.arange(n)[None, :, None], grid[:, :, None], np.arange(3)[None, None, :]
    ]
    means = picked.sum(axis=1) / n
    return ((means - means.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)


def _verify_permutation(matrix: np.ndarray, frozen: dict, tag: str) -> dict:
    """Re-enumerate the exact null and refuse to draw if it disagrees."""
    null = _exact_null(matrix)
    t_obs = _T(matrix)
    n_ge = int((null >= t_obs - 1e-12).sum())
    p = n_ge / null.size
    if null.size != int(frozen["n_permutations"]):
        raise ValueError(
            f"S20 {tag}: enumerated {null.size} relabellings, artifact froze "
            f"{frozen['n_permutations']}"
        )
    if not np.isclose(t_obs, frozen["T_obs"], rtol=0, atol=1e-12):
        raise ValueError(
            f"S20 {tag}: recomputed T_obs={t_obs!r} != frozen "
            f"{frozen['T_obs']!r}; refusing to plot a local value"
        )
    if f"{p:.6g}" != f"{float(frozen['p_perm']):.6g}":
        raise ValueError(
            f"S20 {tag}: recomputed p_perm={p!r} != frozen "
            f"{frozen['p_perm']!r}; refusing to plot a local value"
        )
    return {"null": null, "T_obs": t_obs, "p": p, "n_ge": n_ge}


def _verify_k0(frozen: dict) -> dict:
    """Recompute max |dr_m| across representations at K = 0 from the phases."""
    worst = 0.0
    n_comparisons = 0
    seeds = sorted(int(p.name.rsplit("_s", 1)[1])
                   for p in R2_CLAUDE.glob("N17_K+0_s*"))
    for s in seeds:
        phases = {rep: np.load(R2_CLAUDE / _cell_id(0.0, s) / rep / "phases.npy")
                  for rep in REP_ORDER}
        for m in M_ORDERS:
            series = {rep: _order_parameter(phases[rep], m) for rep in REP_ORDER}
            for a, b in itertools.combinations(REP_ORDER, 2):
                worst = max(worst, float(np.max(np.abs(series[a] - series[b]))))
                n_comparisons += 1
    tol = float(frozen["tolerance"])
    if worst > tol:
        raise ValueError(
            f"S20 panel a: recomputed max|dr_m|={worst!r} exceeds the frozen "
            f"tolerance {tol!r}"
        )
    if not np.isclose(worst, float(frozen["max_abs_diff_rm"]), rtol=0, atol=tol):
        raise ValueError(
            f"S20 panel a: recomputed max|dr_m|={worst!r} != frozen "
            f"{frozen['max_abs_diff_rm']!r}"
        )
    return {"max_abs_diff_rm": worst, "tolerance": tol, "n_seeds": len(seeds),
            "n_comparisons": n_comparisons, "seed": seeds[0]}


def _k0_actions() -> dict:
    """Pooled (p-, p0, p+) action fractions per representation at K = 0."""
    out = {}
    for rep in REP_ORDER:
        counts = np.zeros(3)
        for path in sorted(R2_CLAUDE.glob(f"N17_K+0_s*/{rep}/actions.npy")):
            a = np.load(path)
            for i, v in enumerate((-1, 0, 1)):
                counts[i] += float((a == v).sum())
        out[rep] = counts / counts.sum()
    return out


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def _spread(values, center: float, half: float = 0.15) -> np.ndarray:
    n = len(values)
    if n <= 1:
        return np.full(n, center)
    return center + np.linspace(-half, half, n)


def _decollide(values: dict, gap: float) -> dict:
    """Push overlapping right-hand line labels apart, largest first."""
    out, prev = {}, None
    for key, v in sorted(values.items(), key=lambda kv: kv[1], reverse=True):
        if prev is not None and prev - v < gap:
            v = prev - gap
        out[key] = v
        prev = v
    return out


# ---------------------------------------------------------------------------
# Panel a - exact K = 0 control
# ---------------------------------------------------------------------------
def _panel_a(fig, k0: dict, actions: dict) -> None:
    ax = mm_axes(fig, COL_L[0], R1_TOP, COL_W, A1_H)
    panel_label_at(fig, LETTER_X[0], LETTER_Y[0], "a",
                   "Exact $K=0$ engine control")

    seed = k0["seed"]
    phases = {rep: np.load(R2_CLAUDE / _cell_id(0.0, seed) / rep / "phases.npy")
              for rep in REP_ORDER}
    # One physical trajectory per m, drawn three times - once per observation
    # map - in decreasing weight, so the exact coincidence reads as three
    # differently dashed strokes sharing a single path.
    styles = (("solid", LW_TRACE), (_DASH, LW_LINE), (_DOT, LW_LINE))
    ends = {}
    for m in M_ORDERS:
        for rep, (ls, lw) in zip(REP_ORDER, styles):
            y = _order_parameter(phases[rep], m)
            ax.plot(np.arange(len(y)), y, color=REP[rep], lw=lw, ls=ls, zorder=3 + m)
        ends[m] = float(y[-1])
    n_t = len(y) - 1
    for m, yv in _decollide(ends, 0.052).items():
        ax.annotate(rf"$r_{m}$", xy=(n_t, yv), xytext=(1.8, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=FS_TINY, color=INK, annotation_clip=False)

    ax.set_xlim(0, n_t)
    ax.set_ylim(0, 0.50)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$r_m(t)$")
    trim_spines(ax)
    ax.text(0.015, 0.955, f"core seed {seed + 1}", transform=ax.transAxes,
            ha="left", va="center", fontsize=FS_TINY, color=MUTED)
    # The three maps are distinguishable only by dash weight: they share one
    # path exactly, which is the whole content of the control.
    for i, rep in enumerate(REP_ORDER):
        ls, lw = styles[i]
        x = 0.30 + i * 0.235
        ax.plot([x, x + 0.056], [0.955, 0.955], transform=ax.transAxes,
                color=REP[rep], lw=lw, ls=ls)
        ax.text(x + 0.068, 0.955, REP_ABBR[rep], transform=ax.transAxes,
                ha="left", va="center", fontsize=FS_TINY, color=REP[rep],
                fontweight="bold")

    # The control itself is a constant, so it is stated, not drawn.
    fig_text_mm(
        fig, COL_L[0] + COL_W / 2.0, A_BADGE_Y,
        r"$\max|\Delta r_m| = " f"{k0['max_abs_diff_rm']:g}$"
        f"   ({k0['n_comparisons']} comparisons, tol "
        + _sci(k0["tolerance"]) + ")",
        ha="center", va="center", fontsize=FS_TINY, color="#12572A",
        bbox=dict(facecolor="#EFF6EF", edgecolor=PASS_GREEN, lw=LW_HAIR,
                  boxstyle="round,pad=0.34"),
    )

    # ... yet the actions the three maps elicit are not the same.
    fig_text_mm(fig, COL_L[0] - 0.5, A_HEAD_Y,
                "identical physics, different actions "
                "(fraction of moves)",
                ha="left", va="baseline", fontsize=FS_TINY, color=INK)
    axa = mm_axes(fig, COL_L[0], A2_TOP, COL_W, A2_H)
    cols = (ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"])
    for i, rep in enumerate(REP_ORDER):
        left = 0.0
        for k, col in enumerate(cols):
            w = actions[rep][k]
            axa.barh([2 - i], [w], left=[left], height=A_BAR_H, color=col,
                     edgecolor="white", linewidth=LW_THIN)
            if w > 0.09:
                axa.text(left + w / 2.0, 2 - i, f"{w:.2f}"[1:], ha="center",
                         va="center", fontsize=FS_TINY,
                         color=INK if k == 1 else "white")
            left += w
    axa.set_xlim(0, 1)
    axa.set_ylim(-0.5, 2.5)
    axa.set_yticks([2, 1, 0])
    axa.set_yticklabels([REP_ABBR[r] for r in REP_ORDER], fontsize=FS_TINY)
    for tick, r in zip(axa.get_yticklabels(), REP_ORDER):
        tick.set_color(REP[r])
        tick.set_fontweight("bold")
    axa.set_xticks([])
    axa.tick_params(axis="y", length=0, pad=1.2)
    for sp in axa.spines.values():
        sp.set_visible(False)
    # A swatch key rather than labels on one row's segments: p0 is a sliver in
    # the "mom" row, so segment labels fixed under "int" sat nowhere near the
    # bands they name. "action" states the grammar explicitly, because blue is
    # also the moments colour in the row labels beside this chart.
    fig_text_mm(fig, COL_L[0], A_KEY_Y, "action", ha="left", va="baseline",
                fontsize=FS_TINY, color=MUTED)
    for i, (lab, col, word) in enumerate(
            zip(ACTION_LABELS, cols, ("retard", "stay", "advance"))):
        x = COL_L[0] + A_KEY_X0 + i * A_KEY_DX
        fig.add_artist(mpatches.Rectangle(
            (x / 180.0, (H_MM - A_KEY_Y + 0.15) / H_MM),
            2.0 / 180.0, 1.8 / H_MM, transform=fig.transFigure,
            facecolor=col, edgecolor="#8C8C8C", lw=LW_HAIR, clip_on=False))
        fig_text_mm(fig, x + 2.9, A_KEY_Y, f"{lab} {word}", ha="left",
                    va="baseline", fontsize=FS_TINY,
                    color=MUTED if col == ACTION["p_zero"] else col,
                    fontweight="bold")


# ---------------------------------------------------------------------------
# Panels b, c - the two endpoints, seed by seed
# ---------------------------------------------------------------------------
def _endpoint_panel(fig, col: int, letter: str, title: str, core: np.ndarray,
                    held: np.ndarray, frozen: dict, ylab: str, ylim, yticks,
                    mfmt: str) -> None:
    ax = mm_axes(fig, COL_L[col], R1_TOP, COL_W, BC_H)
    panel_label_at(fig, LETTER_X[col], LETTER_Y[0], letter, title)

    offs = np.linspace(-0.15, 0.15, core.shape[0])
    for s in range(core.shape[0]):
        xs = np.arange(3) - 0.18 + offs[s]
        ax.plot(xs, core[s], color=MUTED, lw=LW_THIN, alpha=0.45, zorder=2)
        for i, rep in enumerate(REP_ORDER):
            ax.scatter([xs[i]], [core[s, i]], marker="o", c=REP[rep],
                       s=S_POINT, linewidths=0, zorder=3)
    for i in range(3):
        errorbar_mean(ax, i + 0.14, core[:, i], color=INK, ms=MS_MEAN,
                      lw=LW_LINE, capsize=CAPSIZE, zorder=6)
        ax.scatter(_spread(held[:, i], i + 0.40, half=0.07), held[:, i],
                   marker="^", facecolors="white",
                   edgecolors=REP_LIGHT[REP_ORDER[i]], linewidths=LW_THIN,
                   s=S_POINT, zorder=4)

    means = frozen["mean_by_rep"]
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        [f"{REP_SHORT[r]}\n{mfmt.format(means[r])}" for r in REP_ORDER],
        fontsize=FS_TICK, linespacing=1.3,
    )
    for tick, r in zip(ax.get_xticklabels(), REP_ORDER):
        tick.set_color(REP[r])
    ax.set_xlim(-0.66, 2.70)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylab)
    trim_spines(ax, x=False)
    ax.text(0.985, 0.02,
            f"exact $p_{{{frozen['endpoint']}}}$ = {_pfmt(frozen['p_perm'])}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS_SMALL, color=INK)


def _panels_bc(fig, rows, inf: dict) -> None:
    _endpoint_panel(
        fig, 1, "b", "Sustained-lock score $L$",
        _endpoint_matrix(rows, "core", "sustained_lock"),
        _endpoint_matrix(rows, "heldout", "sustained_lock"),
        inf["primary_sustained_lock"],
        r"$L_{sR}$", (-0.16, 1.14), [0, 0.5, 1.0], "{:.2f}",
    )
    _endpoint_panel(
        fig, 2, "c", "Final polar order $Y$",
        _endpoint_matrix(rows, "core", "final_r1"),
        _endpoint_matrix(rows, "heldout", "final_r1"),
        inf["secondary_final_r1"],
        r"$Y_{sR}$", (0.34, 1.09), [0.4, 0.6, 0.8, 1.0], "{:.3f}",
    )

    key = mm_panel(fig, COL_L[1], BC_KEY_TOP, COL_L[2] + COL_W - COL_L[1], 6.4)
    key.set_xlim(0, 1)
    key.scatter([0.008], [0.80], marker="o", c=MUTED, s=S_POINT, linewidths=0,
                clip_on=False)
    key.text(0.026, 0.80, "core seed; a grey line pairs one seed", ha="left",
             va="center", fontsize=FS_TINY, color=MUTED)
    key.scatter([0.520], [0.80], marker="^", facecolors="white",
                edgecolors=MUTED, linewidths=LW_THIN, s=S_POINT, clip_on=False)
    key.text(0.538, 0.80, "held-out seed (direction only)", ha="left",
             va="center", fontsize=FS_TINY, color=MUTED)
    key.plot([0.010], [0.28], marker="D", color=INK, ms=MS_POINT, clip_on=False)
    key.text(0.026, 0.28,
             "representation mean $\\pm$ 95% CI over the six core seeds; the "
             "printed means are the stored artifact values",
             ha="left", va="center", fontsize=FS_TINY, color=MUTED)


# ---------------------------------------------------------------------------
# Panels d, e - the exact permutation nulls
# ---------------------------------------------------------------------------
def _null_panel(fig, col: int, letter: str, title: str, res: dict,
                frozen: dict, tlab: str, discrete: bool, xlim, xticks) -> None:
    ax = mm_axes(fig, COL_L[col], R2_TOP, COL_W, DE_H)
    panel_label_at(fig, LETTER_X[col], LETTER_Y[1], letter, title)

    null = res["null"]
    t_obs = res["T_obs"]
    if discrete:
        vals, counts = np.unique(np.round(null, 10), return_counts=True)
        width = float(np.diff(np.sort(vals)).min()) * 0.72
        cols = [STOP_RED if v >= t_obs - 1e-12 else _NULL_GREY for v in vals]
        ax.bar(vals, counts, width=width, color=cols, linewidth=0)
    else:
        # Anchor a bin edge exactly on T_obs. With evenly spaced bins the tail
        # relabellings fell inside a bin whose *centre* was below T_obs, so the
        # rejection tail drew grey and hid behind the observed line; the note
        # then promised a red bar that was not there. Counts and p are
        # untouched - only where the bin boundary falls changes.
        width = float(null.max()) * 1.02 / 60.0
        n_below = max(int(round(t_obs / width)), 1)
        edges = np.concatenate([np.linspace(0.0, t_obs, n_below + 1),
                                [t_obs + width]])
        counts, _ = np.histogram(null, bins=edges)
        lefts = edges[:-1]
        cols = [STOP_RED if v >= t_obs - 1e-12 else _NULL_GREY for v in lefts]
        ax.bar(lefts, counts, width=np.diff(edges) * 0.96, color=cols,
               linewidth=0, align="edge")
    top = float(counts.max())

    ax.axvspan(t_obs, xlim[1], color="#FBEAE9", lw=0, zorder=0)
    ax.axvline(t_obs, color=INK, lw=LW_LINE, zorder=6)
    ax.set_yscale("log")
    ax.set_ylim(0.6, top * _HEADROOM)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1", "10", "$10^2$", "$10^3$", "$10^4$"])
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(tlab)
    ax.set_ylabel("relabellings")
    trim_spines(ax, x=False)

    ax.text(0.02, 0.995, f"{tlab} = {frozen['T_obs']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_TINY,
            color=INK)
    ax.text(0.02, 0.895,
            f"$p$ = {res['n_ge']}/{_thousands(null.size)} = "
            f"{_pfmt(frozen['p_perm'])}",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_TINY,
            color=STOP_RED)
    ax.annotate("observed", xy=(t_obs, 0.60), xycoords=("data", "axes fraction"),
                xytext=(-3.2, 0), textcoords="offset points", ha="right",
                va="center", fontsize=FS_TINY, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=LW_HAIR,
                                mutation_scale=4, shrinkA=1.0, shrinkB=1.0))


def _panels_de(fig, res_L: dict, res_Y: dict, inf: dict) -> None:
    _null_panel(fig, 0, "d", "Exact null for $T_L$", res_L,
                inf["primary_sustained_lock"], "$T_L$", True,
                (-0.014, 0.50), [0, 0.1, 0.2, 0.3, 0.4])
    _null_panel(fig, 1, "e", "Exact null for $T_Y$", res_Y,
                inf["secondary_final_r1"], "$T_Y$", False,
                (-0.0025, 0.092), [0, 0.02, 0.04, 0.06, 0.08])


# ---------------------------------------------------------------------------
# Panel f - complete phenotype matrix
# ---------------------------------------------------------------------------
def _panel_f(fig, rows) -> None:
    ax = mm_axes(fig, COL_L[2], R2_TOP, F_W, F_H)
    panel_label_at(fig, LETTER_X[2], LETTER_Y[1], "f", "Phenotype matrix")

    core_seeds = _seeds(rows, "core")
    held_seeds = _seeds(rows, "heldout")
    seed_rows = ([("core", s) for s in core_seeds]
                 + [("heldout", s) for s in held_seeds])
    cols = [(rep, k) for rep in REP_ORDER for k in K_POS]

    rgb = np.zeros((len(seed_rows), len(cols), 3))
    lock = {"core": {}, "heldout": {}}
    for j, (rep, k) in enumerate(cols):
        for panel in ("core", "heldout"):
            sel = [r for r in rows if r["panel"] == panel
                   and r["representation"] == rep
                   and np.isclose(r["coupling"], k)]
            lock[panel][(rep, k)] = (int(sum(r["sustained_lock"] for r in sel)),
                                     len(sel))
        for i, (panel, seed) in enumerate(seed_rows):
            row = next(r for r in rows if r["panel"] == panel
                       and r["seed_index"] == seed
                       and r["representation"] == rep
                       and np.isclose(r["coupling"], k))
            rgb[i, j] = hex_rgb(PHENOTYPE[row["cluster_phenotype"]])

    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(seed_rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=LW_HAIR)
    ax.tick_params(which="minor", length=0)
    for x in (1.5, 3.5):
        ax.axvline(x, color="white", lw=_LW_BLOCK_SEP)
    ax.axhline(len(core_seeds) - 0.5, color="white", lw=_LW_GROUP_SEP)
    ax.axhline(len(core_seeds) - 0.5, color=INK, lw=LW_HAIR)

    ax.set_yticks(range(len(seed_rows)))
    ax.set_yticklabels([f"seed {i + 1}" for i in range(len(seed_rows))],
                       fontsize=FS_TINY)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{k:g}" for _, k in cols], fontsize=FS_TINY)
    ax.tick_params(axis="both", length=0, pad=1.2)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cell_h = F_H / len(seed_rows)
    for label, i0, n in (("core", 0, len(core_seeds)),
                         ("held-out", len(core_seeds), len(held_seeds))):
        fig_text_mm(fig, COL_L[2] + F_W + 1.4,
                    R2_TOP + (i0 + n / 2.0) * cell_h - 0.5, label,
                    ha="left", va="center", fontsize=FS_TINY,
                    color=INK if label == "core" else MUTED, fontweight="bold")

    bottom = R2_TOP + F_H
    cell_w = F_W / len(cols)
    for lab, key, y_off, palette in ((r"$K$", None, 0.7, None),
                                     ("core lock", "core", 4.0, REP),
                                     ("held lock", "heldout", 7.2, REP_LIGHT)):
        fig_text_mm(fig, COL_L[2] - 1.4, bottom + y_off, lab, ha="right",
                    va="top", fontsize=FS_TINY, color=MUTED)
        if key is None:
            continue
        for j, (rep, k) in enumerate(cols):
            n_lock, n_tot = lock[key][(rep, k)]
            fig_text_mm(fig, COL_L[2] + cell_w * (j + 0.5), bottom + y_off,
                        f"{n_lock}/{n_tot}", ha="center", va="top",
                        fontsize=FS_TINY, color=palette[rep], fontweight="bold")
    for jj, rep in enumerate(REP_ORDER):
        fig_text_mm(fig, COL_L[2] + cell_w * (2 * jj + 1), bottom + 10.4,
                    REP_ABBR[rep], ha="center", va="top", color=REP[rep],
                    fontweight="bold", fontsize=FS_SMALL)

    present = {r["cluster_phenotype"] for r in rows if r["coupling"] > 0}
    axL = mm_panel(fig, COL_L[0], LEGEND_TOP, 104.0, 4.2)
    handles = [mpatches.Patch(facecolor=PHENOTYPE[k], edgecolor="none",
                              label=PHENOTYPE_LABEL[k])
               for k in PHENOTYPE if k in present]
    axL.legend(handles=handles, loc="center left", ncol=3, frameon=False,
               fontsize=FS_TINY, handlelength=1.0, handleheight=1.0,
               columnspacing=1.1, handletextpad=0.4, borderpad=0.0,
               borderaxespad=0.0)


# ---------------------------------------------------------------------------
# Panel g - pairwise seed-level contrasts
# ---------------------------------------------------------------------------
def _contrast_axes(fig, top: float, contrasts: list[dict], xlab: str,
                   xlim, xticks, tag: str) -> None:
    ax = mm_axes(fig, COL_L[0], top, COL_W, G_H)
    by_name = {c["contrast"]: c for c in contrasts}
    ax.axvline(0.0, color=RULE, lw=LW_HAIR, zorder=0)
    for gi, name in enumerate(CONTRASTS):
        c = by_name[name]
        diffs = np.asarray(c["seed_diffs"], float)
        ax.scatter(diffs, _spread(diffs, gi - 0.26, half=0.12), marker="o",
                   c=MUTED, s=S_POINT * 0.7, linewidths=0, zorder=3)
        m = float(c["mean_diff"])
        lo, hi = (float(v) for v in c["bootstrap_ci_95"])
        ax.errorbar([m], [gi + 0.16], xerr=[[m - lo], [hi - m]], fmt="D",
                    color=INK, ecolor=INK, ms=MS_SERIES, elinewidth=LW_LINE,
                    capsize=CAPSIZE, capthick=LW_LINE, zorder=5)
        ax.text(0.985, gi + 0.16, f"$p$ = {_pfmt(c['signflip_p'])}",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=FS_TINY, color=MUTED)
    ax.set_yticks(range(3))
    ax.set_yticklabels([n.replace("-", "–") for n in CONTRASTS],
                       fontsize=FS_TICK)
    ax.set_ylim(2.62, -0.70)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:g}".replace("-", "−") for v in xticks])
    ax.set_xlabel(xlab)
    ax.tick_params(axis="y", length=0)
    trim_spines(ax, y=False)
    ax.spines["left"].set_visible(False)
    fig_text_mm(fig, COL_L[0] - 0.5, top - 1.4, tag, ha="left", va="baseline",
                fontsize=FS_SMALL, color=INK, fontweight="bold")


def _panel_g(fig, inf: dict) -> None:
    panel_label_at(fig, LETTER_X[0], LETTER_Y[2], "g",
                   "Pairwise seed-level contrasts")
    _contrast_axes(fig, G_TOP, inf["contrasts_L"],
                   r"paired difference in $L$", (-1.20, 1.06),
                   [-1.0, -0.5, 0, 0.5], "endpoint $L$")
    _contrast_axes(fig, G2_TOP, inf["contrasts_Y"],
                   r"paired difference in $Y$", (-0.58, 0.54),
                   [-0.4, -0.2, 0, 0.2], "endpoint $Y$")
    n_flip = int(inf["contrasts_L"][0]["n_signflip"])
    fig_text_mm(
        fig, COL_L[0] - 0.5, G_NOTE_Y,
        "Grey dots: the six core-seed differences.\n"
        "Diamond: mean with the stored bootstrap 95% CI.\n"
        f"$p$: exact sign-flip test over all {n_flip} sign patterns.",
        ha="left", va="top", fontsize=FS_TINY, color=MUTED, linespacing=1.45,
    )


# ---------------------------------------------------------------------------
# Panel h - core versus held-out direction
# ---------------------------------------------------------------------------
def _panel_h(fig, rows) -> None:
    ax = mm_axes(fig, COL_L[1], H_TOP, COL_W, H_H)
    panel_label_at(fig, LETTER_X[1], LETTER_Y[2], "h",
                   "Core vs held-out direction")
    fig_text_mm(fig, COL_L[1] - 0.5, H_TOP - 1.4, "sustained lock",
                ha="left", va="baseline", fontsize=FS_SMALL, color=INK,
                fontweight="bold")

    cells = [(i * 2.4 + j, rep, k) for i, rep in enumerate(REP_ORDER)
             for j, k in enumerate(K_POS)]
    for x, rep, k in cells:
        vals = {}
        for panel in ("core", "heldout"):
            sel = [r for r in rows if r["panel"] == panel
                   and r["representation"] == rep
                   and np.isclose(r["coupling"], k)]
            vals[panel] = (float(np.mean([r["sustained_lock"] for r in sel])),
                           int(sum(r["sustained_lock"] for r in sel)), len(sel))
        ax.scatter([x], [vals["core"][0]], marker="o", facecolors=REP[rep],
                   edgecolors=REP[rep], linewidths=LW_THIN, s=S_SERIES,
                   zorder=5, clip_on=False)
        ax.scatter([x], [vals["heldout"][0]], marker="^", facecolors="white",
                   edgecolors=REP_LIGHT[rep], linewidths=LW_LINE, s=S_SERIES,
                   zorder=6, clip_on=False)
        hi = max(vals["core"][0], vals["heldout"][0])
        lo = min(vals["core"][0], vals["heldout"][0])
        ax.text(x, hi + 0.11, f"{vals['core'][1]}/{vals['core'][2]}",
                ha="center", va="bottom", fontsize=FS_TINY, color=REP[rep],
                fontweight="bold")
        ax.text(x, lo - 0.12, f"{vals['heldout'][1]}/{vals['heldout'][2]}",
                ha="center", va="top", fontsize=FS_TINY, color=MUTED)
    ax.set_xlim(-0.55, 6.15)
    ax.set_ylim(-0.34, 1.34)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("lock fraction")
    ax.set_xticks([c[0] for c in cells])
    ax.set_xticklabels([f"{c[2]:g}" for c in cells], fontsize=FS_TINY)
    ax.tick_params(axis="x", length=0, pad=0.8)
    trim_spines(ax, x=False)
    ax.spines["bottom"].set_visible(False)
    fig_text_mm(fig, COL_L[1] - 1.2, H_TOP + H_H + 1.0, r"$K$", ha="right",
                va="top", fontsize=FS_TINY, color=MUTED)
    for i, rep in enumerate(REP_ORDER):
        fig_text_mm(fig, COL_L[1] + COL_W * (i * 2.4 + 0.5 + 0.55) / 6.70,
                    H_TOP + H_H + 4.6, REP_ABBR[rep], ha="center", va="top",
                    fontsize=FS_SMALL, color=REP[rep], fontweight="bold")

    ax2 = mm_axes(fig, COL_L[1], H2_TOP, COL_W, H2_H)
    fig_text_mm(fig, COL_L[1] - 0.5, H2_TOP - 1.4, "continuous endpoint",
                ha="left", va="baseline", fontsize=FS_SMALL, color=INK,
                fontweight="bold")
    for i, rep in enumerate(REP_ORDER):
        for panel, marker, face, edge, dx in (
            ("core", "o", REP[rep], REP[rep], -0.26),
            ("heldout", "^", "white", REP_LIGHT[rep], 0.26),
        ):
            vals = np.array([r["final_r1"] for r in rows
                             if r["panel"] == panel
                             and r["representation"] == rep
                             and r["coupling"] > 0])
            ax2.scatter([i + dx], [vals.mean()], marker=marker,
                        facecolors=face, edgecolors=edge, linewidths=LW_LINE,
                        s=S_SERIES, zorder=4, clip_on=False)
            ax2.text(i + dx, vals.mean() - 0.05, f"{vals.mean():.2f}",
                     ha="center", va="top", fontsize=FS_TINY,
                     color=REP[rep] if panel == "core" else MUTED)
    ax2.set_xlim(-0.62, 2.62)
    ax2.set_ylim(0.40, 1.12)
    ax2.set_yticks([0.5, 0.75, 1.0])
    ax2.set_ylabel(r"mean final $r_1$")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([REP_ABBR[r] for r in REP_ORDER], fontsize=FS_SMALL)
    for tick, r in zip(ax2.get_xticklabels(), REP_ORDER):
        tick.set_color(REP[r])
        tick.set_fontweight("bold")
    ax2.tick_params(axis="x", length=0, pad=1.4)
    trim_spines(ax2, x=False)
    ax2.spines["bottom"].set_visible(False)

    fig_text_mm(fig, COL_L[1] - 0.5, H_NOTE_Y,
                "Filled circle: core six. Open triangle: held-out four.\n"
                "Counts above each column are core, below are held-out.",
                ha="left", va="top", fontsize=FS_TINY, color=MUTED,
                linespacing=1.45)


# ---------------------------------------------------------------------------
# Panel i - prespecified gate verdicts
# ---------------------------------------------------------------------------
def _panel_i(fig, inf: dict, res_L: dict, res_Y: dict) -> None:
    ax = mm_panel(fig, COL_L[2], I_TOP, COL_W, I_H)
    panel_label_at(fig, LETTER_X[2], LETTER_Y[2], "i",
                   "Prespecified gate verdicts")

    gates = inf["gates"]
    k0 = inf["k0_control"]
    lock_frac = {d["K"]: d["lock_fractions"] for d in gates["gate_B"]["details"]}
    r1 = gates["gate_C"]["mean_final_r1"]
    entries = [
        ("$K=0$ control", bool(k0["pass"]), "engine and matching valid",
         r"$\max|\Delta r_m|$ = " f"{k0['max_abs_diff_rm']:g}, tol "
         + _sci(float(k0["tolerance"]))),
        ("Gate A", bool(gates["gate_A"]["pass"]),
         "macro representation effect",
         f"$p_L$ = {_pfmt(res_L['p'])},  $p_Y$ = {_pfmt(res_Y['p'])}"),
        ("Gate B", bool(gates["gate_B"]["pass"]),
         "qualitative phase separation",
         "lock fraction at $K$ = 0.08: "
         + ", ".join(f"{lock_frac[0.08][r]:.2f}" for r in REP_ORDER)),
        ("Gate C", bool(gates["gate_C"]["pass"]),
         "GPT phenotype map not replicated",
         r"final $r_1$ reverses: "
         + " < ".join(f"{r1[r]:.2f}" for r in REP_ORDER)),
    ]

    y = 0.985
    dy = 0.250
    for name, ok, interp, detail in entries:
        col = PASS_GREEN if ok else STOP_RED
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.068), 0.225, 0.068,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            facecolor=col, edgecolor="none", transform=ax.transAxes, zorder=2))
        ax.text(0.1125, y - 0.036, "PASS" if ok else "FAIL", ha="center",
                va="center", fontsize=FS_SMALL, fontweight="bold",
                color="white", zorder=3)
        ax.text(0.262, y - 0.036, name, ha="left", va="center",
                fontsize=FS_BODY, color=INK, fontweight="bold")
        ax.text(0.0, y - 0.113, interp, ha="left", va="center",
                fontsize=FS_TINY, color=INK)
        ax.text(0.0, y - 0.166, detail, ha="left", va="center",
                fontsize=FS_TINY, color=MUTED)
        y -= dy
        if y > 0.05:
            ax.plot([0.0, 1.0], [y + 0.048, y + 0.048], color=RULE,
                    lw=LW_HAIR, zorder=0)


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    rows = _load("trajectory_rows.json")
    inf = _load("primary_inference.json")
    k0_frozen = _load("k0_control.json")

    res_L = _verify_permutation(
        _endpoint_matrix(rows, "core", "sustained_lock"),
        inf["primary_sustained_lock"], "panel d (T_L)")
    res_Y = _verify_permutation(
        _endpoint_matrix(rows, "core", "final_r1"),
        inf["secondary_final_r1"], "panel e (T_Y)")
    k0 = _verify_k0(k0_frozen)

    fig = new_figure(H_MM)
    _panel_a(fig, k0, _k0_actions())
    _panels_bc(fig, rows, inf)
    _panels_de(fig, res_L, res_Y, inf)
    _panel_f(fig, rows)
    _panel_g(fig, inf)
    _panel_h(fig, rows)
    _panel_i(fig, inf, res_L, res_Y)
    return save_fig(fig, "figS20_claude_r2_gates")
