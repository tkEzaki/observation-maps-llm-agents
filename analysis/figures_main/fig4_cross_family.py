"""Figure 4 - Observation maps select model-family-specific collective outcomes.

Rebuilt from scratch on the Claude macroscopic R2 acquisition
(``runs/stage_c/r2_claude_macro/...``).

Panels a-e are new:

* a  the matched R2 design (same physical seeds -> three observation maps ->
  Claude Haiku 4.5 -> collective dynamics), core grid and held-out extension
  kept visually separate, plus the K = 0 engine-matching control badge;
* b  Claude collective trajectories at K = 0.08 and K = 0.15;
* c  the all-seed phenotype matrix, core seeds above the separator, held-out
  below;
* d  the prespecified macro inference on the core six seeds (terminal-lock
  score L and final polar order Y);
* e  the GPT-Claude matched macro map on the six shared physical seeds, which
  is where the representation-to-phenotype mapping reverses.

Panel f is the previous figure's three-family microscopic operator forest plot,
unchanged in data and statistics, plus the scope statement.

Every printed statistic is read from ``r2_inference/primary_inference.json``,
``r2_inference/trajectory_rows.json`` or ``r1_primary/decision.json``; nothing
is hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from analysis.figures_main.style import (
    CAPSIZE,
    DATA_DIR,
    FAMILY_EDGE,
    FAMILY_LABEL,
    FAMILY_MARKER,
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
    PHENOTYPE,
    PHENOTYPE_LABEL,
    REP,
    REP_LIGHT,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    apply_style,
    cbar_mm,
    errorbar_mean,
    fig_text_mm,
    hex_rgb,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label,
    panel_label_at,
    paired_ci,
    save_fig,
    text_on,
    trim_spines,
)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
PRIMARY = ROOT / "analysis" / "matched_rep_collective" / "r1_primary"
PAIRED = ROOT / "analysis" / "matched_rep_collective" / "paired_analysis"
R2 = (
    ROOT
    / "runs"
    / "stage_c"
    / "r2_claude_macro"
    / "matched-rep-collective-r2-claude-v0.1_ff2ef27f0dfe"
    / "20260725T050234Z"
)

K_POS = (0.08, 0.15)
CORE_SEEDS = 6
HELD_SEEDS = 4

# Scatter takes an area; derive it from the shared marker diameters so dot
# sizes match the MS_* scale used by every other figure in the set.
S_POINT = MS_POINT ** 2
S_SERIES = MS_SERIES ** 2

# Non-data strokes with no equivalent in the shared LW_* scale.
_LW_BLOCK_SEP = 2.4    # white mask between representation blocks in panel c
_LW_GROUP_SEP = 1.6    # core / held-out separator in panel c
_DASH = (0, (2.2, 1.4))

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm-wide page)
# ---------------------------------------------------------------------------
H_MM = 164.0

R1_TOP = 9.6
A_L, A_W, A_H = 4.0, 74.0, 42.0
B_L, B_W, B_H = 86.0, 32.0, 32.0
B2_L = 132.0

R2_TOP = 61.0
C_L, C_W, C_H = 13.0, 50.0, 34.0
LEG_TOP, LEG_H = 106.5, 4.0
D1_L, D2_L, D_W, D_H = 86.0, 132.0, 34.0, 32.0

R3_TOP = 116.0
E_L, E_W, E_H = 13.0, 52.0, 26.0
E_CB_L, E_CB_W = 67.0, 2.6
F_L, F_W, F_H = 86.0, 86.0, 30.0

LETTER_X_LEFT = 4.4          # shared letter column, left stack (a, c, e)
LETTER_X_RIGHT = B_L - 8.6   # shared letter column, right stack (b, d, f)
LETTER_Y1 = R1_TOP - 1.6
LETTER_Y2 = R2_TOP - 1.6
LETTER_Y3 = R3_TOP - 1.6

SCOPE_TOP = 153.0
SCOPE_H = 10.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _rows() -> list[dict]:
    return json.loads((R2 / "r2_inference" / "trajectory_rows.json").read_text(encoding="utf-8"))


def _inference() -> dict:
    return json.loads((R2 / "r2_inference" / "primary_inference.json").read_text(encoding="utf-8"))


def _k0_control() -> dict:
    return json.loads((R2 / "r2_inference" / "k0_control.json").read_text(encoding="utf-8"))


def _series(cell_id: str, rep: str) -> np.ndarray:
    meta = json.loads((R2 / cell_id / rep / "run_meta.json").read_text(encoding="utf-8"))
    return np.asarray(meta["r1_series"], dtype=float)


def _cell_id(k: float, seed: int) -> str:
    return f"N17_K+{k:g}_s{seed}"


def _seed_endpoint(rows, panel: str, seed: int, rep: str, field: str) -> float:
    """Mean of ``field`` over the positive couplings for one seed x representation."""
    sel = [
        r for r in rows
        if r["panel"] == panel and r["seed_index"] == seed
        and r["representation"] == rep and r["coupling"] > 0
    ]
    return float(np.mean([r[field] for r in sel]))


def _pfmt(p: float) -> str:
    return f"{p:.3g}".replace("-", "−")


# ---------------------------------------------------------------------------
# Small drawing helpers (private to this figure)
# ---------------------------------------------------------------------------
def _rbox(ax, x, y, w, h, fc, ec, lw=LW_THIN, zorder=2, ls="solid"):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
            facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder, linestyle=ls,
            mutation_aspect=1.0,
        )
    )


def _arrow(ax, x0, y0, x1, y1, color="#9A9A9A", lw=LW_THIN):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=5,
                        shrinkA=0, shrinkB=0),
    )


def _spread(values, center, half=0.085):
    n = len(values)
    if n == 1:
        return np.array([center])
    return center + np.linspace(-half, half, n)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _panel_a(fig, k0: dict) -> None:
    ax = mm_panel(fig, A_L, R1_TOP, A_W, A_H)
    ax.set_xlim(0, A_W)
    ax.set_ylim(0, A_H)
    panel_label_at(fig, LETTER_X_LEFT, LETTER_Y1, "a", "Claude matched replication design")

    xc = A_W / 2.0

    # tier 1 - the physical substrate, held fixed
    _rbox(ax, 6.0, 36.0, A_W - 12.0, 5.4, "#F7F7F7", "#A0A0A0")
    ax.text(xc, 38.7, r"GPT-matched initial conditions within each $K$  ($N=17$)",
            ha="center", va="center", fontsize=FS_SMALL, color=INK)

    # tier 2 - the intervention: three observation maps
    box_w, gap = 20.0, 2.4
    x0 = xc - (3 * box_w + 2 * gap) / 2.0
    for j, rep in enumerate(REP_ORDER):
        bx = x0 + j * (box_w + gap)
        _arrow(ax, xc, 36.0, bx + box_w / 2.0, 31.8)
        _rbox(ax, bx, 26.6, box_w, 5.2, "white", REP[rep], lw=LW_LINE)
        ax.text(bx + box_w / 2.0, 29.2, REP_SHORT[rep], ha="center", va="center",
                fontsize=FS_SMALL, fontweight="bold", color=REP[rep])
    ax.text(xc, 33.9, "three observation maps", ha="center", va="center",
            fontsize=FS_TINY, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

    # tier 3 - the frozen model
    _rbox(ax, xc - 20.0, 19.0, 40.0, 5.4, "#F2EDF9", "#7059A8", lw=LW_LINE)
    ax.text(xc, 21.7, "Claude Haiku 4.5  (pretrained)", ha="center", va="center",
            fontsize=FS_SMALL, fontweight="bold", color="#57407F")
    for j in range(3):
        bx = x0 + j * (box_w + gap) + box_w / 2.0
        _arrow(ax, bx, 26.6, xc + (j - 1) * 9.0, 24.4)

    # tier 4 - the readout
    _arrow(ax, xc, 19.0, xc, 16.6, color="#7A7A7A")
    ax.text(xc, 15.0, r"collective dynamics  $r_1(t)$  $\rightarrow$  phenotype",
            ha="center", va="center", fontsize=FS_SMALL, color=INK)

    # tier 5 - the matched grid: core and held-out are separate designs
    _rbox(ax, 0.5, 6.4, 36.0, 6.4, "#FFFFFF", "#8C8C8C", lw=LW_LINE)
    ax.text(18.5, 10.6, "core", ha="center", va="center", fontsize=FS_SMALL,
            fontweight="bold", color=INK)
    ax.text(18.5, 7.9, r"6 GPT-shared seeds · $K\in\{0,0.08,0.15\}$",
            ha="center", va="center", fontsize=FS_TINY - 0.6, color="#333333")

    _rbox(ax, A_W - 36.5, 6.4, 36.0, 6.4, "#FBFBFB", "#B0B0B0", lw=LW_THIN,
          ls=_DASH)
    ax.text(A_W - 18.5, 10.6, "held-out extension", ha="center", va="center",
            fontsize=FS_SMALL, fontweight="bold", color=MUTED)
    ax.text(A_W - 18.5, 7.9, r"4 new seeds · $K\in\{0.08,0.15\}$",
            ha="center", va="center", fontsize=FS_TINY - 0.6, color="#333333")

    # tier 6 - engine-matching control badge
    _rbox(ax, 0.5, 0.2, A_W - 1.0, 5.0, "#EFF6EF", "#1A7F37", lw=LW_THIN)
    ax.text(xc, 2.7,
            r"control  $K=0$:  $r_1,\ r_2,\ r_3$ coincide exactly across encodings"
            f"  (max |$\\Delta$| = {k0['max_abs_diff_rm']:g})",
            ha="center", va="center", fontsize=FS_TINY, color="#12572A")


def _panel_b(fig, rows) -> None:
    core_seeds = sorted({r["seed_index"] for r in rows if r["panel"] == "core"})
    held_seeds = sorted({r["seed_index"] for r in rows if r["panel"] == "heldout"})

    axes = []
    for facet, (k, left) in enumerate(zip(K_POS, (B_L, B2_L))):
        ax = mm_axes(fig, left, R1_TOP, B_W, B_H)
        axes.append(ax)
        ends = {}
        for rep in REP_ORDER:
            stack = []
            for seed in core_seeds + held_seeds:
                y = _series(_cell_id(k, seed), rep)
                stack.append(y)
                ax.plot(
                    np.arange(len(y)), y, color=REP[rep], lw=LW_THIN, alpha=0.30,
                    ls="solid" if seed in core_seeds else _DASH, zorder=2,
                )
            agg = np.mean(np.vstack(stack), axis=0)
            ax.plot(np.arange(len(agg)), agg, color=REP[rep], lw=LW_TRACE, zorder=4)
            ends[rep] = float(agg[-1])

        # direct line-end labels, de-collided downwards
        order = sorted(ends.items(), key=lambda kv: kv[1], reverse=True)
        placed, prev = {}, None
        for rep, v in order:
            if prev is not None and prev - v < 0.085:
                v = prev - 0.085
            placed[rep] = v
            prev = v
        for rep, yv in placed.items():
            ax.annotate(
                REP_SHORT[rep], xy=(100, yv), xytext=(2.2, 0),
                textcoords="offset points", color=REP[rep], fontsize=FS_SMALL,
                fontweight="bold", va="center", annotation_clip=False,
            )

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel(r"$t$")
        # inside the axes: the top-left corner is empty at every K, and a
        # facet title above the axes would collide with the panel-b title
        ax.text(0.03, 0.99, rf"$K={k:g}$", transform=ax.transAxes, ha="left",
                va="top", fontsize=FS_BODY, color=INK)
        if facet == 0:
            ax.set_ylabel(r"$r_1(t)$")
        else:
            ax.set_yticklabels([])

    panel_label(axes[0], "b", "Claude collective trajectories")

    # line convention, stated once between the facets
    key = mm_panel(fig, B_L, R1_TOP + B_H + 5.2, B2_L + B_W - B_L, 4.4)
    key.set_xlim(0, 1)
    for i, (ls, lw, txt) in enumerate((
        ("solid", LW_THIN, "core seed"),
        (_DASH, LW_THIN, "held-out seed"),
        ("solid", LW_TRACE, "10-seed mean"),
    )):
        x = 0.10 + i * 0.31
        key.plot([x, x + 0.045], [0.5, 0.5], color=MUTED, lw=lw, ls=ls,
                 clip_on=False)
        key.text(x + 0.058, 0.5, txt, ha="left", va="center", fontsize=FS_TINY,
                 color=MUTED)


def _panel_c(fig, rows) -> dict:
    ax = mm_axes(fig, C_L, R2_TOP, C_W, C_H)
    panel_label(ax, "c", "Claude phenotype matrix")

    core_seeds = sorted({r["seed_index"] for r in rows if r["panel"] == "core"})
    held_seeds = sorted({r["seed_index"] for r in rows if r["panel"] == "heldout"})
    seed_rows = [("core", s) for s in core_seeds] + [("heldout", s) for s in held_seeds]
    cols = [(rep, k) for rep in REP_ORDER for k in K_POS]

    rgb = np.zeros((len(seed_rows), len(cols), 3))
    lock = {"core": {}, "heldout": {}}
    for j, (rep, k) in enumerate(cols):
        for panel in ("core", "heldout"):
            sel = [
                r for r in rows
                if r["panel"] == panel and r["representation"] == rep
                and np.isclose(r["coupling"], k)
            ]
            lock[panel][(rep, k)] = (
                int(sum(r["polar_locked"] for r in sel)), len(sel),
            )
        for i, (panel, seed) in enumerate(seed_rows):
            row = next(
                r for r in rows
                if r["panel"] == panel and r["seed_index"] == seed
                and r["representation"] == rep and np.isclose(r["coupling"], k)
            )
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
                       fontsize=FS_SMALL)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{k:g}" for _, k in cols], fontsize=FS_SMALL)
    ax.tick_params(axis="both", length=0, pad=1.4)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # row-group brackets on the right
    cell_h = C_H / len(seed_rows)
    for label, sub, i0, n in (
        ("core", "GPT-shared", 0, len(core_seeds)),
        ("held-out", "new seeds", len(core_seeds), len(held_seeds)),
    ):
        y_top = R2_TOP + i0 * cell_h
        fig_text_mm(fig, C_L + C_W + 2.0, y_top + n * cell_h / 2.0 - 0.9,
                    label, ha="left", va="center", fontsize=FS_SMALL,
                    color=INK if label == "core" else MUTED, fontweight="bold")
        fig_text_mm(fig, C_L + C_W + 2.0, y_top + n * cell_h / 2.0 + 1.9,
                    sub, ha="left", va="center", fontsize=FS_TINY, color=MUTED)

    # K row, core lock counts row, representation names
    c_bottom = R2_TOP + C_H
    fig_text_mm(fig, C_L - 1.6, c_bottom + 0.7, r"$K$", ha="right", va="top",
                fontsize=FS_SMALL, color=MUTED)
    fig_text_mm(fig, C_L - 1.6, c_bottom + 4.0, "core lock", ha="right",
                va="top", fontsize=FS_TINY, color=MUTED)
    cell_w = C_W / len(cols)
    for j, (rep, k) in enumerate(cols):
        n_lock, n_tot = lock["core"][(rep, k)]
        fig_text_mm(fig, C_L + cell_w * (j + 0.5), c_bottom + 4.0,
                    f"{n_lock}/{n_tot}", ha="center", va="top",
                    fontsize=FS_TINY, color=REP[rep], fontweight="bold")
    for jj, rep in enumerate(REP_ORDER):
        fig_text_mm(fig, C_L + cell_w * (2 * jj + 1), c_bottom + 7.5,
                    REP_SHORT[rep], ha="center", va="top", color=REP[rep],
                    fontweight="bold", fontsize=FS_BODY)

    # phenotype key (only the classes that occur at K > 0)
    present = {r["cluster_phenotype"] for r in rows if r["coupling"] > 0}
    axL = mm_panel(fig, C_L, LEG_TOP, C_W + 14.0, LEG_H)
    handles = [
        mpatches.Patch(facecolor=PHENOTYPE[k], edgecolor="none",
                       label=PHENOTYPE_LABEL[k])
        for k in PHENOTYPE if k in present
    ]
    axL.legend(handles=handles, loc="center left", ncol=3, frameon=False,
               fontsize=FS_TINY, handlelength=1.0, handleheight=1.0,
               columnspacing=1.0, handletextpad=0.4, borderpad=0.0,
               borderaxespad=0.0)
    return lock


def _panel_d(fig, rows, inf: dict) -> dict:
    core_seeds = sorted({r["seed_index"] for r in rows if r["panel"] == "core"})
    held_seeds = sorted({r["seed_index"] for r in rows if r["panel"] == "heldout"})

    specs = (
        ("primary_sustained_lock", "sustained_lock", D1_L, "d",
         "Does encoding affect locking in Claude?",
         r"terminal-lock score $L$", (-0.10, 1.42), [0, 0.5, 1.0],
         r"$p_L$", "{:.2f}"),
        ("secondary_final_r1", "final_r1", D2_L, None, None,
         r"final polar order $Y$", (0.30, 1.22), [0.4, 0.6, 0.8, 1.0],
         r"$p_Y$", "{:.3f}"),
    )
    computed = {}
    for key, field, left, letter, title, ylab, ylim, yticks, plab, mfmt in specs:
        ax = mm_axes(fig, left, R2_TOP, D_W, D_H)
        if letter:
            panel_label(ax, letter, title)
        stats = inf[key]
        core_means = {}
        for i, rep in enumerate(REP_ORDER):
            core_v = np.array([_seed_endpoint(rows, "core", s, rep, field)
                               for s in core_seeds])
            held_v = np.array([_seed_endpoint(rows, "heldout", s, rep, field)
                               for s in held_seeds])
            core_means[rep] = float(core_v.mean())
            ax.scatter(_spread(core_v, i - 0.17), core_v, marker="o",
                       c=REP[rep], s=S_POINT, linewidths=0, zorder=3)
            errorbar_mean(ax, i - 0.17, core_v, color=INK, ms=MS_MEAN,
                          lw=LW_LINE, capsize=CAPSIZE, zorder=5)
            ax.scatter(_spread(held_v, i + 0.30, half=0.07), held_v, marker="^",
                       facecolors="white", edgecolors=REP_LIGHT[rep],
                       linewidths=0.7, s=S_POINT, zorder=3)
        computed[key] = core_means

        # The group means live in the tick labels: printing them above the
        # columns collided with the panel title, and this keeps each number
        # attached to the representation it belongs to.
        means = stats["mean_by_rep"]
        ax.set_xticks(range(3))
        ax.set_xticklabels(
            [f"{REP_SHORT[r]}\n{mfmt.format(means[r])}" for r in REP_ORDER],
            fontsize=FS_TICK, linespacing=1.3,
        )
        for tick, r in zip(ax.get_xticklabels(), REP_ORDER):
            tick.set_color(REP[r])
        ax.set_xlim(-0.62, 2.62)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylab)
        trim_spines(ax, x=False)

        # printed statistics, read from the frozen inference file
        ax.text(0.5, 0.995, f"{plab} = {_pfmt(stats['p_perm'])}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=FS_SMALL, color=INK)

        # Claim boundary: the supported contrast is moments vs the two
        # histogram encodings, so the histogram pair is shaded as one group
        # instead of being bracketed below the axis - a bracket there costs
        # vertical space and reads as a centers-vs-intervals significance bar.
        ax.axvspan(0.58, 2.42, color="#F2F2F2", lw=0, zorder=0)
        ax.text(1.50, 0.012, "histogram encodings",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=FS_TINY, color=MUTED)

    # marker key, permutation note and the claim boundary sentence
    axk = mm_panel(fig, D1_L, R2_TOP + D_H + 9.4, D2_L + D_W - D1_L, 10.4)
    axk.set_xlim(0, 1)
    axk.set_ylim(0, 1)
    prim = inf["primary_sustained_lock"]
    n_perm = f"{prim['n_permutations']:,}".replace(",", " ")
    # Reviewer pass: the key now names what each mark *is* rather than what
    # role it plays in the design - one dot is one physical seed, the black
    # diamond is the group mean with its interval, and the held-out triangles
    # are an independent directional check that no confirmatory p value uses.
    axk.scatter([0.012], [0.90], marker="o", c=MUTED, s=S_POINT, linewidths=0,
                clip_on=False)
    axk.text(0.033, 0.90, "core seed = one physical seed", ha="left",
             va="center", fontsize=FS_TINY, color=MUTED)
    axk.scatter([0.400], [0.90], marker="D", c=INK, s=S_POINT, linewidths=0,
                clip_on=False)
    axk.text(0.421, 0.90, "mean and seed-bootstrap 95% CI (6 core seeds)", ha="left",
             va="center", fontsize=FS_TINY, color=MUTED)
    axk.scatter([0.012], [0.66], marker="^", facecolors="white",
                edgecolors=MUTED, linewidths=0.7, s=S_POINT, clip_on=False)
    axk.text(0.033, 0.66,
             "held-out seed = independent directional check, not in the "
             "confirmatory p-values",
             ha="left", va="center", fontsize=FS_TINY, color=MUTED)
    axk.text(0.0, 0.42,
             f"Exact global permutation, {n_perm} relabellings of the "
             f"{prim['n_seeds']} core seeds; ticks show group means.",
             ha="left", va="center", fontsize=FS_TINY, color=MUTED)
    axk.text(0.0, 0.18,
             "Supported contrast: moments vs the histogram encodings; "
             "centers vs intervals not established.",
             ha="left", va="center", fontsize=FS_TINY, color=INK, style="italic")
    return computed


def _panel_e(fig, rows) -> dict:
    ax = mm_axes(fig, E_L, R3_TOP, E_W, E_H)
    panel_label(ax, "e", "GPT\u2013Claude matched macro map")

    ep = pd.read_csv(PAIRED / "endpoints_long.csv")
    shared = sorted({r["seed_index"] for r in rows if r["panel"] == "core"})
    gpt = ep[(ep["coupling"] > 0) & (ep["seed_index"].isin(shared))]

    mat = np.zeros((2, 3))
    txt = [["", "", ""], ["", "", ""]]
    for j, rep in enumerate(REP_ORDER):
        g = gpt[gpt["representation"] == rep]
        mat[0, j] = float(g["final_r1"].mean())
        txt[0][j] = f"{int(g['locking'].sum())}/{len(g)}"

        c = [r for r in rows if r["panel"] == "core"
             and r["representation"] == rep and r["coupling"] > 0]
        mat[1, j] = float(np.mean([r["final_r1"] for r in c]))
        txt[1][j] = f"{sum(r['polar_locked'] for r in c)}/{len(c)}"

    # A neutral greyscale carries the continuous endpoint, so the
    # representation grammar (blue/orange/green) and the model-family grammar
    # reserved for panel f are not recruited a third time inside one panel.
    im = ax.imshow(mat, cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto",
                   interpolation="nearest")
    for i in range(2):
        for j in range(3):
            col = text_on(im.cmap(im.norm(mat[i, j])))
            ax.text(j, i - 0.15, txt[i][j], ha="center", va="center",
                    fontsize=FS_BODY, fontweight="bold", color=col)
            ax.text(j, i + 0.22, f"$r_1$ {mat[i, j]:.2f}", ha="center",
                    va="center", fontsize=FS_TINY, color=col)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.9)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([REP_SHORT[r] for r in REP_ORDER], fontsize=FS_SMALL)
    for tick, r in zip(ax.get_xticklabels(), REP_ORDER):
        tick.set_color(REP[r])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GPT", "Claude"], fontsize=FS_BODY, fontweight="bold")
    ax.tick_params(axis="both", length=0, pad=1.8)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = cbar_mm(fig, im, E_CB_L, R3_TOP + 1.0, E_CB_W, E_H - 2.0,
                 ticks=[0, 0.5, 1.0])
    cb.set_label(r"mean final $r_1$ ($K>0$)", fontsize=FS_TINY, labelpad=1.6)

    d_gpt = mat[0, 0] - (mat[0, 1] + mat[0, 2]) / 2.0
    d_cla = mat[1, 0] - (mat[1, 1] + mat[1, 2]) / 2.0

    y0 = R3_TOP + E_H + 6.0
    fig_text_mm(fig, E_L - 8.6, y0, "the encoding-to-locking map reverses",
                ha="left", va="baseline", fontsize=FS_SMALL, color=INK,
                fontweight="bold")
    fig_text_mm(fig, E_L - 8.6, y0 + 3.2,
                "6 matched seed-index blocks, $K>0$ \u00b7 cell text = polar-lock runs",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)
    fig_text_mm(fig, E_L - 8.6, y0 + 6.9,
                r"secondary matched summary  "
                r"$\Delta_{\mathrm{M-H}}=Y_M-(Y_C+Y_I)/2$",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)
    fig_text_mm(fig, E_L - 8.6, y0 + 10.1,
                f"GPT ${d_gpt:+.2f}$".replace("-", "\u2212")
                + "   \u00b7   "
                + f"Claude ${d_cla:+.2f}$".replace("-", "\u2212"),
                ha="left", va="baseline", fontsize=FS_TINY, color=INK)
    return {"gpt": d_gpt, "claude": d_cla}


def _panel_f(fig) -> None:
    dec = json.loads((PRIMARY / "decision.json").read_text(encoding="utf-8"))
    fams = ["gpt", "claude", "gemini"]
    pairs = [
        ("moments_m1_m3__vs__centers_24_standard", "M–C"),
        ("moments_m1_m3__vs__intervals_24_decimal6", "M–I"),
        ("centers_24_standard__vs__intervals_24_decimal6", "C–I"),
    ]
    ax = mm_axes(fig, F_L, R3_TOP, F_W, F_H)
    panel_label(ax, "f", "Three-family microscopic operator replication")

    ROW = 0.85
    GRP = 3.95
    col_x = (LETTER_X_RIGHT - F_L) / F_W
    ind_x = (LETTER_X_RIGHT + 2.3 - F_L) / F_W
    ytr = ax.get_yaxis_transform()
    for gi, (pair_key, pair_lab) in enumerate(pairs):
        y0 = gi * GRP
        ax.text(col_x, y0, pair_lab, transform=ytr, ha="left", va="center",
                fontsize=FS_BODY, color=INK, clip_on=False)
        for fi, fam in enumerate(fams):
            yy = y0 + (fi + 1) * ROW
            boot = dec["families"][fam]["primary_target"]["pairwise_bootstrap"][pair_key]
            m, lo, hi = boot["mean"], boot["ci95"][0], boot["ci95"][1]
            ax.errorbar(
                [m], [yy], xerr=[[m - lo], [hi - m]], fmt=FAMILY_MARKER[fam],
                color=FAMILY_EDGE[fam], ecolor=FAMILY_EDGE[fam],
                markerfacecolor="white", markeredgewidth=0.9, ms=MS_SERIES,
                capsize=CAPSIZE, elinewidth=LW_LINE, capthick=0.8, clip_on=False,
            )
            ax.text(ind_x, yy, FAMILY_LABEL[fam], transform=ytr, ha="left",
                    va="center", fontsize=FS_SMALL, color=INK, clip_on=False)
        if gi < len(pairs) - 1:
            ysep = y0 + 3 * ROW + (GRP - 3 * ROW) / 2.0
            ax.plot([col_x, 1.0], [ysep, ysep], transform=ytr, color=RULE,
                    lw=LW_HAIR, clip_on=False, zorder=0)

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.75, 2 * GRP + 3 * ROW + 0.75)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.6)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # "field-cluster 95% CI" named the resampling unit in method language; the
    # interval and its 48-field cluster unit are unchanged, only the wording.
    ax.set_xlabel(r"mean $d_{\mathrm{TV}}$ (95% CI from resampling the 48 physical fields)",
                  fontsize=FS_BODY)
    for gx in (0.1, 0.2, 0.3, 0.4, 0.5):
        ax.axvline(gx, color="#EDEDED", lw=0.4, zorder=0)


def _scope(fig) -> None:
    axS = mm_panel(fig, F_L, SCOPE_TOP, F_W, SCOPE_H)
    axS.set_xlim(0, 1)
    axS.set_ylim(0, 1)
    # plain rectangle in axes coordinates: a rounded box drawn in these units
    # would have a corner radius the size of the whole panel
    axS.add_patch(
        mpatches.Rectangle((0, 0), 1, 1, transform=axS.transAxes,
                           facecolor="#F7F7F7", edgecolor="#B0B0B0",
                           lw=LW_THIN, zorder=0)
    )
    lines = (
        ("Macro encoding dependence", "GPT + Claude", INK, "bold"),
        ("Microscopic operator dependence", "GPT + Claude + Gemini", INK, "bold"),
    )
    for i, (lhs, rhs, col, wt) in enumerate(lines):
        y = 0.70 - i * 0.40
        axS.text(0.030, y, lhs, ha="left", va="center", fontsize=FS_TINY,
                 color=col)
        axS.text(0.480, y, rhs, ha="left", va="center", fontsize=FS_TINY,
                 color=col, fontweight=wt)


# ---------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    rows = _rows()
    inf = _inference()
    k0 = _k0_control()

    fig = new_figure(H_MM)
    _panel_a(fig, k0)
    _panel_b(fig, rows)
    _panel_c(fig, rows)
    _panel_d(fig, rows, inf)
    _panel_e(fig, rows)
    _panel_f(fig)
    _scope(fig)
    return save_fig(fig, "fig4_cross_family")
