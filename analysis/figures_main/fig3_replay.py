"""Figure 3 - Cross-encoding replay on identical physical states.

Geometry pass only: same data, same statistics, same printed numbers and the
same CVD-validated colours as the earlier variant. What changes is the
canvas (a true 180 mm page placed in millimetres from the top-left), the
inter-panel gutters, the relative area given to schematic vs evidence panels,
and the placement of every annotation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from analysis.figures_main.style import (
    ACCENT_RED,
    DATA_DIR,
    FS_BODY,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    INK,
    MUTED,
    REP,
    REP_ABBR,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    VERDICT,
    annotate_cells,
    apply_style,
    cbar_mm,
    despine,
    mm_axes,
    mm_panel,
    new_figure,
    LW_FAINT,
    LW_TRACE,
    LW_THIN,
    LW_LINE,
    MS_MEAN,
    CAPSIZE,
    panel_label,
    panel_label_at,
    save_fig,
    swarm_x,
    trim_spines,
)

PRIMARY = ROOT / "analysis" / "matched_rep_collective" / "replay_primary"

PAIR_COLOR = "#5B6770"  # single neutral ink; pair identity lives on the axis

# --- canvas plan (mm from top-left) ----------------------------------------
H_MM = 130.0

ROW1_TOP = 8.0
ROW2_TOP = 51.0
ROW3_TOP = 94.0
LETTER_DY = 1.6          # matches style.LETTER_DY_MM
LET_Y1 = ROW1_TOP - LETTER_DY
LEFT_LETTER = 5.4        # shared optical column for the left-most letters

A_LEFT, A_W, A_H = 8.0, 53.0, 30.0
HM = 29.0                # heat-map side (b and c are exact twins)
B_LEFT, C_LEFT = 74.0, 127.0
CB_GAP, CB_W = 2.5, 2.4

D_LEFT, D_W, ROW2_H = 14.0, 92.0, 29.0
E_LEFT, E_W = 121.0, 53.0

F_LEFT, F_W, ROW3_H = 14.0, 58.0, 26.0
G_LEFT, G_W = 88.0, 86.0


def _heatmap(fig, left, mat, cmap, vmin, vmax, fmt):
    """One of the two 3x3 twins: identical cell size, identical colourbar."""
    ax = mm_axes(fig, left, ROW1_TOP, HM, HM)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    annotate_cells(ax, mat, fmt=fmt, cmap=im.cmap, norm=im.norm, fontsize=FS_SMALL)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="both", length=0, pad=1.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels([REP_ABBR[r] for r in REP_ORDER], fontsize=FS_TICK)
    ax.set_yticklabels([REP_ABBR[r] for r in REP_ORDER], fontsize=FS_TICK)
    for tick, r in zip(ax.get_xticklabels(), REP_ORDER):
        tick.set_color(REP[r])
    for tick, r in zip(ax.get_yticklabels(), REP_ORDER):
        tick.set_color(REP[r])
    ax.set_xlabel("presented", labelpad=1.4)
    ax.set_ylabel("source", labelpad=1.4)
    cbar_mm(fig, im, left + HM + CB_GAP, ROW1_TOP, CB_W, HM)
    return ax


def _design_schematic(fig):
    """Panel a: compressed 3x3 source x target grid."""
    ax = mm_panel(fig, A_LEFT, ROW1_TOP, A_W, A_H)
    ax.set_xlim(0, A_W)
    ax.set_ylim(A_H, 0)          # mm, top-down

    cell_w, cell_h = 11.0, 5.4
    gap_x, gap_y = 1.6, 1.1
    x0, y0 = 15.0, 7.2
    xs = [x0 + k * (cell_w + gap_x) for k in range(3)]
    ys = [y0 + k * (cell_h + gap_y) for k in range(3)]
    right = xs[-1] + cell_w

    ax.text((x0 + right) / 2, 1.4, "presented encoding", ha="center", va="center",
            fontsize=FS_TINY, color=MUTED)
    for k, r in enumerate(REP_ORDER):
        ax.text(xs[k] + cell_w / 2, 4.6, REP_SHORT[r], ha="center", va="center",
                fontsize=FS_TINY, color=REP[r])
        ax.text(x0 - 1.6, ys[k] + cell_h / 2, REP_SHORT[r], ha="right",
                va="center", fontsize=FS_TINY, color=REP[r])
    ax.text(1.0, (ys[0] + ys[-1] + cell_h) / 2, "source trajectory", rotation=90,
            ha="center", va="center", fontsize=FS_TINY, color=MUTED)

    for yy in ys:
        for xx in xs:
            ax.add_patch(mpatches.FancyBboxPatch(
                (xx, yy), cell_w, cell_h, boxstyle="round,pad=0,rounding_size=0.7",
                facecolor="#F7F7F7", edgecolor="#B4B4B4", lw=0.5))
            ax.text(xx + cell_w / 2, yy + cell_h / 2, r"$\rho$", ha="center",
                    va="center", fontsize=FS_TICK, color="#4D4D4D")

    ax.text((x0 + right) / 2, ys[-1] + cell_h + 3.0,
            r"$48\times3\times32=4608$ replays", ha="center", va="center",
            fontsize=FS_TINY, color=MUTED)
    return ax


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    dec = json.loads((PRIMARY / "decision.json").read_text(encoding="utf-8"))
    fields = pd.DataFrame(json.loads((PRIMARY / "field_pairwise_tv.json").read_text(encoding="utf-8")))
    noise_pairs = pd.read_csv(DATA_DIR / "replay_field_noise_pairs.csv")
    null_tv = np.load(DATA_DIR / "replay_null_tv.npy")
    null_meta = json.loads((DATA_DIR / "replay_null_meta.json").read_text(encoding="utf-8"))
    tgt = dec["primary_global_tests"]["1_target_main_effect"]
    verdict = dec["mechanistic_verdict"]
    matA = dec["matrix_3x3"]["A"]
    matA0 = dec["matrix_3x3"]["a0"]

    fig = new_figure(H_MM)

    # ---- a  design ---------------------------------------------------------
    _design_schematic(fig)
    panel_label_at(fig, LEFT_LETTER, LET_Y1, "a", "3×3 replay design")

    # ---- b  activity heatmap ----------------------------------------------
    A = np.array([[matA[s][t] for t in REP_ORDER] for s in REP_ORDER], dtype=float)
    axB = _heatmap(fig, B_LEFT, A, "Reds", 0.7, 1.0, "{:.2f}")
    panel_label(axB, "b", r"Activity $A_{S,\mathcal{R}}$")

    # ---- c  signed action --------------------------------------------------
    A0 = np.array([[matA0[s][t] for t in REP_ORDER] for s in REP_ORDER], dtype=float)
    lim = max(0.35, float(np.max(np.abs(A0))))
    axC = _heatmap(fig, C_LEFT, A0, "RdBu_r", -lim, lim, "{:+.2f}")
    panel_label(axC, "c", r"Signed action $a_{0,S,\mathcal{R}}$")

    # ---- d  pairwise dTV swarm --------------------------------------------
    axD = mm_axes(fig, D_LEFT, ROW2_TOP, D_W, ROW2_H)
    panel_label(axD, "d", r"Pairwise presented $d_{\mathrm{TV}}$ (48 fields)")
    pairs = [
        ("dTV_MC", "M–C", tgt["pairwise_bootstrap"]["moments_m1_m3__vs__centers_24_standard"]),
        ("dTV_MI", "M–I", tgt["pairwise_bootstrap"]["moments_m1_m3__vs__intervals_24_decimal6"]),
        ("dTV_CI", "C–I", tgt["pairwise_bootstrap"]["centers_24_standard__vs__intervals_24_decimal6"]),
    ]
    for i, (col, lab, boot) in enumerate(pairs):
        vals = fields[col].to_numpy()
        axD.scatter(swarm_x(vals, i, 0.22), vals, s=4.0, c=PAIR_COLOR, alpha=0.42,
                    zorder=2, linewidths=0)
        m, lo, hi = boot["mean"], boot["ci95"][0], boot["ci95"][1]
        axD.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="D", color=INK,
                     ms=MS_MEAN, capsize=CAPSIZE, elinewidth=LW_LINE, capthick=LW_LINE, zorder=4)
        # mean printed on one shared baseline above the panel, centred on its group
        axD.text(i, 1.075, f"{m:.3f}", ha="center", va="center", fontsize=FS_SMALL,
                 color=INK, fontweight="bold")
    axD.set_xticks([0, 1, 2])
    axD.set_xticklabels([p[1] for p in pairs])
    axD.set_xlim(-0.5, 2.5)
    axD.set_ylabel(r"$d_{\mathrm{TV}}$")
    axD.set_ylim(0, 1.15)
    axD.set_yticks([0, 0.5, 1.0])
    despine(axD)
    trim_spines(axD, x=False)

    # ---- e  separation vs block drift -------------------------------------
    axE = mm_axes(fig, E_LEFT, ROW2_TOP, E_W, ROW2_H)
    panel_label(axE, "e", "Encoding separation versus test-retest variation (block noise)")
    for _, row in noise_pairs.iterrows():
        axE.plot([0, 1], [row["within_block_TV"], row["between_target_TV"]],
                 color="#D9D9D9", lw=LW_FAINT, zorder=1)
    axE.scatter(np.zeros(len(noise_pairs)), noise_pairs["within_block_TV"],
                s=3.5, c="#AAAAAA", zorder=2, linewidths=0)
    axE.scatter(np.ones(len(noise_pairs)), noise_pairs["between_target_TV"],
                s=3.5, c="#555555", zorder=2, linewidths=0)
    axE.plot([0, 1],
             [noise_pairs["within_block_TV"].mean(), noise_pairs["between_target_TV"].mean()],
             "o-", color=ACCENT_RED, lw=LW_TRACE, ms=MS_MEAN, zorder=4)
    axE.set_xticks([0, 1])
    axE.set_xticklabels(["same encoding\nrepeated", "different\nencodings"])
    axE.set_xlim(-0.3, 1.3)
    axE.set_ylabel(r"$d_{\mathrm{TV}}$")
    axE.set_ylim(-0.02, 1.02)
    axE.set_yticks([0, 0.5, 1.0])
    despine(axE)
    trim_spines(axE, x=False)
    nf = tgt["noise_floor"]
    axE.text(0.10, 0.99,
             f"{nf['mean_between_target_TV']:.3f} vs {nf['mean_within_target_block_TV']:.3f}\n"
             f"{nf['ratio']:.2f}×, $p$=0.0002",
             transform=axE.transAxes, ha="left", va="top", fontsize=FS_SMALL,
             fontweight="bold", color=ACCENT_RED, linespacing=1.35)

    # ---- f  permutation null ----------------------------------------------
    axF = mm_axes(fig, F_LEFT, ROW3_TOP, F_W, ROW3_H)
    panel_label(axF, "f", "Separation when encoding labels do not matter")
    axF.hist(null_tv, bins=50, color="#C9CDD2", edgecolor="white", linewidth=0.2)
    obs = null_meta["observed_mean_pairwise_TV"]
    axF.axvline(obs, color=ACCENT_RED, lw=LW_TRACE, zorder=4)
    axF.axvline(null_meta["null_mean"], color="#666666", lw=LW_THIN, ls=(0, (2.5, 1.8)))
    axF.set_xlabel(r"mean pairwise $d_{\mathrm{TV}}$")
    axF.set_ylabel("count")
    axF.set_xlim(0.05, 0.40)
    axF.set_xticks([0.1, 0.2, 0.3, 0.4])
    despine(axF)
    ytop = axF.get_ylim()[1]
    axF.set_ylim(0, ytop)
    trim_spines(axF, x=False)
    axF.text(null_meta["null_mean"] + 0.007, ytop * 0.99, "null mean",
             fontsize=FS_TINY, color="#666666", ha="left", va="top")
    # annotation block sits in the empty middle of the panel and points at the
    # observed line, instead of being right-aligned on top of it
    axF.text(
        0.205, ytop * 0.99,
        f"observed = {obs:.3f}\n$p_{{\\mathrm{{perm}}}}$ = 0.0002\n(5,000 permutations)",
        fontsize=FS_TINY, color=ACCENT_RED, ha="left", va="top", linespacing=1.5,
    )

    # ---- g  verdict table --------------------------------------------------
    axG = mm_axes(fig, G_LEFT, ROW3_TOP, G_W, ROW3_H)
    panel_label(axG, "g", "Questions tested")
    axG.set_xlim(0, G_W)
    axG.set_ylim(ROW3_H, 0)       # mm, top-down
    axG.axis("off")

    x_eff, x_p, x_int = 1.6, 49.0, 57.5
    # The frozen decision artifact labels the source effect
    # "suggested_ns_at_0.05" and the interaction "not_established". Both are
    # non-significant at the prespecified alpha, so the panel reports them at
    # equal status; the "_ns_at_0.05" qualifier is not dropped here.
    #
    # Reviewer pass: the left column now states each prespecified test as the
    # question a reader would ask, instead of naming the design term ("Target",
    # "Source", "Source x target"). The verdict words and every p value are
    # unchanged - only the wording of the row label moved. The questions are
    # hand-wrapped to two lines so each line clears the p column at x_p.
    rows = [
        ("Does re-encoding change the response\nto the same field?",
         f"{verdict['target_p']:.4f}", "Supported", VERDICT["supported"]),
        ("Do fields generated under different encodings\ndiffer on average?",
         f"{verdict['source_p']:.3f}", "Not conclusive",
         VERDICT["not_established"]),
        ("Does that effect depend on which encoding\ngenerated the field?",
         f"{verdict['interaction_p']:.3f}", "Not conclusive",
         VERDICT["not_established"]),
    ]
    # the prespecified alpha moves from the panel title into the verdict column
    # header, so making the title a question costs the reader no information
    for x, head in ((x_eff, "Question"), (x_p, r"$p$"),
                    (x_int, r"Verdict ($\alpha$ = 0.05)")):
        axG.text(x, 2.0, head, fontsize=FS_SMALL, fontweight="bold", va="center",
                 ha="left", color=INK)
    axG.plot([0, G_W], [3.9, 3.9], color=RULE, lw=0.6, clip_on=False)

    box_h, pitch, y_first = 5.4, 6.3, 7.2
    for k, (name, p, verd, col) in enumerate(rows):
        yc = y_first + k * pitch
        axG.add_patch(mpatches.FancyBboxPatch(
            (0.0, yc - box_h / 2), G_W, box_h,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor=col, alpha=0.10, edgecolor=col, lw=0.55))
        axG.text(x_eff, yc, name, fontsize=FS_SMALL, va="center", ha="left",
                 color=INK, linespacing=1.05)
        axG.text(x_p, yc, p, fontsize=FS_BODY, va="center", ha="left", color=INK)
        axG.text(x_int, yc, verd, fontsize=FS_BODY, va="center", ha="left",
                 fontweight="bold", color=col)

    axG.text(G_W / 2, y_first + 2 * pitch + box_h / 2 + 2.4,
             r"Strong presented-encoding effect at a fixed field."
             "\n"
             r"No conclusive evidence either way for source-dependent modulation.",
             ha="center", va="center", fontsize=FS_TINY, style="italic",
             color=MUTED, linespacing=1.25)

    return save_fig(fig, "fig3_replay_operator")
