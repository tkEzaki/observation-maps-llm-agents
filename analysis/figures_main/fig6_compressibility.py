"""Figure 6 - Compressibility does not ensure closed-loop support.

Geometry pass only: identical data, statistics and printed numbers as the
earlier build. The canvas is now a true 180 mm journal double
column (the earlier version cropped out to 206 mm, over the journal maximum),
and panel a's 4-stage flow plus its PASS/STOP table are folded into a single
aligned stage x representation grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from analysis.figures_main.style import (
    FS_BODY,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    INK,
    MUTED,
    PASS_GREEN,
    REP,
    REP_LIGHT,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    STOP_RED,
    apply_style,
    blank,
    fig_text_mm,
    mm_axes,
    new_figure,
    paired_ci,
    panel_label_at,
    rule_mm,
    save_fig,
    trim_spines,
)

SYN = ROOT / "analysis" / "synthesis"
SC = ROOT / "analysis" / "stage_c_artifacts" / "stage_c_v0_2_combined"

BUCKET_COLOR = {
    "local_noncompressible": "#C0392B",
    "active_feature_gap": "#E8A13C",
    "supported": "#4C9A6E",
}
BUCKET_LABEL = {
    "local_noncompressible": "locally non-compressible",
    "active_feature_gap": "active feature gap",
    "supported": "supported",
}

# Private to this figure (style.py is shared and must not be edited).
CHIP_PASS_BG = "#E7F2EA"
CHIP_STOP_BG = "#FAEAEA"

# Reader-facing wording for the prespecified dispositions recorded in
# transport_funnel.csv (values are read from the file, never hard-coded).
DISPOSITION_TEXT = {
    "frozen_stage_c": "collective path frozen",
    "stopped_microscopic_archive": "stopped; microscopic archive",
}

# --- canvas plan (mm from top-left) -----------------------------------------
H_MM = 135.5
COL1_X, COL2_X = 4.0, 96.0        # panel-letter columns
AX1_X, AX2_X = 15.0, 107.0        # plot-box columns
AX_W = 68.0                       # shared plot-box width for b/c/d/e

A_TOP, A_H = 8.8, 28.0
R2_TOP, R3_TOP = 45.0, 88.0
R3_H = 19.5
C_H = 32.0                        # panel c: the two-line y label sets the height
F_RULE_Y = 115.5                  # footer hairline
F_TOP, F_BOX_H = 121.0, 6.4       # disposition status-box strip


def _share_axis(ax, n_label: str) -> None:
    """Matched x-axis treatment for the paired share panels d and e."""
    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.set_xlabel(f"share of local diagnoses ($n$={n_label})", fontsize=FS_BODY)
    ax.tick_params(axis="x", labelsize=FS_TICK)


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    funnel = pd.read_csv(SYN / "transport_funnel.csv")
    comp = pd.read_csv(SYN / "fig2_compressibility.csv")
    scatter = pd.read_csv(SYN / "fig4_support_scatter.csv")
    runs = pd.read_csv(SC / "combined_endpoints_per_run.csv")
    runs = runs[(runs["n_agents"] == 17) & (runs["coupling"] > 0)].copy()

    fig = new_figure(H_MM)

    # ==================================================================
    # a  transportability funnel: stages as columns, representations as rows
    # ==================================================================
    axA = mm_axes(fig, COL1_X, A_TOP, 173.0, A_H)
    blank(axA)
    axA.set_xlim(0, 173)
    axA.set_ylim(A_H, 0)          # y grows downward, 1 unit == 1 mm
    panel_label_at(fig, COL1_X, A_TOP - 2.3, "a",
                   "Empirical surrogate-evaluation funnel")

    # Column names follow the four evaluation levels, not the internal
    # project stage names; the first column is training data, not a test.
    stages = ["Training\nlabels", "In-domain\ngrouped OOF",
              "Closed-loop\nsupport", "Prospective\nevaluation"]
    col_w, box_w = 38.4, 34.0
    grid_x0 = 19.0
    xs = [grid_x0 + col_w * i + col_w / 2 for i in range(4)]

    for x, s in zip(xs, stages):
        axA.add_patch(
            mpatches.FancyBboxPatch(
                (x - box_w / 2, 0.4), box_w, 6.0,
                boxstyle="round,pad=0,rounding_size=1.0",
                facecolor="#F5F5F5", edgecolor="#8C8C8C", lw=0.55,
                mutation_aspect=1.0,
            )
        )
        axA.text(x, 3.4, s, ha="center", va="center", fontsize=FS_SMALL,
                 linespacing=1.35, color=INK)
        if x != xs[-1]:
            axA.annotate(
                "", xy=(x + col_w / 2 + 2.1, 3.4), xytext=(x + box_w / 2 + 0.4, 3.4),
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.7,
                                mutation_scale=6, shrinkA=0, shrinkB=0),
            )

    axA.plot([0, 173], [8.3, 8.3], color=RULE, lw=0.5, clip_on=False)

    paths = {
        "moments_m1_m3": ("fitted", "PASS", "PASS after repair",
                          "PASS, performed"),
        "centers_24_standard": ("fitted", "PASS", "STOP: feature gap",
                                "not performed"),
        "intervals_24_decimal6": ("fitted", "PASS", "STOP: noncompressible",
                                  "not performed"),
    }
    row_y = [11.6, 16.0, 20.4]
    for y, rep in zip(row_y, REP_ORDER):
        axA.text(grid_x0 - 2.0, y, REP_SHORT[rep], color=REP[rep], fontsize=FS_BODY,
                 va="center", ha="right", fontweight="bold")
        for x, lab in zip(xs, paths[rep]):
            ok = lab.startswith("PASS")
            stop = lab.startswith("STOP")
            if ok or stop:
                axA.add_patch(
                    mpatches.FancyBboxPatch(
                        (x - box_w / 2, y - 1.75), box_w, 3.5,
                        boxstyle="round,pad=0,rounding_size=0.8",
                        facecolor=CHIP_PASS_BG if ok else CHIP_STOP_BG,
                        edgecolor="none", lw=0.0, mutation_aspect=1.0, zorder=1,
                    )
                )
            color = PASS_GREEN if ok else (STOP_RED if stop else "#9A9A9A")
            axA.text(x, y, lab, ha="center", va="center", fontsize=FS_SMALL,
                     color=color, fontweight="bold" if (ok or stop) else "normal",
                     zorder=2)

    axA.text(
        xs[0] - box_w / 2, 22.6,
        "Moments localized regime error was diagnosed by replay and repaired "
        "prospectively.\nCenters and intervals stop rules were prespecified.",
        ha="left", va="top", fontsize=FS_TINY, style="italic", color=MUTED,
        linespacing=1.45,
    )

    # ==================================================================
    # b  IID OOF evidence forest
    # ==================================================================
    axB = mm_axes(fig, AX1_X, R2_TOP, AX_W, 20.0)
    panel_label_at(fig, COL1_X, R2_TOP - 2.3, "b",
                   "Histogram encodings: field-grouped in-domain OOF evidence")
    b_reps = ("centers_24_standard", "intervals_24_decimal6")
    yticks, ylabs = [], []
    for y, rep in enumerate(b_reps):
        row = comp[comp["representation"] == rep].iloc[0]
        m = float(row["delta_ll_mean"])
        lo = float(row["ci_low_field_cluster"])
        hi = float(row["ci_high_field_cluster"])
        axB.errorbar(
            [m], [y], xerr=[[m - lo], [hi - m]], fmt="o", color=REP[rep],
            ms=3.2, capsize=2.0, elinewidth=0.9, capthick=0.9, clip_on=False,
        )
        yticks.append(y)
        ylabs.append(REP_SHORT[rep])
    axB.axvline(0, color="#B0B0B0", lw=0.5, zorder=0)
    axB.set_yticks(yticks)
    axB.set_yticklabels(ylabs)
    for tick, rep in zip(axB.get_yticklabels(), b_reps):
        tick.set_color(REP[rep])
    axB.tick_params(axis="y", length=0)
    axB.set_ylim(-0.62, 1.62)
    axB.set_xlim(-0.02, 0.5)
    axB.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axB.set_xlabel(r"$\Delta$ log-loss (field-cluster 95% CI)")
    axB.spines["left"].set_visible(False)
    trim_spines(axB, y=False)

    fig_text_mm(
        fig, AX1_X, R2_TOP + 30.0,
        "moments: in-domain criteria met under its own prespecified gates\n"
        "(no peer-global ΔLL row in this table)",
        fontsize=FS_TINY, color=REP["moments_m1_m3"], va="top", ha="left",
        linespacing=1.45,
    )

    # ==================================================================
    # c  moments prospective repair v1 -> v2
    # ==================================================================
    axC = mm_axes(fig, AX2_X, R2_TOP, AX_W, C_H)
    panel_label_at(fig, COL2_X, R2_TOP - 2.3, "c",
                   "Moments: prospective repair (v1→v2)")
    metrics = [("eA_v1", "eA_v2", "activity"), ("estay_v1", "estay_v2", "stay"),
               ("er1_v1", "er1_v2", r"final $r_1$")]
    for i, (c1, c2, lab) in enumerate(metrics):
        v1 = runs[c1].to_numpy()
        v2 = runs[c2].to_numpy()
        for a, b in zip(v1, v2):
            axC.plot([i - 0.16, i + 0.16], [a, b], color="#DCDCDC", lw=0.4, zorder=1)
        axC.scatter(np.full(len(v1), i - 0.16), v1, s=4.5,
                    c=REP_LIGHT["moments_m1_m3"], zorder=2, linewidths=0)
        axC.scatter(np.full(len(v2), i + 0.16), v2, s=4.5,
                    c=REP["moments_m1_m3"], zorder=2, linewidths=0)
        for vals, dx, color in ((v1, -0.16, "#555555"), (v2, 0.16, "k")):
            m, lo, hi = paired_ci(vals)
            axC.errorbar(
                [i + dx], [m], yerr=[[m - lo], [hi - m]], fmt="D", color=color,
                ms=2.0, capsize=1.6, elinewidth=0.7, capthick=0.7, zorder=4,
            )
    axC.set_xticks(range(3))
    axC.set_xticklabels([m[2] for m in metrics])
    axC.set_xlim(-0.45, 2.45)
    axC.tick_params(axis="x", length=0)
    # The quantity is |compressed-surrogate prediction − LLM-driven outcome|
    # on held-out prospective collective cells; set over two lines so the label
    # stays inside the panel box and clear of the panel letter.
    axC.set_ylabel("absolute prediction error on\nprospective LLM collective cells",
                   fontsize=FS_SMALL, linespacing=1.35)
    trim_spines(axC, x=False)

    # v1 / v2 key, lifted out of the plot area onto the panel-title line
    for dx, col, lab in ((0.0, REP_LIGHT["moments_m1_m3"], "v1"),
                         (7.0, REP["moments_m1_m3"], "v2")):
        fig.text((AX2_X + AX_W - 14.0 + dx) / 180.0,
                 (H_MM - (R2_TOP - 2.5)) / H_MM, "●", color=col,
                 fontsize=FS_SMALL, ha="left", va="baseline")
        fig.text((AX2_X + AX_W - 10.6 + dx) / 180.0,
                 (H_MM - (R2_TOP - 2.3)) / H_MM, lab, color=INK,
                 fontsize=FS_SMALL, ha="left", va="baseline")

    # ==================================================================
    # d  centers: feature-gap transfer failure
    # ==================================================================
    axD = mm_axes(fig, AX1_X, R3_TOP, AX_W, R3_H)
    panel_label_at(fig, COL1_X, R3_TOP - 2.3, "d",
                   "Centers: closed-loop support failure (feature gap)")
    sub_c = scatter[scatter["representation"].str.contains("centers", case=False,
                                                           na=False)]
    n_c = len(sub_c)
    cnt_c = sub_c["diagnosis_bucket"].value_counts()
    order_d = ["active_feature_gap", "local_noncompressible", "supported"]
    cnts_d = [int(cnt_c.get(k, 0)) for k in order_d]
    vals_d = [c / n_c for c in cnts_d]
    ys = np.arange(len(order_d), dtype=float)
    axD.barh(ys, vals_d, color=[BUCKET_COLOR[k] for k in order_d], height=0.42)
    for yy, v, c, k in zip(ys, vals_d, cnts_d, order_d):
        axD.text(0.004, yy - 0.40, BUCKET_LABEL[k], va="baseline", ha="left",
                 fontsize=FS_SMALL, color=BUCKET_COLOR[k])
        axD.text(v + 0.012, yy, f"{v:.0%} ($n$={c})", va="center", ha="left",
                 fontsize=FS_SMALL, color=INK)
    axD.set_yticks([])
    axD.set_ylim(2.62, -0.72)
    axD.spines["left"].set_visible(False)
    _share_axis(axD, str(n_c))
    trim_spines(axD, y=False)

    # ==================================================================
    # e  intervals: local noncompressibility
    # ==================================================================
    axE = mm_axes(fig, AX2_X, R3_TOP, AX_W, R3_H)
    panel_label_at(fig, COL2_X, R3_TOP - 2.3, "e",
                   "Intervals: local noncompressibility")
    sub_i = scatter[scatter["representation"].str.contains("intervals", case=False,
                                                           na=False)]
    n_i = len(sub_i)
    cnt_i = sub_i["diagnosis_bucket"].value_counts()
    order_e = ["local_noncompressible", "active_feature_gap", "supported"]
    cnts_e = [int(cnt_i.get(k, 0)) for k in order_e]
    vals_e = [c / n_i for c in cnts_e]
    left = 0.0
    y_bar = 1.0                      # vertically centred against d's three bars
    for v, c, o in zip(vals_e, cnts_e, order_e):
        axE.barh([y_bar], [v], left=left, color=BUCKET_COLOR[o], height=1.05,
                 edgecolor="white", linewidth=0.6)
        if v > 0.10:
            axE.text(left + v / 2, y_bar, f"{BUCKET_LABEL[o]}\n{v:.0%} ($n$={c})",
                     ha="center", va="center", fontsize=FS_TINY, color="white",
                     fontweight="bold", linespacing=1.3)
        else:
            axE.plot([left + v / 2, left + v / 2], [y_bar + 0.56, y_bar + 0.80],
                     color=BUCKET_COLOR[o], lw=0.5, clip_on=False)
            axE.text(min(left + v / 2 + 0.01, 1.0), y_bar + 0.88,
                     f"{BUCKET_LABEL[o]} {v:.0%} ($n$={c})", ha="right", va="top",
                     fontsize=FS_SMALL, color=BUCKET_COLOR[o])
        left += v
    axE.set_yticks([])
    axE.set_ylim(2.62, -0.72)
    axE.spines["left"].set_visible(False)
    _share_axis(axE, str(n_i))
    trim_spines(axE, y=False)

    axD.text(1.0, -0.66, "ID structure ≠ finite-peer abstention", ha="right",
             va="baseline", fontsize=FS_TINY, color=MUTED, style="italic")
    axE.text(1.0, -0.66, "neighbors exist; responses disagree", ha="right",
             va="baseline", fontsize=FS_TINY, color=MUTED, style="italic")

    # ==================================================================
    # f  final disposition: one status box per representation, then the claim
    # ==================================================================
    rule_mm(fig, COL1_X, 177.0, F_RULE_Y)
    panel_label_at(fig, COL1_X, F_RULE_Y + 4.0, "f", "Final disposition")

    axF = mm_axes(fig, COL1_X, F_TOP, 173.0, F_BOX_H)
    blank(axF)
    axF.set_xlim(0, 173)
    axF.set_ylim(F_BOX_H, 0)          # y grows downward, 1 unit == 1 mm
    gap_f = 2.5
    box_w_f = (173.0 - 2 * gap_f) / 3.0
    for i, rep_key in enumerate(REP_ORDER):
        x0 = i * (box_w_f + gap_f)
        code = funnel.loc[funnel["representation"] == rep_key,
                          "disposition"].iloc[0]
        axF.add_patch(
            mpatches.FancyBboxPatch(
                (x0, 0.35), box_w_f, F_BOX_H - 0.7,
                boxstyle="round,pad=0,rounding_size=1.0",
                facecolor=to_rgba(REP_LIGHT[rep_key], 0.45),
                edgecolor=REP[rep_key], lw=0.55, mutation_aspect=1.0,
            )
        )
        axF.text(x0 + 3.0, F_BOX_H / 2.0, REP_SHORT[rep_key], ha="left",
                 va="center", fontsize=FS_SMALL, fontweight="bold",
                 color=REP[rep_key])
        axF.text(x0 + 14.5, F_BOX_H / 2.0,
                 DISPOSITION_TEXT.get(code, code.replace("_", " ")),
                 ha="left", va="center", fontsize=FS_SMALL, color=INK)

    fig_text_mm(fig, COL1_X, F_TOP + F_BOX_H + 4.4,
                "compressibility PASS does not guarantee prospective eligibility",
                fontsize=FS_SMALL, color=INK, ha="left", va="baseline",
                fontweight="bold")

    return save_fig(fig, "fig6_compressibility_transportability")
