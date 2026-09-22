"""Figure 5 - Task-information-fixed presentation control.

Geometry redesign only: identical data, statistics and printed numbers as the
earlier variant. Laid out on a fixed 180 mm canvas with millimetre panel
placement.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from analysis.plot_main.style import (
    ACTION,
    ACTION_LABELS,
    CAPSIZE,
    DATA_DIR,
    FS_BODY,
    FS_SMALL,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    LW_TRACE,
    MS_MEAN,
    MUTED,
    OMAP,
    OMAP_MID,
    ROOT,
    apply_style,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label,
    panel_label_at,
    save_fig,
    swarm_x,
    trim_spines,
)

PRIMARY = ROOT / "analysis" / "matched_rep_collective" / "omap_primary"
CROSSREP = "#5B6770"

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm wide sheet)
# ---------------------------------------------------------------------------
H_MM = 129.0

A_LEFT, A_TOP, A_W, A_H = 5.0, 7.6, 170.0, 17.5      # row 1: schematic band
R2_TOP, R2_H = 34.6, 32.0                            # row 2: b | c
B_LEFT, B_W = 13.0, 92.0
C_LEFT, C_W = 130.0, 45.0
R3_TOP, R3_H = 81.6, 30.0                            # row 3: d | e | f
D_LEFT, D_W = 13.0, 45.0
E_LEFT, E_W = 70.0, 53.0
F_LEFT, F_W = 130.0, 47.0

LETTER_X = 4.4          # shared optical column for the left-hand letters
ROW1_LETTER_Y = 6.0
ROW3_LETTER_Y = R3_TOP - 1.6


def _mm_swatch(fig, x_mm, y_mm, w_mm, h_mm, color):
    """Small filled key swatch placed in mm from the canvas top-left."""
    W, H = 180.0, H_MM
    r = mpatches.Rectangle(
        (x_mm / W, (H - y_mm - h_mm) / H), w_mm / W, h_mm / H,
        transform=fig.transFigure, facecolor=color, edgecolor="#8C8C8C",
        lw=LW_HAIR, clip_on=False,
    )
    fig.add_artist(r)
    return r


def _seg_labels(ax, x, tri, thresh=0.12, fontsize=FS_TINY):
    """Consistent in-bar action labels: every segment above ``thresh`` is named."""
    base = 0.0
    for k, lab in enumerate(ACTION_LABELS):
        v = tri[k]
        if v > thresh:
            col = "white" if k in (0, 2) else "#333333"
            ax.text(x, base + v / 2.0, lab, ha="center", va="center",
                    color=col, fontsize=fontsize)
        base += v


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    dec = json.loads((PRIMARY / "decision.json").read_text(encoding="utf-8"))
    lead = dec["lead_with_pairwise"]
    pw = dec["same_info_pairwise_bootstrap"]
    field_tv = pd.read_csv(DATA_DIR / "omap_field_pairwise_tv.csv")
    null_tv = np.load(DATA_DIR / "omap_null_tv.npy")
    null_meta = json.loads((DATA_DIR / "omap_null_meta.json").read_text(encoding="utf-8"))

    fig = new_figure(H_MM)

    # ------------------------------------------------------------------
    # a  design of the same-task-information control (compact flow band)
    # ------------------------------------------------------------------
    axA = mm_panel(fig, A_LEFT, A_TOP, A_W, A_H)
    axA.set_xlim(0, A_W)
    axA.set_ylim(0, A_H)
    panel_label_at(fig, LETTER_X, ROW1_LETTER_Y, "a",
                   r"Same task-information control: identical $(\tilde z_1,\tilde z_2,\tilde z_3)$")

    src_x1, bus_x, box_x0, box_x1 = 42.0, 47.0, 50.5, A_W
    y_mid = A_H / 2.0
    axA.add_patch(
        mpatches.FancyBboxPatch(
            (0.6, y_mid - 4.0), src_x1 - 0.6, 8.0,
            boxstyle="round,pad=0.0,rounding_size=0.8",
            facecolor="#ECECEC", edgecolor="#444444", lw=LW_THIN,
        )
    )
    axA.text((0.6 + src_x1) / 2.0, y_mid, "task-relevant moments\nheld fixed",
             ha="center", va="center", fontsize=FS_BODY, linespacing=1.35, color=INK)

    variants = [
        ("original", OMAP["moments_original"], "narrative moments block"),
        ("reformatted", OMAP["moments_reformatted"], r"table layout of same $\tilde z_k$"),
        ("task-irrelevant padded", OMAP["moments_length_matched"], r"same $\tilde z_k$ + task-irrelevant context"),
    ]
    box_h = 4.6
    y_centers = [A_H - 2.9 - i * 6.0 for i in range(3)]  # 14.6, 8.6, 2.6

    # a short vertical bus instead of a long fan
    axA.add_line(mlines.Line2D([src_x1 + 0.8, bus_x], [y_mid, y_mid],
                               color="#767676", lw=LW_THIN, solid_capstyle="butt"))
    axA.add_line(mlines.Line2D([bus_x, bus_x], [min(y_centers), max(y_centers)],
                               color="#767676", lw=LW_THIN, solid_capstyle="butt"))

    # name/description pair is optically centred in the band (descriptions run
    # longer than the names, so nudge the split point left of the true centre)
    text_c = (box_x0 + box_x1) / 2.0 - 7.0
    for (name, col, body), yc in zip(variants, y_centers):
        axA.annotate("", xy=(box_x0, yc), xytext=(bus_x, yc),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=LW_LINE,
                                     mutation_scale=6))
        axA.add_patch(
            mpatches.FancyBboxPatch(
                (box_x0 + 0.6, yc - box_h / 2.0), box_x1 - box_x0 - 0.6, box_h,
                boxstyle="round,pad=0.0,rounding_size=0.8",
                facecolor=col, alpha=0.10, edgecolor=col, lw=LW_LINE,
            )
        )
        axA.text(text_c - 2.0, yc, name, ha="right", va="center", color=col,
                 fontweight="bold", fontsize=FS_SMALL)
        axA.text(text_c + 2.0, yc, body, ha="left", va="center",
                 fontsize=FS_SMALL, color="#333333")

    # ------------------------------------------------------------------
    # b  pairwise same-task-information swarm + mean
    # ------------------------------------------------------------------
    axB = mm_axes(fig, B_LEFT, R2_TOP, B_W, R2_H)
    panel_label(axB, "b", r"Pairwise same-task-info $d_{\mathrm{TV}}$")
    items = [
        ("dTV_orig_reformat", "original__vs__reformatted", "orig–reformat",
         OMAP["moments_reformatted"]),
        ("dTV_orig_pad", "original__vs__length_matched", "orig–padded",
         OMAP["moments_length_matched"]),
        ("dTV_reformat_pad", "reformatted__vs__length_matched", "reformat–padded",
         OMAP_MID),
    ]
    label_y = 1.09   # single shared baseline for the three mean values
    for i, (col, boot_key, lab, color) in enumerate(items):
        vals = field_tv[col].to_numpy()
        axB.scatter(swarm_x(vals, i, width=0.17), vals, s=4.0, c=color,
                    alpha=0.55, zorder=2, linewidths=0)
        boot = pw[boot_key]
        m, lo, hi = boot["mean"], boot["ci95"][0], boot["ci95"][1]
        axB.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="D", color="k",
                     ms=MS_MEAN, capsize=CAPSIZE, elinewidth=LW_LINE,
                     capthick=LW_LINE, zorder=5)
        axB.text(i, label_y, f"{m:.3f}", ha="center", va="center",
                 fontsize=FS_SMALL, fontweight="bold", color=color)
    axB.set_xticks([0, 1, 2])
    axB.set_xticklabels([it[2] for it in items])
    axB.set_xlim(-0.5, 2.5)
    axB.set_ylabel(r"$d_{\mathrm{TV}}$")
    axB.set_ylim(0, 1.16)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    axB.tick_params(axis="x", length=0)
    trim_spines(axB, x=False)
    axB.text(-0.46, label_y, "mean", ha="left", va="center",
             fontsize=FS_TINY, color=MUTED)

    # ------------------------------------------------------------------
    # c  scale comparison
    # ------------------------------------------------------------------
    axC = mm_axes(fig, C_LEFT, R2_TOP, C_W, R2_H)
    panel_label(axC, "c", "Scale comparison", dx_mm=-16.0)
    scales = [
        ("block noise", lead["block_noise"], "#BBBBBB"),
        ("orig–reformat", lead["original_vs_reformatted"], OMAP["moments_reformatted"]),
        ("pairwise mean", lead["same_info_mean"], "#666666"),
        ("cross-encoding GPT", lead["cross_rep_mean"], CROSSREP),
        ("reformat–pad", lead["reformatted_vs_padded"], OMAP_MID),
        ("orig–padded", lead["original_vs_padded"], OMAP["moments_length_matched"]),
    ]
    scales = sorted(scales, key=lambda s: s[1])
    ys = np.arange(len(scales))
    axC.barh(ys, [s[1] for s in scales], color=[s[2] for s in scales], height=0.60)
    axC.set_yticks(ys)
    axC.set_yticklabels([s[0] for s in scales], fontsize=FS_SMALL)
    axC.set_ylim(-0.7, len(scales) - 0.3)
    axC.tick_params(axis="y", length=0)
    axC.set_xlabel(r"$d_{\mathrm{TV}}$")
    for y, (_, v, _) in enumerate(scales):
        axC.text(v + 0.014, y, f"{v:.3f}", va="center", ha="left",
                 fontsize=FS_SMALL, color=INK)
    axC.set_xlim(0, 0.56)
    axC.set_xticks([0, 0.2, 0.4])
    trim_spines(axC, y=False)

    # ------------------------------------------------------------------
    # d  variant-label permutation null
    # ------------------------------------------------------------------
    axD = mm_axes(fig, D_LEFT, R3_TOP, D_W, R3_H)
    # Reviewer pass: state what the null *means* rather than naming the
    # procedure. The full "Separation if presentation labels did not matter"
    # measures 53.8 mm from the shared left letter column and would collide
    # with panel e's letter at 61.4 mm, so the subject noun is dropped - the
    # separated quantity is already named on this panel's x axis.
    panel_label(axD, "d", "If presentation labels did not matter")
    # bins span the null itself and carry no white edges, so the narrow null stays visible
    axD.hist(null_tv, bins=np.linspace(np.min(null_tv), np.max(null_tv), 25), color="#8F969D",
             edgecolor="none", density=True)
    obs = null_meta["observed_mean_pairwise_TV"]
    axD.axvline(obs, color=OMAP["moments_length_matched"], lw=LW_TRACE)
    axD.axvline(null_meta["null_mean"], color="#666666", lw=LW_THIN,
                ls=(0, (3, 2)))
    axD.set_xlabel(r"mean pairwise $d_{\mathrm{TV}}$")
    axD.set_ylabel("density")
    axD.set_xlim(0.045, 0.335)
    axD.set_xticks([0.1, 0.2, 0.3])
    top = axD.get_ylim()[1]
    axD.text(np.max(null_tv) + 0.006, top * 0.99, "null",
             fontsize=FS_TINY, color="#666666", ha="left", va="top")
    axD.text(obs - 0.009, top * 0.99, f"observed\n{obs:.3f}\n$p$=0.0002",
             fontsize=FS_TINY, color=OMAP["moments_length_matched"],
             ha="right", va="top", linespacing=1.45, fontweight="bold")
    trim_spines(axD)
    # state the resampling explicitly (value read from omap_null_meta.json)
    fig_text_mm(fig, D_LEFT + D_W / 2.0, R3_TOP + R3_H + 7.6,
                f"{null_meta['n_perm']:,} field-blocked permutations",
                ha="center", va="top", fontsize=FS_TINY, color=MUTED)

    # ------------------------------------------------------------------
    # e  typical response patterns
    # ------------------------------------------------------------------
    axE = mm_axes(fig, E_LEFT, R3_TOP, E_W, R3_H)
    panel_label(axE, "e", "Typical response patterns")
    med_ref = field_tv.iloc[
        (field_tv["dTV_orig_reformat"] - field_tv["dTV_orig_reformat"].median()).abs().argmin()
    ]
    med_pad = field_tv.iloc[
        (field_tv["dTV_orig_pad"] - field_tv["dTV_orig_pad"].median()).abs().argmin()
    ]

    def _p(row, key):
        v = row[key]
        if isinstance(v, str):
            return tuple(ast.literal_eval(v))
        return tuple(v)

    # Each example is the field nearest the median of that contrast; the
    # value printed beneath it is the contrast's median over all 48 fields.
    med_ref_v = float(field_tv["dTV_orig_reformat"].median())
    med_pad_v = float(field_tv["dTV_orig_pad"].median())
    dtv = r"$d_{\mathrm{TV}}$"
    groups = [
        (0.0,
         [_p(med_ref, "p_orig"), _p(med_ref, "p_reformat"), _p(med_ref, "p_pad")],
         "median orig–reformat field", f"{dtv} = {med_ref_v:.2f}"),
        (4.0,
         [_p(med_pad, "p_orig"), _p(med_pad, "p_reformat"), _p(med_pad, "p_pad")],
         "median orig–padded field", f"{dtv} = {med_pad_v:.2f}"),
    ]
    cols = [ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"]]
    for x0, tris, _lab, _val in groups:
        bottoms = np.zeros(3)
        for k, col in enumerate(cols):
            vals = [t[k] for t in tris]
            axE.bar(np.arange(3) + x0, vals, bottom=bottoms, color=col, width=0.7,
                    edgecolor="white", linewidth=0.6)
            bottoms = bottoms + np.array(vals)
        for j, tri in enumerate(tris):
            _seg_labels(axE, x0 + j, tri)

    axE.set_ylim(0, 1.0)
    axE.set_yticks([0, 0.5, 1.0])
    axE.set_xlim(-0.75, 6.75)
    axE.set_xticks([0, 1, 2, 4, 5, 6])
    axE.set_xticklabels(["orig", "ref", "pad", "orig", "ref", "pad"], fontsize=FS_TINY)
    axE.tick_params(axis="x", length=0, pad=1.2)
    axE.set_ylabel("probability")
    trim_spines(axE, x=False)

    # grouped-axis treatment for the two representative fields
    tr = axE.get_xaxis_transform()
    for (x0, _tris, lab, val) in groups:
        a, b = x0 - 0.42, x0 + 2.42
        axE.plot([a, b], [-0.155, -0.155], transform=tr, color=MUTED, lw=LW_HAIR,
                 clip_on=False, solid_capstyle="butt")
        for xe in (a, b):
            axE.plot([xe, xe], [-0.155, -0.125], transform=tr, color=MUTED,
                     lw=LW_HAIR, clip_on=False, solid_capstyle="butt")
        axE.text((a + b) / 2.0, -0.20, lab, transform=tr, ha="center", va="top",
                 fontsize=FS_TINY, color=INK, clip_on=False)
        axE.text((a + b) / 2.0, -0.30, val, transform=tr, ha="center", va="top",
                 fontsize=FS_TINY, color=INK, fontweight="bold", clip_on=False)

    # what the two examples mean, stated plainly
    fig_text_mm(fig, E_LEFT, R3_TOP + R3_H + 12.8,
                "reformatting leaves the median field unchanged; padding does not",
                ha="left", va="top", fontsize=FS_TINY, color=MUTED)

    # compact action-colour key, right-aligned on the panel-title line
    key_x, key_y = E_LEFT + E_W - 17.6, R3_TOP - 3.3
    for k, (col, lab) in enumerate(zip(cols, ACTION_LABELS)):
        x = key_x + k * 7.4
        _mm_swatch(fig, x, key_y - 1.0, 2.0, 2.0, col)
        fig_text_mm(fig, x + 2.7, key_y, lab, ha="left", va="center",
                    fontsize=FS_SMALL, color=INK)

    # ------------------------------------------------------------------
    # f  effective observation map
    # ------------------------------------------------------------------
    axF = mm_panel(fig, F_LEFT, R3_TOP, F_W, R3_H)
    panel_label_at(fig, F_LEFT, ROW3_LETTER_Y, "f",
                   r"Effective observation map $\mathcal{R}(\rho)$")
    boxes = [
        (r"physical $\rho$", "#F5F5F5", "#555555"),
        ("retained task information", "#F5F5F5", "#555555"),
        ("+ serialization / context", "#EFEAF7", OMAP["moments_length_matched"]),
        (r"effective $\mathcal{R}(\rho)$", "#F5F5F5", "#555555"),
        (r"operator $g_{\mathcal{R}}$", "#F5F5F5", "#555555"),
        ("collective dynamics", "#F5F5F5", "#555555"),
    ]
    bh, step, y0 = 0.115, 0.170, 0.935
    for i, (lab, fc, ec) in enumerate(boxes):
        y = y0 - i * step
        axF.add_patch(
            mpatches.FancyBboxPatch(
                (0.08, y - bh / 2.0), 0.84, bh,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                facecolor=fc, edgecolor=ec, lw=LW_THIN,
            )
        )
        axF.text(0.5, y, lab, ha="center", va="center", fontsize=FS_SMALL)
        if i < len(boxes) - 1:
            axF.annotate("", xy=(0.5, y - step + bh / 2.0), xytext=(0.5, y - bh / 2.0),
                         arrowprops=dict(arrowstyle="-|>", color="#6E6E6E", lw=LW_THIN,
                                         mutation_scale=5, shrinkA=0, shrinkB=0))

    return save_fig(fig, "fig5_observation_map_control")
