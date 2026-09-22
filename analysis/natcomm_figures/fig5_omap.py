"""Figure 5 (was 6) — Task-information-fixed presentation control."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from analysis.natcomm_figures.style import (
    ACTION,
    ACTION_LABELS,
    DATA_DIR,
    FIGSIZE,
    OMAP,
    ROOT,
    apply_style,
    beeswarm_x,
    panel_label,
    save_fig,
)

PRIMARY = ROOT / "analysis" / "matched_rep_collective" / "omap_primary"


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    dec = json.loads((PRIMARY / "decision.json").read_text(encoding="utf-8"))
    lead = dec["lead_with_pairwise"]
    pw = dec["same_info_pairwise_bootstrap"]
    field_tv = pd.read_csv(DATA_DIR / "omap_field_pairwise_tv.csv")
    null_tv = np.load(DATA_DIR / "omap_null_tv.npy")
    null_meta = json.loads((DATA_DIR / "omap_null_meta.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.05, 1.15, 1.15], hspace=0.42, wspace=0.35)

    # A+B design with snippets
    axA = fig.add_subplot(gs[0, :])
    panel_label(axA, "A", r"Same task-information control: identical $(z_1,z_2,z_3)$")
    axA.axis("off")
    axA.set_xlim(0, 12)
    axA.set_ylim(0, 6)
    axA.add_patch(
        mpatches.FancyBboxPatch((3.8, 4.5), 4.4, 1.1, boxstyle="round,pad=0.04", facecolor="#e8e8e8", edgecolor="#333")
    )
    axA.text(6.0, 5.05, r"task-relevant moments held fixed", ha="center", va="center", fontsize=9, fontweight="bold")
    variants = [
        ("original", OMAP["moments_original"], "narrative moments\nblock"),
        ("reformatted", OMAP["moments_reformatted"], "table layout\nof same $z_k$"),
        ("length-matched", OMAP["moments_length_matched"], "same $z_k$ +\nneutral padding"),
    ]
    for i, (name, col, body) in enumerate(variants):
        x = 1.0 + 3.8 * i
        axA.annotate("", xy=(x + 1.2, 3.5), xytext=(6.0, 4.5), arrowprops=dict(arrowstyle="->", color=col, lw=1.3))
        axA.add_patch(
            mpatches.FancyBboxPatch((x, 0.6), 3.2, 2.8, boxstyle="round,pad=0.04", facecolor=col, alpha=0.12, edgecolor=col)
        )
        axA.text(x + 1.6, 3.0, name, ha="center", color=col, fontweight="bold", fontsize=9)
        axA.text(x + 1.6, 1.7, body, ha="center", va="center", fontsize=8)

    # C pairwise beeswarm
    axC = fig.add_subplot(gs[1, 0:2])
    panel_label(axC, "B", r"Pairwise same-task-info $d_{\mathrm{TV}}$")
    items = [
        ("dTV_orig_reformat", "original__vs__reformatted", "orig–reformat", OMAP["moments_reformatted"]),
        ("dTV_orig_pad", "original__vs__length_matched", "orig–padded", OMAP["moments_length_matched"]),
        ("dTV_reformat_pad", "reformatted__vs__length_matched", "reformat–padded", "#8e44ad"),
    ]
    rng = np.random.default_rng(1)
    for i, (col, boot_key, lab, color) in enumerate(items):
        vals = field_tv[col].to_numpy()
        axC.scatter(beeswarm_x(len(vals), i, 0.18, rng), vals, s=16, c=color, alpha=0.5, zorder=2)
        boot = pw[boot_key]
        m, lo, hi = boot["mean"], boot["ci95"][0], boot["ci95"][1]
        axC.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="D", color="k", ms=5.5, capsize=4, zorder=4)
        axC.text(i, hi + 0.03, f"{m:.3f}", ha="center", fontsize=8)
    axC.set_xticks([0, 1, 2])
    axC.set_xticklabels([it[2] for it in items])
    axC.set_ylabel(r"$d_{\mathrm{TV}}$")
    axC.set_ylim(0, 0.9)

    # D scale
    axD = fig.add_subplot(gs[1, 2])
    panel_label(axD, "C", "Scale comparison")
    scales = [
        ("block noise", lead["block_noise"], "#bbbbbb"),
        ("orig–reformat", lead["original_vs_reformatted"], OMAP["moments_reformatted"]),
        ("reformat–pad", lead["reformatted_vs_padded"], "#8e44ad"),
        ("pairwise mean", lead["same_info_mean"], "#666666"),
        ("cross-rep GPT", lead["cross_rep_mean"], "#1f77b4"),
        ("orig–padded", lead["original_vs_padded"], OMAP["moments_length_matched"]),
    ]
    ys = np.arange(len(scales))
    axD.barh(ys, [s[1] for s in scales], color=[s[2] for s in scales], height=0.7)
    axD.set_yticks(ys)
    axD.set_yticklabels([s[0] for s in scales], fontsize=7)
    axD.set_xlabel(r"$d_{\mathrm{TV}}$")
    for y, (_, v, _) in enumerate(scales):
        axD.text(v + 0.008, y, f"{v:.3f}", va="center", fontsize=7)
    axD.set_xlim(0, 0.55)

    # D real null
    axE = fig.add_subplot(gs[2, 0])
    panel_label(axE, "D", "Variant-label permutation null")
    axE.hist(null_tv, bins=50, color="#bdbdbd", edgecolor="white")
    axE.axvline(null_meta["observed_mean_pairwise_TV"], color="#6b3fa0", lw=2.2)
    axE.axvline(null_meta["null_mean"], color="#666", lw=1.2, ls="--")
    axE.set_xlabel(r"mean pairwise $d_{\mathrm{TV}}$")
    axE.set_ylabel("count")
    axE.set_xlim(0.05, 0.40)
    axE.text(
        0.03,
        0.92,
        f"observed = {null_meta['observed_mean_pairwise_TV']:.3f}\n"
        r"$p_{\mathrm{perm}}=2.0\times10^{-4}$" + "\n(5,000 field-blocked permutations)",
        transform=axE.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color="#6b3fa0",
    )

    # E representative responses — field nearest mean contrast (not median; median can be 0)
    axF = fig.add_subplot(gs[2, 1])
    panel_label(axF, "E", "Representative response changes")
    import ast

    def _p(row, key):
        v = row[key]
        if isinstance(v, str):
            return tuple(ast.literal_eval(v))
        return tuple(v)

    def _nearest(col: str, target: float) -> pd.Series:
        return field_tv.iloc[(field_tv[col] - target).abs().argmin()]

    # Use reported mean contrasts (lead_with_pairwise), not sample median (often 0)
    typ_ref = _nearest("dTV_orig_reformat", float(lead["original_vs_reformatted"]))
    typ_pad = _nearest("dTV_orig_pad", float(lead["original_vs_padded"]))

    def _bar_stack(ax, x0, tris, title, dtv):
        bottoms = np.zeros(3)
        cols = [ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"]]
        for k, col in enumerate(cols):
            vals = [t[k] for t in tris]
            ax.bar(np.arange(3) + x0, vals, bottom=bottoms, color=col, width=0.7, edgecolor="white")
            bottoms = bottoms + np.array(vals)
        ax.text(x0 + 1, 1.08, f"{title}\n$d_{{\\mathrm{{TV}}}}={dtv:.3f}$", ha="center", fontsize=7.5)

    _bar_stack(
        axF,
        0,
        [_p(typ_ref, "p_orig"), _p(typ_ref, "p_reformat"), _p(typ_ref, "p_pad")],
        "typical orig–reformat",
        float(typ_ref["dTV_orig_reformat"]),
    )
    _bar_stack(
        axF,
        4,
        [_p(typ_pad, "p_orig"), _p(typ_pad, "p_reformat"), _p(typ_pad, "p_pad")],
        "typical orig–pad",
        float(typ_pad["dTV_orig_pad"]),
    )
    axF.set_ylim(0, 1.28)
    axF.set_xticks([0, 1, 2, 4, 5, 6])
    axF.set_xticklabels(["orig", "ref", "pad", "orig", "ref", "pad"], fontsize=7)
    axF.set_ylabel("probability")
    handles = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip([ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"]], ACTION_LABELS)
    ]
    axF.legend(handles=handles, frameon=False, fontsize=7, loc="upper right", ncol=3)

    # F concept boxes
    axG = fig.add_subplot(gs[2, 2])
    panel_label(axG, "F", r"Effective observation map $\mathcal{R}(\rho)$")
    axG.axis("off")
    boxes = [
        (0.5, 0.92, r"physical $\rho$"),
        (0.5, 0.72, "retained task information"),
        (0.5, 0.58, "+ serialization / context"),
        (0.5, 0.40, r"effective $\mathcal{R}(\rho)$"),
        (0.5, 0.24, r"operator $g_{\mathcal{R}}$"),
        (0.5, 0.08, "collective dynamics"),
    ]
    for i, (x, y, lab) in enumerate(boxes):
        axG.add_patch(
            mpatches.FancyBboxPatch(
                (0.12, y - 0.06), 0.76, 0.11, boxstyle="round,pad=0.01", facecolor="#f5f5f5", edgecolor="#555"
            )
        )
        axG.text(x, y, lab, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            lw = 2.2 if i == len(boxes) - 2 else 1.2
            axG.annotate(
                "",
                xy=(0.5, boxes[i + 1][1] + 0.06),
                xytext=(0.5, y - 0.06),
                arrowprops=dict(arrowstyle="->", color="#333", lw=lw),
            )

    return save_fig(fig, "fig5_observation_map_control")
