"""Figure 4 — Cross-family operator replication."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from analysis.natcomm_figures.style import (
    DATA_DIR,
    FAMILY_EDGE,
    FAMILY_LABEL,
    FAMILY_MARKER,
    FIGSIZE,
    ROOT,
    apply_style,
    panel_label,
    save_fig,
)

PRIMARY = ROOT / "analysis" / "matched_rep_collective" / "r1_primary"
FIELDS_META = ROOT / "analysis" / "matched_rep_collective" / "replay_fields_v0_1.json"


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    dec = json.loads((PRIMARY / "decision.json").read_text(encoding="utf-8"))
    fieldwise = pd.read_csv(DATA_DIR / "r1_fieldwise_mean_tv.csv")
    fields_meta = {f["field_id"]: f for f in json.loads(FIELDS_META.read_text(encoding="utf-8"))["fields"]}
    fams = ["gpt", "claude", "gemini"]
    pairs = [
        ("moments_m1_m3__vs__centers_24_standard", "M–C", "#1f77b4"),
        ("moments_m1_m3__vs__intervals_24_decimal6", "M–I", "#2ca02c"),
        ("centers_24_standard__vs__intervals_24_decimal6", "C–I", "#ff7f0e"),
    ]

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.15, 1.25], hspace=0.4, wspace=0.55)

    # A design with clear flow
    axA = fig.add_subplot(gs[0, 0:2])
    panel_label(axA, "A", "Cross-family replication design")
    axA.axis("off")
    axA.set_xlim(0, 10)
    axA.set_ylim(0, 7)
    axA.add_patch(mpatches.FancyBboxPatch((2.5, 5.2), 5.0, 1.2, boxstyle="round,pad=0.04", facecolor="#f0f0f0", edgecolor="#333"))
    axA.text(5.0, 5.8, "48 physical fields", ha="center", va="center", fontsize=9, fontweight="bold")
    axA.annotate("", xy=(5.0, 4.5), xytext=(5.0, 5.15), arrowprops=dict(arrowstyle="->", color="#333", lw=1.4))
    axA.text(5.0, 4.7, "3 observation maps", ha="center", fontsize=8, color="#555")
    axA.annotate("", xy=(5.0, 3.3), xytext=(5.0, 4.35), arrowprops=dict(arrowstyle="->", color="#333", lw=1.4))
    for fam, x in zip(fams, [2.0, 5.0, 8.0]):
        axA.add_patch(
            mpatches.FancyBboxPatch(
                (x - 1.1, 1.6), 2.2, 1.5, boxstyle="round,pad=0.04", facecolor="white", edgecolor=FAMILY_EDGE[fam], lw=2
            )
        )
        axA.text(x, 2.35, FAMILY_LABEL[fam], ha="center", va="center", fontsize=9, fontweight="bold", color=FAMILY_EDGE[fam])
    axA.text(5, 0.7, "GPT $n$=32 · Claude/Gemini $n$=16", ha="center", fontsize=7.5)
    axA.text(5, 0.2, r"all families: $p_{\mathrm{perm}}=2.0\times10^{-4}$", ha="center", fontsize=8, fontweight="bold")

    # B forest
    axB = fig.add_subplot(gs[0, 2:])
    panel_label(axB, "B", r"Pairwise $d_{\mathrm{TV}}$ by model family")
    y = 0
    yticks, ylabs = [], []
    for pair_key, pair_lab, col in pairs:
        for fam in fams:
            boot = dec["families"][fam]["primary_target"]["pairwise_bootstrap"][pair_key]
            m, lo, hi = boot["mean"], boot["ci95"][0], boot["ci95"][1]
            axB.errorbar(
                [m],
                [y],
                xerr=[[m - lo], [hi - m]],
                fmt=FAMILY_MARKER[fam],
                color=col,
                ecolor=col,
                markeredgecolor=FAMILY_EDGE[fam],
                markeredgewidth=1.3,
                ms=7,
                capsize=3,
            )
            yticks.append(y)
            ylabs.append(f"{pair_lab} · {FAMILY_LABEL[fam]}")
            y += 1
        y += 0.35
    axB.set_yticks(yticks)
    axB.set_yticklabels(ylabs, fontsize=7)
    axB.set_xlabel(r"mean $d_{\mathrm{TV}}$ (field-cluster CI)")
    axB.set_xlim(0, 0.65)
    axB.invert_yaxis()

    # C between/within with family markers
    axC = fig.add_subplot(gs[1, 0:2])
    panel_label(axC, "C", "Separation relative to noise")
    for i, fam in enumerate(fams):
        nf = dec["families"][fam]["primary_target"]["noise_floor"]
        b = nf["mean_between_target_TV"]
        w = nf["mean_within_target_block_TV"]
        axC.plot([i, i], [w, b], color="#888", lw=2, zorder=1)
        axC.scatter([i], [w], s=55, facecolors="white", edgecolors=FAMILY_EDGE[fam], lw=1.5, marker=FAMILY_MARKER[fam], zorder=3)
        axC.scatter([i], [b], s=60, c=FAMILY_EDGE[fam], marker=FAMILY_MARKER[fam], zorder=3)
        axC.text(i, b + 0.02, f"{nf['ratio']:.1f}×", ha="center", fontsize=7.5, fontweight="bold")
    axC.set_xticks(range(3))
    axC.set_xticklabels([FAMILY_LABEL[f] for f in fams])
    axC.set_ylabel(r"$d_{\mathrm{TV}}$")
    axC.set_ylim(0, 0.55)
    axC.scatter([], [], s=55, facecolors="white", edgecolors="#333", marker="o", label="within")
    axC.scatter([], [], s=55, c="#333", marker="o", label="between")
    axC.legend(frameon=False, fontsize=7, loc="upper left")

    # D fieldwise with stratum separators — wider
    axD = fig.add_subplot(gs[1, 2:5])
    panel_label(axD, "D", "Fieldwise operator sensitivity")
    gpt = fieldwise[fieldwise["family"] == "gpt"].sort_values("mean_pairwise_TV")
    order = gpt["field_id"].tolist()
    mat = np.zeros((48, 3))
    strata = [fields_meta.get(fid, {}).get("stratum", "") for fid in order]
    for j, fam in enumerate(fams):
        sub = fieldwise[fieldwise["family"] == fam].set_index("field_id")
        mat[:, j] = [float(sub.loc[fid, "mean_pairwise_TV"]) for fid in order]
    im = axD.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.6, float(mat.max())))
    # stratum boundary lines where stratum changes
    for i in range(1, 48):
        if strata[i] != strata[i - 1]:
            axD.axhline(i - 0.5, color="white", lw=0.8, alpha=0.8)
    axD.set_xticks([0, 1, 2])
    axD.set_xticklabels([FAMILY_LABEL[f] for f in fams])
    axD.set_ylabel("fields (sorted by GPT)")
    axD.set_yticks([])
    plt.colorbar(im, ax=axD, fraction=0.046, pad=0.04, label=r"mean $d_{\mathrm{TV}}$")

    # E compact scope
    axE = fig.add_subplot(gs[1, 5])
    panel_label(axE, "E", "Scope")
    axE.axis("off")
    axE.add_patch(
        mpatches.FancyBboxPatch((0.05, 0.58), 0.9, 0.32, boxstyle="round,pad=0.03", facecolor="#e3f2fd", edgecolor="#1565c0")
    )
    axE.text(0.5, 0.74, "Macro phases\nGPT only", ha="center", va="center", fontsize=8)
    axE.add_patch(
        mpatches.FancyBboxPatch((0.05, 0.18), 0.9, 0.32, boxstyle="round,pad=0.03", facecolor="#f3e5f5", edgecolor="#6b3fa0")
    )
    axE.text(0.5, 0.34, "Micro operators\n3 families", ha="center", va="center", fontsize=8)
    axE.text(0.5, 0.05, "macro×family\nnot tested", ha="center", fontsize=6.5, style="italic", color="#666")

    return save_fig(fig, "fig4_cross_family")
