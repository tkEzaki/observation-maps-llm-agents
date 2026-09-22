"""Figure 6 — Compressibility does not ensure transportability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from analysis.natcomm_figures.style import (
    DIAGNOSIS,
    FIGSIZE,
    REP,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    apply_style,
    paired_ci,
    panel_label,
    save_fig,
)

SYN = ROOT / "analysis" / "synthesis"
SC = ROOT / "analysis" / "stage_c_artifacts" / "stage_c_v0_2_combined"


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    funnel = pd.read_csv(SYN / "transport_funnel.csv")
    comp = pd.read_csv(SYN / "fig2_compressibility.csv")
    scatter = pd.read_csv(SYN / "fig4_support_scatter.csv")
    runs = pd.read_csv(SC / "combined_endpoints_per_run.csv")
    runs = runs[(runs["n_agents"] == 17) & (runs["coupling"] > 0)].copy()

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.15, 1.1, 1.15], hspace=0.4, wspace=0.3)

    # A funnel — larger tracks
    axA = fig.add_subplot(gs[0, :])
    panel_label(axA, "A", "Empirical transportability funnel")
    axA.set_xlim(0, 12)
    axA.set_ylim(0, 4.6)
    axA.axis("off")
    stages = ["Stage B\nlabels", "Grouped\nOOF", "Support /\nreplay", "Prospective\ncollective"]
    xs = [1.6, 4.4, 7.2, 10.0]
    for x, s in zip(xs, stages):
        axA.add_patch(
            mpatches.FancyBboxPatch(
                (x - 1.05, 3.05), 2.1, 1.15, boxstyle="round,pad=0.04", facecolor="#f5f5f5", edgecolor="#777"
            )
        )
        axA.text(x, 3.6, s, ha="center", va="center", fontsize=8.5)
        if x < xs[-1]:
            axA.annotate("", xy=(x + 1.15, 3.6), xytext=(x + 1.05, 3.6), arrowprops=dict(arrowstyle="->", color="#666", lw=1.4))

    paths = {
        "moments_m1_m3": ("PASS", "PASS", "PASS*", "PASS · frozen C"),
        "centers_24_standard": ("PASS", "PASS", "STOP · feature-gap", "—"),
        "intervals_24_decimal6": ("PASS", "PASS", "STOP · noncomp.", "—"),
    }
    for i, rep in enumerate(REP_ORDER):
        y = 2.35 - 0.7 * i
        axA.plot([0.9, 11.1], [y, y], color=REP[rep], lw=3.5, alpha=0.25, solid_capstyle="round")
        axA.text(0.15, y, REP_SHORT[rep], color=REP[rep], fontsize=9, va="center", fontweight="bold")
        for x, lab in zip(xs, paths[rep]):
            stop = lab.startswith("STOP")
            ok = lab.startswith("PASS")
            color = "#1a7f37" if ok else ("#b00020" if stop else "#999")
            if stop:
                axA.scatter([x], [y], s=120, c="#b00020", marker="|", linewidths=3, zorder=4)
            axA.text(x, y, lab, ha="center", va="center", fontsize=8, color=color, fontweight="bold", zorder=5)
    axA.text(
        6,
        0.25,
        "*moments: localized regime error repaired prospectively · stop rules preregistered",
        ha="center",
        fontsize=7.5,
        style="italic",
    )

    # B histogram branches only
    axB = fig.add_subplot(gs[1, 0])
    panel_label(axB, "B", r"Histogram branches: IID OOF improvement")
    y = 0
    yticks, ylabs = [], []
    for rep in ("centers_24_standard", "intervals_24_decimal6"):
        row = comp[comp["representation"] == rep].iloc[0]
        m = float(row["delta_ll_mean"])
        lo = float(row["ci_low_field_cluster"])
        hi = float(row["ci_high_field_cluster"])
        axB.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="o", color=REP[rep], ms=8, capsize=4)
        yticks.append(y)
        ylabs.append(REP_SHORT[rep])
        y += 1
    axB.axvline(0, color="#888", lw=0.8)
    axB.set_yticks(yticks)
    axB.set_yticklabels(ylabs)
    axB.set_xlabel(r"$\Delta$ log-loss vs peer-global (field-cluster CI)")
    axB.set_xlim(-0.02, 0.5)

    # C moments repair
    axC = fig.add_subplot(gs[1, 1])
    panel_label(axC, "C", "Moments: prospective repair (v1→v2)")
    metrics = [("eA_v1", "eA_v2", "activity"), ("estay_v1", "estay_v2", "stay"), ("er1_v1", "er1_v2", r"final $r_1$")]
    for i, (c1, c2, lab) in enumerate(metrics):
        v1 = runs[c1].to_numpy()
        v2 = runs[c2].to_numpy()
        for a, b in zip(v1, v2):
            axC.plot([i - 0.15, i + 0.15], [a, b], color="#cccccc", lw=0.7, zorder=1)
        axC.scatter(np.full(len(v1), i - 0.15), v1, s=18, c="#a6c8e8", zorder=2, label="v1" if i == 0 else None)
        axC.scatter(np.full(len(v2), i + 0.15), v2, s=18, c=REP["moments_m1_m3"], zorder=2, label="v2" if i == 0 else None)
        for vals, dx, color in ((v1, -0.15, "#555"), (v2, 0.15, "k")):
            m, lo, hi = paired_ci(vals)
            axC.errorbar([i + dx], [m], yerr=[[m - lo], [hi - m]], fmt="D", color=color, ms=4, capsize=2, zorder=4)
    axC.set_xticks(range(3))
    axC.set_xticklabels([m[2] for m in metrics])
    axC.set_ylabel(r"$|$error$|$ vs LLM teacher")
    axC.legend(frameon=False, fontsize=8)

    # D centers with counts
    axD = fig.add_subplot(gs[2, 0])
    panel_label(axD, "D", "Centers: feature-gap transfer failure")
    sub_c = scatter[scatter["representation"].str.contains("centers", case=False, na=False)]
    vc = sub_c["diagnosis_bucket"].value_counts()
    order = ["active_feature_gap", "local_noncompressible", "supported"]
    n = len(sub_c)
    counts = [int(vc.get(k, 0)) for k in order]
    fracs = [c / n for c in counts]
    labels = [f"{o.replace('_', ' ')}\n{c}/{n}" for o, c in zip(order, counts)]
    axD.barh(labels, fracs, color=[DIAGNOSIS[o] for o in order])
    axD.set_xlim(0, 1)
    axD.set_xlabel(f"share of local diagnoses (n={n})")

    # E intervals unified colors
    axE = fig.add_subplot(gs[2, 1])
    panel_label(axE, "E", "Intervals: local noncompressibility")
    sub_i = scatter[scatter["representation"].str.contains("intervals", case=False, na=False)]
    vc = sub_i["diagnosis_bucket"].value_counts(normalize=True)
    order = ["local_noncompressible", "active_feature_gap", "supported"]
    vals = [float(vc.get(k, 0)) for k in order]
    left = 0.0
    for v, o in zip(vals, order):
        axE.barh([0], [v], left=left, color=DIAGNOSIS[o], label=f"{o.replace('_', ' ')} ({v:.0%})")
        if v > 0.08:
            axE.text(left + v / 2, 0, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left += v
    axE.set_yticks([])
    axE.set_xlim(0, 1)
    axE.set_xlabel(f"share of local diagnoses (n={len(sub_i)})")
    axE.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)

    # F summary strip
    axF = fig.add_axes([0.08, 0.02, 0.84, 0.07])
    axF.axis("off")
    boxes = [
        (0.02, "Moments", "frozen collective path", REP["moments_m1_m3"]),
        (0.28, "Centers", "microscopic archive", REP["centers_24_standard"]),
        (0.54, "Intervals", "microscopic archive", REP["intervals_24_decimal6"]),
    ]
    for x, title, body, c in boxes:
        axF.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.15), 0.24, 0.75, boxstyle="round,pad=0.02", facecolor=c, alpha=0.12, edgecolor=c, transform=axF.transAxes
            )
        )
        axF.text(x + 0.12, 0.65, title, ha="center", va="center", fontsize=8, fontweight="bold", color=c, transform=axF.transAxes)
        axF.text(x + 0.12, 0.35, body, ha="center", va="center", fontsize=7, transform=axF.transAxes)
    axF.text(
        0.90,
        0.5,
        r"Compressibility $\neq$" + "\ntransportability",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        transform=axF.transAxes,
    )
    _ = funnel  # disposition embedded in A/F

    return save_fig(fig, "fig6_compressibility_transportability")
