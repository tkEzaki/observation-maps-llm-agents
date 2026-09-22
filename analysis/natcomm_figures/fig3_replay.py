"""Figure 3 (was 4) — Cross-encoding replay on identical physical states."""

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
    FIGSIZE,
    REP,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    VERDICT,
    apply_style,
    beeswarm_x,
    panel_label,
    save_fig,
)

PRIMARY = ROOT / "analysis" / "matched_rep_collective" / "replay_primary"


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

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # A design — moments→centers→intervals on BOTH axes (moments at top, matching B/C)
    axA = fig.add_subplot(gs[0, 0])
    panel_label(axA, "A", "3×3 replay design")
    axA.set_xlim(-0.5, 2.5)
    axA.set_ylim(2.5, -0.5)  # invert: index 0 (moments) at top
    axA.set_xticks([0, 1, 2])
    axA.set_yticks([0, 1, 2])
    axA.set_xticklabels([REP_SHORT[r] for r in REP_ORDER])
    axA.set_yticklabels([REP_SHORT[r] for r in REP_ORDER])
    axA.set_xlabel("target encoding")
    axA.set_ylabel("source trajectory")
    for i in range(3):
        for j in range(3):
            axA.add_patch(plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, fill=False, edgecolor="#888"))
            axA.text(j, i, r"$\rho$", ha="center", va="center", fontsize=10, color="#555")
    axA.text(0.5, 0.02, r"$48\times3\times32=4608$", transform=axA.transAxes, ha="center", fontsize=7)

    # B Activity — moments first row/col
    axB = fig.add_subplot(gs[0, 1])
    panel_label(axB, "B", r"Activity $A_{S,\mathcal{R}}$")
    A = np.array([[matA[s][t] for t in REP_ORDER] for s in REP_ORDER], dtype=float)
    im = axB.imshow(A, cmap="YlOrRd", vmin=0.7, vmax=1.0)
    for i in range(3):
        for j in range(3):
            axB.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axB.set_xticks([0, 1, 2])
    axB.set_yticks([0, 1, 2])
    axB.set_xticklabels([REP_SHORT[r] for r in REP_ORDER])
    axB.set_yticklabels([REP_SHORT[r] for r in REP_ORDER])
    axB.set_xlabel("target")
    axB.set_ylabel("source")
    plt.colorbar(im, ax=axB, fraction=0.046, pad=0.04)

    # C a0
    axC = fig.add_subplot(gs[0, 2])
    panel_label(axC, "C", r"Signed action $a_{0,S,\mathcal{R}}$")
    A0 = np.array([[matA0[s][t] for t in REP_ORDER] for s in REP_ORDER], dtype=float)
    lim = max(0.35, float(np.max(np.abs(A0))))
    im2 = axC.imshow(A0, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(3):
        for j in range(3):
            axC.text(j, i, f"{A0[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    axC.set_xticks([0, 1, 2])
    axC.set_yticks([0, 1, 2])
    axC.set_xticklabels([REP_SHORT[r] for r in REP_ORDER])
    axC.set_yticklabels([REP_SHORT[r] for r in REP_ORDER])
    axC.set_xlabel("target")
    axC.set_ylabel("source")
    plt.colorbar(im2, ax=axC, fraction=0.046, pad=0.04)

    # D pairwise TV
    axD = fig.add_subplot(gs[1, 0:2])
    panel_label(axD, "D", r"Pairwise target $d_{\mathrm{TV}}$ (48 fields)")
    pairs = [
        ("dTV_MC", "M–C", tgt["pairwise_bootstrap"]["moments_m1_m3__vs__centers_24_standard"], "#1f77b4"),
        ("dTV_MI", "M–I", tgt["pairwise_bootstrap"]["moments_m1_m3__vs__intervals_24_decimal6"], "#2ca02c"),
        ("dTV_CI", "C–I", tgt["pairwise_bootstrap"]["centers_24_standard__vs__intervals_24_decimal6"], "#ff7f0e"),
    ]
    rng = np.random.default_rng(0)
    for i, (col, lab, boot, color) in enumerate(pairs):
        vals = fields[col].to_numpy()
        axD.scatter(beeswarm_x(len(vals), i, 0.18, rng), vals, s=18, c=color, alpha=0.55, zorder=2)
        m, lo, hi = boot["mean"], boot["ci95"][0], boot["ci95"][1]
        axD.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="D", color="k", ms=5.5, capsize=4, zorder=4)
        axD.text(i, hi + 0.04, f"{m:.3f}", ha="center", fontsize=8)
    axD.set_xticks([0, 1, 2])
    axD.set_xticklabels([p[1] for p in pairs])
    axD.set_ylabel(r"$d_{\mathrm{TV}}$")
    axD.set_ylim(0, 1.05)

    # E paired noise
    axE = fig.add_subplot(gs[1, 2])
    panel_label(axE, "E", "Operator separation vs block drift")
    for _, row in noise_pairs.iterrows():
        axE.plot([0, 1], [row["within_block_TV"], row["between_target_TV"]], color="#cccccc", lw=0.7, zorder=1)
    axE.scatter(np.zeros(len(noise_pairs)), noise_pairs["within_block_TV"], s=16, c="#999", zorder=2)
    axE.scatter(np.ones(len(noise_pairs)), noise_pairs["between_target_TV"], s=16, c="#333", zorder=2)
    axE.plot(
        [0, 1],
        [noise_pairs["within_block_TV"].mean(), noise_pairs["between_target_TV"].mean()],
        "o-",
        color="#b00020",
        lw=2,
        ms=7,
        zorder=4,
    )
    axE.set_xticks([0, 1])
    axE.set_xticklabels(["within\nblock", "between\ntarget"])
    axE.set_ylabel(r"$d_{\mathrm{TV}}$")
    nf = tgt["noise_floor"]
    axE.text(
        0.5,
        0.95,
        f"{nf['ratio']:.2f}× · $p$=0.0002",
        transform=axE.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )

    # F real null
    axF = fig.add_subplot(gs[2, 0])
    panel_label(axF, "F", "Target-label permutation null")
    axF.hist(null_tv, bins=50, color="#bdbdbd", edgecolor="white", density=False)
    axF.axvline(null_meta["observed_mean_pairwise_TV"], color="#b00020", lw=2.2, label="observed")
    axF.axvline(null_meta["null_mean"], color="#666", lw=1.2, ls="--", label="null mean")
    axF.set_xlabel(r"mean pairwise $d_{\mathrm{TV}}$")
    axF.set_ylabel("count")
    axF.set_xlim(0.05, 0.40)
    axF.legend(frameon=False, fontsize=7, loc="upper right")
    axF.text(
        0.03,
        0.92,
        f"observed = {null_meta['observed_mean_pairwise_TV']:.3f}\n"
        r"$p_{\mathrm{perm}}=2.0\times10^{-4}$" + "\n(5,000 permutations)",
        transform=axF.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="#b00020",
    )

    # G+H combined verdict
    axG = fig.add_subplot(gs[2, 1:])
    panel_label(axG, "G", "Preregistered mechanistic verdict")
    axG.axis("off")
    rows = [
        ("Target", f"{verdict['target_p']:.4f}", "Supported", VERDICT["supported"]),
        ("Source", f"{verdict['source_p']:.3f}", "Suggested", VERDICT["suggested"]),
        ("Source × target", f"{verdict['interaction_p']:.3f}", "Not established", VERDICT["not_established"]),
    ]
    # table header
    axG.text(0.05, 0.88, "Effect", fontweight="bold", fontsize=9)
    axG.text(0.35, 0.88, r"$p$", fontweight="bold", fontsize=9)
    axG.text(0.55, 0.88, "Interpretation", fontweight="bold", fontsize=9)
    y = 0.68
    for name, p, verd, col in rows:
        axG.add_patch(
            mpatches.FancyBboxPatch(
                (0.03, y - 0.08), 0.94, 0.18, boxstyle="round,pad=0.01", facecolor=col, alpha=0.1, edgecolor=col, lw=1.2
            )
        )
        axG.text(0.05, y, name, fontsize=9, va="center", color=col)
        axG.text(0.35, y, p, fontsize=9, va="center")
        axG.text(0.55, y, verd, fontsize=9, va="center", fontweight="bold", color=col)
        y -= 0.24
    axG.text(
        0.5,
        0.08,
        r"Confirmed: $\mathcal{R}(\rho)\rightarrow$ different $g_{\mathcal{R}}(\rho)$."
        + "\nNot established: feedback / mediation.",
        ha="center",
        fontsize=8.5,
    )

    return save_fig(fig, "fig3_replay_operator")
