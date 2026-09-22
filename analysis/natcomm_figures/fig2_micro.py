"""Figure 2 — Equivalent physical fields elicit different microscopic operators."""

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
    apply_style,
    panel_label,
    save_fig,
    stacked_trinomial,
)


def _mean_trinomial(curves: pd.DataFrame, profile: str, offset_index: int | None = 0) -> dict:
    sub = curves[curves["profile"] == profile]
    if offset_index is not None:
        sub = sub[sub["offset_index"] == offset_index]
    out = {}
    for rep in REP_ORDER:
        g = sub[sub["representation"] == rep]
        out[rep] = (
            float(g["p_retard"].mean()),
            float(g["p_stay"].mean()),
            float(g["p_advance"].mean()),
        )
    return out


def _from_atlas(atlas: pd.DataFrame, profile: str) -> dict:
    """Recover (p-,p0,p+) from activity and a0."""
    out = {}
    for rep_short, rep in (("moments", "moments_m1_m3"), ("centers", "centers_24_standard"), ("intervals", "intervals_24_decimal6")):
        row = atlas[(atlas["representation"] == rep_short) & (atlas["profile"] == profile)].iloc[0]
        a = float(row["activity"])
        a0 = float(row["a0"])
        p_plus = 0.5 * (a + a0)
        p_minus = 0.5 * (a - a0)
        p0 = 1.0 - a
        # numerical clip
        p_plus = float(np.clip(p_plus, 0, 1))
        p_minus = float(np.clip(p_minus, 0, 1))
        p0 = float(np.clip(p0, 0, 1))
        s = p_minus + p0 + p_plus
        out[rep] = (p_minus / s, p0 / s, p_plus / s)
    return out


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    curves = pd.read_csv(ROOT / "analysis/complex_kernel_stimulus_manifold/offset_action_curves.csv")
    atlas = pd.read_csv(ROOT / "analysis/stage_b_offline_p3_p4/phenotype_atlas.csv")
    eps = pd.read_csv(ROOT / "analysis/complex_kernel_antipodal_weight/epsilon_trajectories.csv")
    prompts = json.loads((DATA_DIR / "prompt_excerpts_unimodal.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.05, 1.1, 1.25], hspace=0.4, wspace=0.4)

    # Shared unimodal band behind A–C conceptually via annotation
    # A physical field
    axA = fig.add_subplot(gs[0, 0:2])
    panel_label(axA, "A", r"Same unimodal field $\rho$")
    axA.set_xlim(-1.4, 1.4)
    axA.set_ylim(-1.4, 1.4)
    axA.set_aspect("equal")
    axA.axis("off")
    axA.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#333", lw=1.2))
    axA.scatter([1.0], [0.0], s=70, c="k", zorder=5)
    rng = np.random.default_rng(0)
    ph = rng.normal(0.0, 0.35, size=16)
    axA.scatter(np.cos(ph), np.sin(ph), s=28, c="#666", zorder=4)
    axA.text(0, -1.28, "unimodal · focal at 0", ha="center", fontsize=8)
    axA.annotate("", xy=(1.35, 0), xytext=(1.15, 0), arrowprops=dict(arrowstyle="->", color="#888", lw=1.2), annotation_clip=False)

    # B actual prompts — short, larger
    axB = fig.add_subplot(gs[0, 2:])
    panel_label(axB, "B", "Three observations of the same field")
    axB.axis("off")
    axB.set_xlim(0, 10)
    axB.set_ylim(0, 6)
    excerpts = {
        "moments_m1_m3": "moment_1_cos = …\nmoment_1_sin = …\nmoment_2_cos = …\n…",
        "centers_24_standard": "bin_00 center … mass …\nbin_01 center … mass …\nbin_02 center … mass …\n…",
        "intervals_24_decimal6": "bin_00 [lo, hi): …\nbin_01 [lo, hi): …\nbin_02 [lo, hi): …\n…",
    }
    # Prefer real data lines if available, else fallback structure
    for i, rep in enumerate(REP_ORDER):
        x = 0.25 + 3.25 * i
        c = REP[rep]
        axB.add_patch(
            mpatches.FancyBboxPatch((x, 0.5), 3.05, 5.0, boxstyle="round,pad=0.04", facecolor=c, alpha=0.08, edgecolor=c)
        )
        axB.text(x + 1.5, 5.1, REP_SHORT[rep], ha="center", color=c, fontweight="bold", fontsize=10)
        text = prompts.get(rep, "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and ln.strip() != "…"]
        data_lines = [ln for ln in lines if any(k in ln for k in ("moment_", "bin_"))][:3]
        if len(data_lines) < 3:
            shown = excerpts[rep]
        else:
            # shorten numbers
            shortened = []
            for ln in data_lines:
                parts = ln.split()
                if len(parts) > 6:
                    ln = " ".join(parts[:6]) + " …"
                shortened.append(ln)
            shown = "\n".join(shortened + ["…"])
        axB.text(x + 1.52, 2.7, shown, ha="center", va="center", fontsize=8, family="monospace", linespacing=1.45)
    axB.annotate("", xy=(0.02, -0.08), xytext=(0.98, -0.08), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.0))
    axB.text(0.5, -0.14, "A → B → C: same unimodal field", transform=axB.transAxes, ha="center", fontsize=7.5, color="#555")

    # C unimodal
    axC = fig.add_subplot(gs[1, 0:2])
    panel_label(axC, "C", "Unimodal response")
    stacked_trinomial(axC, _mean_trinomial(curves, "unimodal_k9", 0), show_legend=True)

    # D antipodal
    axD = fig.add_subplot(gs[1, 2:4])
    panel_label(axD, "D", "Exact antipodal response")
    stacked_trinomial(axD, _from_atlas(atlas, "antipodal_equal_k6"), show_legend=False)

    # E near-zero epsilon zoom
    axE = fig.add_subplot(gs[1, 4:])
    panel_label(axE, "E", r"Near-zero imbalance activation")
    for rep in REP_ORDER:
        # map representation names in epsilon file
        sub = eps[eps["representation"] == rep].sort_values("epsilon")
        near = sub[sub["epsilon"].isin([-0.02, 0.0, 0.02])]
        axE.plot(near["epsilon"], near["activity"], "-o", color=REP[rep], ms=6, lw=1.8, label=REP_SHORT[rep])
    axE.axvline(0, color="#aaa", lw=0.8)
    axE.set_xlabel(r"$\varepsilon$")
    axE.set_ylabel(r"$A$")
    axE.set_ylim(0, 1.05)
    axE.legend(frameon=False, fontsize=7)

    # F full signed sweep two rows
    axF1 = fig.add_subplot(gs[2, 0:3])
    axF2 = fig.add_subplot(gs[2, 3:5])
    panel_label(axF1, "F", r"Signed sweep: activity $A(\varepsilon)$")
    panel_label(axF2, "F′", r"Signed polar response $a_1(\varepsilon)=\mathrm{Re}\,C_1$")
    for rep in REP_ORDER:
        sub = eps[eps["representation"] == rep].sort_values("epsilon")
        axF1.plot(sub["epsilon"], sub["activity"], "-o", color=REP[rep], ms=4, lw=1.5, label=REP_SHORT[rep])
        axF2.plot(sub["epsilon"], sub["a1"], "-o", color=REP[rep], ms=4, lw=1.5, label=REP_SHORT[rep])
    for ax in (axF1, axF2):
        ax.axvline(0, color="#aaa", lw=0.8)
        ax.set_xlabel(r"$\varepsilon$")
    axF1.set_ylabel(r"$A$")
    axF2.set_ylabel(r"$a_1$")
    axF2.axhline(0, color="#aaa", lw=0.8)
    axF1.set_ylim(0, 1.05)
    axF1.legend(frameon=False, fontsize=7, loc="lower right")

    # G operator TV heatmap from phenotype atlas families
    axG = fig.add_subplot(gs[2, 5])
    panel_label(axG, "G", r"Operator $d_{\mathrm{TV}}$ map")
    families = [
        "unimodal_k9",
        "antipodal_equal_k6",
        "asymmetric_w075_sep2_k6",
        "sparse_N8_unimodal_k6",
        "sparse_N16_antipodal_k6",
        "bimodal_equal_sep2_k6",
    ]
    labels = ["unimodal", "antipodal", "imbalance", "sparse uni.", "sparse anti.", "bimodal"]
    mat = np.zeros((len(families), 3))
    pair_names = ["M–C", "M–I", "C–I"]
    for i, fam in enumerate(families):
        probs = _from_atlas(atlas, fam)
        pm, pc, pi = [np.array(probs[r]) for r in REP_ORDER]
        mat[i, 0] = 0.5 * np.abs(pm - pc).sum()
        mat[i, 1] = 0.5 * np.abs(pm - pi).sum()
        mat[i, 2] = 0.5 * np.abs(pc - pi).sum()
    im = axG.imshow(mat, cmap="viridis", vmin=0, vmax=max(0.4, mat.max()), aspect="auto")
    axG.set_yticks(range(len(families)))
    axG.set_yticklabels(labels, fontsize=7)
    axG.set_xticks([0, 1, 2])
    axG.set_xticklabels(pair_names, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(3):
            axG.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="w" if mat[i, j] > 0.25 else "k", fontsize=6)
    plt.colorbar(im, ax=axG, fraction=0.05, pad=0.04)

    return save_fig(fig, "fig2_microscopic_operators")
