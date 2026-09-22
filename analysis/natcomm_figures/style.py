"""Shared Nat Commun figure style (submission polish)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures" / "natcomm"
DATA_DIR = Path(__file__).resolve().parent / "data"

# Unified full-width canvas (inches)
FIGSIZE = (14.0, 10.5)

REP = {
    "moments_m1_m3": "#1f77b4",
    "centers_24_standard": "#ff7f0e",
    "intervals_24_decimal6": "#2ca02c",
}
REP_SHORT = {
    "moments_m1_m3": "moments",
    "centers_24_standard": "centers",
    "intervals_24_decimal6": "intervals",
}
REP_ORDER = list(REP.keys())

# Action colors — separate from representation greens
ACTION = {
    "p_minus": "#8e0152",
    "p_zero": "#d9d9d9",
    "p_plus": "#2166ac",
}
ACTION_LABELS = (r"$p_-$", r"$p_0$", r"$p_+$")

OMAP = {
    "moments_original": "#4c4c4c",
    "moments_reformatted": "#7b68a6",
    "moments_length_matched": "#9b59b6",
}

FAMILY_EDGE = {"gpt": "#333333", "claude": "#6b3fa0", "gemini": "#b8860b"}
FAMILY_MARKER = {"gpt": "o", "claude": "s", "gemini": "D"}
FAMILY_LABEL = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini"}

# Phenotype palette — avoid overlapping moments blue
PHENOTYPE = {
    "polar_locked": "#1a1a1a",
    "partial_polar_order": "#9e9ac8",
    "high_r2_nonpolar": "#c51b8a",
    "low_polar_active": "#bdbdbd",
}

# Support / diagnosis buckets (Fig. 6)
DIAGNOSIS = {
    "local_noncompressible": "#b2182b",
    "active_feature_gap": "#fdae61",
    "supported": "#35978f",
}

VERDICT = {
    "supported": "#1a1a1a",
    "suggested": "#666666",
    "not_established": "#999999",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=6, fontsize=10)


def save_fig(fig, stem: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def paired_ci(values: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    m = float(np.mean(v))
    if len(v) < 2:
        return m, m, m
    se = float(np.std(v, ddof=1) / np.sqrt(len(v)))
    return m, m - 1.96 * se, m + 1.96 * se


def beeswarm_x(
    n: int, center: float, width: float = 0.12, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return center + rng.uniform(-width, width, size=n)


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def stacked_trinomial(
    ax, probs_by_rep: dict[str, tuple[float, float, float]], show_legend: bool = True
) -> None:
    bottoms = np.zeros(3)
    cols = [ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"]]
    for k, lab, col in zip(range(3), ACTION_LABELS, cols):
        vals = [probs_by_rep[r][k] for r in REP_ORDER]
        ax.bar(
            [0, 1, 2],
            vals,
            bottom=bottoms,
            color=col,
            edgecolor="white",
            width=0.65,
            label=lab if show_legend else None,
        )
        bottoms = bottoms + np.array(vals)
    for i, r in enumerate(REP_ORDER):
        p = probs_by_rep[r]
        a = p[0] + p[2]
        a0 = p[2] - p[0]
        ax.text(i, 1.04, f"$A$={a:.2f}\n$a_0$={a0:+.2f}", ha="center", va="bottom", fontsize=8)
        ax.plot(
            [i - 0.32, i + 0.32],
            [-0.04, -0.04],
            color=REP[r],
            lw=3,
            solid_capstyle="butt",
            clip_on=False,
        )
    ax.set_ylim(0, 1.32)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([REP_SHORT[r] for r in REP_ORDER])
    for tick, r in zip(ax.get_xticklabels(), REP_ORDER):
        tick.set_color(REP[r])
        tick.set_fontweight("bold")
    if show_legend:
        ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=8)
