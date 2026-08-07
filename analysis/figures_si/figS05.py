"""Supplementary Figure S5 - full concentration-dependent microscopic transmutation map.

Stage B measured one unimodal relative-phase field at five frozen
concentrations through three observation representations, in two independent
seed blocks. This figure is the complete operator map behind main Fig. 2:
activity, signed mean action, both low-order complex harmonics with their
frozen phase-reporting gates, the complex-plane trajectories with bootstrap
ellipses, the frozen phenotype classification and the between-block
replication.

Every number is read from the frozen artifacts:

* ``analysis/transmutation_real/transmutation_endpoints.csv``
* ``analysis/transmutation_real/transmutation_analysis.json``
* ``analysis/complex_kernel_transmutation/complex_endpoints.csv``
* ``analysis/complex_kernel_transmutation/complex_ellipses.csv``
* ``analysis/complex_kernel_transmutation/delta_c1_bootstrap_tests.csv``
* ``experiments/stage_b_response_law/protocol_transmutation_v0_1_block_1.json``

Blocks are distinguished by marker and line style only (filled circle / solid
for block 1, open square / dashed for block 2); colour is reserved for the
representation grammar.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from analysis.figures_si.style import (
    CAPSIZE,
    FS_BODY,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    MS_POINT,
    MS_SERIES,
    MUTED,
    REP,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    STOP_RED,
    apply_style,
    despine,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label,
    panel_label_at,
    save_fig,
    trim_spines,
)

# --------------------------------------------------------------------------
# Layout board (mm from the top-left of a 180 mm canvas)
# --------------------------------------------------------------------------
H_MM = 167.0

COL_X = (13.0, 71.4, 129.8)      # shared left edge of every panel column
COL_W = 46.0

KEY_Y = 4.8                      # one shared key line above the first row
LETTER_Y = (10.2, 60.0, 111.0)   # shared letter baseline per row
ROW_TOP = (15.4, 65.2, 116.2)    # shared top edge per row

TALL_H = 34.0                    # a, b, e, f - a single drawing surface
MAIN_H = 23.5                    # c, d - harmonic magnitude / projection
STRIP_GAP = 2.2
STRIP_H = 8.3                    # c, d - phase strip

G_LABEL_X = 8.0                  # left edge of the panel-g row labels
G_X = 35.0
G_W = 60.0
G_H = 22.0
G_CHIP_X = 98.0

H_SIDE = 29.0                    # panel h is a square identity scatter

BLOCKS = ("block_1", "block_2")
BLOCK_STYLE = {                  # blocks differ by mark, never by colour
    "block_1": dict(marker="o", fillstyle="full", ls="-"),
    "block_2": dict(marker="s", fillstyle="none", ls="--"),
}
BLOCK_LABEL = {"block_1": "block 1", "block_2": "block 2"}
DX = 0.115                       # horizontal split of the two block estimates

# Frozen class names for the three measured representation classes, as recorded
# in docs/STAGE_B_TRANSMUTATION_MAP.md ("Frozen design").
CLASS_NAME = {
    "moments_m1_m3": "polar-preserving",
    "centers_24_standard": "sign-reversing",
    "intervals_24_decimal6": "switching",
}


# --------------------------------------------------------------------------
# Artifact readers - nothing below is hard-coded
# --------------------------------------------------------------------------
def _load():
    base = ROOT / "analysis"
    endp = pd.read_csv(base / "transmutation_real/transmutation_endpoints.csv")
    ana = json.loads(
        (base / "transmutation_real/transmutation_analysis.json").read_text(encoding="utf-8")
    )
    cx = pd.read_csv(base / "complex_kernel_transmutation/complex_endpoints.csv")
    ell = pd.read_csv(base / "complex_kernel_transmutation/complex_ellipses.csv")
    dc1 = pd.read_csv(base / "complex_kernel_transmutation/delta_c1_bootstrap_tests.csv")
    proto = json.loads(
        (ROOT / "experiments/stage_b_response_law"
         / "protocol_transmutation_v0_1_block_1.json").read_text(encoding="utf-8")
    )
    return endp, ana, cx, ell, dc1, proto


def _kappas(cx: pd.DataFrame) -> list[float]:
    return sorted(cx["concentration"].unique())


def _cell(df: pd.DataFrame, block: str, rep: str) -> pd.DataFrame:
    return df[(df["block"] == block) & (df["representation"] == rep)].sort_values(
        "concentration"
    )


def _boot(ana: dict, block: str, rep: str, profile: str, key: str) -> tuple[float, float]:
    b = ana["blocks"][block]["representations"][rep]["conditions"][profile]["bootstrap"][key]
    return float(b["ci95_low"]), float(b["ci95_high"])


def _pooled(cx: pd.DataFrame, rep: str, col: str) -> np.ndarray:
    """Across-block mean of a per-cell estimate, ordered by concentration."""
    sub = cx[cx["representation"] == rep]
    return sub.groupby("concentration")[col].mean().sort_index().to_numpy()


# --------------------------------------------------------------------------
# Shared drawing idioms
# --------------------------------------------------------------------------
def _kappa_axis(ax, kappas, label: bool = True) -> None:
    ax.set_xlim(-0.42, len(kappas) - 0.58)
    ax.set_xticks(range(len(kappas)))
    if label:
        ax.set_xticklabels([f"{k:g}" for k in kappas])
        ax.set_xlabel(r"concentration $\kappa$", fontsize=FS_BODY)
    else:
        ax.set_xticklabels([])


def _block_series(ax, x, y, lo, hi, rep, block, zorder=4):
    st = BLOCK_STYLE[block]
    ax.errorbar(
        x, y, yerr=[np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)],
        color=REP[rep], marker=st["marker"], fillstyle=st["fillstyle"],
        markerfacecolor=REP[rep] if st["fillstyle"] == "full" else "white",
        markeredgecolor=REP[rep], markersize=MS_SERIES, ls=st["ls"],
        lw=LW_LINE, elinewidth=LW_HAIR, capsize=CAPSIZE, capthick=LW_HAIR,
        zorder=zorder, clip_on=True,
    )


def _shared_key(fig) -> None:
    """One key line for both grammars: representation colour and block mark."""
    fig_text_mm(fig, COL_X[0], KEY_Y, "encodings:", ha="left",
                va="baseline", fontsize=FS_SMALL, color=MUTED)
    for rep, x in zip(("moments_m1_m3", "centers_24_standard",
                       "intervals_24_decimal6"), (32.5, 47.0, 60.0)):
        fig_text_mm(fig, x, KEY_Y, REP_SHORT[rep], ha="left", va="baseline",
                    fontsize=FS_SMALL, color=REP[rep], fontweight="bold")
    fig_text_mm(fig, 75.0, KEY_Y,
                "blocks:  ● solid = block 1     □ dashed = block 2"
                "     pale line = across-block pooled mean",
                ha="left", va="baseline", fontsize=FS_SMALL, color=MUTED)


def _label_kappa(ax, fig, pts, labels, color, placed) -> None:
    """Annotate a complex-plane path with kappa, nudging away from earlier labels.

    The three representation paths cross, so a fixed offset collides. Each
    label takes the first candidate offset that clears every label already
    placed in the panel.
    """
    ppd = fig.dpi / 72.0
    for (xa, yb), lab in zip(pts, labels):
        x0, y0 = ax.transData.transform((xa, yb))
        for dx, dy in ((0.0, 3.4), (0.0, -3.4), (7.5, 0.8), (-7.5, 0.8),
                       (0.0, 8.0), (0.0, -8.0)):
            px, py = x0 + dx * ppd, y0 + dy * ppd
            if all((px - qx) ** 2 + (py - qy) ** 2 > (8.5 * ppd) ** 2
                   for qx, qy in placed):
                placed.append((px, py))
                ax.annotate(f"{lab:g}", (xa, yb), textcoords="offset points",
                            xytext=(dx, dy), ha="center",
                            va="bottom" if dy >= 0 else "top",
                            fontsize=FS_TINY, color=color, zorder=6)
                break


# --------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    endp, ana, cx, ell, dc1, proto = _load()
    kap = _kappas(cx)
    prof = {k: f"kappa_{k:g}" for k in kap}

    fig = new_figure(H_MM)

    # ---- a  activity A(kappa) ---------------------------------------------
    axA = mm_axes(fig, COL_X[0], ROW_TOP[0], COL_W, TALL_H)
    panel_label(axA, "a", r"Activity $A(\kappa)$")
    for rep in REP_ORDER:
        pool = _pooled(cx, rep, "activity")
        axA.plot(range(len(kap)), pool, color=REP[rep], lw=LW_THIN, alpha=0.55,
                 zorder=2)
        for j, blk in enumerate(BLOCKS):
            sub = _cell(cx, blk, rep)
            y = sub["activity"].to_numpy()
            ci = [_boot(ana, blk, rep, prof[k], "mean_activity") for k in kap]
            lo = [c[0] for c in ci]
            hi = [c[1] for c in ci]
            _block_series(axA, np.arange(len(kap)) + (j - 0.5) * 2 * DX, y, lo, hi,
                          rep, blk)
    _kappa_axis(axA, kap)
    axA.set_ylabel(r"$A$", fontsize=FS_BODY)
    axA.set_ylim(0.9615, 1.0045)
    axA.set_yticks([0.97, 0.98, 0.99, 1.00])
    despine(axA)
    trim_spines(axA)

    # ---- b  signed mean action a0(kappa) ----------------------------------
    axB = mm_axes(fig, COL_X[1], ROW_TOP[0], COL_W, TALL_H)
    panel_label(axB, "b", r"Signed mean action $a_0(\kappa)$")
    axB.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=0)
    for rep in REP_ORDER:
        pool = _pooled(cx, rep, "a0")
        axB.plot(range(len(kap)), pool, color=REP[rep], lw=LW_THIN, alpha=0.55,
                 zorder=2)
        for j, blk in enumerate(BLOCKS):
            sub = _cell(cx, blk, rep)
            y = sub["a0"].to_numpy()
            ci = [_boot(ana, blk, rep, prof[k], "a0") for k in kap]
            _block_series(axB, np.arange(len(kap)) + (j - 0.5) * 2 * DX, y,
                          [c[0] for c in ci], [c[1] for c in ci], rep, blk)
    _kappa_axis(axB, kap)
    axB.set_ylabel(r"$a_0$", fontsize=FS_BODY)
    axB.set_ylim(-0.50, 0.44)
    axB.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    despine(axB)
    trim_spines(axB)

    # ---- c / d  low-order harmonics with the frozen phase gate -------------
    for panel, m, col, letter, title in (
        ("c", 1, COL_X[2], "c", "First harmonic"),
        ("d", 2, COL_X[0], "d", "Second harmonic"),
    ):
        row = 0 if m == 1 else 1
        top = ROW_TOP[row]
        axM = mm_axes(fig, col, top, COL_W, MAIN_H)
        axP = mm_axes(fig, col, top + MAIN_H + STRIP_GAP, COL_W, STRIP_H)
        panel_label(axM, letter,
                    f"{title}: $a_{m}$, $R_{m}$" + r" and $\phi_%d$" % m)
        axM.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=0)
        for rep in REP_ORDER:
            axM.plot(range(len(kap)), _pooled(cx, rep, f"R{m}"), color=REP[rep],
                     lw=LW_THIN, ls=(0, (1, 1.2)), alpha=0.9, zorder=2)
            for j, blk in enumerate(BLOCKS):
                sub = _cell(endp, blk, rep)
                y = sub[f"a{m}"].to_numpy()
                lo = sub[f"a{m}_ci_low"].to_numpy()
                hi = sub[f"a{m}_ci_high"].to_numpy()
                _block_series(axM, np.arange(len(kap)) + (j - 0.5) * 2 * DX,
                              y, lo, hi, rep, blk)
                # phase strip: only cells that pass the frozen reporting gate
                cs = _cell(cx, blk, rep)
                gate = cs[f"phi{m}_reportable"].to_numpy() == 1
                st = BLOCK_STYLE[blk]
                axP.plot(
                    (np.arange(len(kap)) + (j - 0.5) * 2 * DX)[gate],
                    cs[f"phi{m}_degrees"].to_numpy()[gate],
                    ls="none", marker=st["marker"], markersize=MS_POINT,
                    markerfacecolor=REP[rep] if st["fillstyle"] == "full" else "white",
                    markeredgecolor=REP[rep], markeredgewidth=LW_HAIR, clip_on=False,
                )
        _kappa_axis(axM, kap, label=False)
        axM.set_ylabel(f"$a_{m}$", fontsize=FS_BODY)
        if m == 1:
            axM.set_ylim(-0.58, 1.16)
            axM.set_yticks([-0.5, 0.0, 0.5, 1.0])
        else:
            axM.set_ylim(-0.32, 0.72)
            axM.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
        despine(axM)
        trim_spines(axM, x=False)
        axM.spines["bottom"].set_visible(False)
        axM.tick_params(axis="x", length=0)
        _kappa_axis(axP, kap)
        axP.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=0)
        axP.set_ylim(-195, 195)
        axP.set_yticks([-180, 0, 180])
        axP.set_yticklabels([r"$-180$", "0", "180"])
        axP.set_ylabel(r"$\phi_%d\,(\degree)$" % m, fontsize=FS_BODY)
        despine(axP)
        trim_spines(axP, x=False)
        n_gate = int(cx[f"phi{m}_reportable"].sum())
        axP.text(-0.36, -148.0,
                 f"{n_gate}/{len(cx)} cells pass the prespecified gate",
                 ha="left", va="center", fontsize=FS_TINY, color=MUTED)
        ylo, yhi = axM.get_ylim()
        # the mark convention is identical in c and d, so it is stated once,
        # in the only free band either panel has (the foot of c)
        if m == 1:
            axM.text(-0.40, ylo + 0.204 * (yhi - ylo),
                     r"markers: $a_m$;  dotted: $R_m$",
                     ha="left", va="center", fontsize=FS_TINY, color=MUTED)

    # ---- e / f  complex C1 and C2 trajectories -----------------------------
    ax_e = None
    for m, col, letter in ((1, COL_X[1], "e"), (2, COL_X[2], "f")):
        ax = mm_axes(fig, col, ROW_TOP[1], COL_W, TALL_H)
        if m == 1:
            ax_e = ax
        panel_label(ax, letter, r"Complex $C_%d=a_%d+ib_%d$" % (m, m, m))
        if m == 1:
            ax.set_xlim(-0.62, 1.16)
            span = (1.16 + 0.62) * TALL_H / COL_W
        else:
            ax.set_xlim(-0.33, 0.73)
            span = (0.73 + 0.33) * TALL_H / COL_W
        ax.set_ylim(-span / 2, span / 2)
        ax.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=0)
        ax.axvline(0.0, color=RULE, lw=LW_HAIR, zorder=0)
        placed: list[tuple[float, float]] = []
        for rep in REP_ORDER:
            for blk in BLOCKS:
                st = BLOCK_STYLE[blk]
                sub = _cell(cx, blk, rep)
                eb = ell[(ell["block"] == blk) & (ell["representation"] == rep)
                         & (ell["harmonic"] == m)].sort_values("concentration")
                for _, r in eb.iterrows():
                    ax.add_patch(mpatches.Ellipse(
                        (r["center_a"], r["center_b"]),
                        width=r["axis_major"], height=r["axis_minor"],
                        angle=r["angle_degrees"], facecolor=REP[rep], alpha=0.10,
                        edgecolor=REP[rep], lw=LW_HAIR, zorder=1))
                ax.plot(sub[f"a{m}"], sub[f"b{m}"], color=REP[rep], ls=st["ls"],
                        lw=LW_LINE, marker=st["marker"], markersize=MS_POINT,
                        markerfacecolor=REP[rep] if st["fillstyle"] == "full" else "white",
                        markeredgecolor=REP[rep], markeredgewidth=LW_HAIR, zorder=4)
            # kappa labels on the across-block mean, so the path is annotated once
            pa = _pooled(cx, rep, f"a{m}")
            pb = _pooled(cx, rep, f"b{m}")
            _label_kappa(ax, fig, list(zip(pa, pb)), kap, REP[rep], placed)
        ax.set_xlabel(f"$a_{m}$", fontsize=FS_BODY)
        ax.set_ylabel(f"$b_{m}$", fontsize=FS_BODY)
        despine(ax)
        trim_spines(ax)
    n_dc1 = int((dc1["contains_origin"] == 0).sum())
    ax_e.text(0.02, 0.035,
              "ellipses: per-block bootstrap; point labels are $\\kappa$\n"
              f"{n_dc1}/{len(dc1)} pairwise $\\Delta C_1$ ellipses exclude the origin",
              transform=ax_e.transAxes, ha="left", va="bottom",
              fontsize=FS_TINY, color=MUTED, linespacing=1.4)

    # ---- g  frozen phenotype classification --------------------------------
    axG = mm_axes(fig, G_X, ROW_TOP[2], G_W, G_H)
    panel_label_at(fig, COL_X[0] - 8.6, LETTER_Y[2], "g",
                   "Prespecified phenotype classification")
    axG.set_xlim(0, len(kap))
    axG.set_ylim(len(REP_ORDER), 0)
    axG.set_xticks(np.arange(len(kap)) + 0.5)
    axG.set_xticklabels([f"{k:g}" for k in kap], fontsize=FS_TICK)
    axG.set_xlabel(r"concentration $\kappa$", fontsize=FS_BODY)
    axG.set_yticks([])
    axG.tick_params(length=0)
    for sp in axG.spines.values():
        sp.set_visible(False)
    gate_min = float(proto["replication_rules"]["minimum_phenotype_probability"])
    for i, rep in enumerate(REP_ORDER):
        for j, k in enumerate(kap):
            axG.add_patch(mpatches.Rectangle(
                (j + 0.04, i + 0.06), 0.92, 0.88, facecolor=REP[rep], alpha=0.07,
                edgecolor="white", lw=LW_THIN, zorder=1))
            cells = cx[(cx["representation"] == rep) & (cx["concentration"] == k)]
            for h, dx in ((1, 0.33), (2, 0.55)):
                for b, blk in enumerate(BLOCKS):
                    ok = int(cells[cells["block"] == blk][f"phi{h}_reportable"].iloc[0])
                    axG.add_patch(mpatches.Rectangle(
                        (j + dx, i + 0.28 + b * 0.24), 0.11, 0.19,
                        facecolor=REP[rep] if ok else "white",
                        edgecolor=REP[rep], lw=LW_HAIR, zorder=3))
        # row margin: representation, frozen class, phenotype probability
        y_mm = ROW_TOP[2] + (i + 0.42) * G_H / len(REP_ORDER)
        fig_text_mm(fig, G_LABEL_X, y_mm, REP_SHORT[rep], ha="left", va="baseline",
                    fontsize=FS_BODY, color=REP[rep], fontweight="bold")
        fig_text_mm(fig, G_LABEL_X, y_mm + 3.0, CLASS_NAME[rep], ha="left",
                    va="baseline", fontsize=FS_SMALL, color=MUTED)
        probs = [ana["blocks"][b]["representations"][rep]["phenotype_probability"]
                 for b in BLOCKS]
        passes = all(ana["blocks"][b]["representations"][rep]["phenotype_pass"]
                     for b in BLOCKS)
        # one grammar per panel: the probability keys its own representation
        # row, so it is set in that row's colour (red only if the gate fails)
        col = REP[rep] if passes else STOP_RED
        fig_text_mm(fig, G_CHIP_X, y_mm,
                    f"$P$ = {probs[0]:.3f} / {probs[1]:.3f}",
                    ha="left", va="baseline", fontsize=FS_SMALL, color=col)
        fig_text_mm(fig, G_CHIP_X, y_mm + 3.0,
                    "blocks 1 / 2" + (r"  $\geq$ " + f"{gate_min:g}" if passes else ""),
                    ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)
    fig_text_mm(fig, G_X, ROW_TOP[2] - 3.1,
                r"per-cell phase-reporting gates: left mark $\phi_1$, right mark "
                r"$\phi_2$;",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)
    fig_text_mm(fig, G_X, ROW_TOP[2] - 0.9,
                "upper row block 1, lower row block 2; filled = gate passed",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)

    # ---- h  between-block replication --------------------------------------
    axH = mm_axes(fig, COL_X[2], ROW_TOP[2], H_SIDE, H_SIDE)
    # bottom-row letters share one baseline: g is set above panel g's own
    # header notes, so h is lifted to the same LETTER_Y rather than the
    # default 1.6 mm above its axes.
    panel_label(axH, "h", "Block replication",
                dy_mm=ROW_TOP[2] - LETTER_Y[2])
    quants = (("activity", "$A$", "o"), ("a0", "$a_0$", "s"),
              ("R1", "$R_1$", "^"), ("R2", "$R_2$", "D"))
    lo_all, hi_all, rvals = [], [], {}
    for col_name, lab, mk in quants:
        piv = cx.pivot_table(index=["representation", "concentration"],
                             columns="block", values=col_name)
        x = piv["block_1"].to_numpy()
        y = piv["block_2"].to_numpy()
        lo_all += [x.min(), y.min()]
        hi_all += [x.max(), y.max()]
        rvals[lab] = float(np.corrcoef(x, y)[0, 1])
        for rep in REP_ORDER:
            sel = piv.index.get_level_values(0) == rep
            axH.plot(x[sel], y[sel], ls="none", marker=mk, markersize=MS_POINT,
                     markerfacecolor="none", markeredgecolor=REP[rep],
                     markeredgewidth=LW_THIN, clip_on=False, zorder=4)
    lo, hi = min(lo_all), max(hi_all)
    pad = 0.09 * (hi - lo)
    axH.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED,
             lw=LW_HAIR, zorder=1)
    axH.set_xlim(lo - pad, hi + pad)
    axH.set_ylim(lo - pad, hi + pad)
    axH.set_xticks([-0.5, 0.0, 0.5, 1.0])
    axH.set_yticks([-0.5, 0.0, 0.5, 1.0])
    axH.set_xlabel("block 1 estimate", fontsize=FS_BODY)
    axH.set_ylabel("block 2 estimate", fontsize=FS_BODY)
    despine(axH)
    trim_spines(axH)
    for n, (lab, mk) in enumerate((("$A$", "o"), ("$a_0$", "s"),
                                   ("$R_1$", "^"), ("$R_2$", "D"))):
        y = 0.30 - n * 0.083
        axH.plot([0.575], [y], marker=mk, markersize=MS_POINT,
                 markerfacecolor="none", markeredgecolor=INK,
                 markeredgewidth=LW_THIN, transform=axH.transAxes,
                 clip_on=False, zorder=6)
        axH.text(0.635, y, f"{lab}  $r$ = {rvals[lab]:.3f}", ha="left",
                 va="center", fontsize=FS_SMALL, color=INK,
                 transform=axH.transAxes, zorder=6)
    axH.text(0.545, 0.385, "identity line; $r$ descriptive only",
             ha="left", va="center", fontsize=FS_TINY, color=MUTED,
             transform=axH.transAxes, zorder=6)

    # ---- shared keys and footnote -----------------------------------------
    _shared_key(fig)

    corr = ana["block_comparison"]
    r1 = [corr[r]["trajectory_correlations"]["a_sin_1"] for r in REP_ORDER]
    r2 = [corr[r]["trajectory_correlations"]["a_sin_2"] for r in REP_ORDER]
    n_rec = sum(ana["blocks"][b]["operational"]["n_valid"] for b in BLOCKS)
    n_coef = len(REP_ORDER) * len(kap) * 3          # a0, a1, a2 per cell
    n_incl = round(ana["coefficient_compatibility_fraction"] * n_coef)
    n_coef = len(REP_ORDER) * len(kap) * 3          # a0, a1, a2 per cell
    n_incl = round(ana["coefficient_compatibility_fraction"] * n_coef)

    return save_fig(fig, "figS05_transmutation_map")
