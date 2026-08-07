"""Supplementary Figure S10 - Negative coupling and nonlocking dynamical pathways.

The point of the figure is to characterise ``K = -0.15`` *without* calling the
resulting state disordered. Three things are true at once and each gets a
panel:

* the polar order parameter stays low for every representation (a), yet
* the action operator is almost always firing (b) and carries a
  representation-specific mean torque whose **sign** differs (c), and
* that torque is exactly the quantity the engine identity turns into a
  collective drift, ``Omega_coll = mean(omega) + K * tau_social`` (d).

Panels e and f close the description: a mechanically selected final phase
configuration per representation next to every final harmonic (e), and the
shared-seed paired contrasts in torque, activity and Q2 (f).

Every printed number is computed here from the frozen artifacts
(``paired_analysis/endpoints_long.csv``, the per-session
``trajectory_harmonics_timeseries.csv``, ``paired_contrasts.csv``) - nothing
is hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures_si.style import (
    CAPSIZE,
    FS_BODY,
    FS_SMALL,
    FS_TINY,
    INK,
    LW_FAINT,
    LW_HAIR,
    LW_LINE,
    LW_TRACE,
    MS_MEAN,
    MS_POINT,
    MUTED,
    REP,
    REP_ABBR,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    apply_style,
    errorbar_mean,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label,
    panel_label_at,
    rep_tick_colors,
    save_fig,
    swarm_x,
    trim_spines,
)

PAIRED = ROOT / "analysis" / "matched_rep_collective" / "paired_analysis"
META = json.loads((PAIRED / "meta.json").read_text(encoding="utf-8"))

K_NEG = -0.15          # the coupling this figure characterises
K_ZERO = 0.0           # the uncoupled baseline panel d contrasts it against
N_SEEDS = 6
W_FINAL = 20           # trailing window (steps) for the descriptive summaries

MINUS = "−"

# Scatter takes an *area*; derive it from the shared marker diameters so dot
# sizes match the MS_* scale the rest of the set uses.
S_POINT = MS_POINT ** 2
S_RING = MS_MEAN ** 2

SEED_ALPHA = 0.22
FAINT = "#CFCFCF"

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm-wide page)
# ---------------------------------------------------------------------------
H_MM = 134.5

R1_TOP, R1_H = 14.0, 48.0
R2_TOP, R2_H = 80.0, 44.0

COL_LEFT = (13.0, 76.0, 132.0)      # shared left edge per column, both rows
COL_LETTER = (4.0, 68.0, 124.0)     # shared letter column, both rows
R1_LETTER_DY = 5.2                  # panel a needs a facet-header band
R2_LETTER_DY = 1.6

# a: 3 representations x 3 harmonics
A_SUB_W, A_SUB_H = 14.0, 14.0
A_GAP_X, A_GAP_Y = 4.5, 3.0
A_HEADER_Y = 12.6

# b, c: trajectory bundle + a narrow final-window summary strip
TRAJ_W = 29.0
STRIP_GAP = 2.0
STRIP_W = 13.0

D_W = 50.0                          # d: two coupling groups on one axis

# e: three phase rings over a final-harmonic scatter
RING_W, RING_GAP = 13.0, 2.5
E_CAP_Y = 97.0
E_SCAT_TOP, E_SCAT_H, E_W = 101.0, 23.0, 44.0

F_W = 44.0                          # f: three paired-contrast groups

ROW1_CAP_Y = 71.6


def _kfmt(k: float) -> str:
    return rf"$K={k:g}$".replace("-", MINUS)


def _session_for(rep: str) -> Path:
    for s in META["sessions"]:
        if s["representation"] == rep:
            return ROOT.joinpath(*str(s["session"]).replace("\\", "/").split("/"))
    raise KeyError(rep)


def _load_ts(rep: str) -> pd.DataFrame:
    return pd.read_csv(
        PAIRED / f"_tmp_session_{REP_ORDER.index(rep)}"
        / "trajectory_harmonics_timeseries.csv"
    )


def _phases(rep: str, run_id: str, t: int) -> np.ndarray:
    arr = np.load(_session_for(rep) / run_id / "phases.npy")
    return np.asarray(arr[min(t, arr.shape[0] - 1)], dtype=float)


def _sci(x: float) -> str:
    """Render a tiny magnitude as maths, e.g. 3 x 10^-17."""
    exp = int(np.floor(np.log10(abs(x))))
    man = abs(x) / 10.0 ** exp
    return rf"{man:.0f}\times10^{{{exp}}}"


def _bare_ring(ax) -> None:
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _bundle(ax, frame: pd.DataFrame, rep: str, col: str, t_max: int) -> None:
    """Six faint seed traces plus the across-seed median, in one colour."""
    d = frame[frame["t"] < t_max] if col != "r1" else frame
    for s in range(N_SEEDS):
        sub = d[d["seed_index"] == s].sort_values("t")
        ax.plot(sub["t"], sub[col], color=REP[rep], alpha=SEED_ALPHA, lw=LW_FAINT)
    g = d.groupby("t")[col].median()
    ax.plot(g.index, g.values, color=REP[rep], lw=LW_TRACE)


def build(_out_dir: Path | None = None) -> Path:
    apply_style()

    ep = pd.read_csv(PAIRED / "endpoints_long.csv")
    pc = pd.read_csv(PAIRED / "paired_contrasts.csv")
    ts = {rep: _load_ts(rep) for rep in REP_ORDER}
    neg = {rep: ts[rep][np.isclose(ts[rep]["coupling"], K_NEG)] for rep in REP_ORDER}
    ep_neg = ep[np.isclose(ep["coupling"], K_NEG)]

    t_max = int(neg[REP_ORDER[0]]["t"].max())
    t_lo = t_max - W_FINAL            # actions exist for t < t_max only

    def _final_window(rep: str, col: str) -> np.ndarray:
        d = neg[rep]
        d = d[(d["t"] >= t_lo) & (d["t"] < t_max)]
        return d.groupby("seed_index")[col].mean().sort_index().to_numpy()

    fig = new_figure(H_MM)

    # =====================================================================
    # a  order parameters: 6 seeds of r1, r2, r3, faceted by representation
    # =====================================================================
    r_cols = ("r1", "r2", "r3")
    r_max = max(float(neg[rep][c].max()) for rep in REP_ORDER for c in r_cols)
    y_top = float(np.ceil(r_max * 10.0) / 10.0)
    axA0 = None
    for j, rep in enumerate(REP_ORDER):
        x_left = COL_LEFT[0] + j * (A_SUB_W + A_GAP_X)
        fig_text_mm(fig, x_left + A_SUB_W / 2.0, A_HEADER_Y, REP_SHORT[rep],
                    ha="center", va="baseline", color=REP[rep],
                    fontweight="bold", fontsize=FS_SMALL)
        for i, col in enumerate(r_cols):
            ax = mm_axes(fig, x_left, R1_TOP + i * (A_SUB_H + A_GAP_Y),
                         A_SUB_W, A_SUB_H)
            _bundle(ax, neg[rep], rep, col, t_max)
            ax.set_xlim(0, t_max)
            ax.set_ylim(0, y_top)
            ax.set_xticks([0, t_max // 2, t_max])
            ax.set_yticks([0, y_top / 2.0, y_top])
            if j == 0:
                ax.set_ylabel(rf"$r_{{{i + 1}}}$", labelpad=1.2)
            else:
                ax.tick_params(labelleft=False)
            if i == len(r_cols) - 1:
                ax.tick_params(labelbottom=True)
                if j == 1:
                    ax.set_xlabel(r"$t$")
            else:
                ax.tick_params(labelbottom=False)
            trim_spines(ax)
            if i == 0 and j == 0:
                axA0 = ax
    panel_label(axA0, "a", f"Harmonic order parameters, {_kfmt(K_NEG)}",
                dx_mm=COL_LETTER[0] - COL_LEFT[0], dy_mm=R1_LETTER_DY)

    # =====================================================================
    # b  activity: near-unit action rate alongside near-zero polar order
    # =====================================================================
    axB = mm_axes(fig, COL_LEFT[1], R1_TOP, TRAJ_W, R1_H)
    panel_label(axB, "b", "Action activity",
                dx_mm=COL_LETTER[1] - COL_LEFT[1], dy_mm=R1_LETTER_DY)
    for rep in REP_ORDER:
        _bundle(axB, neg[rep], rep, "activity", t_max)
    axB.set_xlim(0, t_max)
    axB.set_ylim(0.5, 1.06)
    axB.set_xticks([0, t_max // 2, t_max])
    axB.set_yticks([0.5, 0.75, 1.0])
    axB.set_xlabel(r"$t$")
    axB.set_ylabel(r"activity $A(t)=\langle|f_i|\rangle_i$")
    trim_spines(axB)

    act_fw = {rep: _final_window(rep, "activity") for rep in REP_ORDER}
    axBs = mm_axes(fig, COL_LEFT[1] + TRAJ_W + STRIP_GAP, R1_TOP, STRIP_W, R1_H)
    for i, rep in enumerate(REP_ORDER):
        v = act_fw[rep]
        axBs.scatter(swarm_x(v, i, width=0.22), v, c=REP[rep], s=S_POINT,
                     linewidths=0, zorder=3)
        errorbar_mean(axBs, i, v, color=INK, ms=MS_MEAN, lw=LW_LINE,
                      capsize=CAPSIZE, zorder=4)
    axBs.set_xlim(-0.7, 2.7)
    axBs.set_ylim(0.5, 1.06)
    axBs.set_yticks([0.5, 0.75, 1.0])
    axBs.tick_params(labelleft=False)
    axBs.set_xticks([])
    axBs.set_title(rf"final $t={t_lo}$–${t_max - 1}$", fontsize=FS_TINY,
                   color=MUTED, pad=2.0)
    trim_spines(axBs, x=False)

    # =====================================================================
    # c  social torque: the sign pattern is representation-specific
    # =====================================================================
    axC = mm_axes(fig, COL_LEFT[2], R1_TOP, TRAJ_W, R1_H)
    panel_label(axC, "c", "Social torque",
                dx_mm=COL_LETTER[2] - COL_LEFT[2], dy_mm=R1_LETTER_DY)
    for rep in REP_ORDER:
        _bundle(axC, neg[rep], rep, "tau_social", t_max)
    axC.axhline(0.0, color=MUTED, lw=LW_HAIR, zorder=1)
    axC.set_xlim(0, t_max)
    axC.set_ylim(-1.0, 1.62)
    axC.set_xticks([0, t_max // 2, t_max])
    axC.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axC.set_xlabel(r"$t$")
    axC.set_ylabel(r"torque $\tau_{\mathrm{social}}(t)=\langle f_i\rangle_i$")
    trim_spines(axC)

    tau_fw = {rep: _final_window(rep, "tau_social") for rep in REP_ORDER}
    axCs = mm_axes(fig, COL_LEFT[2] + TRAJ_W + STRIP_GAP, R1_TOP, STRIP_W, R1_H)
    for i, rep in enumerate(REP_ORDER):
        v = tau_fw[rep]
        axCs.scatter(swarm_x(v, i, width=0.22), v, c=REP[rep], s=S_POINT,
                     linewidths=0, zorder=3)
        errorbar_mean(axCs, i, v, color=INK, ms=MS_MEAN, lw=LW_LINE,
                      capsize=CAPSIZE, zorder=4)
    axCs.axhline(0.0, color=MUTED, lw=LW_HAIR, zorder=1)
    axCs.set_xlim(-0.7, 2.7)
    axCs.set_ylim(-1.0, 1.62)
    axCs.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axCs.tick_params(labelleft=False)
    axCs.set_xticks([])
    axCs.set_title(rf"final $t={t_lo}$–${t_max - 1}$", fontsize=FS_TINY,
                   color=MUTED, pad=2.0)
    trim_spines(axCs, x=False)

    # descriptive sign statement, read off the frozen full-run means
    tau_run = {
        rep: ep_neg[ep_neg["representation"] == rep]
        .sort_values("seed_index")["mean_tau_social"].to_numpy()
        for rep in REP_ORDER
    }
    axC.text(0.02, 0.985, "full-run mean torque", transform=axC.transAxes,
             ha="left", va="top", fontsize=FS_TINY, color=MUTED)
    for i, rep in enumerate(REP_ORDER):
        v = tau_run[rep]
        n_neg = int((v < 0).sum())
        agree = max(n_neg, len(v) - n_neg)
        word = "negative" if n_neg > len(v) - n_neg else "positive"
        axC.text(0.02, 0.925 - 0.058 * i,
                 rf"{REP_ABBR[rep]}  $\bar\tau={np.mean(v):+.3f}$".replace("-", MINUS)
                 + f"  ({agree}/{len(v)} {word})",
                 transform=axC.transAxes, ha="left", va="top",
                 fontsize=FS_TINY, color=REP[rep])

    fig_text_mm(fig, COL_LEFT[0], ROW1_CAP_Y,
                f"All of row 1 is {_kfmt(K_NEG)}. Individual seeds faint, "
                f"across-seed median heavy; summary strips in b and c are the "
                f"per-seed mean over the final {W_FINAL} steps.",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)

    # =====================================================================
    # d  collective frequency and the engine accounting identity
    # =====================================================================
    axD = mm_axes(fig, COL_LEFT[0], R2_TOP, D_W, R2_H)
    panel_label(axD, "d", "Collective frequency accounting",
                dx_mm=COL_LETTER[0] - COL_LEFT[0], dy_mm=R2_LETTER_DY)
    group_dx = 4.0
    resid = 0.0
    omega_bar = 0.0
    for gi, k in enumerate((K_NEG, K_ZERO)):
        sel = ep[np.isclose(ep["coupling"], k)]
        for i, rep in enumerate(REP_ORDER):
            row = sel[sel["representation"] == rep].sort_values("seed_index")
            meas = row["omega_coll"].to_numpy()
            pred = row["omega_coll_predicted"].to_numpy()
            resid = max(resid, float(np.abs(row["engine_residual"]).max()))
            omega_bar = max(omega_bar, float(np.abs(row["mean_omega"]).max()))
            x = swarm_x(meas, gi * group_dx + i, width=0.20)
            axD.scatter(x, pred, s=S_RING, facecolors="none",
                        edgecolors=INK, linewidths=LW_HAIR, zorder=3)
            axD.scatter(x, meas, s=S_POINT, c=REP[rep], linewidths=0, zorder=4)
    axD.axhline(0.0, color=MUTED, lw=LW_HAIR, zorder=1)
    axD.axvline((group_dx + 2) / 2.0, color=FAINT, lw=LW_HAIR, ls=(0, (3, 2)))
    axD.set_xlim(-0.7, group_dx + 2.7)
    axD.set_xticks([0, 1, 2, group_dx, group_dx + 1, group_dx + 2])
    axD.set_xticklabels([REP_ABBR[r] for r in REP_ORDER] * 2)
    rep_tick_colors(axD, REP_ORDER * 2)
    axD.set_ylabel(r"$\Omega_{\mathrm{coll}}$ (rad per step)")
    axD.set_ylim(-0.022, 0.088)
    axD.set_yticks([-0.02, 0.0, 0.02, 0.04, 0.06])
    trim_spines(axD, x=False)
    for gi, k in enumerate((K_NEG, K_ZERO)):
        axD.text(gi * group_dx + 1, 0.084, _kfmt(k), ha="center", va="center",
                 fontsize=FS_BODY, color=INK)
    axD.text(
        0.53, 0.90,
        r"$\Omega_{\mathrm{coll}}=\bar\omega+K\,\tau_{\mathrm{social}}$"
        "\nfilled: measured\nring: reconstructed\n"
        rf"residual $\leq{_sci(resid)}$" "\n"
        rf"$\bar\omega={_sci(omega_bar)}$",
        transform=axD.transAxes, ha="left", va="top", fontsize=FS_TINY,
        color=MUTED, linespacing=1.6,
    )

    # =====================================================================
    # e  final phase configurations + every final harmonic
    # =====================================================================
    panel_label_at(fig, COL_LETTER[1], R2_TOP - R2_LETTER_DY, "e",
                   "Final configurations and harmonics")
    for i, rep in enumerate(REP_ORDER):
        sub = ep_neg[ep_neg["representation"] == rep]
        s = int(sub.iloc[(sub["final_r1"] - sub["final_r1"].median()).abs()
                         .argmin()]["seed_index"])
        axr = mm_axes(fig, COL_LEFT[1] + i * (RING_W + RING_GAP), R2_TOP,
                      RING_W, RING_W)
        _bare_ring(axr)
        ph = _phases(rep, f"N17_K{K_NEG:+g}_s{s}", t_max)
        axr.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=FAINT, lw=LW_HAIR))
        axr.scatter(np.cos(ph), np.sin(ph), s=S_POINT, c=REP[rep], linewidths=0)
        axr.text(-1.43, 1.43, REP_ABBR[rep], ha="left", va="top",
                 fontsize=FS_TINY, fontweight="bold", color=REP[rep])
        axr.text(0.0, -1.45, f"seed {s}", ha="center", va="bottom",
                 fontsize=FS_TINY, color=MUTED)
    fig_text_mm(fig, COL_LEFT[1], E_CAP_Y,
                rf"$t={t_max}$ phases, median-final-$r_1$ seed",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)

    axE = mm_axes(fig, COL_LEFT[1], E_SCAT_TOP, E_W, E_SCAT_H)
    off = (-0.24, 0.0, 0.24)
    for m in (1, 2, 3):
        for i, rep in enumerate(REP_ORDER):
            v = (ep_neg[ep_neg["representation"] == rep]
                 .sort_values("seed_index")[f"final_r{m}"].to_numpy())
            axE.scatter(swarm_x(v, m + off[i], width=0.075), v, c=REP[rep],
                        s=S_POINT, linewidths=0, zorder=3)
            errorbar_mean(axE, m + off[i], v, color=INK, ms=MS_MEAN,
                          lw=LW_LINE, capsize=CAPSIZE, zorder=4)
    axE.set_xlim(0.6, 3.4)
    axE.set_xticks([1, 2, 3])
    axE.set_xticklabels([r"$r_1$", r"$r_2$", r"$r_3$"])
    axE.set_ylim(0, 0.46)
    axE.set_yticks([0, 0.2, 0.4])
    axE.set_ylabel(r"final $r_m$")
    axE.set_xlabel("harmonic")
    trim_spines(axE, x=False)
    axE.text(0.99, 0.98, f"all {N_SEEDS} seeds · all three encodings",
             transform=axE.transAxes, ha="right", va="top",
             fontsize=FS_TINY, color=MUTED)

    # =====================================================================
    # f  shared-seed paired contrasts
    # =====================================================================
    axF = mm_axes(fig, COL_LEFT[2], R2_TOP, F_W, R2_H)
    panel_label(axF, "f", "Paired contrasts vs moments",
                dx_mm=COL_LETTER[2] - COL_LEFT[2], dy_mm=R2_LETTER_DY)
    cneg = pc[np.isclose(pc["coupling"], K_NEG)]
    others = [REP_ORDER[1], REP_ORDER[2]]
    metrics = (
        ("delta_mean_tau_social", r"$\Delta\tau_{\mathrm{social}}$"),
        ("delta_mean_activity", r"$\Delta A$"),
        ("delta_final_Q2", r"$\Delta Q_2$"),
    )
    gdx = 2.6
    agree_txt = {}
    for gi, (col, _lab) in enumerate(metrics):
        for i, other in enumerate(others):
            name = f"{other}__minus__{REP_ORDER[0]}"
            v = (cneg[cneg["contrast"] == name]
                 .sort_values("seed_index")[col].to_numpy())
            x = gi * gdx + i
            axF.scatter(swarm_x(v, x, width=0.20), v, c=REP[other],
                        s=S_POINT, linewidths=0, zorder=3)
            errorbar_mean(axF, x, v, color=INK, ms=MS_MEAN, lw=LW_LINE,
                          capsize=CAPSIZE, zorder=4)
            n_pos = int((v > 0).sum())
            agree = max(n_pos, len(v) - n_pos)
            agree_txt[(col, other)] = agree
            axF.text(x, 0.545, f"{agree}/{len(v)}", ha="center", va="center",
                     fontsize=FS_TINY,
                     color=INK if agree == len(v) else MUTED)
    axF.axhline(0.0, color=MUTED, lw=LW_HAIR, zorder=1)
    axF.set_xlim(-0.7, 2 * gdx + 1.7)
    axF.set_ylim(-0.40, 0.60)
    axF.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    axF.set_xticks([gi * gdx + 0.5 for gi in range(len(metrics))])
    axF.set_xticklabels([lab for _c, lab in metrics])
    axF.set_ylabel("paired difference (shared seed)")
    trim_spines(axF, x=False)
    for i, other in enumerate(others):
        axF.text(0.02, 0.115 - 0.068 * i,
                 f"{REP_SHORT[other]} {MINUS} {REP_SHORT[REP_ORDER[0]]}",
                 transform=axF.transAxes, ha="left", va="bottom",
                 fontsize=FS_TINY, color=REP[other], fontweight="bold")

    # =====================================================================

    return save_fig(fig, "figS10_negative_coupling")
