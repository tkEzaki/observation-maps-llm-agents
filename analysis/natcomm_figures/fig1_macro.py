"""Figure 1 — Observation representations select distinct collective phases."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from analysis.natcomm_figures.style import (
    FIGSIZE,
    PHENOTYPE,
    REP,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    apply_style,
    hex_rgb,
    paired_ci,
    panel_label,
    save_fig,
)

PAIRED = ROOT / "analysis" / "matched_rep_collective" / "paired_analysis"
META = json.loads((PAIRED / "meta.json").read_text(encoding="utf-8"))


def _session_for(rep: str) -> Path:
    for s in META["sessions"]:
        if s["representation"] == rep:
            return ROOT / Path(s["session"])
    raise KeyError(rep)


def _load_ts(rep: str) -> pd.DataFrame:
    return pd.read_csv(PAIRED / f"_tmp_session_{REP_ORDER.index(rep)}" / "trajectory_harmonics_timeseries.csv")


def _phases(rep: str, run_id: str, t: int) -> np.ndarray:
    arr = np.load(_session_for(rep) / run_id / "phases.npy")
    return np.asarray(arr[min(t, arr.shape[0] - 1)], dtype=float)


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    ep = pd.read_csv(PAIRED / "endpoints_long.csv")
    k_show = 0.08
    # median final_r1 under moments for phase snapshots
    sub_m = ep[(ep["representation"] == "moments_m1_m3") & (np.isclose(ep["coupling"], k_show))]
    seed = int(sub_m.iloc[(sub_m["final_r1"] - sub_m["final_r1"].median()).abs().argmin()]["seed_index"])
    run_id = f"N17_K+0.08_s{seed}"

    fig = plt.figure(figsize=FIGSIZE)
    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[1.15, 0.55, 1.2, 1.15],
        hspace=0.45,
        wspace=0.4,
    )

    # ---- A system + intervention ----
    axAB = fig.add_subplot(gs[0, 0:3])
    panel_label(axAB, "A", "Matched observation-map intervention")
    axAB.set_xlim(0, 12)
    axAB.set_ylim(0, 7.2)
    axAB.axis("off")
    axAB.add_patch(plt.Circle((1.4, 4.2), 0.95, fill=False, lw=1.4, color="#333"))
    th = np.linspace(0, 2 * np.pi, 17, endpoint=False)
    axAB.scatter(1.4 + 0.72 * np.cos(th), 4.2 + 0.72 * np.sin(th), s=14, c="#555")
    axAB.text(1.4, 2.85, r"physical $\rho$", ha="center", fontsize=8)
    for name, y, key in (
        ("moments", 5.6, "moments_m1_m3"),
        ("centers", 4.2, "centers_24_standard"),
        ("intervals", 2.8, "intervals_24_decimal6"),
    ):
        c = REP[key]
        axAB.annotate("", xy=(3.5, y), xytext=(2.4, 4.2), arrowprops=dict(arrowstyle="->", color=c, lw=1.3))
        axAB.text(4.35, y, name, ha="center", va="center", color=c, fontweight="bold", fontsize=9)
    axAB.add_patch(
        mpatches.FancyBboxPatch((5.5, 3.4), 1.9, 1.6, boxstyle="round,pad=0.04", facecolor="#f2f2f2", edgecolor="#333")
    )
    axAB.text(6.45, 4.2, "frozen\nLLM", ha="center", va="center", fontsize=8)
    axAB.annotate("", xy=(8.0, 4.2), xytext=(7.45, 4.2), arrowprops=dict(arrowstyle="->", color="#333"))
    axAB.text(8.9, 4.2, r"$f\in\{-1,0,+1\}$", ha="center", fontsize=8)

    axAB.add_patch(
        mpatches.FancyBboxPatch((0.3, 0.55), 5.5, 1.85, boxstyle="round,pad=0.04", facecolor="#f7f7f7", edgecolor="#888")
    )
    axAB.text(3.05, 2.05, "FIXED", ha="center", fontweight="bold", fontsize=8)
    axAB.text(
        3.05,
        1.15,
        r"$\theta_i(0),\ \omega_i,\ N{=}17,\ K,\ T$"
        + "\nsystem prompt · action contract · integrator",
        ha="center",
        fontsize=7.5,
    )
    axAB.add_patch(
        mpatches.FancyBboxPatch(
            (6.1, 0.55), 5.5, 1.85, boxstyle="round,pad=0.04", facecolor="#fff7eb", edgecolor=REP["moments_m1_m3"]
        )
    )
    axAB.text(8.85, 2.05, "INTERVENED", ha="center", fontweight="bold", fontsize=8, color=REP["moments_m1_m3"])
    axAB.text(8.85, 1.15, r"observation map $\mathcal{R}(\rho)$ only", ha="center", fontsize=8.5)
    axAB.text(
        6.0,
        0.15,
        r"Matched design: $K\in\{-0.15,0,0.08,0.15\}$, $T=100$, 6 paired seeds",
        ha="center",
        fontsize=7.5,
    )

    # ---- B trajectories ----
    axC = fig.add_subplot(gs[0, 3:])
    panel_label(axC, "B", rf"Collective dynamics ($K={k_show}$)")
    for rep in REP_ORDER:
        ts = _load_ts(rep)
        for s in range(6):
            sub = ts[(np.isclose(ts["coupling"], k_show)) & (ts["seed_index"] == s)].sort_values("t")
            axC.plot(sub["t"], sub["r1"], color=REP[rep], alpha=0.18, lw=0.9)
        g = ts[np.isclose(ts["coupling"], k_show)].groupby("t")["r1"].median()
        axC.plot(g.index, g.values, color=REP[rep], lw=2.2, label=REP_SHORT[rep])
    axC.set_ylim(0, 1.05)
    axC.set_xlabel(r"$t$")
    axC.set_ylabel(r"$r_1(t)$")
    axC.legend(frameon=False, loc="upper left", fontsize=8)

    # phase snapshots under B
    for j, rep in enumerate(REP_ORDER):
        axp = fig.add_subplot(gs[1, 3 + j])
        ph = _phases(rep, run_id, 100)
        axp.scatter(np.cos(ph), np.sin(ph), s=16, c=REP[rep])
        axp.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#ccc", lw=0.7))
        axp.set_xlim(-1.3, 1.3)
        axp.set_ylim(-1.3, 1.3)
        axp.set_aspect("equal")
        axp.set_xticks([])
        axp.set_yticks([])
        axp.set_title(REP_SHORT[rep], fontsize=8, color=REP[rep], fontweight="bold", pad=2)
        if j == 0:
            axp.set_ylabel(r"final phase ($t=100$)", fontsize=8)

    # spacer label on left of phase row
    ax_note = fig.add_subplot(gs[1, 0:3])
    ax_note.axis("off")
    ax_note.text(
        0.5,
        0.5,
        f"Final configurations for median-$r_1$ moments seed (seed={seed})",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
        color="#555",
    )

    # ---- C phenotype ----
    axD = fig.add_subplot(gs[2, :])
    panel_label(axD, "C", "All-seed phenotype matrix")
    ks = [-0.15, 0.0, 0.08, 0.15]
    rgb = np.zeros((6, 12, 3))
    for j, rep in enumerate(REP_ORDER):
        for kk, k in enumerate(ks):
            col = 4 * j + kk
            for s in range(6):
                row = ep[
                    (ep["representation"] == rep)
                    & (np.isclose(ep["coupling"], k))
                    & (ep["seed_index"] == s)
                ].iloc[0]
                rgb[s, col] = hex_rgb(PHENOTYPE[row["cluster_phenotype"]])
    axD.imshow(rgb, aspect="auto")
    axD.set_yticks(range(6))
    axD.set_yticklabels([f"seed {i}" for i in range(6)])
    xt, xl = [], []
    for j in range(3):
        for kk, k in enumerate(ks):
            xt.append(4 * j + kk)
            xl.append(f"{k:g}")
    axD.set_xticks(xt)
    axD.set_xticklabels(xl, fontsize=7)
    for j, rep in enumerate(REP_ORDER):
        axD.text(
            4 * j + 1.5,
            -0.85,
            REP_SHORT[rep],
            ha="center",
            color=REP[rep],
            fontweight="bold",
            fontsize=9,
            clip_on=False,
        )
    for x in (3.5, 7.5):
        axD.axvline(x + 0.5, color="white", lw=2.5)
    axD.text(
        2.5,
        6.35,
        r"moments 6/6 lock; centers/intervals 0/6  ·  exact sign $p=0.03125$",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#1a1a1a",
        fontweight="bold",
        clip_on=False,
    )
    handles = [mpatches.Patch(color=c, label=k.replace("_", " ")) for k, c in PHENOTYPE.items()]
    axD.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7)

    # ---- D paired r1 facets ----
    axD0 = fig.add_subplot(gs[3, 0:2])
    axD1 = fig.add_subplot(gs[3, 2:4], sharey=axD0)
    panel_label(axD0, "D", r"Paired final $r_1$")
    axD1.set_title(r"$K=0.15$", loc="center", fontsize=9)
    axD0.set_title(r"$K=0.08$", loc="center", fontsize=9)
    for ax, k in ((axD0, 0.08), (axD1, 0.15)):
        for s in range(6):
            ys = [
                float(
                    ep[
                        (ep["representation"] == rep)
                        & (np.isclose(ep["coupling"], k))
                        & (ep["seed_index"] == s)
                    ]["final_r1"].iloc[0]
                )
                for rep in REP_ORDER
            ]
            ax.plot([0, 1, 2], ys, color="#d0d0d0", lw=0.9, zorder=1)
        for i, rep in enumerate(REP_ORDER):
            vals = (
                ep[(ep["representation"] == rep) & (np.isclose(ep["coupling"], k))]
                .sort_values("seed_index")["final_r1"]
                .to_numpy()
            )
            ax.scatter(np.full(6, i) + np.linspace(-0.06, 0.06, 6), vals, c=REP[rep], s=32, zorder=3)
            m, lo, hi = paired_ci(vals)
            ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="D", color="k", ms=4.5, capsize=3, zorder=4)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([REP_SHORT[r] for r in REP_ORDER])
        for tick, r in zip(ax.get_xticklabels(), REP_ORDER):
            tick.set_color(REP[r])
        ax.set_ylim(0, 1.05)
    axD0.set_ylabel(r"final $r_1$")
    axD1.tick_params(labelleft=False)

    # ---- E Q2 ----
    axF = fig.add_subplot(gs[3, 4])
    panel_label(axF, "E", r"Final $Q_2=r_2-r_1$")
    for facet, k in enumerate((0.08, 0.15)):
        for i, rep in enumerate(REP_ORDER):
            vals = (
                ep[(ep["representation"] == rep) & (np.isclose(ep["coupling"], k))]
                .sort_values("seed_index")["final_Q2"]
                .to_numpy()
            )
            x0 = facet * 3.5 + i
            axF.scatter(np.full(6, x0) + np.linspace(-0.08, 0.08, 6), vals, c=REP[rep], s=22, zorder=3)
            m, lo, hi = paired_ci(vals)
            axF.errorbar([x0], [m], yerr=[[m - lo], [hi - m]], fmt="D", color="k", ms=3.5, capsize=2, zorder=4)
    axF.axhline(0, color="#888", lw=0.8)
    axF.axvline(2.75, color="#ccc", lw=0.8, ls="--")
    axF.set_xticks([1, 4.5])
    axF.set_xticklabels([r"$K=0.08$", r"$K=0.15$"])
    axF.set_ylabel(r"$Q_2$")

    # ---- F K=0: explicit coincidence ----
    axG = fig.add_subplot(gs[3, 5])
    panel_label(axG, "F", r"$K=0$ control")
    mats = []
    for rep in REP_ORDER:
        ts = _load_ts(rep)
        g = ts[np.isclose(ts["coupling"], 0.0)].groupby("t")["r1"].mean()
        mats.append(g.to_numpy())
    mats = np.stack(mats)
    max_diff = float(np.max(np.abs(mats[:, None, :] - mats[None, :, :])))
    t = _load_ts("moments_m1_m3")
    g = t[np.isclose(t["coupling"], 0.0)].groupby("t")["r1"].mean()
    axG.plot(g.index, g.values, color="black", lw=2.2)
    axG.set_ylim(0, 1.0)
    axG.set_xlim(0, 100)
    axG.set_xlabel(r"$t$")
    axG.set_ylabel(r"$r_1(t)$")
    axG.text(
        0.5,
        0.95,
        "three representations coincide exactly\n"
        + rf"$\max_t|\Delta r_1|={max_diff:.1e}$",
        transform=axG.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
    )
    # activity as small inset
    axi = axG.inset_axes([0.55, 0.12, 0.42, 0.35])
    for i, rep in enumerate(REP_ORDER):
        vals = ep[(ep["representation"] == rep) & (np.isclose(ep["coupling"], 0.0))]["mean_activity"].to_numpy()
        axi.scatter(np.full(len(vals), i) + np.linspace(-0.1, 0.1, len(vals)), vals, c=REP[rep], s=14)
    axi.set_xticks([0, 1, 2])
    axi.set_xticklabels(["m", "c", "i"], fontsize=6)
    axi.set_ylabel("activity", fontsize=6)
    axi.tick_params(labelsize=5)
    axi.set_title("activity still differs", fontsize=6, pad=1)

    return save_fig(fig, "fig1_collective_phases")
