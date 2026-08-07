"""Supplementary Figure S8 — Complete GPT matched-collective trajectories.

Three lettered pages under one figure number, one page per observation map:

    S8a  moments      figS08a_gpt_trajectories_moments
    S8b  centers      figS08b_gpt_trajectories_centers
    S8c  intervals    figS08c_gpt_trajectories_intervals

Each page carries the complete trajectory set for its representation: a
3 x 4 grid of the twelve (harmonic, coupling) cells, every cell holding all
six seeds of r_m(t), plus the twenty-four final phase configurations at
t = 100. Nothing is averaged across seeds — the plan asks explicitly that
run-to-run timing variation stay visible — and the y range of a given
harmonic is identical on all three pages so the pages can be read side by
side.

These are the trajectories underlying main Figure 1. Every number on the
canvas (sustained-lock counts, final r_1, model id, valid rate) is read from
``paired_analysis`` or from the run session artifacts; none is hard-coded.
The ring in panel d and the r_1 printed under it are cross-checked against
each other, so a drifted artifact fails the build instead of being drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures_si.style import (
    FS_BODY,
    FS_LETTER,
    FS_SMALL,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_THIN,
    MS_POINT,
    MUTED,
    PASS_GREEN,
    REP,
    REP_LIGHT,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    apply_style,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label,
    save_fig,
    trim_spines,
)

PAIRED = ROOT / "analysis" / "matched_rep_collective" / "paired_analysis"
META = json.loads((PAIRED / "meta.json").read_text(encoding="utf-8"))

# Scatter takes an *area*; derive it from the shared marker diameter so the
# phase dots match the MS_* scale used by main Figure 1.
S_POINT = MS_POINT ** 2

PAGE = {
    "moments_m1_m3": ("S8a", "figS08a_gpt_trajectories_moments"),
    "centers_24_standard": ("S8b", "figS08b_gpt_trajectories_centers"),
    "intervals_24_decimal6": ("S8c", "figS08c_gpt_trajectories_intervals"),
}
# What a reader needs in order to decode the serialization from its name.
REP_GLOSS = {
    "moments_m1_m3": r"circular moments $m_1,\,m_3$ of the peer phases",
    "centers_24_standard": "24-bin phase histogram, bin centres in standard notation",
    "intervals_24_decimal6": "24-bin phase histogram, bin intervals to six decimals",
}

HARMONICS = (
    ("r1", r"$r_1(t)$", "Polar order"),
    ("r2", r"$r_2(t)$", "Second harmonic"),
    ("r3", r"$r_3(t)$", "Third harmonic"),
)
KS = (-0.15, 0.0, 0.08, 0.15)
SEEDS = tuple(range(6))
T_FINAL = 100

# The prespecified polar-locking threshold behind ``t_r1_gt_0.9_sustained``
# and ``sustained_lock`` in endpoints_long.csv.
LOCK_R1 = 0.9

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm-wide page)
# ---------------------------------------------------------------------------
H_MM = 165.5

GRID_LEFT, COL_W, COL_PITCH = 13.8, 35.6, 41.5   # four coupling columns
ROW_TOP, ROW_H, ROW_PITCH = 27.5, 24.5, 32.0     # three harmonic rows
COL_HEAD_Y = 21.5                                # K = ... over each column
LOCK_STRIP_Y = 54.9                              # lock counts under row a

TITLE_Y, TITLE_X2, SUB_Y = 8.4, 26.0, 13.4
KEY_Y, KEY_LEFT, KEY_PITCH, KEY_SWATCH = 8.4, 116.0, 10.2, 3.4

RING_LABEL_Y = 127.4                             # panel d letter baseline
RING_TOP = (130.6, 145.4)                        # two ring rows per column
RING_W, RING_PITCH = 10.6, 12.5                  # three rings across a column
RING_CAP_DY = 12.3                               # final r_1 under each ring
RING_K_DY = 3.4                                  # K restated under the band


def _session_for(rep: str) -> Path:
    for s in META["sessions"]:
        if s["representation"] == rep:
            # normalise Windows-style separators so the build runs on any OS
            return ROOT.joinpath(*str(s["session"]).replace("\\", "/").split("/"))
    raise KeyError(rep)


def _load_ts(rep: str) -> pd.DataFrame:
    return pd.read_csv(
        PAIRED / f"_tmp_session_{REP_ORDER.index(rep)}"
        / "trajectory_harmonics_timeseries.csv"
    )


def _acquisition(rep: str) -> dict:
    """Model id and call accounting for one representation, from its session."""
    d = _session_for(rep)
    cfg = json.loads((d / "resolved_config.json").read_text(encoding="utf-8"))
    summ = json.loads((d / "session_summary.json").read_text(encoding="utf-8"))
    proto = json.loads((d / "protocol.json").read_text(encoding="utf-8"))
    calls = int(summ["total_calls"])
    valid = int(summ["total_valid"])
    return {
        "model": str(cfg["model"]),
        "n_runs": int(summ["n_runs"]),
        "n_agents": int(proto["n_agents"][0]),
        "n_steps": int(proto["n_steps"]),
        "calls": calls,
        "valid_rate": valid / calls if calls else float("nan"),
    }


def _seed_ramp(rep: str) -> list[str]:
    """Six distinguishable shades of the representation hue, light to dark."""
    base = np.array(mcolors.to_rgb(REP[rep]))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{rep}_seeds", [REP_LIGHT[rep], REP[rep], tuple(base * 0.52)]
    )
    return [mcolors.to_hex(cmap(v)) for v in np.linspace(0.10, 1.0, len(SEEDS))]


def _run_id(k: float, seed: int) -> str:
    """Run directory / run_id spelling used by the acquisition (e.g. N17_K+0_s3)."""
    return f"N17_K{k:+g}_s{seed}"


def _phases_final(rep: str, run_id: str) -> np.ndarray:
    arr = np.load(_session_for(rep) / run_id / "phases.npy")
    return np.asarray(arr[min(T_FINAL, arr.shape[0] - 1)], dtype=float)


def _row(ep: pd.DataFrame, rep: str, k: float, seed: int) -> pd.Series:
    sel = ep[
        (ep["representation"] == rep)
        & (np.isclose(ep["coupling"], k))
        & (ep["seed_index"] == seed)
    ]
    if len(sel) != 1:
        raise ValueError(
            f"expected one endpoint row for {rep} K={k} s={seed}, got {len(sel)}"
        )
    return sel.iloc[0]


def _check_phase_endpoint(ph: np.ndarray, row: pd.Series, run_id: str) -> None:
    """The drawn ring and the printed r_1 must describe the same final state."""
    r1 = float(np.abs(np.exp(1j * ph).mean()))
    if abs(r1 - float(row["final_r1"])) > 1e-6:
        raise ValueError(
            f"{run_id}: r1 recomputed from phases.npy ({r1:.9f}) disagrees with "
            f"endpoints_long.csv final_r1 ({float(row['final_r1']):.9f})"
        )


# ---------------------------------------------------------------------------
# One page = one observation map
# ---------------------------------------------------------------------------
def _header(fig, rep: str, acq: dict, ramp: list[str]) -> None:
    fig_text_mm(fig, 4.0, TITLE_Y, REP_SHORT[rep], ha="left", va="baseline",
                fontsize=FS_LETTER, fontweight="bold", color=REP[rep])
    fig_text_mm(fig, TITLE_X2, TITLE_Y,
                "observation map: every coupling, every seed, "
                "underlying main Fig. 1",
                ha="left", va="baseline", fontsize=FS_BODY, color=INK)
    fig_text_mm(
        fig, 4.0, SUB_Y,
        REP_GLOSS[rep]
        + rf"  ·  $N={acq['n_agents']}$ oscillators, {acq['n_steps']} steps"
        + f"  ·  {acq['model']}  ·  {acq['n_runs']} runs, "
        + f"{acq['calls']:,} calls, {acq['valid_rate'] * 100:.1f}% valid",
        ha="left", va="baseline", fontsize=FS_SMALL, color=MUTED,
    )

    fig_text_mm(fig, KEY_LEFT - 2.4, KEY_Y, "seed", ha="right", va="baseline",
                fontsize=FS_SMALL, color=MUTED)
    for s, c in zip(SEEDS, ramp):
        x = KEY_LEFT + s * KEY_PITCH
        axk = mm_axes(fig, x, KEY_Y - 1.5, KEY_SWATCH, 1.0)
        axk.set_xticks([])
        axk.set_yticks([])
        for sp in axk.spines.values():
            sp.set_visible(False)
        axk.set_facecolor(c)
        fig_text_mm(fig, x + KEY_SWATCH + 0.9, KEY_Y, str(s), ha="left",
                    va="baseline", fontsize=FS_SMALL, color=INK)


def _grid(fig, rep: str, ts: pd.DataFrame, ep: pd.DataFrame,
          ramp: list[str]) -> None:
    for j, k in enumerate(KS):
        fig_text_mm(fig, GRID_LEFT + j * COL_PITCH + COL_W / 2.0, COL_HEAD_Y,
                    rf"$K={k:g}$", ha="center", va="baseline",
                    fontsize=FS_BODY, color=INK)

    for i, (key, ylab, title) in enumerate(HARMONICS):
        top = ROW_TOP + i * ROW_PITCH
        last_row = i == len(HARMONICS) - 1
        for j, k in enumerate(KS):
            ax = mm_axes(fig, GRID_LEFT + j * COL_PITCH, top, COL_W, ROW_H)
            sub_k = ts[np.isclose(ts["coupling"], k)]
            if key == "r1":
                ax.axhline(LOCK_R1, color=RULE, lw=LW_HAIR, ls=(0, (3, 2)),
                           zorder=1)
            for s in SEEDS:
                run = sub_k[sub_k["seed_index"] == s].sort_values("t")
                ax.plot(run["t"], run[key], color=ramp[s], lw=LW_THIN,
                        zorder=2 + s)
            ax.set_xlim(0, T_FINAL)
            ax.set_ylim(0, 1.04)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticks([0, 50, 100])
            trim_spines(ax)
            if j == 0:
                ax.set_ylabel(ylab)
                panel_label(ax, "abc"[i], title)
            else:
                ax.tick_params(labelleft=False)
            if last_row:
                ax.set_xlabel(r"$t$")
            else:
                ax.tick_params(labelbottom=False)

            if key == "r1":
                # the locking threshold, named once on the first cell only
                if j == 0:
                    ax.text(0.985, LOCK_R1, f"{LOCK_R1:g}",
                            transform=ax.get_yaxis_transform(), ha="right",
                            va="bottom", fontsize=FS_TINY, color=MUTED)
                if k == 0.0:
                    ax.text(0.97, 0.96, "negative control",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=FS_TINY, color=MUTED)
                n_lock = sum(int(_row(ep, rep, k, s)["sustained_lock"])
                             for s in SEEDS)
                fig_text_mm(
                    fig, GRID_LEFT + j * COL_PITCH + COL_W / 2.0, LOCK_STRIP_Y,
                    f"terminal lock {n_lock}/{len(SEEDS)}", ha="center",
                    va="baseline", fontsize=FS_TINY,
                    fontweight="bold" if n_lock else "normal",
                    color=PASS_GREEN if n_lock else MUTED,
                )


def _rings(fig, rep: str, ep: pd.DataFrame, ramp: list[str]) -> None:
    fig_text_mm(fig, 4.0, RING_LABEL_Y, "d", ha="left", va="baseline",
                fontsize=FS_LETTER, fontweight="bold", color=INK)
    fig_text_mm(fig, 8.6, RING_LABEL_Y,
                rf"Final phase configuration at $t={T_FINAL}$, "
                r"with final $r_1$ printed under each ring",
                ha="left", va="baseline", fontsize=FS_BODY, color="#2B2B2B")

    for j, k in enumerate(KS):
        for s in SEEDS:
            r, c = divmod(s, 3)
            left = GRID_LEFT + j * COL_PITCH + c * RING_PITCH
            top = RING_TOP[r]
            run_id = _run_id(k, s)
            row = _row(ep, rep, k, s)
            ph = _phases_final(rep, run_id)
            _check_phase_endpoint(ph, row, run_id)

            axp = mm_axes(fig, left, top, RING_W, RING_W)
            axp.add_patch(plt.Circle((0, 0), 1.0, fill=False,
                                     color="#9E9E9E", lw=LW_HAIR))
            axp.scatter(np.cos(ph), np.sin(ph), s=S_POINT, c=ramp[s],
                        linewidths=0, zorder=3)
            axp.set_xlim(-1.5, 1.5)
            axp.set_ylim(-1.5, 1.5)
            axp.set_xticks([])
            axp.set_yticks([])
            axp.patch.set_visible(False)
            for sp in axp.spines.values():
                sp.set_visible(False)
            axp.text(-1.48, 1.48, str(s), ha="left", va="top",
                     fontsize=FS_TINY, color=MUTED)
            fig_text_mm(fig, left + RING_W / 2.0, top + RING_CAP_DY,
                        f"{float(row['final_r1']):.2f}", ha="center",
                        va="baseline", fontsize=FS_TINY, color=INK)
        fig_text_mm(fig, GRID_LEFT + j * COL_PITCH + COL_W / 2.0,
                    RING_TOP[1] + RING_CAP_DY + RING_K_DY, rf"$K={k:g}$",
                    ha="center", va="baseline", fontsize=FS_SMALL, color=MUTED)


def _page(rep: str, ts: pd.DataFrame, ep: pd.DataFrame) -> Path:
    tag, stem = PAGE[rep]
    ramp = _seed_ramp(rep)

    fig = new_figure(H_MM)
    _header(fig, rep, _acquisition(rep), ramp)
    _grid(fig, rep, ts, ep, ramp)
    _rings(fig, rep, ep, ramp)
    return save_fig(fig, stem)


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    ep = pd.read_csv(PAIRED / "endpoints_long.csv")
    paths = []
    for rep in REP_ORDER:
        ts = _load_ts(rep)
        missing = [
            (k, s) for k in KS for s in SEEDS
            if not len(ts[np.isclose(ts["coupling"], k) & (ts["seed_index"] == s)])
        ]
        if missing:
            raise ValueError(f"{rep}: missing trajectory cells {missing}")
        paths.append(_page(rep, ts, ep))
    return paths[0]


if __name__ == "__main__":
    print(build())
