"""Figure 1 - Observation representations select distinct collective phases.

Panel plan alignment pass. The accepted mm grid, colours and statistics are
unchanged; what changes is the panel inventory:

* panel a closes the physical loop - the flow now runs
  rho -> R(rho) -> pretrained LLM -> f in {-1,0,+1} -> x_i(t+1) = x_i(t) + w_i + K f_i(t)
  and the integrator feeds the next state back to rho, so the reader can see
  how the sampled action re-enters the physics;
* (round 2b) panel a's "physical rho" is no longer a stimulus-catalog field:
  it draws the *same* final phase configuration that panel b's bottom strip
  shows for the intervals arm (t = 100, K = 0.08, median seed), through the
  same ring helper, and the three one-line encoding examples are computed at
  build time from that exact array by the circlemap encoders - so the drawn
  state and the three strings are one state by construction;
* panel b is faceted at K = 0.08 and K = 0.15, the two positive couplings the
  plan asks for, on a shared y axis;
* the K = 0 negative control is a single panel f with a left half (the one
  black r1(t) curve, identical across representations) and a right half (mean
  activity by representation), so the figure ends at f;
* the exact sign-test p value is printed at its stored precision.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from analysis.figures_main.style import (
    CAPSIZE,
    FS_BODY,
    FS_SMALL,
    FS_TINY,
    GRAY_BOX,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    LW_TRACE,
    MS_MEAN,
    MS_POINT,
    MUTED,
    PHENOTYPE,
    PHENOTYPE_LABEL,
    REP,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    apply_style,
    errorbar_mean,
    fig_text_mm,
    hex_rgb,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label,
    rule_mm,
    save_fig,
    trim_spines,
)

# Panel a encodes the very state it draws through the experiment's own
# encoders - the canonical 24-bin relative-phase field and the three payload
# serialisers. Importing them (rather than re-implementing the binning or the
# serialisation) keeps the printed examples bit-identical to what the
# experiment's LLM actually received for this configuration.
from circlemap.observation import relative_phase_histogram
from circlemap.representations import build_representation_prompt_from_histogram

PAIRED = ROOT / "analysis" / "matched_rep_collective" / "paired_analysis"
META = json.loads((PAIRED / "meta.json").read_text(encoding="utf-8"))

# Scatter takes an *area*; derive it from the shared marker diameters so the
# dot sizes match the MS_* scale used by every other figure in the set.
S_POINT = MS_POINT ** 2

# The observation is defined relative to one focal agent; panel a always
# marks (and encodes for) the same deterministic agent.
FOCAL_INDEX = 0

# Non-data strokes that have no equivalent in the shared LW_* scale.
_LW_BLOCK_SEP = 2.4   # white mask between representation blocks in panel c

# Frozen statistics printed on the figure (never recomputed here).
P_EXACT_SIGN = 0.03125          # decision.json primary_evidence
GREY_TRACE = "#BBBBBB"

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm-wide page)
# ---------------------------------------------------------------------------
H_MM = 151.5

R1_TOP = 9.5                        # row 1: a schematic | b faceted timeseries
A_LEFT, A_W, A_H = 4.0, 78.0, 56.5  # schematic carries the closed physical loop

# Panel b: two facets on a shared y axis. Its letter sits one line higher than
# the axes box so the per-facet K captions have their own band.
B_TOP, B_H, B_W = 13.6, 30.0, 26.0
B_FACET_LEFT = (96.0, 137.0)
B_LETTER_DY = 5.7
B_KS = (0.08, 0.15)

# Final-phase snapshots, centred under the two facets.
RING_TOP, RING_W = 50.0, 11.5
RING_LEFT = (105.25, 123.75, 142.25)
RING_CAP_Y = 64.5

R2_TOP, R2_H = 73.0, 22.0           # row 2: c phenotype matrix, full width
C_LEFT, C_W = 13.0, 164.0
LEGEND_TOP, LEGEND_H = 101.0, 4.2   # horizontal phenotype legend strip

R3_TOP, R3_H = 113.0, 25.0          # row 3: d e f
#            axes left, width, letter x
R3 = {
    "d": (13.5, 42.0, 4.0),
    "e": (70.0, 29.0, 61.0),
    "f": (113.0, 31.0, 104.0),
}
F2_LEFT, F2_W = 156.0, 20.0         # right half of the merged panel f

PAIR_RULE_Y = 144.5                 # tie-line under the two halves of panel f
PAIR_NOTE_Y = 148.2


def _session_for(rep: str) -> Path:
    for s in META["sessions"]:
        if s["representation"] == rep:
            # normalise Windows-style separators so the build runs on any OS
            return ROOT.joinpath(*str(s["session"]).replace("\\", "/").split("/"))
    raise KeyError(rep)


def _load_ts(rep: str) -> pd.DataFrame:
    return pd.read_csv(
        PAIRED / f"_tmp_session_{REP_ORDER.index(rep)}" / "trajectory_harmonics_timeseries.csv"
    )


def _phases(rep: str, run_id: str, t: int) -> np.ndarray:
    arr = np.load(_session_for(rep) / run_id / "phases.npy")
    return np.asarray(arr[min(t, arr.shape[0] - 1)], dtype=float)


def _phase_ring(ax, ph: np.ndarray, color: str, *,
                focal_index: int | None = None) -> np.ndarray:
    """Draw one final-phase ring in the panel-b snapshot style.

    Panel a's physical-rho circle and panel b's bottom strip both call this,
    so circle weight, dot size and geometry are shared by construction. When
    ``focal_index`` is given, that agent is marked with a larger filled dot.
    Returns the exact array that was drawn, so the caller can hand the
    identical object to the encoders.
    """
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#9E9E9E", lw=LW_HAIR))
    ax.scatter(np.cos(ph), np.sin(ph), s=S_POINT, c=color, linewidths=0, zorder=3)
    if focal_index is not None:
        ax.plot([np.cos(ph[focal_index])], [np.sin(ph[focal_index])], "o",
                color=color, ms=MS_MEAN, zorder=4)
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.42, 1.42)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ph


def _rep_ticklabels(ax, order=None) -> None:
    """Colour the mom / cen / int tick labels by representation."""
    for tick, r in zip(ax.get_xticklabels(), order or REP_ORDER):
        tick.set_color(REP[r])


def _final_r1(ep, rep: str, k: float, seed: int) -> float:
    return float(
        ep[
            (ep["representation"] == rep)
            & (np.isclose(ep["coupling"], k))
            & (ep["seed_index"] == seed)
        ]["final_r1"].iloc[0]
    )


def _spec_block(ax, x0, y0, x1, y1, header, body, *, accent=None,
                facecolor="#FFFFFF", edgecolor="#AAAAAA", lw=LW_THIN):
    """One labelled specification block in panel a (header above body)."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.05",
            facecolor=facecolor, edgecolor=edgecolor, lw=lw,
        )
    )
    ax.text(x0 + 0.34, y1 - 0.34, header, ha="left", va="center",
            fontweight="bold", fontsize=FS_BODY, color=accent or INK)
    ax.text(x0 + 0.34, y1 - 0.86, body, ha="left", va="center",
            fontsize=FS_SMALL, color="#333333", linespacing=1.35)


# First data row of each serialised payload, as the encoders emit it.
_MOM_ROW_RE = re.compile(r"moment_1_cos:\s*([+-]\d+\.\d+)")
_CEN_ROW_RE = re.compile(r"center\s*([+-]\d+\.\d+):\s*(\d+\.\d+)")
_INT_ROW_RE = re.compile(r"\[([+-]\d+\.\d+),([+-]\d+\.\d+)\):\s*(\d+\.\d+)")


def _encoding_examples(ph: np.ndarray,
                       focal_index: int = FOCAL_INDEX) -> dict[str, str]:
    """One real data row per representation, encoded from the drawn state.

    The focal agent's canonical 24-bin relative-phase field and all three
    payloads come from the circlemap encoders themselves (the exact code the
    experiment ran); each string quotes the first data row of the resulting
    payload, rounded to 3 decimals to fit the schematic. Nothing is
    hard-coded. moments keeps a trailing ellipsis so it reads as the first of
    six features rather than the whole observation.
    """
    hist = relative_phase_histogram(ph, focal_index)
    prompt = {
        rep: build_representation_prompt_from_histogram(rep, hist)
        for rep in REP_ORDER
    }
    m1 = float(_MOM_ROW_RE.search(prompt["moments_m1_m3"]).group(1))
    c0, m_c = (float(g) for g in
               _CEN_ROW_RE.search(prompt["centers_24_standard"]).groups())
    lo, hi, m_i = (float(g) for g in
                   _INT_ROW_RE.search(prompt["intervals_24_decimal6"]).groups())
    return {
        "moments_m1_m3": f"m1_cos = {m1:.3f}, ...",
        "centers_24_standard": f"center {c0:.3f}: {m_c:.3f}",
        "intervals_24_decimal6": f"[{lo:.3f}, {hi:.3f}): {m_i:.3f}",
    }


def _label_offsets(finals: dict[str, float], gap: float = 0.11) -> dict[str, float]:
    """Push overlapping line-end labels apart without moving the lines."""
    out = {r: 0.0 for r in finals}
    ordered = sorted(finals.items(), key=lambda kv: kv[1])
    for (lo_r, lo_v), (hi_r, hi_v) in zip(ordered, ordered[1:]):
        deficit = gap - (hi_v + out[hi_r] - (lo_v + out[lo_r]))
        if deficit > 0:
            out[hi_r] += deficit
    return out


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    ep = pd.read_csv(PAIRED / "endpoints_long.csv")
    k_show = 0.08

    fig = new_figure(H_MM)

    # The final phase configuration shown twice: once per representation in
    # panel b's bottom strip and - for the intervals arm - as the physical
    # state in panel a. Each array is loaded exactly once through _phases,
    # so the two drawings can never diverge.
    sub_m = ep[(ep["representation"] == "moments_m1_m3") & (np.isclose(ep["coupling"], k_show))]
    seed = int(sub_m.iloc[(sub_m["final_r1"] - sub_m["final_r1"].median()).abs().argmin()]["seed_index"])
    run_id = f"N17_K+0.08_s{seed}"
    ring_phases = {rep: _phases(rep, run_id, 100) for rep in REP_ORDER}
    print(f"fig1 a physical state: run {run_id}, intervals arm, t = 100 "
          "(the configuration panel b's bottom strip draws)")

    # =====================================================================
    # a  schematic: matched intervention on the SAME physical state.
    #    The state is real - the intervals-arm final configuration that
    #    panel b's bottom strip draws (t = 100, K = 0.08, median seed) -
    #    and each branch quotes one data row encoded at build time from
    #    that exact array, so "same state, three strings" is shown, not
    #    asserted.
    # =====================================================================
    axA = mm_panel(fig, A_LEFT, R1_TOP, A_W, A_H)
    # square data units: A_W / 12 mm per x unit, matched on y so circles are round
    ux = A_W / 12.0
    axA.set_xlim(0, 12)
    axA.set_ylim(0, A_H / ux)
    panel_label(axA, "a",
                "Matched observation-map intervention on the same physical state",
                dx_mm=0.0)

    # --- the physical state, drawn exactly as panel b's bottom strip draws
    #     its intervals ring (same helper, same array), in ink. The larger
    #     filled dot marks the focal agent i whose observation the three
    #     branches encode.
    cx, cy = 0.85, 7.30                  # ring centre in axA data units
    cr = (RING_W / 2.0) / 1.42 / ux      # drawn circle radius in data units
    axAr = mm_axes(fig,
                   A_LEFT + cx * ux - RING_W / 2.0,
                   R1_TOP + A_H - cy * ux - RING_W / 2.0,
                   RING_W, RING_W)
    ph_state = ring_phases["intervals_24_decimal6"]
    drawn = _phase_ring(axAr, ph_state, INK, focal_index=FOCAL_INDEX)
    # identity guard: the encoders below must receive the identical array
    # object that was just drawn - same state by construction, not by luck
    assert drawn is ph_state
    examples = _encoding_examples(drawn)
    axAr.text(1.26 * np.cos(drawn[FOCAL_INDEX]), 1.26 * np.sin(drawn[FOCAL_INDEX]),
              r"$i$", ha="center", va="center", fontsize=FS_TINY, color=INK)
    axA.text(cx, 8.24, r"physical $\rho$", ha="center", va="center",
             fontsize=FS_SMALL)

    # --- three branches: name + one real data row encoded from the drawn state
    EX_X = 2.28
    for key, ry in (
        ("moments_m1_m3", 8.20),
        ("centers_24_standard", 7.30),
        ("intervals_24_decimal6", 6.40),
    ):
        c = REP[key]
        axA.annotate(
            "",
            xy=(2.12, ry),
            xytext=(1.64, cy),
            arrowprops=dict(arrowstyle="-|>", color=c, lw=LW_LINE, mutation_scale=6),
        )
        axA.text(EX_X, ry + 0.22, REP_SHORT[key], ha="left", va="center",
                 color=c, fontweight="bold", fontsize=FS_BODY)
        axA.text(EX_X, ry - 0.24, examples[key], ha="left", va="center",
                 fontsize=FS_TINY, family="monospace", color="#333333")
    axA.text(EX_X, 5.72, "serialized excerpts of one field in Fig. 2a; complete payloads in Supplementary Fig. S2",
             ha="left", va="center", fontsize=FS_TINY, color=MUTED)

    axA.add_patch(
        mpatches.FancyBboxPatch(
            (7.05, cy - 0.61), 1.50, 1.22, boxstyle="round,pad=0.05", **GRAY_BOX
        )
    )
    axA.text(7.80, cy, "pretrained\nLLM", ha="center", va="center",
             fontsize=FS_SMALL, linespacing=1.25)
    axA.annotate(
        "",
        xy=(9.15, cy),
        xytext=(8.60, cy),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=LW_LINE, mutation_scale=6),
    )
    axA.text(9.28, cy, r"$f\in\{-1,\,0,\,+1\}$", ha="left", va="center", fontsize=FS_SMALL)

    # --- the integrator closes the loop back onto the physical state --------
    int_y = 5.02
    axA.add_patch(
        mpatches.FancyBboxPatch(
            (4.60, int_y - 0.36), 5.80, 0.72, boxstyle="round,pad=0.05", **GRAY_BOX
        )
    )
    axA.text(7.50, int_y, r"$x_i(t{+}1)=x_i(t)+\omega_i+K\,f_i(t)$",
             ha="center", va="center", fontsize=FS_SMALL)
    axA.annotate(                      # the sampled action enters the integrator
        "",
        xy=(10.30, int_y + 0.40), xytext=(10.30, cy - 0.30),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=LW_LINE,
                        mutation_scale=6, shrinkA=0, shrinkB=0),
    )
    axA.annotate(                      # the integrated state returns to the ensemble
        "",
        xy=(cx, cy - cr - 0.16), xytext=(4.52, int_y),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=LW_LINE,
                        mutation_scale=6, shrinkA=0, shrinkB=0,
                        connectionstyle="angle,angleA=0,angleB=90,rad=2.2"),
    )
    axA.text(2.55, int_y - 0.20, r"$\rho(t{+}1)$", ha="center", va="top",
             fontsize=FS_TINY, color=MUTED)

    # --- three specification blocks -------------------------------------
    # K is swept, not held fixed, and it is not intervened on either: it defines
    # the matched experimental grid. Keeping it out of both of the first two
    # blocks is the whole point of the split.
    bx0, bx1 = 0.05, 10.75
    _spec_block(
        axA, bx0, 3.08, bx1, 4.30,
        "Fixed",
        r"$\theta_i(0),\ \omega_i,\ N{=}17,\ T$" + "  ·  "
        + "system prompt · action contract · integrator",
        facecolor="#F7F7F7", edgecolor="#AAAAAA",
    )
    _spec_block(
        axA, bx0, 1.58, bx1, 2.80,
        "Intervened",
        r"observation map $\mathcal{R}(\rho)$ only",
        accent=REP["moments_m1_m3"],
        facecolor="#EBF3FA", edgecolor=REP["moments_m1_m3"], lw=LW_LINE,
    )
    _spec_block(
        axA, bx0, 0.08, bx1, 1.30,
        "Matched experimental grid",
        r"$K\in\{-0.15,\,0,\,0.08,\,0.15\}$ · $T=100$ · 6 paired seeds",
        facecolor="#FFFFFF", edgecolor="#AAAAAA",
    )

    # =====================================================================
    # b  collective dynamics, faceted over the two positive couplings
    # =====================================================================
    ts_by_rep = {rep: _load_ts(rep) for rep in REP_ORDER}
    axB0 = None
    for j, (k, b_left) in enumerate(zip(B_KS, B_FACET_LEFT)):
        ax = mm_axes(fig, b_left, B_TOP, B_W, B_H)
        finals = {}
        for rep in REP_ORDER:
            kk = ts_by_rep[rep][np.isclose(ts_by_rep[rep]["coupling"], k)]
            for seed in range(6):
                sub = kk[kk["seed_index"] == seed].sort_values("t")
                ax.plot(sub["t"], sub["r1"], color=REP[rep], alpha=0.16, lw=LW_THIN)
            g = kk.groupby("t")["r1"].median()
            ax.plot(g.index, g.values, color=REP[rep], lw=LW_TRACE)
            finals[rep] = float(g.iloc[-1])
        # direct labels at line ends (no legend box)
        offsets = _label_offsets(finals)
        for rep in REP_ORDER:
            ax.annotate(
                REP_SHORT[rep],
                xy=(100, finals[rep] + offsets[rep]),
                xytext=(2.2, 0),
                textcoords="offset points",
                color=REP[rep],
                fontsize=FS_SMALL,
                fontweight="bold",
                va="center",
                annotation_clip=False,
            )
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel(r"$t$")
        ax.set_title(rf"$K={k}$", fontsize=FS_BODY, pad=2.6)
        if j == 0:
            ax.set_ylabel(r"$r_1(t)$")
            axB0 = ax
        else:
            ax.tick_params(labelleft=False)
    panel_label(axB0, "b", "Collective dynamics at positive coupling",
                dx_mm=-9.0, dy_mm=B_LETTER_DY)

    # final-phase snapshots, centred under the two facets (arrays preloaded
    # at the top of build; the intervals one is the state panel a draws)
    for rep, r_left in zip(REP_ORDER, RING_LEFT):
        axp = mm_axes(fig, r_left, RING_TOP, RING_W, RING_W)
        _phase_ring(axp, ring_phases[rep], REP[rep])
        axp.text(-1.40, 1.40, REP_SHORT[rep], ha="left", va="top",
                 fontsize=FS_TINY, fontweight="bold", color=REP[rep])
    fig_text_mm(
        fig, (RING_LEFT[0] + RING_LEFT[-1] + RING_W) / 2.0, RING_CAP_Y,
        rf"final phase configuration at $t=100$, $K={k_show}$ (median seed)",
        ha="center", va="baseline", fontsize=FS_TINY, color=MUTED,
    )

    # =====================================================================
    # c  all-seed phenotype matrix
    # =====================================================================
    axC = mm_axes(fig, C_LEFT, R2_TOP, C_W, R2_H)
    panel_label(axC, "c", "All-seed phenotype matrix", dx_mm=-9.0)
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
    axC.imshow(rgb, aspect="auto", interpolation="nearest")
    # thin white cell grid + strong block separators
    axC.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    axC.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    axC.grid(which="minor", color="white", lw=LW_HAIR)
    axC.tick_params(which="minor", length=0)
    for x in (3.5, 7.5):
        axC.axvline(x, color="white", lw=_LW_BLOCK_SEP)
    axC.set_yticks(range(6))
    axC.set_yticklabels([f"seed {i}" for i in range(6)])
    xt = [4 * j + kk for j in range(3) for kk in range(4)]
    axC.set_xticks(xt)
    axC.set_xticklabels([f"{k:g}" for _ in range(3) for k in ks], fontsize=FS_SMALL)
    axC.tick_params(axis="both", length=0, pad=1.4)
    for sp in axC.spines.values():
        sp.set_visible(False)

    # one clean labelled axis: "K" flush against the tick row, block names below it
    c_bottom = R2_TOP + R2_H
    fig_text_mm(fig, C_LEFT - 1.6, c_bottom + 0.7, r"$K$", ha="right", va="top",
                fontsize=FS_SMALL, color=MUTED)
    cell_w = C_W / 12.0
    for j, rep in enumerate(REP_ORDER):
        fig_text_mm(fig, C_LEFT + cell_w * (4 * j + 2), c_bottom + 3.5,
                    REP_SHORT[rep], ha="center", va="top", color=REP[rep],
                    fontweight="bold", fontsize=FS_BODY)

    # headline statistic, top-right (opposite the panel title)
    axC.annotate(
        "moments 6/6 lock · centers/intervals 0/6 · "
        rf"exact sign $p={P_EXACT_SIGN:.5f}$",
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="baseline",
        fontsize=FS_BODY,
        color=REP["moments_m1_m3"],
        fontweight="bold",
        annotation_clip=False,
    )

    # phenotype key as a compact horizontal strip under the matrix
    axL = mm_panel(fig, C_LEFT, LEGEND_TOP, C_W, LEGEND_H)
    handles = [
        mpatches.Patch(facecolor=c, edgecolor="none", label=PHENOTYPE_LABEL[k])
        for k, c in PHENOTYPE.items()
    ]
    axL.legend(
        handles=handles, loc="center", ncol=4, frameon=False, fontsize=FS_SMALL,
        handlelength=1.0, handleheight=1.0, columnspacing=1.4, handletextpad=0.45,
        borderpad=0.0, borderaxespad=0.0,
    )

    # =====================================================================
    # d  paired final r1, faceted by coupling
    # =====================================================================
    d_left, d_w, d_lx = R3["d"]
    axD = mm_axes(fig, d_left, R3_TOP, d_w, R3_H)
    panel_label(axD, "d", r"Paired final $r_1$", dx_mm=d_lx - d_left)
    facet_dx = 3.6
    for facet, k in enumerate(B_KS):
        x0 = facet * facet_dx
        for s in range(6):
            ys = [_final_r1(ep, rep, k, s) for rep in REP_ORDER]
            axD.plot([x0, x0 + 1, x0 + 2], ys, color="#D5D5D5", lw=LW_HAIR, zorder=1)
        for i, rep in enumerate(REP_ORDER):
            vals = (
                ep[(ep["representation"] == rep) & (np.isclose(ep["coupling"], k))]
                .sort_values("seed_index")["final_r1"]
                .to_numpy()
            )
            axD.scatter(
                np.full(6, x0 + i) + np.linspace(-0.07, 0.07, 6), vals,
                c=REP[rep], s=S_POINT, zorder=3, linewidths=0,
            )
            errorbar_mean(axD, x0 + i, vals, color="k", ms=MS_MEAN, lw=LW_LINE,
                          capsize=CAPSIZE, zorder=4)
        axD.text(x0 + 1, 1.125, rf"$K={k}$", ha="center", va="center",
                 fontsize=FS_BODY, color=INK)
    axD.axvline(2.8, color="#CCCCCC", lw=LW_HAIR, ls=(0, (3, 2)))
    axD.set_xticks([0, 1, 2, 3.6, 4.6, 5.6])
    axD.set_xticklabels(["mom", "cen", "int"] * 2)
    _rep_ticklabels(axD, REP_ORDER * 2)
    axD.set_xlim(-0.55, 6.15)
    axD.set_ylim(0, 1.22)
    axD.set_yticks([0, 0.5, 1.0])
    axD.set_ylabel(r"final $r_1$")
    trim_spines(axD, x=False)

    # =====================================================================
    # e  Q2
    # =====================================================================
    e_left, e_w, e_lx = R3["e"]
    axE = mm_axes(fig, e_left, R3_TOP, e_w, R3_H)
    panel_label(axE, "e", r"Final $Q_2=r_2-r_1$", dx_mm=e_lx - e_left)
    for facet, k in enumerate(B_KS):
        for i, rep in enumerate(REP_ORDER):
            vals = (
                ep[(ep["representation"] == rep) & (np.isclose(ep["coupling"], k))]
                .sort_values("seed_index")["final_Q2"]
                .to_numpy()
            )
            x0 = facet * facet_dx + i
            axE.scatter(
                np.full(6, x0) + np.linspace(-0.09, 0.09, 6), vals,
                c=REP[rep], s=S_POINT, zorder=3, linewidths=0,
            )
            errorbar_mean(axE, x0, vals, color="k", ms=MS_MEAN, lw=LW_LINE,
                          capsize=CAPSIZE, zorder=4)
    axE.axhline(0, color="#999999", lw=LW_HAIR)
    axE.axvline(2.8, color="#CCCCCC", lw=LW_HAIR, ls=(0, (3, 2)))
    axE.set_xticks([1, 4.6])
    axE.set_xticklabels([r"$K=0.08$", r"$K=0.15$"])
    axE.set_ylabel(r"$Q_2$")

    # =====================================================================
    # f  K = 0 negative control: one panel, two halves.
    #    left  - the collective is identical across representations
    #    right - the microscopic action statistics are not
    # =====================================================================
    f_left, f_w, f_lx = R3["f"]
    axF = mm_axes(fig, f_left, R3_TOP, f_w, R3_H)
    panel_label(axF, "f", r"$K=0$ negative control", dx_mm=f_lx - f_left)
    # The three representations give bit-identical r1(t) here, so drawing three
    # coloured curves would hide two of them behind whichever is drawn last.
    k0 = ts_by_rep[REP_ORDER[0]]
    k0 = k0[np.isclose(k0["coupling"], 0.0)]
    for s in range(6):
        sub = k0[k0["seed_index"] == s].sort_values("t")
        axF.plot(sub["t"], sub["r1"], color=GREY_TRACE, lw=LW_THIN, zorder=1)
    g0 = k0.groupby("t")["r1"].mean()
    axF.plot(g0.index, g0.values, color=INK, lw=LW_TRACE, zorder=3)
    axF.set_xlim(0, 100)
    axF.set_ylim(0, 1.0)
    axF.set_yticks([0, 0.5, 1.0])
    axF.set_xticks([0, 50, 100])
    axF.set_xlabel(r"$t$")
    axF.set_ylabel(r"$r_1$")
    axF.text(0.50, 0.985, "all three order-parameter trajectories\ncoincide exactly",
             transform=axF.transAxes, ha="center", va="top", fontsize=FS_TINY,
             color=INK, linespacing=1.35)

    axFb = mm_axes(fig, F2_LEFT, R3_TOP, F2_W, R3_H)
    for i, rep in enumerate(REP_ORDER):
        vals = ep[(ep["representation"] == rep) & (np.isclose(ep["coupling"], 0.0))][
            "mean_activity"
        ].to_numpy()
        axFb.scatter(
            np.full(6, i) + np.linspace(-0.09, 0.09, 6), vals,
            c=REP[rep], s=S_POINT, linewidths=0, zorder=3,
        )
        errorbar_mean(axFb, i, vals, color="k", ms=MS_MEAN, lw=LW_LINE,
                      capsize=CAPSIZE, zorder=4)
    axFb.set_xticks([0, 1, 2])
    axFb.set_xticklabels(["mom", "cen", "int"])
    _rep_ticklabels(axFb)
    axFb.set_xlim(-0.5, 2.5)
    axFb.margins(y=0.14)
    axFb.set_ylabel("mean activity")

    # the two halves are a matched pair, which is the point of the control
    rule_mm(fig, f_left, F2_LEFT + F2_W, PAIR_RULE_Y, color=RULE, lw=LW_HAIR)
    fig_text_mm(
        fig, (f_left + F2_LEFT + F2_W) / 2.0, PAIR_NOTE_Y,
        "the collective is identical; the microscopic operator is not",
        ha="center", va="baseline", fontsize=FS_TINY, color=MUTED,
    )

    return save_fig(fig, "fig1_collective_phases")
