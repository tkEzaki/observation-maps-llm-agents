"""Supplementary Figure S6 - Antipodal symmetry and dense signed-imbalance response.

The dense signed weight axis puts two von Mises modes at 0 and pi with weights
(0.5 + eps) and (0.5 - eps) at kappa = 6, and sweeps
eps in {-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10, 0.20} across three
representations, 36 offsets, 12 samples and 2 independent seed blocks
(20,736 calls; ``runs/antipodal_weight``).

Panels

* a  the physical field at exact balance, at the two smallest signed
  imbalances, and at the largest imbalance on the grid - the mode weights are
  printed because the +/-0.02 fields are not visually separable;
* b  the full trinomial operator at exact balance, with both seed blocks shown
  and activity, signed bias and response entropy stated;
* c  activity A(eps) over the full signed sweep, block by block;
* d  signed action bias a0(eps);
* e  signed first-harmonic response a1(eps) - the same quantity as main
  Fig. 2f';
* f  second-harmonic amplitude R2(eps);
* g  pairwise operator total variation at every eps, not only at eps = 0;
* h  the near-zero window, with the unmeasured band marked.

Data honesty
------------
Panels b-f and h read the frozen artifacts under
``analysis/complex_kernel_antipodal_weight/`` (``epsilon_trajectories.csv``,
``complex_endpoints.csv``, ``susceptibilities.csv``) column for column.

The frozen distance file ``distances_at_epsilon0.csv`` stores total variation
**only at eps = 0**, so panel g recomputes the per-eps pairwise TV from the two
raw trace blocks using the frozen analysis functions
(``analyze_transmutation.load_block``, ``complex_kernel.summarize_condition``,
``complex_kernel.mean_total_variation``, ``analyze_complex_kernel._probability_matrix``)
rather than re-implementing them. The module asserts at build time that every
condition cell holds exactly ``n_offsets x repetitions`` responses and that the
recomputed eps = 0 values reproduce the frozen file; a mismatch raises rather
than plotting a private number.

Panel a reads the stimulus specifications from the production catalog
(``circlemap.stimuli``), which is the same code path the acquisition used.

Claim boundary: the near-zero behaviour is described as sharp / threshold-like,
matching main Fig. 2e. No mathematical discontinuity and no derivative at
eps = 0 is claimed; the half-activation scale is reported only as a
grid-limited upper bound, because the sweep has no points inside |eps| < 0.02.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.analyze_complex_kernel import _probability_matrix
from analysis.analyze_transmutation import load_block
from analysis.complex_kernel import mean_total_variation, summarize_condition
from circlemap.stimuli import (
    STIMULUS_CATALOG,
    antipodal_weight_epsilon_id,
    build_stimulus_histogram,
    register_antipodal_weight_epsilon_grid,
)

from analysis.natcomm_si_opus.style import (
    ACTION_LABELS,
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
    RUNS,
    annotate_cells,
    apply_style,
    cbar_mm,
    despine,
    direct_label,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label,
    panel_label_at,
    save_fig,
    stacked_trinomial,
    trim_spines,
)

DATA = ROOT / "analysis" / "complex_kernel_antipodal_weight"

# --------------------------------------------------------------------------
# Layout board (mm from the top-left of a 180 mm canvas)
# --------------------------------------------------------------------------
H_MM = 158.0

COL_X = (11.0, 69.0, 127.0)       # shared left edge of every panel column
COL_W = 49.0
LET_DX = -7.0                     # letter column = COL_X + LET_DX = 4.0 mm
G_W = 36.0                        # panel g is a heatmap + its own colourbar

R1_LETTER = 6.4
R1_TOP = 11.0
R1_H = 31.0

R2_LETTER = 53.5
R2_TOP = 58.3
R2_H = 34.0

R3_LETTER = 103.5
R3_TOP = 108.3
R3_H = 34.0

A_X = (11.0, 38.0, 65.0, 92.0)    # four circular fields
A_D = 22.0                        # circle panel side
A_TOP = 15.4
A_EPS_Y = 13.9                    # baseline of the eps caption above a circle
A_W_Y = (39.9, 43.0)              # baselines of the two mode-weight lines

EPS_TICKS = [-0.1, 0.0, 0.1, 0.2]
EPS_XLIM = (-0.128, 0.252)

FIELD_FILL = "#C9C9C9"
FIELD_EDGE = "#5A5A5A"
BAND = "#EDEDED"

# short pair names, identical to main Fig. 2g
PAIRS = (
    ("moments_m1_m3", "centers_24_standard", "M–C"),
    ("moments_m1_m3", "intervals_24_decimal6", "M–I"),
    ("centers_24_standard", "intervals_24_decimal6", "C–I"),
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _tv_by_epsilon(frozen_zero: pd.DataFrame) -> pd.DataFrame:
    """Per-eps pairwise operator TV, recomputed from the two raw trace blocks.

    ``distances_at_epsilon0.csv`` freezes TV at eps = 0 only. This walks the
    same code path the frozen analyser used and refuses to return unless the
    eps = 0 slice reproduces the frozen file.
    """
    run_dirs = sorted(
        p for p in (RUNS / "antipodal_weight").glob("*/*")
        if (p / "trace.jsonl").exists()
    )
    if len(run_dirs) != 2:
        raise RuntimeError(f"expected 2 antipodal_weight blocks, found {len(run_dirs)}")

    loaded = [load_block(p) for p in run_dirs]
    protocol = loaded[0][0]["protocol"]
    offsets = np.asarray(protocol["offset_radians"], dtype=np.float64)
    profiles = list(protocol["stimulus_profiles"])
    reps = list(protocol["representations"])
    eps_by_profile = protocol["epsilon_by_profile"]
    primary = int(protocol["primary_fourier_order"])
    sensitivity = [int(o) for o in protocol.get("sensitivity_fourier_orders", [2, 4, 8, 12])]
    expected_cell = len(offsets) * int(protocol["repetitions_per_condition"])

    rows = []
    for config, counts in loaded:
        block_id = config["protocol"]["seed_block_id"]
        matrices = {}
        for rep in reps:
            for profile in profiles:
                cell = counts[rep][profile]
                total = int(cell.sum())
                if total != expected_cell:
                    raise RuntimeError(
                        f"{block_id}/{rep}/{profile}: {total} responses, "
                        f"expected {expected_cell}"
                    )
                summary = summarize_condition(
                    cell, offsets, primary_order=primary,
                    sensitivity_orders=sensitivity,
                )
                matrices[(rep, profile)] = _probability_matrix(summary)
        for profile in profiles:
            for left, right, label in PAIRS:
                rows.append(
                    {
                        "block": block_id,
                        "profile": profile,
                        "epsilon": float(eps_by_profile[profile]),
                        "pair": label,
                        "mean_tv": mean_total_variation(
                            matrices[(left, profile)], matrices[(right, profile)]
                        ),
                    }
                )
    tv = pd.DataFrame(rows)

    # --- reproduction check against the frozen eps = 0 file -----------------
    frozen = frozen_zero.copy()
    label_of = {frozenset((a, b)): lab for a, b, lab in PAIRS}
    frozen["pair"] = [
        label_of[frozenset((a, b))]
        for a, b in zip(frozen["representation_a"], frozen["representation_b"])
    ]
    merged = frozen.merge(
        tv[tv["epsilon"] == 0.0], on=["block", "profile", "pair"],
        suffixes=("_frozen", "_recomputed"),
    )
    if len(merged) != len(frozen):
        raise RuntimeError(
            f"eps = 0 join covered {len(merged)}/{len(frozen)} frozen rows"
        )
    delta = float(np.max(np.abs(merged["mean_tv_frozen"] - merged["mean_tv_recomputed"])))
    if delta > 1e-9:
        raise RuntimeError(
            "recomputed eps = 0 total variation does not reproduce "
            f"distances_at_epsilon0.csv (max |delta| = {delta:.3e})"
        )
    return tv


def _trinomial_from(activity: float, a0: float) -> tuple[float, float, float]:
    """(p-, p0, p+) from activity and signed bias - main Fig. 2's reconstruction."""
    p_plus = float(np.clip(0.5 * (activity + a0), 0.0, 1.0))
    p_minus = float(np.clip(0.5 * (activity - a0), 0.0, 1.0))
    p_zero = float(np.clip(1.0 - activity, 0.0, 1.0))
    s = p_minus + p_zero + p_plus
    return p_minus / s, p_zero / s, p_plus / s


def _series(ax, traj, ends, column, *, points: bool = True) -> dict:
    """Across-block trajectory plus both raw seed-block values, per representation."""
    last = {}
    for rep in REP_ORDER:
        t = traj[traj["representation"] == rep].sort_values("epsilon")
        ax.plot(t["epsilon"], t[column], "-o", color=REP[rep],
                ms=MS_SERIES, lw=LW_LINE, zorder=4)
        if points:
            e = ends[ends["representation"] == rep]
            ax.plot(e["epsilon"], e[column], linestyle="none", marker="o",
                    ms=MS_POINT, mfc="white", mec=REP[rep], mew=LW_THIN,
                    zorder=3)
        last[rep] = (float(t["epsilon"].iloc[-1]), float(t[column].iloc[-1]))
    return last


def _eps_axis(ax) -> None:
    ax.axvline(0, color="#BBBBBB", lw=LW_HAIR, zorder=0)
    ax.set_xlabel(r"weight imbalance $\varepsilon$")
    ax.set_xlim(*EPS_XLIM)
    ax.set_xticks(EPS_TICKS)
    despine(ax)


# --------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()

    traj = pd.read_csv(DATA / "epsilon_trajectories.csv")
    ends = pd.read_csv(DATA / "complex_endpoints.csv")
    susc = pd.read_csv(DATA / "susceptibilities.csv").set_index("representation")
    frozen_zero = pd.read_csv(DATA / "distances_at_epsilon0.csv")

    epsilons = sorted(float(v) for v in traj["epsilon"].unique())
    register_antipodal_weight_epsilon_grid(epsilons)
    tv = _tv_by_epsilon(frozen_zero)

    fig = new_figure(H_MM)

    # ---- a  the physical field at four points on the signed axis -----------
    panel_label_at(fig, COL_X[0] + LET_DX, R1_LETTER, "a",
                   "Antipodal field on the signed weight axis")
    show_eps = [0.0, -0.02, 0.02, max(epsilons)]
    hists, specs = [], []
    for e in show_eps:
        pid = antipodal_weight_epsilon_id(e)
        specs.append(STIMULUS_CATALOG[pid])
        hists.append(build_stimulus_histogram(pid, 0.0))
    fmax = max(float(np.max(h.fractions)) for h in hists)
    r0, rs = 0.34, 0.64

    for i, (e, hist, spec) in enumerate(zip(show_eps, hists, specs)):
        ax = mm_axes(fig, A_X[i], A_TOP, A_D, A_D)
        ax.set_xlim(-1.16, 1.16)
        ax.set_ylim(-1.16, 1.16)
        ax.set_aspect("equal")
        ax.axis("off")
        edges = np.asarray(hist.edges, dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        frac = np.asarray(hist.fractions, dtype=np.float64)
        # circular histogram: one wedge per serialised bin, drawn outward from
        # the phase circle so the two mode weights are read as bar heights
        for lo, hi, f in zip(edges[:-1], edges[1:], frac):
            pad = 0.11 * (hi - lo)
            arc = np.linspace(lo + pad, hi - pad, 6)
            r_out = r0 + rs * float(f) / fmax
            xs = np.concatenate([r0 * np.cos(arc), r_out * np.cos(arc[::-1])])
            ys = np.concatenate([r0 * np.sin(arc), r_out * np.sin(arc[::-1])])
            ax.fill(xs, ys, facecolor=FIELD_FILL, edgecolor=FIELD_EDGE,
                    lw=LW_HAIR, zorder=2)
        ax.add_patch(plt.Circle((0, 0), r0, fill=False, color=FIELD_EDGE,
                                lw=LW_THIN, zorder=4))
        for mode in spec.mode_offsets:
            ax.plot([r0 * np.cos(mode)], [r0 * np.sin(mode)], marker="o",
                    ms=MS_POINT, color=INK, zorder=5, linestyle="none")
        sign = "0" if e == 0 else f"{e:+.2f}".replace("+", "{+}").replace("-", "-")
        fig_text_mm(fig, A_X[i] + A_D / 2, A_EPS_Y,
                    rf"$\varepsilon = {sign}$", ha="center", va="baseline",
                    fontsize=FS_BODY, color=INK)
        w0, wpi = float(spec.weights[0]), float(spec.weights[1])
        fig_text_mm(fig, A_X[i] + A_D / 2, A_W_Y[0],
                    rf"$w(0) = {w0:.3f}$", ha="center", va="baseline",
                    fontsize=FS_TINY, color=INK)
        fig_text_mm(fig, A_X[i] + A_D / 2, A_W_Y[1],
                    rf"$w(\pi) = {wpi:.3f}$", ha="center", va="baseline",
                    fontsize=FS_TINY, color=INK)
        if i == 0:
            ax.text(1.10, 0.0, "0", ha="left", va="center",
                    fontsize=FS_TINY, color=MUTED, clip_on=False)
            ax.text(-1.10, 0.0, r"$\pi$", ha="right", va="center",
                    fontsize=FS_TINY, color=MUTED, clip_on=False)

    fig_text_mm(fig, A_X[0], A_EPS_Y - 2.9,
                "relative-phase mass, $\\kappa = 6$; the $\\pm 0.02$ fields are "
                "not visually separable · serialisations in Supplementary Fig. S2",
                ha="left", va="baseline", fontsize=FS_SMALL, color=MUTED)

    # ---- b  full trinomial operator at exact balance -----------------------
    axB = mm_axes(fig, COL_X[2], R1_TOP, COL_W, R1_H)
    panel_label(axB, "b", "Operator at exact balance", dx_mm=LET_DX,
                dy_mm=R1_TOP - R1_LETTER)
    zero = ends[ends["epsilon"] == 0.0]
    tri, stats = {}, {}
    for rep in REP_ORDER:
        g = zero[zero["representation"] == rep]
        a = float(g["activity"].mean())
        a0 = float(g["a0"].mean())
        tri[rep] = _trinomial_from(a, a0)
        stats[rep] = (a, a0, float(g["mean_entropy"].mean()),
                      [_trinomial_from(float(r.activity), float(r.a0))
                       for r in g.itertuples()])
    stacked_trinomial(axB, tri, show_legend=False, annotate=False)
    for i, rep in enumerate(REP_ORDER):
        a, a0, h, per_block = stats[rep]
        axB.text(i, 1.045, f"$A$={a:.2f}\n$a_0$={a0:+.2f}\n$H$={h:.2f}",
                 ha="center", va="bottom", fontsize=FS_TINY,
                 linespacing=1.18, color=INK)
        for p in per_block:          # both seed blocks, at the segment seams
            axB.plot([i, i], [p[0], p[0] + p[1]], linestyle="none", marker="o",
                     ms=MS_POINT, mfc="white", mec=INK, mew=LW_THIN, zorder=6)
    axB.set_ylim(0, 1.42)
    axB.set_ylabel("probability")
    despine(axB)
    trim_spines(axB, x=False)
    pi_ = tri["intervals_24_decimal6"]
    axB.text(2, pi_[0] / 2, ACTION_LABELS[0], ha="center", va="center",
             color="white", fontsize=FS_SMALL)
    axB.text(2, pi_[0] + pi_[1] / 2, ACTION_LABELS[1], ha="center", va="center",
             color="#333333", fontsize=FS_SMALL)
    axB.text(2, pi_[0] + pi_[1] + pi_[2] / 2, ACTION_LABELS[2], ha="center",
             va="center", color="white", fontsize=FS_SMALL)
    fig_text_mm(fig, COL_X[2], R1_TOP + R1_H + 7.4,
                "$H$ in nats (max $\\ln 3 = 1.10$); rings mark both seed blocks",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)

    # ---- c  activity across the signed sweep -------------------------------
    axC = mm_axes(fig, COL_X[0], R2_TOP, COL_W, R2_H)
    panel_label(axC, "c", r"Activity $A(\varepsilon)$", dx_mm=LET_DX)
    _series(axC, traj, ends, "activity")
    _eps_axis(axC)
    axC.set_ylabel(r"$A$")
    axC.set_ylim(0, 1.06)
    axC.set_yticks([0, 0.5, 1.0])
    for rep in REP_ORDER:
        y = float(traj[(traj["representation"] == rep) & (traj["epsilon"] == 0.0)]["activity"].iloc[0])
        direct_label(axC, 0.026, y, REP_SHORT[rep], REP[rep], fontsize=FS_SMALL)
    trim_spines(axC)

    # ---- d  signed action bias ---------------------------------------------
    axD = mm_axes(fig, COL_X[1], R2_TOP, COL_W, R2_H)
    panel_label(axD, "d", r"Signed bias $a_0(\varepsilon)$", dx_mm=LET_DX)
    _series(axD, traj, ends, "a0")
    _eps_axis(axD)
    axD.axhline(0, color="#BBBBBB", lw=LW_HAIR, zorder=0)
    axD.set_ylabel(r"$a_0$")
    axD.set_ylim(-0.28, 0.28)
    axD.set_yticks([-0.2, 0.0, 0.2])
    trim_spines(axD)

    # ---- e  signed first-harmonic response ---------------------------------
    axE = mm_axes(fig, COL_X[2], R2_TOP, COL_W, R2_H)
    panel_label(axE, "e", r"Signed response $a_1(\varepsilon)$", dx_mm=LET_DX)
    _series(axE, traj, ends, "a1")
    _eps_axis(axE)
    axE.axhline(0, color="#BBBBBB", lw=LW_HAIR, zorder=0)
    axE.set_ylabel(r"$a_1$")
    axE.set_ylim(-1.35, 1.35)
    axE.set_yticks([-1, 0, 1])
    for rep, y in zip(REP_ORDER, (1.30, 1.03, 0.76)):
        direct_label(axE, -0.121, y, REP_SHORT[rep], REP[rep], fontsize=FS_SMALL)
    trim_spines(axE)

    # ---- f  second-harmonic amplitude --------------------------------------
    axF = mm_axes(fig, COL_X[0], R3_TOP, COL_W, R3_H)
    panel_label(axF, "f", r"Second harmonic $R_2(\varepsilon)$", dx_mm=LET_DX)
    _series(axF, traj, ends, "R2")
    _eps_axis(axF)
    axF.set_ylabel(r"$R_2$")
    axF.set_ylim(0, 0.52)
    axF.set_yticks([0, 0.25, 0.5])
    trim_spines(axF)

    # ---- g  pairwise operator TV at every epsilon --------------------------
    axG = mm_axes(fig, COL_X[1], R3_TOP, G_W, R3_H)
    panel_label(axG, "g", r"Operator $d_{\mathrm{TV}}$", dx_mm=LET_DX)
    pair_labels = [lab for *_ , lab in PAIRS]
    mat = np.array([
        [float(tv[(tv["epsilon"] == e) & (tv["pair"] == lab)]["mean_tv"].mean())
         for lab in pair_labels]
        for e in epsilons
    ])
    vmax = float(np.ceil(mat.max() * 20.0) / 20.0)
    im = axG.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto",
                    interpolation="nearest")
    axG.set_xticks(np.arange(-0.5, len(pair_labels), 1), minor=True)
    axG.set_yticks(np.arange(-0.5, len(epsilons), 1), minor=True)
    axG.grid(which="minor", color="white", lw=LW_HAIR)
    axG.tick_params(which="minor", length=0)
    axG.tick_params(which="major", length=0)
    for sp in axG.spines.values():
        sp.set_visible(False)
    axG.set_xticks(range(len(pair_labels)))
    axG.set_xticklabels(pair_labels, fontsize=FS_TICK)
    axG.set_yticks(range(len(epsilons)))
    axG.set_yticklabels([f"{e:.2f}".replace("-", "−") for e in epsilons],
                        fontsize=FS_TICK)
    axG.set_ylabel(r"$\varepsilon$")
    annotate_cells(axG, mat, fmt="{:.2f}", cmap=plt.get_cmap("viridis"),
                   norm=im.norm, fontsize=FS_TINY)
    zero_row = epsilons.index(0.0)
    for label, e in zip(axG.get_yticklabels(), epsilons):
        if e == 0.0:
            label.set_fontweight("bold")
    axG.add_patch(mpatches.Rectangle(
        (-0.5, zero_row - 0.5), len(pair_labels), 1.0, fill=False,
        edgecolor=INK, lw=LW_LINE, zorder=6))
    cbar_mm(fig, im, COL_X[1] + G_W + 2.2, R3_TOP + 3.0, 2.2, R3_H - 3.0,
            ticks=[0, round(vmax / 2, 2), vmax])
    fig_text_mm(fig, COL_X[1] + G_W + 3.3, R3_TOP + 1.4,
                r"$d_{\mathrm{TV}}$", ha="center", va="baseline",
                fontsize=FS_SMALL)
    fig_text_mm(fig, COL_X[1], R3_TOP + R3_H + 7.0,
                "outlined row: exact balance", ha="left", va="baseline",
                fontsize=FS_TINY, color=MUTED)

    # ---- h  near-zero window -----------------------------------------------
    axH = mm_axes(fig, COL_X[2], R3_TOP, COL_W, R3_H)
    panel_label(axH, "h", "Near-zero activation", dx_mm=LET_DX)
    near = [-0.02, 0.0, 0.02]
    for e in near:                    # the three measured points on this window
        axH.axvline(e, color="#DEDEDE", lw=LW_HAIR, ls=(0, (1.4, 1.4)), zorder=0)
    for rep in REP_ORDER:
        t = traj[(traj["representation"] == rep) & (traj["epsilon"].isin(near))].sort_values("epsilon")
        axH.plot(t["epsilon"], t["activity"], "-o", color=REP[rep],
                 ms=MS_SERIES, lw=LW_LINE, zorder=4)
        e = ends[(ends["representation"] == rep) & (ends["epsilon"].isin(near))]
        axH.plot(e["epsilon"], e["activity"], linestyle="none", marker="o",
                 ms=MS_POINT, mfc="white", mec=REP[rep], mew=LW_THIN, zorder=3)
        a_at_0 = float(susc.loc[rep, "activity_at_0"])
        axH.text(-0.0022, a_at_0, f"{a_at_0:.2f}", ha="right", va="center",
                 fontsize=FS_TINY, color=REP[rep])
    axH.axvline(0, color="#BBBBBB", lw=LW_HAIR, zorder=1)
    axH.set_xlabel(r"weight imbalance $\varepsilon$")
    axH.set_ylabel(r"$A$")
    axH.set_xlim(-0.029, 0.029)
    axH.set_ylim(0, 1.06)
    axH.set_xticks([-0.02, 0.0, 0.02])
    axH.set_yticks([0, 0.5, 1.0])
    eps_half = sorted({float(v) for v in susc["eps_half_activation"]})
    if len(eps_half) != 1:
        raise RuntimeError(f"expected one shared half-activation scale, got {eps_half}")
    despine(axH)
    trim_spines(axH)
    # Set below the panel, on the footnote baseline panels f and g already
    # use: inside the panel this note was drawn straight through the dotted
    # grid line at eps = +0.02, which split "grid-limited / upper bound".
    fig_text_mm(fig, COL_X[2], R3_TOP + R3_H + 11.4,
                rf"$\varepsilon_{{1/2}} \leq {eps_half[0]:.2f}$"
                " \u00b7 grid-limited upper bound",
                ha="left", va="baseline", fontsize=FS_TINY, color=INK)


    return save_fig(fig, "figS06_antipodal_imbalance")
