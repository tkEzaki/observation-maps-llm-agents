"""Supplementary Figure S19 — Complete Claude R2 matched-collective trajectories.

Three lettered pages under one figure number, one page per observation map:

    S19a  moments      figS19a_claude_trajectories_moments
    S19b  centers      figS19b_claude_trajectories_centers
    S19c  intervals    figS19c_claude_trajectories_intervals

Each page carries the complete Claude R2 trajectory set for its representation
as small multiples over (coupling, seed): three coupling rows x ten seed
columns, every cell holding r_1, r_2 and r_3 for one physical seed, plus a
seed-aggregate column and the final phase configuration of every cell.

The Claude R2 grid is **not rectangular**. K = 0 was run for the six core
seeds only; the held-out four seeds exist at K = 0.08 and K = 0.15. That is
26 cells, not 30. The four absent cells in the K = 0 row are labelled as a
design decision (``heldout_panel.note`` in ``protocol.json``) rather than
drawn as blanks a reader could mistake for missing data.

Every number on the canvas — the polar-lock threshold, the lock counts, the
mean final r_1, the cell census, the model id, the K = 0 control tolerance —
is read from ``protocol.json``, ``resolved.json``, ``r2_inference/`` or the
per-cell run artifacts. r_1 and r_2 recomputed from ``phases.npy`` are checked
against ``run_meta.json`` and against ``trajectory_rows.json``, so a drifted
artifact fails the build instead of being drawn. r_3 is not stored anywhere
and is recomputed here.

The thick aggregate r_1 line at K = 0.08 and K = 0.15 is the same 10-seed mean
that main Figure 4b draws, computed the same way from the same series.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_supplement.style import (
    FS_BODY,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    INK,
    LW_FAINT,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    LW_TRACE,
    MS_POINT,
    MUTED,
    PASS_GREEN,
    R2_CLAUDE,
    REP,
    REP_LIGHT,
    REP_ORDER,
    REP_SHORT,
    RULE,
    apply_style,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label,
    panel_label_at,
    rule_mm,
    save_fig,
    trim_spines,
)

# ---------------------------------------------------------------------------
# Frozen acquisition description
# ---------------------------------------------------------------------------
PROTO = json.loads((R2_CLAUDE / "protocol.json").read_text(encoding="utf-8"))
RESOLVED = json.loads((R2_CLAUDE / "resolved.json").read_text(encoding="utf-8"))

CORE_SEEDS = tuple(PROTO["core_panel"]["seed_indices"])
HELD_SEEDS = tuple(PROTO["heldout_panel"]["seed_indices"])
SEEDS = CORE_SEEDS + HELD_SEEDS
KS = tuple(float(k) for k in PROTO["core_panel"]["couplings"])
HELD_KS = tuple(float(k) for k in PROTO["heldout_panel"]["couplings"])
N_AGENTS = int(PROTO["n_agents"])
N_STEPS = int(PROTO["n_steps"])
# "final_r1 >= 0.9" -> 0.9; the prespecified polar-lock threshold.
LOCK_R1 = float(PROTO["phenotypes"]["rules"]["polar_locked"].split(">=")[1])

PAGE = {
    "moments_m1_m3": ("S19a", "figS19a_claude_trajectories_moments"),
    "centers_24_standard": ("S19b", "figS19b_claude_trajectories_centers"),
    "intervals_24_decimal6": ("S19c", "figS19c_claude_trajectories_intervals"),
}
# What a reader needs in order to decode the serialization from its name.
REP_GLOSS = {
    "moments_m1_m3": r"circular moments $m_1,\,m_3$ of the peer phases",
    "centers_24_standard": "24-bin phase histogram, bin centres in standard notation",
    "intervals_24_decimal6": "24-bin phase histogram, bin intervals to six decimals",
}

HARMONIC_LABELS = (r"$r_1$", r"$r_2$", r"$r_3$")
CURVE_LW = (LW_LINE, LW_THIN, LW_THIN)
MEAN_LW = (LW_TRACE, LW_LINE, LW_LINE)
_DASH = (0, (2.2, 1.4))          # held-out seed, as in main Fig. 4b
_RULE_DASH = (0, (3, 2))         # the polar-lock reference rule

# Scatter takes an *area*; derive it from the shared marker diameter so the
# phase dots match the MS_* scale used by main Figure 4.
S_POINT = MS_POINT ** 2

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm-wide page)
# ---------------------------------------------------------------------------
H_MM = 167.0

GRID_LEFT, COL_W, COL_PITCH = 14.6, 12.05, 13.35   # ten seed columns
GRID_RIGHT = GRID_LEFT + 9 * COL_PITCH + COL_W     # 146.8
AGG_LEFT, AGG_W = 152.6, 24.4                      # seed-aggregate column
ROW_TOP, ROW_H, ROW_PITCH = 30.0, 21.0, 27.5       # three coupling rows

TITLE_Y, TITLE_X2, SUB_Y = 8.6, 25.0, 13.6
KEY_L, KEY_TOP, KEY_W, KEY_H = 111.0, 4.4, 66.0, 11.6
GROUP_Y, GROUP_RULE_Y, COL_HEAD_Y = 18.8, 20.2, 24.4

RING_LABEL_Y = 114.5
RING_TOP, RING_PITCH, RING_W = 117.5, 14.2, 11.0
RING_CAP_DY = 11.8
NOTE_Y0, NOTE_DY = 161.5, 3.4


def _cell_id(k: float, seed: int) -> str:
    """Cell directory spelling used by the acquisition (e.g. N17_K+0.08_s3)."""
    return f"N17_K{k:+g}_s{seed}"


def _seeds_at(k: float) -> tuple[int, ...]:
    """The seeds actually run at this coupling — K = 0 is core-only."""
    return SEEDS if k in HELD_KS else CORE_SEEDS


def _rows() -> list[dict]:
    return json.loads(
        (R2_CLAUDE / "r2_inference" / "trajectory_rows.json").read_text(encoding="utf-8")
    )


def _k0_control() -> dict:
    return json.loads(
        (R2_CLAUDE / "r2_inference" / "k0_control.json").read_text(encoding="utf-8")
    )


def _row(rows: list[dict], rep: str, k: float, seed: int) -> dict:
    sel = [
        r for r in rows
        if r["representation"] == rep and np.isclose(r["coupling"], k)
        and r["seed_index"] == seed
    ]
    if len(sel) != 1:
        raise ValueError(
            f"expected one endpoint row for {rep} K={k:g} s={seed}, got {len(sel)}"
        )
    return sel[0]


def _harmonics(rep: str, k: float, seed: int, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(3, T) array of r_1, r_2, r_3 and the final phase vector for one cell.

    r_1 and r_2 are stored in ``run_meta.json``; r_3 is not stored anywhere in
    the acquisition, so all three are recomputed from ``phases.npy`` and the
    two stored series are used as a check on the recomputation.
    """
    d = R2_CLAUDE / _cell_id(k, seed) / rep
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    ph = np.asarray(np.load(d / "phases.npy"), dtype=float)
    rm = np.stack([np.abs(np.exp(1j * m * ph).mean(axis=1)) for m in (1, 2, 3)])

    for i, key in enumerate(("r1_series", "r2_series")):
        stored = np.asarray(meta[key], dtype=float)
        if stored.shape != rm[i].shape or np.max(np.abs(stored - rm[i])) > 1e-9:
            raise ValueError(
                f"{_cell_id(k, seed)}/{rep}: {key} disagrees with phases.npy "
                f"(max |Δ| = {np.max(np.abs(stored - rm[i])):.3e})"
            )
    end = _row(rows, rep, k, seed)
    if abs(rm[0, -1] - float(end["final_r1"])) > 1e-9:
        raise ValueError(
            f"{_cell_id(k, seed)}/{rep}: final r_1 from phases.npy "
            f"({rm[0, -1]:.9f}) disagrees with trajectory_rows.json "
            f"({float(end['final_r1']):.9f})"
        )
    return rm, ph[min(N_STEPS, ph.shape[0] - 1)]


def _harmonic_colors(rep: str) -> list[str]:
    """Three steps of the representation hue: r_1 dark, r_2 mid, r_3 light.

    One colour grammar per page — the page already *is* the representation, so
    the harmonics are separated inside its own hue rather than by importing a
    second palette.
    """
    base = np.array(mcolors.to_rgb(REP[rep]))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{rep}_harmonics", [REP_LIGHT[rep], REP[rep], tuple(base * 0.52)]
    )
    return [mcolors.to_hex(cmap(v)) for v in (1.0, 0.55, 0.16)]


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------
def _header(fig, rep: str, n_cells: int) -> None:
    fig_text_mm(fig, 4.0, TITLE_Y, REP_SHORT[rep], ha="left", va="baseline",
                fontsize=FS_BODY, fontweight="bold", color=REP[rep])
    fig_text_mm(fig, TITLE_X2, TITLE_Y,
                "observation map: every coupling, every seed, "
                "underlying main Fig. 4b",
                ha="left", va="baseline", fontsize=FS_BODY, color=INK)
    fig_text_mm(
        fig, 4.0, SUB_Y,
        REP_GLOSS[rep]
        + rf"  ·  $N={N_AGENTS}$ oscillators, {N_STEPS} steps  ·  "
        + f"{n_cells} Claude cells",
        ha="left", va="baseline", fontsize=FS_SMALL, color=MUTED,
    )


def _key(fig, colors: list[str]) -> None:
    ax = mm_panel(fig, KEY_L, KEY_TOP, KEY_W, KEY_H)
    for i, lab in enumerate(HARMONIC_LABELS):
        x = 0.015 + i * 0.125
        ax.plot([x, x + 0.062], [0.78, 0.78], color=colors[i], lw=CURVE_LW[i],
                solid_capstyle="butt")
        ax.text(x + 0.072, 0.78, lab, ha="left", va="center", fontsize=FS_TINY,
                color=INK)
    ax.text(0.435, 0.78,
            r"$r_m=|\langle e^{\mathrm{i}m\theta_j}\rangle|$", ha="left",
            va="center", fontsize=FS_TINY, color=MUTED)
    for i, (ls, lw, txt) in enumerate((
        ("solid", LW_LINE, "core seed"),
        (_DASH, LW_LINE, "held-out seed"),
        ("solid", LW_TRACE, "seed mean"),
    )):
        x = 0.015 + i * 0.345
        ax.plot([x, x + 0.062], [0.22, 0.22], color=MUTED, lw=lw, ls=ls)
        ax.text(x + 0.072, 0.22, txt, ha="left", va="center", fontsize=FS_TINY,
                color=MUTED)


def _column_heads(fig) -> None:
    fig_text_mm(fig, GRID_LEFT - 2.6, COL_HEAD_Y, "seed", ha="right",
                va="baseline", fontsize=FS_SMALL, color=MUTED)
    for j, s in enumerate(SEEDS):
        fig_text_mm(fig, GRID_LEFT + j * COL_PITCH + COL_W / 2.0, COL_HEAD_Y,
                    str(s), ha="center", va="baseline", fontsize=FS_SMALL,
                    color=INK if s in CORE_SEEDS else MUTED)
    fig_text_mm(fig, AGG_LEFT + AGG_W / 2.0, COL_HEAD_Y, "all seeds",
                ha="center", va="baseline", fontsize=FS_SMALL, color=INK)

    n_core = len(CORE_SEEDS)
    spans = (
        (GRID_LEFT, GRID_LEFT + (n_core - 1) * COL_PITCH + COL_W,
         f"core · {n_core} GPT-shared seeds", INK),
        (GRID_LEFT + n_core * COL_PITCH, GRID_RIGHT,
         f"held-out · {len(HELD_SEEDS)} new seeds", MUTED),
    )
    for x0, x1, txt, col in spans:
        rule_mm(fig, x0, x1, GROUP_RULE_Y, color=RULE, lw=LW_HAIR)
        fig_text_mm(fig, (x0 + x1) / 2.0, GROUP_Y, txt, ha="center",
                    va="baseline", fontsize=FS_TINY, color=col)


def _absent_band(fig, top: float, height: float) -> None:
    """Label the four (K = 0, held-out seed) cells that were never run.

    These cells are absent by design, not lost: the held-out panel was frozen
    with K = 0 omitted because the core panel already covers it. Leaving the
    slot blank would read as missing data, so it is named instead.
    """
    n_core = len(CORE_SEEDS)
    left = GRID_LEFT + n_core * COL_PITCH
    ax = mm_panel(fig, left, top, GRID_RIGHT - left, height)
    ax.add_patch(
        mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor="#F7F7F7", edgecolor=RULE, lw=LW_THIN,
                           ls=_DASH, zorder=0)
    )
    lo, hi = HELD_SEEDS[0], HELD_SEEDS[-1]
    ax.text(0.5, 0.72, f"seeds {lo}–{hi} were not run at $K=0$",
            ha="center", va="center", fontsize=FS_SMALL, color=INK)
    ax.text(0.5, 0.46, "absent by design, not missing data", ha="center",
            va="center", fontsize=FS_TINY, color=MUTED)
    # the protocol's own wording, trimmed to the clause that names the reason
    reason = PROTO["heldout_panel"]["note"].split(";")[-1].strip().rstrip(".")
    ax.text(0.5, 0.24, f"held-out panel: “{reason}”", ha="center",
            va="center", fontsize=FS_TINY, color=MUTED)


# ---------------------------------------------------------------------------
# Panels a-c: one coupling row each
# ---------------------------------------------------------------------------
def _rows_grid(fig, rep: str, rows: list[dict], colors: list[str],
               store: dict) -> None:
    titles = {
        0.0: r"$K=0$: social term inactive; identical under all three maps",
        0.08: r"$K=0.08$: six core seeds and four held-out seeds",
        0.15: r"$K=0.15$: six core seeds and four held-out seeds",
    }
    for i, k in enumerate(KS):
        top = ROW_TOP + i * ROW_PITCH
        last_row = i == len(KS) - 1
        present = _seeds_at(k)
        stack = []
        for j, s in enumerate(SEEDS):
            if s not in present:
                continue
            rm, ph = _harmonics(rep, k, s, rows)
            store[(k, s)] = (rm, ph)
            stack.append(rm)
            ls = "solid" if s in CORE_SEEDS else _DASH
            ax = mm_axes(fig, GRID_LEFT + j * COL_PITCH, top, COL_W, ROW_H)
            ax.axhline(LOCK_R1, color=RULE, lw=LW_HAIR, ls=_RULE_DASH, zorder=1)
            t = np.arange(rm.shape[1])
            for m in range(3):
                ax.plot(t, rm[m], color=colors[m], lw=CURVE_LW[m], ls=ls,
                        zorder=4 - m)
            ax.set_xlim(0, N_STEPS)
            ax.set_ylim(0, 1.04)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticks([0, 50, 100])
            trim_spines(ax)
            if j == 0:
                ax.set_ylabel(r"$r_m(t)$")
                panel_label(ax, "abc"[i], titles[k])
                if i == 0:
                    # named once, on the one cell whose curves stay clear of it
                    ax.text(0.985, LOCK_R1, f"{LOCK_R1:g}",
                            transform=ax.get_yaxis_transform(), ha="right",
                            va="bottom", fontsize=FS_TINY, color=MUTED)
            else:
                ax.tick_params(labelleft=False)
            if last_row and j == 0:
                ax.set_xlabel(r"$t$")
            else:
                ax.tick_params(labelbottom=False)

        if len(present) < len(SEEDS):
            _absent_band(fig, top, ROW_H)

        # seed-aggregate cell: every seed thin, the seed mean thick
        axa = mm_axes(fig, AGG_LEFT, top, AGG_W, ROW_H)
        axa.axhline(LOCK_R1, color=RULE, lw=LW_HAIR, ls=_RULE_DASH, zorder=1)
        t = np.arange(stack[0].shape[1])
        for s in present:
            axa.plot(t, store[(k, s)][0][0], color=colors[0], lw=LW_FAINT,
                     alpha=0.42, ls="solid" if s in CORE_SEEDS else _DASH,
                     zorder=2)
        mean = np.mean(np.stack(stack), axis=0)
        for m in range(3):
            axa.plot(t, mean[m], color=colors[m], lw=MEAN_LW[m], zorder=6 - m)
        axa.set_xlim(0, N_STEPS)
        axa.set_ylim(0, 1.04)
        axa.set_yticks([0, 0.5, 1.0])
        axa.set_xticks([0, 50, 100])
        axa.tick_params(labelleft=False, labelbottom=False)
        trim_spines(axa)
        # named under the cell: inside it the label would sit on the bundle
        fig_text_mm(fig, AGG_LEFT + AGG_W / 2.0, top + ROW_H + 3.4,
                    f"{len(present)}-seed mean", ha="center", va="baseline",
                    fontsize=FS_TINY, color=MUTED)


# ---------------------------------------------------------------------------
# Panel d: final phase configurations
# ---------------------------------------------------------------------------
def _rings(fig, rep: str, rows: list[dict], colors: list[str],
           store: dict) -> None:
    panel_label_at(
        fig, GRID_LEFT - 8.6, RING_LABEL_Y, "d",
        rf"Final phase configuration at $t={N_STEPS}$, with final $r_1$ "
        "printed under each ring",
    )
    for i, k in enumerate(KS):
        top = RING_TOP + i * RING_PITCH
        present = _seeds_at(k)
        fig_text_mm(fig, 7.4, top + RING_W / 2.0, rf"$K={k:g}$", ha="center",
                    va="center", rotation=90, fontsize=FS_SMALL, color=MUTED)
        for j, s in enumerate(SEEDS):
            if s not in present:
                continue
            ph = store[(k, s)][1]
            left = GRID_LEFT + j * COL_PITCH + (COL_W - RING_W) / 2.0
            axp = mm_axes(fig, left, top, RING_W, RING_W)
            axp.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#9E9E9E",
                                     lw=LW_HAIR))
            axp.scatter(np.cos(ph), np.sin(ph), s=S_POINT, c=colors[0],
                        linewidths=0, zorder=3)
            axp.set_xlim(-1.5, 1.5)
            axp.set_ylim(-1.5, 1.5)
            axp.set_xticks([])
            axp.set_yticks([])
            axp.patch.set_visible(False)
            for sp in axp.spines.values():
                sp.set_visible(False)
            end = _row(rows, rep, k, s)
            locked = bool(end["polar_locked"])
            fig_text_mm(fig, left + RING_W / 2.0, top + RING_CAP_DY,
                        f"{float(end['final_r1']):.2f}", ha="center",
                        va="baseline", fontsize=FS_TINY,
                        color=PASS_GREEN if locked else INK,
                        fontweight="bold" if locked else "normal")
        if len(present) < len(SEEDS):
            _absent_band(fig, top, RING_W)

        # per-coupling endpoint census, read from the inference rows
        sel = [r for r in rows if r["representation"] == rep
               and np.isclose(r["coupling"], k)]
        n = len(sel)
        n_lock = sum(int(r["polar_locked"]) for r in sel)
        n_sus = sum(int(r["sustained_lock"]) for r in sel)
        mean_r1 = float(np.mean([r["final_r1"] for r in sel]))
        for d, txt, col in (
            (2.2, f"polar locked  {n_lock}/{n}",
             PASS_GREEN if n_lock else MUTED),
            (5.6, f"sustained lock  {n_sus}/{n}",
             PASS_GREEN if n_sus else MUTED),
            (9.0, f"mean final $r_1$  {mean_r1:.2f}", INK),
        ):
            fig_text_mm(fig, AGG_LEFT, top + d, txt, ha="left", va="baseline",
                        fontsize=FS_TINY, color=col)


# ---------------------------------------------------------------------------
def _page(rep: str, rows: list[dict], k0: dict) -> Path:
    tag, stem = PAGE[rep]
    colors = _harmonic_colors(rep)
    store: dict = {}
    n_cells = sum(len(_seeds_at(k)) for k in KS)

    fig = new_figure(H_MM)
    _header(fig, rep, n_cells)
    _key(fig, colors)
    _column_heads(fig)
    _rows_grid(fig, rep, rows, colors, store)
    _rings(fig, rep, rows, colors, store)

    for i, line in enumerate((
        r"At $K=0$ the social term is inactive, so all three observation maps "
        "drive identical trajectories (engine-matching control, "
        rf"max $|\Delta r_m|$ = {k0['max_abs_diff_rm']:g}).",
    )):
        fig_text_mm(fig, 4.0, NOTE_Y0 + i * NOTE_DY, line, ha="left",
                    va="baseline", fontsize=FS_TINY, color=MUTED)
    return save_fig(fig, stem)


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    rows = _rows()
    k0 = _k0_control()
    if not k0["pass"]:
        raise ValueError("K = 0 engine-matching control did not pass")

    # structural audit: the grid is deliberately ragged, so check both that
    # every expected cell exists and that no unexpected one does
    for rep in REP_ORDER:
        for k in KS:
            for s in SEEDS:
                d = R2_CLAUDE / _cell_id(k, s) / rep
                if s in _seeds_at(k):
                    if not (d / "phases.npy").exists():
                        raise ValueError(f"missing acquisition cell {d}")
                elif d.exists():
                    raise ValueError(
                        f"unexpected acquisition cell {d}: the held-out panel "
                        "is not defined at this coupling"
                    )

    # the pages state that the three maps drive identical dynamics at K = 0;
    # check the phases themselves, not only the r_m tolerance in k0_control
    for k in KS:
        if k != 0.0:
            continue
        for s in _seeds_at(k):
            ref = np.load(R2_CLAUDE / _cell_id(k, s) / REP_ORDER[0] / "phases.npy")
            for other in REP_ORDER[1:]:
                arr = np.load(R2_CLAUDE / _cell_id(k, s) / other / "phases.npy")
                if not np.array_equal(ref, arr):
                    raise ValueError(
                        f"{_cell_id(k, s)}: phases differ between "
                        f"{REP_ORDER[0]} and {other} at K = 0"
                    )

    paths = [_page(rep, rows, k0) for rep in REP_ORDER]
    return paths[0]


if __name__ == "__main__":
    print(build())
