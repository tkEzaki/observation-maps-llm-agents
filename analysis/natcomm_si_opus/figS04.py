"""Supplementary Figure S4 - complete representation-invariance screen.

Purpose
-------
Document Stage B's original invariance test over all eight screened observation
contracts, and the evidence that motivated treating the serialized observation
representation as an intervention variable rather than a neutral channel.

Data provenance
---------------
* **a** - ``circlemap/representations.py::REPRESENTATION_ALIASES``. Every
  attribute in the inventory (retained feature, bin count, numeric precision,
  count/fraction convention, origin shift) is read off the frozen
  ``RepresentationSpec`` objects, and each example strip is the *actual*
  serialized payload returned by ``build_representation_prompt`` for one common
  field, parsed back into numbers. Nothing in this panel is transcribed by hand.
* **b-f** - ``analysis/representation_grid_real/representation_grid_analysis.json``
  (bootstrap intervals, joint switching-rule probability, PASS flags, and the
  frozen ``primary_rule`` thresholds) cross-checked against
  ``analysis/representation_grid_real/representation_grid_endpoints.csv``.
* **b, c** block-specific baseline points - rebuilt from the two independent
  baseline acquisitions named in the JSON's own ``baseline_runs`` field, binned
  and reduced with ``analysis.analyze_representation_screen.coefficients``. The
  module asserts that pooling the two blocks reproduces the frozen JSON
  coefficients bit-for-bit before it draws the per-block points.

Honesty notes carried in the figure itself
------------------------------------------
* Plan panel **g** (a full 8 x 8 operator-distance heat map plus a between-block
  strip) is **omitted**. Two things block it. (i) Only the baseline encoding was
  acquired twice, so there is no between-block strip for the seven alternates.
  (ii) The frozen ``analysis/stage_b_offline_p3_p4/full_operator_between_tv.csv``
  is computed on the six-profile complex-kernel stimulus manifold with three
  representations, whereas this screen is a broad/narrow von Mises sweep; a
  matrix recomputed from ``runs/response_law_representation`` therefore cannot
  reproduce that file's 3 x 3 sub-block, and substituting the three-representation
  matrix would misdescribe it as the full screen. Panels are lettered a-f with
  no gap.
* Only the baseline reaches the 36/48-offset grid at 100 responses per
  condition; the seven alternates carry 40. Both are printed in panel a.
* ``narrow_minus_broad_a2`` has no stored bootstrap interval, so panel d draws
  it as a point estimate with no error bar and says so.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from analysis.analyze_representation_screen import ACTION_INDEX, coefficients
from analysis.natcomm_si_opus.style import (
    FS_BODY,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    MS_MEAN,
    MS_POINT,
    MS_SERIES,
    MUTED,
    PASS_GREEN,
    REP,
    ROOT,
    RULE,
    STOP_RED,
    apply_style,
    despine,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label_at,
    rule_mm,
    save_fig,
    trim_spines,
)
from circlemap.representations import (
    REPRESENTATION_ALIASES,
    build_representation_prompt,
)

# --------------------------------------------------------------------------
# Layout board (mm from the top-left of a 180 mm canvas)
# --------------------------------------------------------------------------
H_MM = 176.0

X_LEFT = 4.0                 # outer left margin / shared panel-letter column
X_RIGHT = 177.0
LAB_RIGHT = 24.6             # right edge of the shared encoding-label gutter
COL_L = 26.5                 # left edge of the leftmost data panel (b, e)
COL_M = 82.0                 # left edge of the middle panel (c) and of f-block
COL_R = 133.0                # left edge of the right panel (d, f)
W_NARROW = 40.0
W_RIGHT = 44.0

LETTER_Y = (8.5, 55.0, 112.0)   # one shared letter baseline per row
ROW_TOP = (10.6, 57.5, 114.5)
ROW_H = 40.0

# Panel a table columns (mm)
A_NAME = 7.6
A_SHORT = 52.0
A_GLYPH0, A_GLYPH1 = 71.0, 93.0
A_FEAT = 96.0
A_BINS = 130.0               # right-aligned
A_PREC = 133.0
A_CONV = 154.0
A_HEAD_Y = 13.6
A_ROW0 = 18.2
A_ROW_DY = 3.9

# Panel e sub-blocks
E_CRIT_L, E_CRIT_W = COL_L, 26.0
E_PROB_L, E_PROB_W = 60.0, 56.0
E_VERDICT_X = 118.0

# Shared categorical axis: eight encodings plus a key strip above them.
Y_LIM = (7.6, -1.55)
Y_KEY = -1.05
Y_RULE = (7.5, -0.55)   # reference rules stop short of the key strip

# Screen order: the frozen baseline first, then the seven alternates in the
# order the Stage B protocol lists them.
ORDER = [
    "baseline_intervals_24_decimal6",
    "centers_24_standard",
    "centers_24_half_shift",
    "intervals_12",
    "intervals_48",
    "intervals_24_decimal4",
    "counts_100",
    "moments_m1_m3",
]
# Alias into circlemap: the screen's "baseline" is the frozen interval contract.
SPEC_ALIAS = {"baseline_intervals_24_decimal6": "intervals_24_decimal6"}

LONG = {
    "baseline_intervals_24_decimal6": "24 intervals, 6 decimals (fixed baseline)",
    "centers_24_standard": "24 bin centers",
    "centers_24_half_shift": "24 bin centers, half-bin origin shift",
    "intervals_12": "12 intervals",
    "intervals_48": "48 intervals",
    "intervals_24_decimal4": "24 intervals, 4 decimals",
    "counts_100": "24 integer counts, fixed total",
    "moments_m1_m3": "circular moments",
}
SHORT = {
    "baseline_intervals_24_decimal6": "baseline",
    "centers_24_standard": "centers 24",
    "centers_24_half_shift": "centers ½-shift",
    "intervals_12": "intervals 12",
    "intervals_48": "intervals 48",
    "intervals_24_decimal4": "intervals d4",
    "counts_100": "counts 100",
    "moments_m1_m3": "moments",
}
# Colour grammar: the representation families of the main set. The count
# encoding belongs to no main-set family, so it takes neutral ink rather than a
# fourth hue.
FAMILY_COLOR = {
    "intervals": REP["intervals_24_decimal6"],
    "centers": REP["centers_24_standard"],
    "moments": REP["moments_m1_m3"],
    "counts": MUTED,
}
FEATURE = {
    "intervals": "interval edges + mass",
    "centers": "bin center + mass",
    "counts": "interval edges + count",
    "moments": "cos/sin moments",
}

EX_KAPPA = 2.0
EX_OFFSET = float(np.pi / 4.0)
GREY_BAND = "#F2F2F2"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _load_screen() -> tuple[dict, pd.DataFrame]:
    report = json.loads(
        (ROOT / "analysis/representation_grid_real/representation_grid_analysis.json")
        .read_text(encoding="utf-8")
    )
    endpoints = pd.read_csv(
        ROOT / "analysis/representation_grid_real/representation_grid_endpoints.csv",
        float_precision="round_trip",
    ).set_index("representation")
    assert set(endpoints.index) == set(ORDER) == set(report["results"]), (
        "the screened encodings in the JSON, the CSV and this module disagree"
    )
    for name in ORDER:
        res = report["results"][name]
        row = endpoints.loc[name]
        for csv_key, node, json_key in (
            ("broad_a1", "broad", "a_sin_1"),
            ("narrow_a1", "narrow", "a_sin_1"),
            ("broad_a2", "broad", "a_sin_2"),
            ("narrow_a2", "narrow", "a_sin_2"),
            ("broad_a0", "broad", "a0"),
            ("narrow_a0", "narrow", "a0"),
        ):
            assert float(row[csv_key]) == float(res[node][json_key]), (
                f"{name}/{csv_key} differs between the CSV and the JSON"
            )
        assert np.isclose(
            float(row["narrow_minus_broad_a2"]),
            float(row["narrow_a2"]) - float(row["broad_a2"]),
        )
        assert bool(row["passes"]) == bool(res["bootstrap"]["passes_probability_gate"])
    return report, endpoints


def _baseline_blocks(report: dict) -> dict[str, dict[str, float]]:
    """Per-block baseline coefficients, verified against the frozen pooled JSON."""
    run_dirs = [ROOT / Path(p.replace("\\", "/")) for p in report["baseline_runs"]]
    grid = int(report["baseline_grid_size"])
    per_block: list[dict[str, np.ndarray]] = []
    for run_dir in run_dirs:
        grouped: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
        with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not record.get("valid"):
                    continue
                offsets = grouped[record["profile"]]
                index = int(record["offset_index"])
                counts = offsets.setdefault(index, np.zeros(3, dtype=np.int64))
                counts[ACTION_INDEX[record["action_label"]]] += 1
        per_block.append(
            {
                profile: np.stack([offsets[i] for i in range(grid)])
                for profile, offsets in grouped.items()
            }
        )
    delta = -np.pi + np.arange(grid) * 2.0 * np.pi / grid
    frozen = report["results"]["baseline_intervals_24_decimal6"]
    out: dict[str, dict[str, float]] = {}
    for profile, node in (("broad_kappa_2", "broad"), ("narrow_kappa_12", "narrow")):
        pooled = coefficients(sum(block[profile] for block in per_block), delta)
        for key in ("a0", "a_sin_1", "a_sin_2"):
            assert np.isclose(pooled[key], frozen[node][key], rtol=0, atol=1e-12), (
                "pooled baseline blocks do not reproduce the frozen coefficients"
            )
        out[node] = [coefficients(block[profile], delta) for block in per_block]
    return out


def _example_series(name: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Positions and values actually serialized for one common field."""
    alias = SPEC_ALIAS.get(name, name)
    spec = REPRESENTATION_ALIASES[alias]
    text = build_representation_prompt(alias, EX_OFFSET, EX_KAPPA)
    xs: list[float] = []
    vs: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("moment_"):
            vs.append(float(line.rsplit(":", 1)[1]))
            xs.append(len(vs) - 0.5)
        elif line.startswith("bin_"):
            vs.append(float(line.rsplit(":", 1)[1]))
            if " center " in line:
                xs.append(float(line.split(" center ")[1].split(":")[0]))
            else:
                lo, hi = line.split("[", 1)[1].split(")", 1)[0].split(",")
                xs.append(0.5 * (float(lo) + float(hi)))
    assert vs, f"no serialized payload parsed for {name}"
    position = np.asarray(xs, dtype=float)
    if spec.kind == "moments":
        position = position / float(len(vs))
    else:
        position = (position + np.pi) / (2.0 * np.pi)
    return spec.kind, position, np.asarray(vs, dtype=float)


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def _category_axis(ax, labels: bool) -> None:
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(range(len(ORDER)))
    if labels:
        ax.set_yticklabels([SHORT[n] for n in ORDER], fontsize=FS_TICK)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    despine(ax)
    ax.spines["left"].set_visible(False)


def _rule_line(ax, x: float) -> None:
    ax.plot([x, x], list(Y_RULE), color="#9A9A9A", lw=LW_HAIR,
            ls=(0, (2.4, 1.6)), zorder=1, clip_on=False)


def _rule_band(ax, x0: float, x1: float) -> None:
    ax.add_patch(mpatches.Rectangle((x0, Y_RULE[1]), x1 - x0,
                                    Y_RULE[0] - Y_RULE[1], facecolor=GREY_BAND,
                                    edgecolor="none", zorder=0))


def _block_key(ax, x: float) -> None:
    ax.plot([x], [Y_KEY], "^", color=INK, ms=MS_POINT, mfc="white",
            mew=LW_THIN, clip_on=False)
    ax.text(x + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.028, Y_KEY,
            "baseline blocks 1, 2", ha="left", va="center", fontsize=FS_TINY,
            color=MUTED)


def _dot(ax, x, y, name, filled=True, ms=MS_MEAN):
    spec = REPRESENTATION_ALIASES[SPEC_ALIAS.get(name, name)]
    color = FAMILY_COLOR[spec.kind]
    ax.plot([x], [y], "o", color=color, ms=ms, mfc=color if filled else "white",
            mew=LW_THIN, zorder=5, clip_on=False)
    return color


# --------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    report, endpoints = _load_screen()
    rule = report["primary_rule"]
    blocks = _baseline_blocks(report)
    fig = new_figure(H_MM)

    # =====================================================================
    # a - screened observation contracts
    # =====================================================================
    a_top, a_bot = 10.6, 47.4
    axA = mm_axes(fig, 0.0, a_top, 180.0, a_bot - a_top)
    axA.set_xlim(0.0, 180.0)
    axA.set_ylim(a_bot, a_top)
    axA.set_xticks([])
    axA.set_yticks([])
    for sp in axA.spines.values():
        sp.set_visible(False)
    axA.patch.set_visible(False)
    panel_label_at(fig, X_LEFT, LETTER_Y[0], "a",
                   "Screened observation contracts")

    for x, text, ha in (
        (A_NAME, "encoding", "left"),
        (A_SHORT, "label in b–f", "left"),
        (A_GLYPH0, "serialized example", "left"),
        (A_FEAT, "feature encoded", "left"),
        (A_BINS, "bins", "right"),
        (A_PREC, "precision", "left"),
        (A_CONV, "value convention", "left"),
    ):
        axA.text(x, A_HEAD_Y, text, ha=ha, va="baseline", fontsize=FS_SMALL,
                 color=MUTED)
    rule_mm(fig, X_LEFT, X_RIGHT, a_top + 4.4)
    rule_mm(fig, X_LEFT, X_RIGHT, a_bot - 0.6)

    for i, name in enumerate(ORDER):
        y = A_ROW0 + i * A_ROW_DY
        spec = REPRESENTATION_ALIASES[SPEC_ALIAS.get(name, name)]
        color = FAMILY_COLOR[spec.kind]
        axA.add_patch(mpatches.Rectangle((X_LEFT, y - 1.8), 1.9, 1.9,
                                         facecolor=color, edgecolor="none"))
        weight = "bold" if i == 0 else "normal"
        axA.text(A_NAME, y, LONG[name], ha="left", va="baseline",
                 fontsize=FS_SMALL, color=INK, fontweight=weight)
        axA.text(A_SHORT, y, SHORT[name], ha="left", va="baseline",
                 fontsize=FS_SMALL, color=color, fontweight="bold")
        axA.text(A_FEAT, y, FEATURE[spec.kind], ha="left", va="baseline",
                 fontsize=FS_SMALL, color=INK)
        axA.text(A_BINS, y, "n/a" if spec.kind == "moments" else str(spec.n_bins),
                 ha="right", va="baseline", fontsize=FS_SMALL, color=INK)
        axA.text(A_PREC, y, f"{spec.decimals} decimals", ha="left", va="baseline",
                 fontsize=FS_SMALL, color=INK)
        if spec.kind == "counts":
            convention = f"integers, Σ = {spec.count_total}"
        elif spec.kind == "moments":
            convention = f"$m$ = 1–{spec.moment_order}, cos/sin"
        else:
            convention = "fractions, Σ = 1"
        axA.text(A_CONV, y, convention, ha="left", va="baseline",
                 fontsize=FS_SMALL, color=INK)

        kind, position, values = _example_series(name)
        span = A_GLYPH1 - A_GLYPH0
        width = span / len(values)
        if kind == "moments":
            base = y - 1.5
            scale = 1.35 / float(np.max(np.abs(values)))
            fill = 0.5          # six values, not a histogram: keep them separate
            axA.plot([A_GLYPH0, A_GLYPH1], [base, base], color=RULE, lw=LW_HAIR,
                     zorder=1)
        else:
            base = y - 0.1
            scale = 2.9 / float(np.max(values))
            fill = 0.84
        for p, v in zip(position, values):
            axA.add_patch(mpatches.Rectangle(
                (A_GLYPH0 + p * span - 0.5 * fill * width, base),
                fill * width, -v * scale,
                facecolor=color, edgecolor="none", alpha=0.92, zorder=2))

    # =====================================================================
    # b - broad-field first harmonic
    # =====================================================================
    axB = mm_axes(fig, COL_L, ROW_TOP[1], W_NARROW, ROW_H)
    panel_label_at(fig, X_LEFT, LETTER_Y[1], "b",
                   "Broad field $\\kappa$ = 2: first harmonic")
    _category_axis(axB, labels=True)
    axB.set_xlim(-0.05, 1.20)
    lo_b = float(rule["minimum_broad_a1"])
    _rule_band(axB, lo_b, 1.20)
    _rule_line(axB, lo_b)
    for i, name in enumerate(ORDER):
        ci = report["results"][name]["bootstrap"]["coefficient_intervals"]["broad_a1"]
        value = float(endpoints.loc[name, "broad_a1"])
        color = _dot(axB, value, i, name)
        axB.plot([ci["ci95_low"], ci["ci95_high"]], [i, i], "-", color=color,
                 lw=LW_LINE, zorder=4)
    for k, marker in enumerate(("^", "v")):
        axB.plot([blocks["broad"][k]["a_sin_1"]], [0], marker, color=INK,
                 ms=MS_POINT, mfc="white", mew=LW_THIN, zorder=6, clip_on=False)
    _block_key(axB, 0.30)
    axB.set_xlabel("$a_1$")
    axB.set_xticks([0.0, 0.5, 1.0])
    axB.text(lo_b + 0.03, 7.45, "$a_1 \\geq 0.2$", ha="left", va="bottom",
             fontsize=FS_TINY, color=MUTED)
    trim_spines(axB, y=False)

    # =====================================================================
    # c - narrow-field first harmonic
    # =====================================================================
    axC = mm_axes(fig, COL_M, ROW_TOP[1], W_NARROW, ROW_H)
    panel_label_at(fig, COL_M - 8.6, LETTER_Y[1], "c",
                   "Narrow field $\\kappa$ = 12: first harmonic")
    _category_axis(axC, labels=False)
    axC.set_xlim(-0.55, 1.20)
    hi_n = float(rule["maximum_abs_narrow_a1"])
    _rule_band(axC, -hi_n, hi_n)
    _rule_line(axC, -hi_n)
    _rule_line(axC, hi_n)
    for i, name in enumerate(ORDER):
        ci = report["results"][name]["bootstrap"]["coefficient_intervals"]["narrow_a1"]
        value = float(endpoints.loc[name, "narrow_a1"])
        color = _dot(axC, value, i, name)
        axC.plot([ci["ci95_low"], ci["ci95_high"]], [i, i], "-", color=color,
                 lw=LW_LINE, zorder=4)
    for k, marker in enumerate(("^", "v")):
        axC.plot([blocks["narrow"][k]["a_sin_1"]], [0], marker, color=INK,
                 ms=MS_POINT, mfc="white", mew=LW_THIN, zorder=6, clip_on=False)
    _block_key(axC, -0.52)
    axC.set_xlabel("$a_1$")
    axC.set_xticks([-0.5, 0.0, 0.5, 1.0])
    axC.text(hi_n + 0.03, 7.45, "$|a_1| \\leq 0.2$", ha="left", va="bottom",
             fontsize=FS_TINY, color=MUTED)
    trim_spines(axC, y=False)

    # =====================================================================
    # d - second harmonic and its concentration increment
    # =====================================================================
    axD = mm_axes(fig, COL_R, ROW_TOP[1], W_RIGHT, ROW_H)
    panel_label_at(fig, COL_R - 8.6, LETTER_Y[1], "d",
                   "Second harmonic")
    _category_axis(axD, labels=False)
    axD.set_xlim(-0.34, 0.56)
    _rule_line(axD, 0.0)
    _rule_line(axD, float(rule["minimum_narrow_minus_broad_a2"]))
    for i, name in enumerate(ORDER):
        ints = report["results"][name]["bootstrap"]["coefficient_intervals"]
        for dy, key, csv_key, filled in ((-0.27, "broad_a2", "broad_a2", False),
                                         (0.0, "narrow_a2", "narrow_a2", True)):
            value = float(endpoints.loc[name, csv_key])
            color = _dot(axD, value, i + dy, name, filled=filled, ms=MS_SERIES)
            axD.plot([ints[key]["ci95_low"], ints[key]["ci95_high"]],
                     [i + dy, i + dy], "-", color=color, lw=LW_THIN, zorder=4)
        spec = REPRESENTATION_ALIASES[SPEC_ALIAS.get(name, name)]
        axD.plot([float(endpoints.loc[name, "narrow_minus_broad_a2"])], [i + 0.27],
                 "D", color=FAMILY_COLOR[spec.kind], ms=MS_POINT, zorder=5,
                 clip_on=False)
    for x, marker, mfc, text in ((-0.31, "o", "white", "broad $a_2$"),
                                 (-0.01, "o", INK, "narrow $a_2$"),
                                 (0.28, "D", INK, "$\\Delta a_2$")):
        axD.plot([x], [Y_KEY], marker, color=INK, ms=MS_POINT, mfc=mfc,
                 mew=LW_THIN, clip_on=False)
        axD.text(x + 0.025, Y_KEY, text, ha="left", va="center", fontsize=FS_TINY,
                 color=MUTED)
    axD.set_xlabel("$a_2$ and $\\Delta a_2$ = narrow $-$ broad")
    axD.set_xticks([-0.2, 0.0, 0.2, 0.4])
    axD.text(0.115, 7.45, "$\\Delta a_2 \\geq 0.1$", ha="left", va="bottom",
             fontsize=FS_TINY, color=MUTED)
    trim_spines(axD, y=False)

    # =====================================================================
    # e - joint gate: frozen criteria and bootstrap probability
    # =====================================================================
    axE1 = mm_axes(fig, E_CRIT_L, ROW_TOP[2], E_CRIT_W, ROW_H)
    panel_label_at(fig, X_LEFT, LETTER_Y[2], "e",
                   "Joint switching gate")
    _category_axis(axE1, labels=True)
    axE1.set_xlim(-0.5, 3.5)
    axE1.set_xticks([])
    axE1.spines["bottom"].set_visible(False)
    n_satisfied = {}
    for i, name in enumerate(ORDER):
        row = endpoints.loc[name]
        met = (
            float(row["broad_a1"]) >= float(rule["minimum_broad_a1"]),
            abs(float(row["narrow_a1"])) <= float(rule["maximum_abs_narrow_a1"]),
            float(row["broad_a2"]) > float(rule["minimum_broad_a2"]),
            float(row["narrow_minus_broad_a2"])
            >= float(rule["minimum_narrow_minus_broad_a2"]),
        )
        n_satisfied[name] = int(sum(met))
        for j, ok in enumerate(met):
            if ok:
                axE1.plot([j], [i], "o", color=PASS_GREEN, ms=MS_SERIES,
                          zorder=4, clip_on=False)
            else:
                axE1.plot([j], [i], "x", color=STOP_RED, ms=MS_SERIES,
                          mew=LW_LINE, zorder=4, clip_on=False)
    for j in range(4):
        axE1.text(j, Y_KEY, str(j + 1), ha="center", va="center",
                  fontsize=FS_TINY, color=MUTED)
    axE1.text(1.5, 7.9, "prespecified criteria", ha="center", va="top",
              fontsize=FS_TINY, color=MUTED)

    axE2 = mm_axes(fig, E_PROB_L, ROW_TOP[2], E_PROB_W, ROW_H)
    _category_axis(axE2, labels=False)
    axE2.set_xlim(0.0, 1.06)
    gate = float(rule["minimum_bootstrap_probability"])
    _rule_line(axE2, gate)
    for i, name in enumerate(ORDER):
        boot = report["results"][name]["bootstrap"]
        probability = float(boot["switching_rule_probability"])
        passes = bool(boot["passes_probability_gate"])
        color = PASS_GREEN if passes else STOP_RED
        axE2.barh(i, probability, height=0.52, color=color, alpha=0.85,
                  edgecolor="none", zorder=3)
        if probability > 0.5:
            axE2.text(probability - 0.02, i, f"{probability:.3f}", ha="right",
                      va="center", fontsize=FS_TINY, color="white", zorder=5)
        else:
            axE2.text(probability + 0.015, i, f"{probability:.3f}", ha="left",
                      va="center", fontsize=FS_TINY, color=INK, zorder=5)
        fig_text_mm(fig, E_VERDICT_X,
                    ROW_TOP[2] + ROW_H * (i - Y_LIM[1]) / (Y_LIM[0] - Y_LIM[1]) + 0.8,
                    "PASS" if passes else "FAIL", ha="left", va="baseline",
                    fontsize=FS_TINY, color=color, fontweight="bold")
    axE2.set_xlabel("bootstrap probability that all four criteria hold")
    axE2.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axE2.text(gate - 0.015, 7.45, f"gate {gate:.2f}", ha="right", va="bottom",
              fontsize=FS_TINY, color=MUTED)
    trim_spines(axE2, y=False)

    # =====================================================================
    # f - offset-averaged action bias
    # =====================================================================
    axF = mm_axes(fig, COL_R, ROW_TOP[2], W_RIGHT, ROW_H)
    panel_label_at(fig, COL_R - 8.6, LETTER_Y[2], "f",
                   "Mean action bias")
    _category_axis(axF, labels=False)
    axF.set_xlim(-0.46, 0.50)
    _rule_line(axF, 0.0)
    for i, name in enumerate(ORDER):
        broad = float(endpoints.loc[name, "broad_a0"])
        narrow = float(endpoints.loc[name, "narrow_a0"])
        spec = REPRESENTATION_ALIASES[SPEC_ALIAS.get(name, name)]
        color = FAMILY_COLOR[spec.kind]
        axF.plot([broad, narrow], [i, i], "-", color=color, lw=LW_THIN, zorder=3)
        _dot(axF, broad, i, name, filled=False, ms=MS_SERIES)
        _dot(axF, narrow, i, name, filled=True, ms=MS_SERIES)
    for x, mfc, text in ((-0.44, "white", "broad $\\kappa$ = 2"),
                         (0.05, INK, "narrow $\\kappa$ = 12")):
        axF.plot([x], [Y_KEY], "o", color=INK, ms=MS_POINT, mfc=mfc, mew=LW_THIN,
                 clip_on=False)
        axF.text(x + 0.025, Y_KEY, text, ha="left", va="center", fontsize=FS_TINY,
                 color=MUTED)
    axF.set_xlabel("$a_0$")
    axF.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    trim_spines(axF, y=False)

    # =====================================================================
    # Foot matter
    # =====================================================================
    base_res = report["results"]["baseline_intervals_24_decimal6"]
    alt_n = {int(report["results"][n]["n_per_condition"]) for n in ORDER[1:]}
    alt_grid = {int(report["results"][n]["n_offsets"]) for n in ORDER[1:]}
    assert len(alt_n) == len(alt_grid) == 1, "the alternates differ in acquisition"
    foot = ROW_TOP[2] + ROW_H + 9.6
    lines = [
        "a  Each example is the payload the fixed encoder actually emits for "
        "one common field ($\\kappa$ = 2, offset $\\pi$/4), scaled to its own "
        "maximum.",
        f"b–f  The seven alternates were acquired at {alt_n.pop()} responses per "
        f"condition on a {alt_grid.pop()}-offset grid, the baseline at "
        f"{int(base_res['n_per_condition'])} on its "
        f"{int(base_res['n_offsets'])}-offset grid.",
        "Points are the stored coefficients, bars bootstrap 95 % intervals; "
        "$\\Delta a_2$ has no stored interval.   e  criteria: 1 broad "
        "$a_1 \\geq 0.2$ · 2 narrow $|a_1| \\leq 0.2$ · 3 broad $a_2 > 0$ · "
        "4 $\\Delta a_2 \\geq 0.1$, on the point estimate.",
    ]
    for k, text in enumerate(lines):
        fig_text_mm(fig, X_LEFT, foot + k * 3.3, text, ha="left", va="baseline",
                    fontsize=FS_TINY, color=MUTED)

    return save_fig(fig, "figS04_representation_screen")
