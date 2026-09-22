"""Supplementary Figure S3 - frozen offset design and aliasing control for
response harmonics.

Purpose
-------
Show that the low-order harmonic estimates reported in the main text are not
artifacts of regular angular subsampling. The argument runs design (a) ->
what a regular design would do (b) -> what it actually did to a measured
response (c) -> what the frozen design recovers (d) -> the symmetry the frozen
design buys (e) -> the uncertainty that survives (f).

Data provenance
---------------
* **a** - ``experiments/stage_b_response_law/protocol_transmutation_v0_1_block_1.json``.
  The 36 design offsets, the ``paired_stratified_jitter`` parameters and the
  generation seed are read from the frozen protocol; the module asserts that
  the offsets in ``offset_action_curves.csv`` are the same set before drawing.
* **b** - analytic. The projection operator is the *actual* estimator
  (``analysis.analyze_transmutation.design_matrix`` + least squares), applied
  to unit-amplitude harmonics of arbitrary phase. Nothing is measured here.
  The amplitude quoted for m = 23 is read from
  ``analysis/stage_b_extended/full_spectrum.csv``.
* **c** - a genuine refit, not the pooled artifact. Counts are rebuilt from
  the ~20 000 records of the two representation-grid runs named inside
  ``analysis/representation_grid_real/representation_grid_analysis.json``,
  binned exactly as ``analysis/analyze_representation_grid.py`` bins them.
  The module asserts that the full-grid coefficients reproduce the frozen JSON
  bit-for-bit before it draws the 12-point refit beside them.
* **d** - simulation on the measured (frozen) offsets, constructed the way
  ``tests/test_representations.py::test_nonuniform_fourier_regression_recovers_low_harmonics``
  constructs its case, but with multinomial sampling at the frozen
  ``repetitions_per_condition`` and with an out-of-band m = 23 nuisance
  harmonic at the measured amplitude.
* **e** - ``analysis/complex_kernel_stimulus_manifold/offset_action_curves.csv``.
* **f** - ``analysis/complex_kernel_transmutation/complex_ellipses.csv``.

Honesty notes carried in the figure itself
------------------------------------------
* The representation-grid acquisition used in **c** is a *regular* 36-point
  grid, not the nonuniform design of **a**. The panel says "full 36-point
  grid", never "frozen design", and the 12-point series is that acquisition's
  own first block, so both fits are of the same measured response.
* **b** therefore shows three designs, not two. A regular 36-point grid is
  clean up to m = 29 and it would be misleading to contrast the frozen design
  only against the 12-point one: the frozen design's virtue is that *no*
  harmonic aliases with unit gain, not that it beats every regular grid.
* **f** uses block 1 only. Bootstrap ellipses in this study are conditional on
  an acquisition block and must not be pooled across blocks.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.analyze_representation_screen import ACTION_INDEX, coefficients
from analysis.analyze_transmutation import design_matrix, fit_fourier
from analysis.natcomm_si_opus.style import (
    ACCENT_RED,
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
    annotate_cells,
    apply_style,
    cbar_mm,
    despine,
    direct_label,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label_at,
    save_fig,
    trim_spines,
)

# --------------------------------------------------------------------------
# Layout board (mm from the top-left of a 180 mm canvas)
# --------------------------------------------------------------------------
H_MM = 162.0

COL_X = (12.6, 71.0, 129.4)     # shared left edge of every panel column
COL_W = 46.0
ROW_TOP = (12.4, 93.0)          # shared top edge of the first axes in a row
LETTER_Y = (7.0, 87.0)          # shared letter baseline per row
KEY_Y = (0.0, 91.4)             # colour-key baseline above the row-2 axes
LETTER_DX = 8.6                 # style.LETTER_DX_MM, applied positively here

FIT_ORDER = 6                   # protocol primary_fourier_order
M_MAX = 24                      # rows of the aliasing map = measured spectrum
CMAP = plt.get_cmap("viridis")

PROTOCOL = (
    ROOT / "experiments/stage_b_response_law/"
    "protocol_transmutation_v0_1_block_1.json"
)
GRID_JSON = ROOT / "analysis/representation_grid_real/representation_grid_analysis.json"
CURVES = ROOT / "analysis/complex_kernel_stimulus_manifold/offset_action_curves.csv"
ELLIPSES = ROOT / "analysis/complex_kernel_transmutation/complex_ellipses.csv"
SPECTRUM = ROOT / "analysis/stage_b_extended/full_spectrum.csv"

R_GATE = 0.15                   # docs/STAGE_B_COMPLEX_KERNEL.md weak-R phase rule
SIGN_REP = "centers_24_standard"
SIGN_PROFILE = "narrow_kappa_12"


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------
def _design() -> dict:
    """The frozen offset design, read from the protocol and self-checked."""
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    offsets = np.sort(np.asarray(protocol["offset_radians"], dtype=float))
    spec = protocol["offset_design"]
    strata = int(spec["positive_strata"])
    low, high = (float(v) for v in spec["jitter_interval"])

    # every offset is one half of an exact mirror pair; there is no anchor
    mirrored = bool(np.allclose(offsets, np.sort(-offsets)))
    positive = offsets[offsets > 0.0]
    width = math.pi / strata
    stratum = np.floor(positive / width).astype(int)
    fraction = positive / width - stratum

    curves = pd.read_csv(CURVES)
    measured = np.sort(np.unique(curves["offset_radians"].to_numpy()))
    if not np.allclose(measured, offsets):
        raise ValueError(
            "offset_action_curves.csv does not use the frozen protocol offsets"
        )
    return {
        "offsets": offsets,
        "n": int(offsets.size),
        "strata": strata,
        "width": width,
        "positive": positive,
        "stratum": stratum,
        "fraction": fraction,
        "jitter_low": low,
        "jitter_high": high,
        "mirrored": mirrored,
        "gaps_deg": np.degrees(np.diff(offsets)),
        "seed": int(spec["generation_seed"]),
        "kind": str(spec["kind"]),
        "version": str(protocol["protocol_version"]),
        "repetitions": int(protocol["repetitions_per_condition"]),
    }


def _regular(n: int) -> np.ndarray:
    """The regular n-point design the grid acquisition and its subset use."""
    return -math.pi + np.arange(n) * 2.0 * math.pi / n


# --------------------------------------------------------------------------
# b - aliasing operator
# --------------------------------------------------------------------------
def _alias_map(offsets: np.ndarray, order: int = FIT_ORDER,
               m_max: int = M_MAX) -> np.ndarray:
    """Worst-case gain from a true harmonic m onto each apparent harmonic.

    A true unit-amplitude harmonic ``cos(m d + phi)`` is fitted with the same
    least-squares operator the study uses. For each apparent harmonic m' the
    recovered ``(a_m', b_m')`` is a linear image of ``(cos phi, sin phi)``, so
    the largest singular value of that 2x2 map is the projection an adversarial
    phase could produce. On a regular N-point grid it is exactly 1 when
    m = +/- m' (mod N) and 0 otherwise.
    """
    inverse = np.linalg.pinv(design_matrix(offsets, order))
    gains = np.zeros((m_max, order))
    for row, m in enumerate(range(1, m_max + 1)):
        signal = np.column_stack([np.cos(m * offsets), -np.sin(m * offsets)])
        beta = inverse @ signal
        for col, apparent in enumerate(range(1, order + 1)):
            pair = beta[[2 * apparent - 1, 2 * apparent], :]
            gains[row, col] = float(np.linalg.svd(pair, compute_uv=False)[0])
    return gains


def _spectrum_amplitude(profile: str, harmonic: int) -> float:
    table = pd.read_csv(SPECTRUM)
    row = table[(table["profile"] == profile) & (table["harmonic"] == harmonic)]
    if row.empty:
        raise ValueError(f"no measured harmonic {harmonic} for {profile}")
    return float(row["amplitude"].iloc[0])


# --------------------------------------------------------------------------
# c - refit of a measured response on a regular 12-point subset
# --------------------------------------------------------------------------
def _grid_index(offset: float, grid_size: int) -> int:
    """Identical binning to analysis/analyze_representation_grid.py."""
    coordinate = (offset + np.pi) * grid_size / (2.0 * np.pi)
    return int(round(coordinate)) % grid_size


def _grid_counts(grid_size: int = 36) -> tuple[dict, dict]:
    """Rebuild per-offset trinomial counts from the frozen acquisition runs."""
    report = json.loads(GRID_JSON.read_text(encoding="utf-8"))
    runs = [
        ROOT / Path(str(path).replace("\\", "/"))
        for path in report["representation_runs"]
    ]
    grouped: dict = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: np.zeros(3, dtype=np.int64))))
    for run in runs:
        with (run / "trace.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not record.get("valid"):
                    continue
                index = _grid_index(float(record["offset_radians"]), grid_size)
                bucket = grouped[record["representation"]][record["profile"]]
                bucket[index][ACTION_INDEX[record["action_label"]]] += 1
    counts = {
        representation: {
            profile: np.stack([offsets[i] for i in range(grid_size)])
            for profile, offsets in profiles.items()
        }
        for representation, profiles in grouped.items()
    }
    return counts, report


def _sign_change(grid_size: int = 36, subset: int = 12) -> dict:
    """Full-grid vs regular-subset fit of one measured response."""
    counts, report = _grid_counts(grid_size)
    block = counts[SIGN_REP][SIGN_PROFILE]
    full_offsets = _regular(grid_size)
    full = coefficients(block, full_offsets)

    frozen = report["results"][SIGN_REP][
        "narrow" if SIGN_PROFILE.startswith("narrow") else "broad"]
    for key, value in frozen.items():
        if abs(float(value) - full[key]) > 1e-12:
            raise ValueError(
                f"refit disagrees with the frozen artifact on {key}: "
                f"{full[key]!r} vs {value!r}"
            )

    step = grid_size // subset
    phases = []
    for start in range(step):
        take = np.arange(start, grid_size, step)
        phases.append(coefficients(block[take], full_offsets[take]))
    totals = np.sum(block, axis=1)
    return {
        "full": full,
        "phases": phases,
        "take": np.arange(0, grid_size, step),
        "offsets": full_offsets,
        "g": (block[:, 2] - block[:, 0]) / totals,
        "n_per_offset": int(totals[0]),
        "n_records": int(totals.sum()),
    }


def _curve(fit: dict, grid: np.ndarray) -> np.ndarray:
    return (fit["a0"]
            + fit["a_sin_1"] * np.sin(grid) + fit["b_cos_1"] * np.cos(grid)
            + fit["a_sin_2"] * np.sin(2 * grid) + fit["b_cos_2"] * np.cos(2 * grid))


# --------------------------------------------------------------------------
# d - recovery under regression on the measured offsets
# --------------------------------------------------------------------------
def _recovery(design: dict, draws: int = 90, seed: int = 20260725) -> dict:
    """Known harmonic mixtures -> multinomial responses -> order-6 refit.

    Built like the frozen unit test, with two additions that make it a control
    rather than an algebra check: the responses are sampled multinomially at
    the protocol's own repetition count, and each mixture carries an
    out-of-band m = 23 nuisance harmonic at the amplitude measured in the
    narrow field.
    """
    rng = np.random.default_rng(seed)
    offsets = design["offsets"]
    reps = design["repetitions"]
    nuisance = _spectrum_amplitude("narrow_kappa_12", 23)
    scale = {1: 1.00, 2: 0.55, 3: 0.22, 4: 0.18, 5: 0.15, 6: 0.12}
    activity = 0.97

    truth: dict[int, list[float]] = defaultdict(list)
    found: dict[int, list[float]] = defaultdict(list)
    for _ in range(draws):
        a0 = float(rng.uniform(-0.12, 0.12))
        sin = {m: float(rng.uniform(-s, s)) for m, s in scale.items()}
        cos = {m: float(rng.uniform(-s, s)) for m, s in scale.items()}
        g = np.full(offsets.size, a0)
        for m in scale:
            g += sin[m] * np.sin(m * offsets) + cos[m] * np.cos(m * offsets)
        phase = float(rng.uniform(-math.pi, math.pi))
        g += nuisance * np.sin(23 * offsets + phase)
        peak = float(np.max(np.abs(g)))
        if peak > activity:                       # keep probabilities legal
            shrink = activity / peak
            g *= shrink
            a0 *= shrink
            sin = {m: v * shrink for m, v in sin.items()}
            cos = {m: v * shrink for m, v in cos.items()}
        plus = np.clip((activity + g) / 2.0, 0.0, 1.0)
        minus = np.clip((activity - g) / 2.0, 0.0, 1.0)
        stay = np.clip(1.0 - plus - minus, 0.0, 1.0)
        probabilities = np.column_stack([minus, stay, plus])
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        counts = np.stack([rng.multinomial(reps, p) for p in probabilities])
        fit = fit_fourier(counts, offsets, FIT_ORDER)
        for m in scale:
            truth[m].extend([sin[m], cos[m]])
            found[m].extend([fit[f"a_sin_{m}"], fit[f"b_cos_{m}"]])
    groups = {
        "first": ([1], "$a_1,\\,b_1$"),
        "second": ([2], "$a_2,\\,b_2$"),
        "higher": ([5, 6], "$a_5,b_5,a_6,b_6$"),
    }
    out = {}
    for key, (orders, label) in groups.items():
        t = np.concatenate([truth[m] for m in orders])
        f = np.concatenate([found[m] for m in orders])
        out[key] = {
            "true": t,
            "found": f,
            "label": label,
            "rmse": float(np.sqrt(np.mean((f - t) ** 2))),
        }
    out["reps"] = reps
    out["draws"] = draws
    out["nuisance"] = nuisance
    return out


# --------------------------------------------------------------------------
# e - mirror-pair balance
# --------------------------------------------------------------------------
def _mirror_parts(design: dict, profile: str = "unimodal_k9",
                  block: str = "block_1") -> dict:
    curves = pd.read_csv(CURVES)
    sub = curves[(curves["block"] == block) & (curves["profile"] == profile)]
    out = {}
    positive = design["positive"]
    for representation in REP_ORDER:
        rows = sub[sub["representation"] == representation]
        lookup = dict(zip(np.round(rows["offset_radians"].to_numpy(), 12),
                          rows["g"].to_numpy()))
        odd, even = [], []
        for delta in positive:
            plus = lookup[round(float(delta), 12)]
            minus = lookup[round(float(-delta), 12)]
            odd.append(0.5 * (plus - minus))
            even.append(0.5 * (plus + minus))
        out[representation] = {
            "delta": np.degrees(positive),
            "odd": np.asarray(odd),
            "even": np.asarray(even),
            "mean_abs_even": float(np.mean(np.abs(even))),
            "mean_abs_odd": float(np.mean(np.abs(odd))),
        }
    return out


# --------------------------------------------------------------------------
# f - bootstrap ellipses
# --------------------------------------------------------------------------
def _ellipses(block: str = "block_1") -> pd.DataFrame:
    table = pd.read_csv(ELLIPSES)
    table = table[(table["block"] == block)
                  & (table["profile"].isin(["kappa_2", "kappa_12"]))].copy()
    table["radius"] = np.hypot(table["center_a"], table["center_b"])
    table["reportable"] = (table["radius"] >= R_GATE) & (table["contains_origin"] == 0)
    return table


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def _draw_a(fig, design: dict) -> None:
    left, top = COL_X[0], ROW_TOP[0]
    panel_label_at(fig, left - LETTER_DX, LETTER_Y[0], "a",
                   "Fixed nonuniform offset design")

    offsets = design["offsets"]
    regular12 = _regular(12)

    ax = mm_axes(fig, left + 12.0, top, 22.0, 22.0)
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.30, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")
    # stratum boundaries: pi/18 strata, mirrored into the lower half
    for k in range(design["strata"] * 2):
        angle = k * design["width"]
        ax.plot([0.90 * math.cos(angle), math.cos(angle)],
                [0.90 * math.sin(angle), math.sin(angle)],
                color=RULE, lw=LW_HAIR, zorder=1)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=MUTED, lw=LW_THIN,
                            zorder=2))
    # mirror pairs joined: delta and -delta share an x, so each chord is vertical
    for delta in design["positive"]:
        ax.plot([math.cos(delta)] * 2, [math.sin(delta), -math.sin(delta)],
                color="#DCDCDC", lw=LW_HAIR, zorder=1)
    ax.plot(np.cos(offsets), np.sin(offsets), "o", color=INK, ms=MS_POINT,
            mew=0.0, zorder=4, clip_on=False)
    for delta in regular12:                      # the comparison design of b/c
        ax.plot([1.10 * math.cos(delta), 1.24 * math.cos(delta)],
                [1.10 * math.sin(delta), 1.24 * math.sin(delta)],
                color=ACCENT_RED, lw=LW_THIN, zorder=3, clip_on=False)

    fig_text_mm(fig, left, top + 25.0,
                "36 fixed offsets, mirror pairs joined",
                ha="left", va="baseline", fontsize=FS_TINY, color=INK)
    fig_text_mm(fig, left, top + 28.0,
                "regular 12-point design (outer ticks)",
                ha="left", va="baseline", fontsize=FS_TINY, color=ACCENT_RED)

    # jitter fraction against the linear offset axis
    axj = mm_axes(fig, left, top + 32.0, COL_W, 16.0)
    axj.axhspan(design["jitter_low"], design["jitter_high"], color="#EDEDED",
                zorder=0)
    for edge in (design["jitter_low"], design["jitter_high"]):
        axj.axhline(edge, color=MUTED, lw=LW_HAIR, ls=(0, (2.4, 1.6)), zorder=1)
    fraction = design["fraction"]
    positive_deg = np.degrees(design["positive"])
    axj.plot(positive_deg, fraction, "o", color=INK, ms=MS_POINT, mew=0.0,
             zorder=3)
    axj.plot(-positive_deg, fraction, "o", mfc="white", mec=INK,
             mew=LW_HAIR, ms=MS_POINT, zorder=3)
    for delta in np.degrees(regular12):
        axj.plot([delta, delta], [-0.17, -0.06], color=ACCENT_RED, lw=LW_THIN,
                 clip_on=False, zorder=3)
    axj.set_xlim(-185, 185)
    axj.set_ylim(-0.05, 1.05)
    axj.set_xticks([-180, -90, 0, 90, 180])
    axj.set_yticks([0, 0.5, 1])
    axj.set_yticklabels(["0", "0.5", "1"])
    axj.set_xlabel("offset $\\delta$ (degrees)", fontsize=FS_BODY, labelpad=2.2)
    axj.set_ylabel("jitter within\nstratum", fontsize=FS_BODY, labelpad=1.4,
                   linespacing=1.1)
    despine(axj)
    trim_spines(axj)

    gaps = design["gaps_deg"]
    fig_text_mm(
        fig, left, top + 57.0,
        f"{design['n']} offsets = {design['strata']} exact mirror pairs; "
        f"no anchor at $\\delta=0$.\n"
        f"Stratified jitter in $\\pi/{design['strata']}$ strata, "
        f"seed {design['seed']}.\n"
        f"Gaps {gaps.min():.2f}\u2013{gaps.max():.2f}\u00b0; "
        f"jitter within [{design['jitter_low']:.2f}, {design['jitter_high']:.2f}].",
        ha="left", va="top", fontsize=FS_TINY, color=MUTED, linespacing=1.45)


def _draw_b(fig, maps: list, amplitude: float, frozen_leak: float) -> None:
    left, top = COL_X[1], ROW_TOP[0]
    panel_label_at(fig, left - LETTER_DX, LETTER_Y[0], "b",
                   "Aliasing onto low-order modes")

    block_w, gap = 13.2, 3.2
    block_top, block_h = top + 5.0, 40.0
    image = None
    for index, (title, gains) in enumerate(maps):
        ax = mm_axes(fig, left + index * (block_w + gap), block_top,
                     block_w, block_h)
        image = ax.imshow(gains, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto",
                          interpolation="nearest",
                          extent=(0.5, FIT_ORDER + 0.5, M_MAX + 0.5, 0.5))
        ax.set_xticks(range(1, FIT_ORDER + 1))
        ax.set_xticklabels(range(1, FIT_ORDER + 1), fontsize=FS_TINY)
        ax.set_xticks(np.arange(1.5, FIT_ORDER), minor=True)
        ax.set_yticks(np.arange(1.5, M_MAX), minor=True)
        ax.grid(which="minor", color="white", lw=LW_HAIR)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0, pad=1.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if index == 0:
            ax.set_yticks([1, 5, 9, 13, 17, 21, 23])
            ax.set_yticklabels([1, 5, 9, 13, 17, 21, 23], fontsize=FS_TINY)
            ax.set_ylabel("true harmonic $m$", fontsize=FS_BODY, labelpad=1.4)
            ax.add_patch(mpatches.Rectangle(
                (0.5, 22.5), 1.0, 1.0, fill=False, edgecolor=ACCENT_RED,
                lw=LW_LINE, zorder=6, clip_on=False))
        else:
            ax.set_yticks([])
        fig_text_mm(fig, left + index * (block_w + gap) + block_w / 2.0,
                    block_top - 1.4, title, ha="center", va="baseline",
                    fontsize=FS_TINY, color=INK)
    fig_text_mm(fig, left + 1.5 * block_w + gap, block_top + block_h + 6.0,
                "apparent harmonic $m^{\\prime}$", ha="center", va="baseline",
                fontsize=FS_BODY)

    bar_left = left + 15.0
    cbar_mm(fig, image, bar_left, block_top + block_h + 8.6, 26.0, 1.8,
            ticks=[0, 0.5, 1.0], orientation="horizontal")
    fig_text_mm(fig, bar_left - 1.6, block_top + block_h + 9.5,
                "|projection|", ha="right", va="center", fontsize=FS_TINY,
                color=INK)
    fig_text_mm(
        fig, left, block_top + block_h + 15.4,
        f"Regular 12-point sampling maps $m=23$ onto $m^{{\\prime}}=1$ with unit "
        f"gain (red cell).\n"
        f"That harmonic carries amplitude {amplitude:.3f} in the narrow field; "
        f"the fixed design\nleaks {frozen_leak:.3f} of it. A regular 36-point "
        f"grid is clean to $m=29$, not beyond.",
        ha="left", va="top", fontsize=FS_TINY, color=MUTED, linespacing=1.45)


def _draw_c(fig, sign: dict) -> None:
    left, top = COL_X[2], ROW_TOP[0] + 3.6
    panel_label_at(fig, left - LETTER_DX, LETTER_Y[0], "c",
                   "Measured sign change")
    fig_text_mm(fig, left, top - 1.4, "all 36", ha="left", va="baseline",
                fontsize=FS_TINY, color=MUTED)
    fig_text_mm(fig, left + 11.0, top - 1.4, "regular 12-point subset",
                ha="left", va="baseline", fontsize=FS_TINY, color=ACCENT_RED)

    ax = mm_axes(fig, left, top, COL_W, 34.0)
    grid = np.linspace(-math.pi, math.pi, 721)
    offsets, g = sign["offsets"], sign["g"]
    take = sign["take"]
    ax.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=0)
    ax.plot(np.degrees(offsets), g, "o", color=MUTED, ms=MS_POINT, mew=0.0,
            zorder=3)
    ax.plot(np.degrees(offsets[take]), g[take], "o", mfc="none",
            mec=ACCENT_RED, mew=LW_LINE, ms=MS_SERIES, zorder=4)
    ax.plot(np.degrees(grid), _curve(sign["full"], grid), "-", color=INK,
            lw=LW_LINE, zorder=5)
    ax.plot(np.degrees(grid), _curve(sign["phases"][0], grid), "--",
            color=ACCENT_RED, lw=LW_LINE, zorder=5)
    ax.set_xlim(-185, 185)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["$-1$", "0", "1"])
    ax.set_xlabel("offset $\\delta$ (degrees)", fontsize=FS_BODY, labelpad=2.2)
    ax.set_ylabel("mean response $g$", fontsize=FS_BODY, labelpad=1.4)
    despine(ax)
    trim_spines(ax)

    def signed(value: float) -> str:
        return f"{value:+.3f}".replace("-", "\u2212")

    full = sign["full"]["a_sin_1"]
    phases = [p["a_sin_1"] for p in sign["phases"]]
    others = ", ".join(signed(v) for v in phases[1:])
    fig_text_mm(
        fig, left, top + 43.0,
        "centers, narrow field $\\kappa=12$; "
        f"{sign['n_per_offset']} per offset.\n"
        f"36-point grid, black fit:   $a_1$ = {signed(full)}\n"
        f"12-point subset, red fit:   $a_1$ = {signed(phases[0])}\n"
        f"other subset phases:   {others}",
        ha="left", va="top", fontsize=FS_TINY, color=MUTED, linespacing=1.45)


def _draw_d(fig, recovery: dict, design: dict) -> None:
    left, top = COL_X[0], ROW_TOP[1]
    panel_label_at(fig, left - LETTER_DX, LETTER_Y[1], "d",
                   "Recovery on measured offsets")

    ax = mm_axes(fig, left, top, 41.0, 41.0)
    ax.set_aspect("equal")
    ax.plot([-0.8, 0.8], [-0.8, 0.8], "-", color=RULE, lw=LW_HAIR, zorder=1)
    styles = (
        ("first", dict(marker="o", mfc=INK, mec=INK, mew=0.0, ms=MS_POINT)),
        ("second", dict(marker="s", mfc="none", mec=MUTED, mew=LW_HAIR,
                        ms=MS_POINT)),
        ("higher", dict(marker="x", mec="#9A9A9A", mew=LW_HAIR, ms=MS_POINT)),
    )
    for key, style in styles:
        group = recovery[key]
        ax.plot(group["true"], group["found"], linestyle="none", zorder=3,
                **style)
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.85, 0.85)
    ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_xticklabels(["$-0.8$", "", "0", "", "0.8"])
    ax.set_yticklabels(["$-0.8$", "", "0", "", "0.8"])
    ax.set_xlabel("true coefficient", fontsize=FS_BODY, labelpad=2.2)
    ax.set_ylabel("recovered coefficient", fontsize=FS_BODY, labelpad=1.4)
    despine(ax)
    trim_spines(ax)

    for row, (key, style) in enumerate(styles):
        y = 0.76 - row * 0.10
        ax.plot([-0.76], [y], linestyle="none", clip_on=False, **style)
        ax.text(-0.68, y,
                f"{recovery[key]['label']}   RMSE {recovery[key]['rmse']:.3f}",
                ha="left", va="center", fontsize=FS_TINY, color=INK)

    fig_text_mm(
        fig, left, top + 53.0,
        f"{recovery['draws']} mixtures, {recovery['reps']} responses per offset,"
        f" order-{FIT_ORDER} fit\n"
        f"on the {design['n']} fixed offsets, with an out-of-band $m=23$\n"
        f"term at the measured amplitude {recovery['nuisance']:.3f}.",
        ha="left", va="top", fontsize=FS_TINY, color=MUTED, linespacing=1.45)


def _rep_key(fig, x_mm: float, y_mm: float, step: float = 11.0) -> None:
    for index, representation in enumerate(REP_ORDER):
        fig_text_mm(fig, x_mm + index * step, y_mm, REP_SHORT[representation],
                    ha="left", va="baseline", fontsize=FS_TINY,
                    color=REP[representation], fontweight="bold")


def _draw_e(fig, parts: dict) -> None:
    left, top = COL_X[1], ROW_TOP[1]
    panel_label_at(fig, left - LETTER_DX, LETTER_Y[1], "e",
                   "Mirror-pair balance")
    _rep_key(fig, left, KEY_Y[1])

    ax_odd = mm_axes(fig, left, top, COL_W, 19.0)
    ax_even = mm_axes(fig, left, top + 23.0, COL_W, 19.0)
    for ax, key, label in (
            (ax_odd, "odd", "odd part\n$[g(\\delta)-g(-\\delta)]/2$"),
            (ax_even, "even", "even part\n$[g(\\delta)+g(-\\delta)]/2$")):
        ax.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=0)
        for representation in REP_ORDER:
            data = parts[representation]
            ax.plot(data["delta"], data[key], "-o", color=REP[representation],
                    lw=LW_LINE, ms=MS_POINT, mew=0.0, zorder=3)
        ax.set_xlim(0, 182)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["$-1$", "0", "1"])
        ax.set_ylabel(label, fontsize=FS_TINY, labelpad=1.4, linespacing=1.2)
        despine(ax)
        trim_spines(ax)
    ax_odd.set_xticklabels([])
    ax_even.set_xlabel("$|\\delta|$ (degrees)", fontsize=FS_BODY, labelpad=2.2)

    summary = ";  ".join(
        f"{REP_SHORT[r]} {parts[r]['mean_abs_even']:.3f}" for r in REP_ORDER)
    fig_text_mm(
        fig, left, top + 53.0,
        "Unimodal $\\kappa=9$, block 1, 18 mirror pairs.\n"
        "The design supplies both halves of every pair; it\n"
        "does not make the response odd. mean |even part|:\n"
        f"{summary}.",
        ha="left", va="top", fontsize=FS_TINY, color=MUTED, linespacing=1.45)


def _draw_f(fig, table) -> None:
    left, top = COL_X[2], ROW_TOP[1]
    panel_label_at(fig, left - LETTER_DX, LETTER_Y[1], "f",
                   "Bootstrap stability of $C_m$")
    _rep_key(fig, left, KEY_Y[1])

    fields = (("kappa_2", "broad $\\kappa=2$"),
              ("kappa_12", "narrow $\\kappa=12$"))
    axes = []
    for index, (profile, title) in enumerate(fields):
        ax = mm_axes(fig, left, top + index * 23.0, COL_W, 20.0)
        axes.append(ax)
        # equal complex-plane aspect is enforced by the mm box, not by
        # set_aspect, which would silently shrink the axes off its column
        ax.set_ylim(-0.42, 0.42)
        ax.set_xlim(-0.68, 0.84 * COL_W / 20.0 - 0.68)
        ax.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=0)
        ax.axvline(0.0, color=RULE, lw=LW_HAIR, zorder=0)
        ax.add_patch(plt.Circle((0, 0), R_GATE, fill=False, color=MUTED,
                                lw=LW_HAIR, ls=(0, (2.0, 1.6)), zorder=1))
        rows = table[table["profile"] == profile]
        for _, row in rows.iterrows():
            colour = REP[row["representation"]]
            solid = int(row["harmonic"]) == 1
            ax.add_patch(mpatches.Ellipse(
                (row["center_a"], row["center_b"]),
                2.0 * row["axis_major"], 2.0 * row["axis_minor"],
                angle=row["angle_degrees"], fill=False, edgecolor=colour,
                lw=LW_THIN, ls="-" if solid else (0, (1.8, 1.4)), zorder=3))
            ax.plot([row["center_a"]], [row["center_b"]], "o",
                    mfc=colour if row["reportable"] else "white", mec=colour,
                    mew=LW_HAIR, ms=MS_POINT, zorder=4)
        ax.set_yticks([-0.4, 0, 0.4])
        ax.set_yticklabels(["$-0.4$", "0", "0.4"])
        ax.set_xticks([-0.5, 0, 0.5, 1.0])
        ax.set_ylabel("$b_m$", fontsize=FS_BODY, labelpad=1.4)
        despine(ax)
        trim_spines(ax)
        fig_text_mm(fig, left + COL_W, top + index * 23.0 + 1.6, title,
                    ha="right", va="baseline", fontsize=FS_TINY, color=INK)
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels(["$-0.5$", "0", "0.5", "1.0"])
    axes[1].set_xlabel("$a_m$", fontsize=FS_BODY, labelpad=2.2)

    fig_text_mm(
        fig, left, top + 53.0,
        "Block 1, 95% multinomial bootstrap. Solid\n"
        "ellipse $C_1$, dashed $C_2$; open centre marks\n"
        f"a suppressed phase ($R_m<{R_GATE:.2f}$, dotted circle,\n"
        "or ellipse covering the origin).",
        ha="left", va="top", fontsize=FS_TINY, color=MUTED, linespacing=1.45)


# --------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    design = _design()
    regular12 = _regular(12)
    regular36 = _regular(36)

    maps = [
        ("regular 12", _alias_map(regular12)),
        ("regular 36", _alias_map(regular36)),
        (f"fixed {design['n']}", _alias_map(design["offsets"])),
    ]
    amplitude = _spectrum_amplitude("narrow_kappa_12", 23)
    frozen_leak = float(maps[2][1][22, 0])
    sign = _sign_change()
    recovery = _recovery(design)
    parts = _mirror_parts(design)
    table = _ellipses()

    fig = new_figure(H_MM)
    _draw_a(fig, design)
    _draw_b(fig, maps, amplitude, frozen_leak)
    _draw_c(fig, sign)
    _draw_d(fig, recovery, design)
    _draw_e(fig, parts)
    _draw_f(fig, table)
    return save_fig(fig, "figS03_offset_design_aliasing")


if __name__ == "__main__":
    print(build())
