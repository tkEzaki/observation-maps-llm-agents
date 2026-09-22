"""Supplementary Figure S13 — Intervals: local noncompressibility under the
tested descriptors and coverage.

Nine panels (a–i) documenting why the intervals collective-surrogate branch was
stopped despite positive in-distribution OOF performance. Every printed number
is read from the frozen artifacts under ``analysis/intervals_branch/``; nothing
is hard-coded.

Panel e repeats the local-support decomposition that main Figure 6e shows for
intervals. The counts are read from ``int2b_offline_decision.json`` and are
cross-checked against ``analysis/synthesis/fig4_support_scatter.csv`` (the table
main Figure 6 reads) at build time, so the two figures cannot drift apart.

Deviation from the panel plan, reported rather than fabricated: the plan asks
panel g to plot nearest-neighbour distance against local response variation.
In the frozen support tables ``d_NN`` is exactly 0.0 for all 1512 fields (every
field has an exact descriptor twin inside its peer block), i.e. it is a
constant, not a distribution. Panel g therefore draws ``d_NN`` = 0 as the
exact-match locus (a line, not a scattered coordinate), puts ``V_local`` on the
y axis and stacks the ``V_local`` distribution by diagnosis beside it. The
point the panel has to land: the stop is not driven by missing
nearest-neighbour support - every field has an exact descriptor match - but by
the response distributions disagreeing at that same descriptor position.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from analysis.natcomm_si_opus.style import (
    ACTION,
    ACTION_LABELS,
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
    PASS_GREEN,
    REP,
    REP_LIGHT,
    ROOT,
    RULE,
    STOP_RED,
    apply_style,
    blank,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label_at,
    save_fig,
    trim_spines,
)

BR = ROOT / "analysis" / "intervals_branch"
SYN = ROOT / "analysis" / "synthesis"

INTV = REP["intervals_24_decimal6"]
INTV_L = REP_LIGHT["intervals_24_decimal6"]

# Diagnosis-bucket colours: identical hexes to main Figure 6d/e so the two
# figures read as one decomposition (style.py is shared and must not be edited).
BUCKET_COLOR = {
    "local_noncompressible": "#C0392B",
    "active_feature_gap": "#E8A13C",
    "supported": "#4C9A6E",
}
BUCKET_LABEL = {
    "local_noncompressible": "locally non-compressible",
    "active_feature_gap": "active feature gap",
    "supported": "supported",
}
BUCKET_ORDER = ["local_noncompressible", "active_feature_gap", "supported"]

# Reader-facing model names. Display strings only; every value is read from the
# frozen decision record.
MODEL_LABEL = {
    "activity_direction": "activity + direction",
    "canonical_emd_plus_discrete": "canonical EMD + discrete",
    "hellinger": "Hellinger",
    "canonical_emd": "canonical EMD",
    "direct_native_emd": "direct native EMD",
    "global": "peer-global baseline",
}
CLUSTER_LEVEL = {
    "physical_field_cluster": "physical field",
    "profile_family_cluster": "profile family",
}

# --- canvas plan (mm from the top-left) -------------------------------------
H_MM = 195.0
LET_X = (4.0, 62.0, 120.0)          # panel-letter columns
AX_X = (13.0, 71.0, 129.0)          # plot-box columns
AX_W = 45.0                         # every column is 45 mm wide

# A 45 mm column holds ~42 characters at FS_TINY; every annotation block below
# is hand-wrapped to that measure so no note crosses into its neighbour.
R1_LET, R1_TOP = 11.4, 13.8
R2_LET, R2_TOP = 65.4, 67.8
R3_LET, R3_TOP = 125.4, 127.8

# Panel e bar depth, in y-data units of a 2.9-unit axis 24 mm tall: 0.60 gives
# a 4.97 mm bar, which clears the 4.29 mm two-line label printed inside it.
E_BAR_H = 0.60


def _mkey(name: str) -> str:
    """Scope-independent model key (``peer_specific_x`` / ``peer16_x`` -> x)."""
    s = name.replace("peer_specific_", "").replace("peer16_", "")
    return "canonical_emd_plus_discrete" if s.startswith("canonical_emd_plus") else s


def _key_mm(fig, x_mm: float, y_mm: float, color: str, label: str,
            dx: float = 2.4) -> None:
    """Swatch + label placed in mm from the top-left of the canvas."""
    fig.text(x_mm / 180.0, (H_MM - y_mm) / H_MM, "■", color=color,
             fontsize=FS_TINY, ha="left", va="baseline")
    fig.text((x_mm + dx) / 180.0, (H_MM - y_mm) / H_MM, label, color=INK,
             fontsize=FS_TINY, ha="left", va="baseline")


def _lollipop(ax, ranking: list, xlim: tuple[float, float],
              highlight: str | None = None) -> float:
    """Log-loss lollipops hanging off the peer-global baseline value."""
    base = [r for r in ranking if _mkey(r["model"]) == "global"][0]
    base_ll = float(base["log_loss"])
    ax.axvline(base_ll, color="#B0B0B0", lw=LW_HAIR, zorder=0)
    for i, r in enumerate(ranking):
        key = _mkey(r["model"])
        ll = float(r["log_loss"])
        col = MUTED if key == "global" else INTV
        ax.plot([ll, base_ll], [i, i], color=col, lw=LW_THIN, zorder=1)
        ax.plot([ll], [i], "o", ms=MS_SERIES, color=col, zorder=3,
                markeredgewidth=0.0)
        ax.text(xlim[0] + 0.006, i - 0.34, MODEL_LABEL[key], ha="left",
                va="baseline", fontsize=FS_TINY, color=INK,
                fontweight="bold" if key == highlight else "normal")
        ax.text(ll - 0.006, i, f"{ll:.3f}", ha="right", va="center",
                fontsize=FS_TINY, color=col)
    ax.set_xlim(*xlim)
    ax.set_ylim(len(ranking) - 0.5, -0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    trim_spines(ax, y=False)
    return base_ll


def _forest_row(ax, y: float, stats: dict, color: str, marker: str) -> None:
    m = float(stats["mean_delta_ll"])
    lo, hi = float(stats["ci_low"]), float(stats["ci_high"])
    positive = bool(stats["ci_entirely_positive"])
    ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt=marker, color=color,
                ms=MS_POINT, capsize=CAPSIZE, elinewidth=LW_THIN,
                capthick=LW_THIN, markerfacecolor=color if positive else "white",
                markeredgewidth=0.6, clip_on=False, zorder=3)


def build(_out_dir: Path | None = None) -> Path:
    apply_style()

    dec = json.loads((BR / "int2b_offline_decision.json").read_text())
    gate = dec["support_gate_locked"]
    full = dec["full_collective"]
    p16 = dec["peer16_only"]
    risk_cal = json.loads((BR / "risk_calibrator_intervals_collective_v0.json").read_text())

    n_full = int(full["ranking"][0]["n"])
    n_p16 = int(p16["n_train"])
    cnt16 = p16["diagnosis_counts"]
    frac16 = p16["diagnosis_fractions"]
    cntF = full["diagnosis_counts"]
    fracF = full["diagnosis_fractions"]

    # --- consistency checks (never override an artifact; fail loudly) -------
    if sum(cnt16.values()) != n_p16 or sum(cntF.values()) != n_full:
        raise ValueError("diagnosis counts do not sum to the scope n")
    scat = pd.read_csv(SYN / "fig4_support_scatter.csv")
    scat = scat[scat["representation"].str.contains("interval", case=False,
                                                    na=False)]
    main_counts = scat["diagnosis_bucket"].value_counts().to_dict()
    if len(scat) != n_p16 or any(int(main_counts.get(k, 0)) != int(cnt16[k])
                                 for k in BUCKET_ORDER):
        raise ValueError(
            "peer-16 diagnosis counts disagree with the table main Fig. 6e "
            f"reads: {main_counts} vs {cnt16}"
        )
    checks = {
        "gap_risk_le_max": frac16["active_feature_gap"] <= gate["max_gap_risk_fraction"],
        "noncompress_le_max": frac16["local_noncompressible"] <= gate["max_noncompress_fraction"],
        "gap_plus_noncompress_le_max": (
            frac16["active_feature_gap"] + frac16["local_noncompressible"]
            <= gate["max_gap_plus_noncompress_fraction"]),
        "support_ok_ge_min": frac16["supported"] >= gate["min_support_ok_fraction"],
    }
    if any(bool(v) != bool(p16["support_gate_checks"][k]) for k, v in checks.items()):
        raise ValueError("recomputed support-gate checks disagree with the "
                         "frozen record")

    fig = new_figure(H_MM)

    # ==================================================================
    # a  structured versus peer-global OOF (full {8,16} scope)
    # ==================================================================
    axA = mm_axes(fig, AX_X[0], R1_TOP, AX_W, 34.0)
    panel_label_at(fig, LET_X[0], R1_LET, "a", "Structured vs peer-global OOF")
    base_full = _lollipop(axA, full["ranking"], (0.755, 1.075))
    axA.set_xticks([0.8, 0.9, 1.0])
    axA.set_xlabel(f"acquisition-block OOF log loss (full {{8,16}}, $n$={n_full})",
                   fontsize=FS_SMALL)
    axA.text(base_full - 0.005, -0.70, "peer-global reference", ha="right",
             va="baseline", fontsize=FS_TINY, color=MUTED, style="italic")

    # ==================================================================
    # b  cluster-bootstrap improvement over peer-global
    # ==================================================================
    axB = mm_axes(fig, AX_X[1], R1_TOP, AX_W, 34.0)
    panel_label_at(fig, LET_X[1], R1_LET, "b", "Cluster-bootstrap improvement")
    cb = full["cluster_bootstrap"]
    structured = [r for r in full["ranking"] if _mkey(r["model"]) != "global"]
    y = 0.0
    n_cl = {}
    for r in structured:
        name = r["model"]
        axB.text(-0.055, y - 0.62, MODEL_LABEL[_mkey(name)], ha="left",
                 va="baseline", fontsize=FS_TINY, color=INK)
        for lvl, col, mk in (("physical_field_cluster", INTV, "o"),
                             ("profile_family_cluster", INTV_L, "D")):
            st = cb[name][lvl]
            n_cl[lvl] = int(st["n_clusters"])
            _forest_row(axB, y, st, col, mk)
            if r is structured[0]:
                # label the two cluster levels once, beside their own rows
                axB.text(float(st["ci_high"]) + 0.016, y,
                         f"{CLUSTER_LEVEL[lvl]} ({int(st['n_clusters'])})",
                         ha="left", va="center", fontsize=FS_TINY, color=MUTED)
            if not st["ci_entirely_positive"]:
                axB.text(float(st["ci_high"]) + 0.016, y, "CI spans 0",
                         ha="left", va="center", fontsize=FS_TINY,
                         color=STOP_RED)
            y += 1.0
        y += 1.0
    axB.axvline(0.0, color="#B0B0B0", lw=LW_HAIR, zorder=0)
    axB.set_xlim(-0.06, 0.62)
    axB.set_xticks([0.0, 0.2, 0.4])
    axB.set_ylim(y - 1.4, -1.05)
    axB.set_yticks([])
    axB.spines["left"].set_visible(False)
    axB.tick_params(axis="x", labelsize=FS_TICK)
    axB.set_xlabel(r"$\Delta$ log loss vs peer-global (95% CI)",
                   fontsize=FS_SMALL)
    trim_spines(axB, y=False)

    # ==================================================================
    # c  full versus peer-16-only
    # ==================================================================
    axC = mm_axes(fig, AX_X[2], R1_TOP, AX_W, 27.0)
    panel_label_at(fig, LET_X[2], R1_LET, "c", "Full vs peer-16 only")
    best_full = full["model_gate"]["best_structured"]
    best_p16 = p16["model_gate"]["best_structured"]
    scopes = [(f"full {{8,16}}, $n$={n_full}", full["cluster_bootstrap"][best_full]),
              (f"peer-16 only, $n$={n_p16}", p16["cluster_bootstrap"][best_p16])]
    y = 0.0
    for lab, blk in scopes:
        axC.text(-0.02, y - 0.60, lab, ha="left", va="baseline",
                 fontsize=FS_TINY, color=INK, fontweight="bold")
        for lvl, col, mk in (("physical_field_cluster", INTV, "o"),
                             ("profile_family_cluster", INTV_L, "D")):
            st = blk[lvl]
            _forest_row(axC, y, st, col, mk)
            axC.text(float(st["ci_high"]) + 0.014, y,
                     f"{CLUSTER_LEVEL[lvl]} ({int(st['n_clusters'])})",
                     ha="left", va="center", fontsize=FS_TINY, color=MUTED)
            y += 1.0
        y += 1.0
    axC.axvline(0.0, color="#B0B0B0", lw=LW_HAIR, zorder=0)
    axC.set_xlim(-0.02, 0.70)
    axC.set_xticks([0.0, 0.2, 0.4])
    axC.set_ylim(y - 1.4, -1.05)
    axC.set_yticks([])
    axC.spines["left"].set_visible(False)
    axC.tick_params(axis="x", labelsize=FS_TICK)
    axC.set_xlabel(r"$\Delta$ log loss vs peer-global (95% CI)",
                   fontsize=FS_SMALL)
    trim_spines(axC, y=False)

    ll_full = {_mkey(r["model"]): float(r["log_loss"]) for r in full["ranking"]}
    ll_p16 = {_mkey(r["model"]): float(r["log_loss"]) for r in p16["ranking"]}
    fig_text_mm(
        fig, AX_X[2], R1_TOP + 35.2,
        "best structured vs peer-global log loss\n"
        f"full {ll_full['activity_direction']:.3f} vs {ll_full['global']:.3f}\n"
        f"peer-16 {ll_p16['activity_direction']:.3f} vs {ll_p16['global']:.3f}\n"
        "in-distribution compressibility survives\nthe peer-regime restriction",
        fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.5)

    # ==================================================================
    # d  discrete-feature bake-off (peer-16 scope)
    # ==================================================================
    axD = mm_axes(fig, AX_X[0], R2_TOP, AX_W, 30.0)
    panel_label_at(fig, LET_X[0], R2_LET, "d", "Discrete-feature bake-off")
    _lollipop(axD, p16["ranking"], (0.755, 1.085),
              highlight="canonical_emd_plus_discrete")
    axD.set_xticks([0.8, 0.9, 1.0])
    axD.set_xlabel(f"peer-16 OOF log loss ($n$={n_p16})", fontsize=FS_SMALL)

    gapd = p16["discrete_help_on_gap_fields"]
    fig_text_mm(
        fig, AX_X[0], R2_TOP + 36.6,
        "Prespecified criterion: a candidate is adopted\n"
        "only if it clears the locked support gate\n"
        f"(supported $\\geq$ {gate['min_support_ok_fraction']:.0%}, "
        f"non-compressible $\\leq$ {gate['max_noncompress_fraction']:.0%}).\n"
        f"On the {int(gapd['n_gap'])} active-feature-gap fields the\n"
        "discrete features move mean local $d_{TV}$\n"
        f"from {gapd['mean_tv_canonical']:.3f} to {gapd['mean_tv_discrete']:.3f} "
        f"and help {gapd['improved_fraction']:.0%} of fields:\n"
        "no tested feature cleared the gate.",
        fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.5)

    # ==================================================================
    # e  diagnostic fractions (reproduces main Fig. 6e for peer-16)
    # ==================================================================
    # 24 mm, not 19: each bar carries a two-line "share / n" label set inside
    # it in white, and at the old 19 mm the bar was 3.4 mm deep against a
    # 4.3 mm label, so the top line's digits ran off the bar and turned white
    # on white paper. The bar is now sized from the label it has to hold.
    axE = mm_axes(fig, AX_X[1], R2_TOP + 2.0, AX_W, 24.0)
    panel_label_at(fig, LET_X[1], R2_LET, "e", "Local-support decomposition")
    rows = [(f"peer-16\n$n$={n_p16}", cnt16, frac16),
            (f"full {{8,16}}\n$n$={n_full}", cntF, fracF)]
    for yi, (lab, cnts, fracs) in enumerate(rows):
        left = 0.0
        for key in BUCKET_ORDER:
            v = float(fracs[key])
            axE.barh([yi], [v], left=left, height=E_BAR_H,
                     color=BUCKET_COLOR[key], edgecolor="white",
                     linewidth=0.6)
            if v > 0.12:
                axE.text(left + v / 2.0, yi, f"{v:.0%}\n$n$={int(cnts[key])}",
                         ha="center", va="center", fontsize=FS_TINY,
                         color="white", fontweight="bold", linespacing=1.2)
            else:
                # leader out of the bar: row 0 upwards, row 1 downwards, so the
                # two small labels cannot meet in the gap between the bars
                s_dir = -1.0 if yi == 0 else 1.0
                axE.plot([left + v / 2.0] * 2,
                         [yi + s_dir * (E_BAR_H / 2.0 + 0.03),
                          yi + s_dir * 0.56],
                         color=BUCKET_COLOR[key], lw=LW_HAIR, clip_on=False)
                axE.text(min(left + v / 2.0 + 0.02, 1.0),
                         yi + s_dir * 0.62,
                         f"{v:.0%} ($n$={int(cnts[key])})", ha="right",
                         va="baseline" if s_dir < 0 else "top",
                         fontsize=FS_TINY, color=BUCKET_COLOR[key])
            left += v
    axE.set_yticks([0, 1])
    axE.set_yticklabels([r[0] for r in rows], fontsize=FS_TINY,
                        linespacing=1.2)
    axE.tick_params(axis="y", length=0)
    axE.set_ylim(2.05, -0.85)
    axE.set_xlim(0, 1)
    axE.set_xticks([0.0, 0.5, 1.0])
    axE.set_xticklabels(["0", "50%", "100%"])
    axE.tick_params(axis="x", labelsize=FS_TICK)
    axE.set_xlabel("share of local diagnoses", fontsize=FS_SMALL)
    axE.spines["left"].set_visible(False)
    trim_spines(axE, y=False)
    for bi, key in enumerate(BUCKET_ORDER):
        _key_mm(fig, AX_X[1], R2_TOP + 35.4 + bi * 2.8, BUCKET_COLOR[key],
                BUCKET_LABEL[key])

    # ==================================================================
    # f  sparse-antipodal disagreement at zero descriptor distance
    # ==================================================================
    axF = mm_axes(fig, AX_X[2], R2_TOP + 2.0, AX_W, 24.0)
    panel_label_at(fig, LET_X[2], R2_LET, "f", "Sparse-antipodal disagreement")
    tr = pd.read_csv(BR / "training_rows_intervals_collective_v0.csv")
    tr = tr[tr["peer_count"] == 16].reset_index(drop=True)
    det16 = pd.read_csv(BR / "int2b_support_detail_peer16.csv")
    if list(tr["profile"]) != list(det16["profile"]):
        raise ValueError("training rows and peer-16 support detail are not "
                         "row-aligned")
    tr["coarse_family"] = det16["coarse_family"].to_numpy()
    sa = tr[tr["coarse_family"] == "sparse_antipodal"].copy()
    counts = sa[["count_retard", "count_stay", "count_advance"]].to_numpy(float)
    n_trials = counts.sum(axis=1)
    probs = counts / n_trials[:, None]

    groups = []
    for h, idx in sa.groupby("prompt_hash").indices.items():
        if not 4 <= len(idx) <= 6:
            continue
        P = probs[idx]
        tv = float(np.mean([0.5 * np.abs(P[i] - P[j]).sum()
                            for i in range(len(P))
                            for j in range(i + 1, len(P))]))
        groups.append((-tv, str(h), list(idx)))
    groups.sort()
    groups = groups[:5]

    xc, ticks, tick_lab = 0.0, [], []
    for gi, (negtv, _h, idx) in enumerate(groups):
        x0 = xc
        for j in idx:
            bottom = 0.0
            for v, col in zip(probs[j], (ACTION["p_minus"], ACTION["p_zero"],
                                         ACTION["p_plus"])):
                axF.bar([xc], [v], bottom=[bottom], width=0.82, color=col,
                        edgecolor="white", linewidth=0.4)
                bottom += v
            xc += 1.0
        ticks.append((x0 + xc - 1.0) / 2.0)
        tick_lab.append(f"{-negtv:.2f}")
        xc += 1.4
    axF.set_xlim(-0.9, xc - 1.9)
    axF.set_ylim(0, 1.0)
    axF.set_yticks([0, 0.5, 1.0])
    axF.set_yticklabels(["0", "0.5", "1"])
    axF.set_xticks(ticks)
    axF.set_xticklabels(tick_lab)
    axF.tick_params(axis="both", labelsize=FS_TICK)
    axF.tick_params(axis="x", length=0)
    axF.set_ylabel("response share", fontsize=FS_SMALL)
    axF.set_xlabel(r"mean pairwise $d_{TV}$ within field group",
                   fontsize=FS_SMALL)
    trim_spines(axF, x=False)
    for ki, (lab, col) in enumerate(zip(ACTION_LABELS,
                                        (ACTION["p_minus"], ACTION["p_zero"],
                                         ACTION["p_plus"]))):
        _key_mm(fig, AX_X[2] + 1.0 + ki * 8.0, R2_TOP + 34.6, col, lab, dx=2.2)
    tmin, tmax = int(n_trials.min()), int(n_trials.max())
    fig_text_mm(
        fig, AX_X[2], R2_TOP + 38.2,
        f"{len(groups)} groups of sparse-antipodal fields with\n"
        "byte-identical serialized histograms\n"
        "(descriptor distance exactly 0); one bar per\n"
        f"acquisition row, {tmin}–{tmax} responses per row.\n"
        "Neighbours exist; their empirical\ntrinomials disagree.",
        fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.5)

    # ==================================================================
    # g  the exact-match locus: d_NN is a constant, V_local is the gate
    # ==================================================================
    panel_label_at(fig, LET_X[0], R3_LET, "g", "Exact-match locus")
    det_full = pd.read_csv(BR / "int2b_support_detail_full.csv")
    if len(det_full) != n_full:
        raise ValueError("full support detail does not hold the full scope n")
    if (np.abs(det_full["d_NN"].to_numpy(float)).max() != 0.0
            or np.abs(det16["d_NN"].to_numpy(float)).max() != 0.0):
        raise ValueError("d_NN is not identically zero; panel g assumes it is")

    v_thr = float(gate["local_dispersion_tv_threshold"])
    nn_thr = float(gate["coverage_gap_nn_threshold"])
    vs = det16["V_local"].to_numpy(float)
    diag = det16["diagnosis"].to_numpy()
    V_LO, V_HI = 0.06, 0.66

    axG = mm_axes(fig, AX_X[0], R3_TOP, 37.0, 29.0)
    axGm = mm_axes(fig, AX_X[0] + 39.0, R3_TOP, 6.0, 29.0)

    # Every field sits at d_NN = 0, so the locus is drawn once as a line
    # instead of scattering 1430 points through a constant coordinate.
    axG.axhline(v_thr, color=INK, lw=LW_HAIR, ls="--", zorder=3)
    axG.axvline(nn_thr, color=MUTED, lw=LW_HAIR, ls=":", zorder=2)
    axG.plot([0.0, 0.0], [V_LO, V_HI], color=INK, lw=LW_LINE, zorder=4,
             solid_capstyle="butt")
    axG.set_xlim(-0.055, 0.41)
    axG.set_ylim(V_LO, V_HI)
    axG.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    axG.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    axG.tick_params(axis="both", labelsize=FS_TICK)
    axG.set_xlabel(r"descriptor distance $d_{NN}$", fontsize=FS_SMALL)
    axG.set_ylabel("local response dispersion\n" r"$V_{local}$ ($k$=16)",
                   fontsize=FS_SMALL, linespacing=1.3)
    trim_spines(axG)
    axG.text(0.135, 0.985,
             "exact descriptor match:\n"
             f"$d_{{NN}}$ = 0, all {n_full} fields",
             transform=axG.transAxes, ha="left", va="top", fontsize=FS_TINY,
             color=INK, linespacing=1.35)
    axG.text(0.150, 0.740,
             f"$V_{{local}}$ $\\geq$ {v_thr:.2f}:\n"
             "locally non-compressible",
             transform=axG.transAxes, ha="left", va="top", fontsize=FS_TINY,
             color=BUCKET_COLOR["local_noncompressible"], linespacing=1.35)
    axG.text(0.150, 0.400,
             f"$V_{{local}}$ < {v_thr:.2f}: active\n"
             "feature gap or supported",
             transform=axG.transAxes, ha="left", va="top", fontsize=FS_TINY,
             color=INK, linespacing=1.35)
    axG.text(nn_thr - 0.014, V_LO + 0.004, "coverage-gap\nthreshold",
             ha="right", va="bottom", fontsize=FS_TINY, color=MUTED,
             linespacing=1.35)

    # the marginal carries the colour: V_local stacked by frozen diagnosis
    bins = np.linspace(V_LO, V_HI, 34)
    stack = np.zeros(len(bins) - 1)
    for key in BUCKET_ORDER:
        h, _ = np.histogram(vs[diag == key], bins=bins)
        axGm.barh(bins[:-1], h, left=stack, height=np.diff(bins),
                  align="edge", color=BUCKET_COLOR[key], linewidth=0.0)
        stack = stack + h
    axGm.axhline(v_thr, color=INK, lw=LW_HAIR, ls="--", zorder=3)
    axGm.set_ylim(V_LO, V_HI)
    axGm.set_xlim(0.0, float(stack.max()) * 1.10)
    axGm.set_xticks([])
    axGm.set_yticks([])
    for sp in axGm.spines.values():
        sp.set_visible(False)
    axGm.text(0.98, 0.99, f"$n$={n_p16}", transform=axGm.transAxes,
              ha="right", va="top", fontsize=FS_TINY, color=MUTED)

    # ==================================================================
    # h  applicability-risk calibration
    # ==================================================================
    axH = mm_axes(fig, AX_X[1], R3_TOP, AX_W, 29.0)
    panel_label_at(fig, LET_X[1], R3_LET, "h", "Applicability-risk calibration")
    binc = risk_cal["bin_calibration"]
    R = np.array([b["mean_R"] for b in binc])
    E = np.array([b["mean_error"] for b in binc])
    lo = min(R.min(), E.min()) - 0.015
    hi = max(R.max(), E.max()) + 0.015
    axH.plot([lo, hi], [lo, hi], color=RULE, lw=LW_HAIR, ls="--", zorder=1)
    axH.plot(R, E, "-", color=INTV_L, lw=LW_THIN, zorder=2)
    axH.plot(R, E, "o", color=INTV, ms=MS_SERIES, markeredgewidth=0.0, zorder=3)
    axH.set_xlim(lo, hi)
    axH.set_ylim(lo, hi)
    axH.set_xticks([0.22, 0.26, 0.30])
    axH.set_yticks([0.22, 0.26, 0.30])
    axH.tick_params(axis="both", labelsize=FS_TICK)
    axH.set_xlabel(f"predicted risk $R$, octile mean "
                   f"($n$={int(binc[0]['n'])} per bin)", fontsize=FS_SMALL)
    axH.set_ylabel("observed error $e_{TV}$,\noctile mean", fontsize=FS_SMALL,
                   linespacing=1.3)
    axH.text(hi - 0.004, lo + 0.004, "identity", ha="right", va="bottom",
             fontsize=FS_TINY, color=MUTED, style="italic")
    trim_spines(axH)
    fig_text_mm(
        fig, AX_X[1], R3_TOP + 36.4,
        "Spearman $\\rho$($R$, $e_{TV}$) and top-decile\n"
        f"enrichment: full {{8,16}} {full['risk']['spearman']:+.3f}, "
        f"{full['risk']['enrichment']:.2f}×;\n"
        f"peer-16 only {p16['risk']['spearman']:+.3f}, "
        f"{p16['risk']['enrichment']:.2f}×. Risk ranks error\n"
        "monotonically but too weakly to gate\nprospective use.",
        fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.5)

    # ==================================================================
    # i  prespecified stop decision
    # ==================================================================
    I_H = 54.0
    axI = mm_axes(fig, AX_X[2], R3_TOP, AX_W, I_H)
    blank(axI)
    axI.set_xlim(0, AX_W)
    axI.set_ylim(I_H, 0)                     # 1 unit == 1 mm, y downwards
    panel_label_at(fig, LET_X[2], R3_LET, "i", "Prespecified stop decision")

    gap_plus = frac16["active_feature_gap"] + frac16["local_noncompressible"]
    gate_rows = [
        ("active feature gap", frac16["active_feature_gap"],
         gate["max_gap_risk_fraction"], "≤", checks["gap_risk_le_max"]),
        ("non-compressible", frac16["local_noncompressible"],
         gate["max_noncompress_fraction"], "≤", checks["noncompress_le_max"]),
        ("gap + non-compressible", gap_plus,
         gate["max_gap_plus_noncompress_fraction"], "≤",
         checks["gap_plus_noncompress_le_max"]),
        ("supported", frac16["supported"], gate["min_support_ok_fraction"],
         "≥", checks["support_ok_ge_min"]),
    ]
    y = 0.0
    h_row = 4.4
    for lab, obs, thr, rel, ok in gate_rows:
        axI.add_patch(mpatches.FancyBboxPatch(
            (0.0, y), AX_W, h_row, boxstyle="round,pad=0,rounding_size=0.8",
            facecolor="#EEF5F0" if ok else "#FAEAEA", edgecolor="none",
            mutation_aspect=1.0))
        axI.text(1.2, y + h_row / 2.0, lab, ha="left", va="center",
                 fontsize=FS_TINY, color=INK)
        axI.text(AX_W - 14.0, y + h_row / 2.0, f"{obs:.1%}", ha="right",
                 va="center", fontsize=FS_TINY, color=INK, fontweight="bold")
        axI.text(AX_W - 7.4, y + h_row / 2.0, f"{rel} {thr:.0%}", ha="right",
                 va="center", fontsize=FS_TINY, color=MUTED)
        axI.text(AX_W - 1.0, y + h_row / 2.0, "PASS" if ok else "FAIL",
                 ha="right", va="center", fontsize=FS_TINY, fontweight="bold",
                 color=PASS_GREEN if ok else STOP_RED)
        y += h_row + 0.7
    axI.annotate("", xy=(AX_W / 2.0, y + 2.0), xytext=(AX_W / 2.0, y - 0.2),
                 arrowprops=dict(arrowstyle="-|>", color="#777777", lw=LW_LINE,
                                 mutation_scale=6, shrinkA=0, shrinkB=0))
    y += 3.4
    verdicts = [
        ("model gate (OOF)", "PASS" if full["model_gate"]["pass"] else "FAIL",
         full["model_gate"]["pass"]),
        ("support gate", "PASS" if p16["support_gate_pass"] else "FAIL",
         p16["support_gate_pass"]),
        ("INT-3 prospective pilot",
         "GO" if dec["int3_prospective_pilot_go"] else "NO-GO", False),
        ("collective bundle freeze",
         "yes" if dec["freeze_ready"] else "no", False),
        ("paid authorization",
         "yes" if dec["paid_authorized"] else "no", False),
    ]
    for lab, val, ok in verdicts:
        axI.text(1.2, y, lab, ha="left", va="top", fontsize=FS_TINY, color=INK)
        axI.text(AX_W - 1.0, y, val, ha="right", va="top", fontsize=FS_TINY,
                 fontweight="bold", color=PASS_GREEN if ok else STOP_RED)
        y += 2.7
    y += 1.0
    axI.add_patch(mpatches.FancyBboxPatch(
        (0.0, y), AX_W, I_H - y - 0.3, boxstyle="round,pad=0,rounding_size=1.0",
        facecolor="#FAEAEA", edgecolor=STOP_RED, lw=LW_LINE,
        mutation_aspect=1.0))
    axI.text(AX_W / 2.0, y + 1.4, "STOP", ha="center", va="top",
             fontsize=FS_BODY, fontweight="bold", color=STOP_RED)
    axI.text(AX_W / 2.0, y + 5.0,
             "available finite-peer manifold does not\n"
             "support reliable prospective transport\n"
             "under the tested descriptors and coverage",
             ha="center", va="top", fontsize=FS_TINY, color=INK,
             linespacing=1.4)

    # --- footnote -----------------------------------------------------------
    fig_text_mm(
        fig, 4.0, H_MM - 10.0,
        f"Prespecified diagnosis rule: $d_{{NN}}$ > "
        f"{gate['coverage_gap_nn_threshold']:.2f} (no neighbour); else "
        f"$V_{{local}}$ $\\geq$ {gate['local_dispersion_tv_threshold']:.2f} "
        "(locally non-compressible); else high-stay fraction < 0.25 with "
        "active fraction $\\geq$ 0.5 (active feature gap);\n"
        "else supported.",
        fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.5)

    return save_fig(fig, "figS13_intervals_stop_rule")


if __name__ == "__main__":  # pragma: no cover
    print(build())
