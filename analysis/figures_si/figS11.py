"""Supplementary Figure S11 — Moments surrogate development, replay diagnosis
and prospective repair.

Two lettered pages, one figure number:

* **S11a** (a–d) — how the moments surrogate was built, how it looked in
  distribution, how independent collective-field replay exposed a localized
  stay-probability error, and what the v2 regime gate actually tests.
* **S11b** (e–h) — the prospective v1→v2 repair on the three Stage C v0.2
  endpoints that carry an LLM ground truth, and the frozen applicability-risk
  calibration, reported as a descriptive diagnostic.

The plan's panels h (social-torque error) and i (locking-time error) are
**omitted** on instruction; the remaining panels are re-lettered consecutively
so there is no gap.

Panel g repeats, cell by cell, the same ``combined_endpoints_per_run.csv``
quantities that main Figure 6c summarises for the ``v0.2a`` block, so the two
figures are read from one table and cannot drift apart. Every printed number
is computed from the frozen artifacts; nothing is hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg

from analysis.figures_si.style import (
    ACCENT_RED,
    CAPSIZE,
    FS_BODY,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    LW_TRACE,
    MS_MEAN,
    MS_POINT,
    MS_SERIES,
    MUTED,
    PASS_GREEN,
    REP,
    REP_LIGHT,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    STOP_RED,
    W_FULL_MM,
    apply_style,
    blank,
    despine,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label_at,
    rule_mm,
    save_fig,
    swarm_x,
    trim_spines,
)

S3A = ROOT / "analysis" / "stage3a_artifacts"
V2 = S3A / "moments_bundle_v2"
SC2 = ROOT / "analysis" / "stage_c_artifacts" / "stage_c_v0_2_combined"

MOM = REP["moments_m1_m3"]
MOM_L = REP_LIGHT["moments_m1_m3"]

# --- local grammars ---------------------------------------------------------
# Panel a groups training rows by *acquisition source*, not by representation
# or model family, so it needs its own ramp. Neutral greys darken toward the
# collective end of the manifold; the 36 production replay fields — the ones
# that broke v1 — carry the house accent red. style.py is shared and must not
# be edited, so the ramp lives here.
SRC_GROUPS = (
    ("stage_b", "Stage B fields", "#C9C9C9",
     ("transmutation", "stimulus_manifold", "antipodal_weight")),
    ("sparse", "sparse peer fields", "#9AA7AE", ("sparse_peer",)),
    ("pilot", "pilot collective fields", "#4F5F68", ("stage3a_pilot",)),
    ("replay", "production collective replay", ACCENT_RED, ("collective_replay",)),
)

# Replay buckets, in the order the check table stores them.
BUCKET_ORDER = ("neg", "pos", "zero", "stage_b_anchor")
BUCKET_LABEL = {
    "neg": "replay, $K<0$",
    "pos": "replay, $K>0$",
    "zero": "replay, $K=0$",
    "stage_b_anchor": "anchors and controls",
}

# post_pilot_model_compare.csv model keys -> reader-facing names + marks.
MODEL_STYLE = {
    "kernel_hurdle": ("kernel hurdle (v1 production)", "o", MOM, MOM),
    "kernel_nn": ("kernel-NN", "s", "white", "#5A5A5A"),
    "softmax_stump_boost": ("stump boost", "^", "white", "#9A9A9A"),
}
MODEL_ORDER = ("kernel_hurdle", "kernel_nn", "softmax_stump_boost")

# Common descriptor indices inside the frozen moments feature matrix. The
# names come from circlemap.field_features.COMMON_FEATURE_NAMES and are
# cross-checked against the bundle's stay_feature_indices at build time.
IDX_ABS_Z1 = 2
IDX_BALANCE = 10

# ---------------------------------------------------------------------------
# page geometry (mm from top-left)
# ---------------------------------------------------------------------------
# House rule: at least 2 mm of clear paper on all four sides. Free text blocks
# are wrapped to an explicit budget derived from these margins rather than
# being set at full canvas width and hoping they fit.
M_SIDE = 4.0
M_FOOT = 3.6
NOTE_BUDGET_MM = W_FULL_MM - 2 * M_SIDE

H_A = 147.0
H_B = 168.0

COL1_X, COL2_X = 4.0, 96.0        # panel-letter columns, page A
AX1_X, AX2_X = 15.0, 107.0        # plot-box columns, page A
AX_W = 68.0

A_ROW1, A_ROW2 = 12.5, 86.0
A_H1, A_H2 = 50.0, 50.0

# Panel c carries its bucket names in the left gutter instead of on a y axis,
# so its plot box starts further right than panel a's; the two panels line up
# on the *ink* (label column vs y-label column), not on the axes box.
C_AX_X, C_AX_W = 34.0, 49.0

B_AX_X = (20.0, 76.0, 132.0)      # plot-box columns, page B
B_AX_W = 44.0
B_LET_DX = -8.6
B_ROW1, B_ROW2 = 12.5, 93.0
B_H1, B_H2 = 62.0, 64.0
B_AX_H_W = 104.0  # panel h plot box width
B_TAB_W = 44.0    # gate-table column to the right of panel h

# Panel e-g header band: two summary lines above the first data row.
B_HEAD_TOP = -2.45
B_HEAD_ALL = -2.25
B_HEAD_CELLS = -1.72
# The "N, K" column header used to share the B_HEAD_CELLS baseline with
# "8/11 cells improved"; it now sits midway to the group label.
B_HEAD_NK = -1.17
B_GROUP_DY = -0.62


def _renderer(fig):
    canvas = fig.canvas
    if not hasattr(canvas, "get_renderer"):
        canvas = FigureCanvasAgg(fig)
    return canvas.get_renderer()


def _text_mm(fig, text: str, fontsize: float, **kw) -> float:
    """Rendered width of ``text`` on this canvas, in millimetres."""
    rend = _renderer(fig)
    probe = fig.text(0.0, -1.0, text, fontsize=fontsize, **kw)
    w = probe.get_window_extent(renderer=rend).width / fig.dpi * 25.4
    probe.remove()
    return w


def _wrap_mm(fig, text: str, width_mm: float, fontsize: float, **kw) -> str:
    """Greedy word wrap to an explicit width budget, measured in millimetres.

    Assumes no spaces inside ``$...$`` spans, which holds for every string
    this figure wraps.
    """
    rend = _renderer(fig)
    probe = fig.text(0.0, -1.0, "", fontsize=fontsize, **kw)
    # An em dash never opens a line: bind it to the word in front of it.
    words: list[str] = []
    for w in text.split():
        if w == "—" and words:
            words[-1] = f"{words[-1]} —"
        else:
            words.append(w)
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        probe.set_text(trial)
        wide = probe.get_window_extent(renderer=rend).width / fig.dpi * 25.4
        if cur and wide > width_mm:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    probe.remove()
    return "\n".join(lines)


def _spearman(a, b) -> float:
    """Spearman rho via ranks; scipy is not a dependency of this repo."""
    ra = pd.Series(np.asarray(a, dtype=float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, dtype=float)).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _fmt_signed(v: float, nd: int = 3) -> str:
    """Signed number with a real U+2212 inside maths."""
    return f"${v:+.{nd}f}$".replace("-", "−")


def _cell_label(n: int, k: float) -> str:
    return f"{int(n)}, ${k:+.2f}$".replace("-", "−")


# ===========================================================================
# S11a — model development and replay diagnosis
# ===========================================================================
def _page_a() -> Path:
    rows = pd.read_csv(V2 / "training_rows_v2.csv")
    rows = rows[rows["representation"] == "moments_m1_m3"].reset_index(drop=True)
    feats = np.load(V2 / "training_arrays_moments.npz")["features"]
    if len(rows) != feats.shape[0]:
        raise RuntimeError("training_rows_v2 and training_arrays_moments disagree")

    hyp = json.loads((V2 / "hyperparameters.json").read_text(encoding="utf-8"))
    prov = json.loads((V2 / "provenance.json").read_text(encoding="utf-8"))
    if hyp["stay_feature_names"][hyp["stay_feature_indices"].index(IDX_ABS_Z1)] != "abs_z1":
        raise RuntimeError("frozen stay-feature index for abs_z1 moved")
    if hyp["stay_feature_names"][
            hyp["stay_feature_indices"].index(IDX_BALANCE)] != "antipodal_balance":
        raise RuntimeError("frozen stay-feature index for antipodal_balance moved")

    bal = feats[:, IDX_BALANCE]
    z1 = feats[:, IDX_ABS_Z1]

    comp = pd.read_csv(S3A / "post_pilot_model_compare.csv")
    comp = comp[comp["representation"] == "moments_m1_m3"]
    holdout_block = str(comp["holdout_block"].iloc[0]).replace("_", " ")

    chk = pd.read_csv(V2 / "replay_stay_check_excl_replay_train.csv")

    fig = new_figure(H_A)

    # ------------------------------------------------------------------
    # a  training region in the two frozen gate descriptors
    # ------------------------------------------------------------------
    axA = mm_axes(fig, AX1_X, A_ROW1, AX_W, A_H1)
    panel_label_at(fig, COL1_X, A_ROW1 - 2.4, "a",
                   "Training region in descriptor space")

    bal_min = float(hyp["balance_min"])
    z1_max = float(hyp["abs_z1_max"])
    axA.add_patch(
        mpatches.Rectangle(
            (bal_min, 0.0), 1.02 - bal_min, z1_max,
            facecolor="#EDEDED", edgecolor="#8C8C8C", lw=LW_HAIR,
            linestyle=(0, (2.2, 1.4)), zorder=0,
        )
    )

    counts = {}
    for key, label, color, families in SRC_GROUPS:
        sel = rows["source_family"].isin(families).to_numpy()
        counts[key] = int(sel.sum())
        big = key in ("pilot", "replay")
        axA.scatter(
            bal[sel], z1[sel],
            s=(MS_SERIES if big else MS_POINT) ** 2,
            c=color, linewidths=0.0, zorder=2 if not big else 3,
            alpha=1.0 if big else 0.75,
        )
    axA.set_xlim(-0.03, 1.03)
    # Headroom above |z1| = 1: at a 1.03 ceiling the top row of markers ran
    # into the panel edge instead of standing clear of it.
    axA.set_ylim(-0.04, 1.08)
    axA.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axA.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axA.set_xlabel("antipodal balance")
    axA.set_ylabel("$|z_1|$")
    trim_spines(axA)

    # The gate rectangle sits in the densest corner of the cloud, so the label
    # is set clear of the marks and leadered in rather than laid over them:
    # in the empty |z1| band (no training row lies between 0.37 and 0.48)
    # with a vertical leader down a balance corridor that carries no marks.
    # The old horizontal leader at |z1| = 0.054 was drawn straight through the
    # dense marker columns at antipodal balance 0.75 and 0.875.
    axA.annotate(
        "v2 abstention regime (panel d)",
        xy=(0.972, z1_max + 0.008), xytext=(0.972, 0.43),
        ha="right", va="center", fontsize=FS_TINY, color="#6E6E6E",
        arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=LW_HAIR,
                        shrinkA=2.4, shrinkB=1.2, relpos=(1.0, 0.0)),
    )

    # Direct key, stacked in the genuinely empty lower-left quadrant. The
    # upper-left it used to occupy carries Stage B rows at |z1| > 0.85.
    for i, (key, label, color, _fams) in enumerate(SRC_GROUPS):
        yk = 0.395 - 0.078 * i
        axA.plot([0.045], [yk], marker="o", color=color, ms=MS_SERIES,
                 linestyle="none")
        axA.text(0.095, yk, f"{label} ($n$={counts[key]})", va="center",
                 ha="left", fontsize=FS_TINY, color=INK)

    # ------------------------------------------------------------------
    # b  v1 held-out fit on the three targets
    # ------------------------------------------------------------------
    axB = mm_axes(fig, AX2_X, A_ROW1, AX_W, A_H1)
    panel_label_at(fig, COL2_X, A_ROW1 - 2.4, "b",
                   "v1 in-distribution fit")

    metric_rows = (
        ("activity_rmse", "activity RMSE", "lower is better", False),
        ("g_rmse", "signed action $g$ RMSE", "lower is better", False),
        ("stay", "stay probability: small imbalance → exact balance",
         "wider gap is better", True),
    )
    printed = {}
    for i, (col, label, hint, is_gap) in enumerate(metric_rows):
        axB.text(0.0, i - 0.42, label, ha="left", va="baseline",
                 fontsize=FS_SMALL, color=INK)
        axB.text(1.0, i - 0.42, hint, ha="right", va="baseline",
                 fontsize=FS_TINY, color=MUTED, style="italic")
        for j, model in enumerate(MODEL_ORDER):
            r = comp[comp["model"] == model].iloc[0]
            _name, marker, face, edge = MODEL_STYLE[model]
            y = i + (j - 1) * 0.235
            if is_gap:
                lo = float(r["moments_stay_small_imbalance"])
                hi = float(r["moments_stay_exact_balance"])
                axB.plot([lo, hi], [y, y], color=edge, lw=LW_THIN, zorder=2)
                axB.plot([lo], [y], marker="|", color=edge, ms=MS_SERIES,
                         mew=LW_LINE, linestyle="none", zorder=3)
                axB.plot([hi], [y], marker=marker, mfc=face, mec=edge,
                         ms=MS_SERIES, mew=LW_THIN, linestyle="none", zorder=3)
                if model == "kernel_hurdle":
                    printed["stay_gap"] = float(r["moments_stay_gap"])
            else:
                v = float(r[col])
                axB.plot([v], [y], marker=marker, mfc=face, mec=edge,
                         ms=MS_SERIES, mew=LW_THIN, linestyle="none", zorder=3)
                if model == "kernel_hurdle":
                    printed[col] = v
    axB.set_xlim(0.0, 1.0)
    axB.set_ylim(2.72, -0.78)
    axB.set_yticks([])
    axB.spines["left"].set_visible(False)
    axB.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axB.set_xlabel(f"held-out value ({holdout_block})")
    trim_spines(axB, y=False)

    # Key entries advanced by their own measured width; the fixed 0.40/0.235
    # advances used before ran the first label straight through the second.
    key_x = 0.0
    for model in MODEL_ORDER:
        name, marker, face, edge = MODEL_STYLE[model]
        bold = model == "kernel_hurdle"
        axB.plot([key_x], [-0.70], marker=marker, mfc=face, mec=edge,
                 ms=MS_SERIES, mew=LW_THIN, linestyle="none", clip_on=False)
        axB.text(key_x + 0.028, -0.70, name, ha="left", va="center",
                 fontsize=FS_TINY, color=INK if bold else MUTED,
                 fontweight="bold" if bold else "normal")
        # axes span 1.0 data unit over AX_W mm
        w = _text_mm(fig, name, FS_TINY,
                     fontweight="bold" if bold else "normal") / AX_W
        key_x += 0.028 + w + 0.045

    fig_text_mm(
        fig, AX2_X, A_ROW1 + A_H1 + 8.4,
        _wrap_mm(
            fig,
            "v1 production model: activity RMSE "
            f"{printed['activity_rmse']:.3f}, $g$ RMSE {printed['g_rmse']:.3f}, "
            f"stay gap {printed['stay_gap']:.3f}: in distribution, nothing "
            "looked broken.",
            AX_W, FS_TINY,
        ),
        fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.35,
    )

    # ------------------------------------------------------------------
    # c  replay stay-probability mismatch
    # ------------------------------------------------------------------
    axC = mm_axes(fig, C_AX_X, A_ROW2, C_AX_W, A_H2)
    panel_label_at(fig, COL1_X, A_ROW2 - 2.4, "c",
                   "Replay stay-probability mismatch")

    # Width budget for the bucket names in the left gutter: from the page
    # margin up to the plot box, less the leader gap. Checked, not assumed.
    lab_budget = C_AX_X - M_SIDE - 2.6
    over = max(_text_mm(fig, s, FS_SMALL) for s in BUCKET_LABEL.values())
    if over > lab_budget:
        raise RuntimeError(
            f"S11c bucket labels need {over:.1f} mm but the gutter is "
            f"{lab_budget:.1f} mm; move C_AX_X right."
        )

    series = (
        ("p_stay_v1", "v1", MOM_L, -0.27),
        ("p_stay_v2_excl_replay", "v2", MOM, 0.0),
        ("replay_p_stay_obs", "observed", INK, 0.27),
    )
    for bi, bucket in enumerate(BUCKET_ORDER):
        sub = chk[chk["source_bucket"] == bucket]
        for col, _lab, color, dy in series:
            vals = sub[col].to_numpy(dtype=float)
            ys = swarm_x(vals, center=bi + dy, width=0.085)
            axC.scatter(vals, ys, s=MS_POINT ** 2, c=color, linewidths=0.0,
                        zorder=3, clip_on=False)
            axC.plot([float(np.mean(vals))], [bi + dy], marker="|", color=color,
                     ms=MS_MEAN * 1.8, mew=LW_LINE, linestyle="none", zorder=4,
                     clip_on=False)
        axC.text(-0.045, bi, BUCKET_LABEL[bucket], ha="right", va="center",
                 fontsize=FS_SMALL, color=INK)
        if bi:
            axC.axhline(bi - 0.5, color=RULE, lw=LW_HAIR, zorder=0)

    coll = chk[chk["source_bucket"] != "stage_b_anchor"]
    m_v1 = float(coll["p_stay_v1"].mean())
    m_v2 = float(coll["p_stay_v2_excl_replay"].mean())
    m_obs = float(coll["replay_p_stay_obs"].mean())
    n_coll = int(len(coll))
    if not np.isclose(m_v1, prov["collective_stay_excl_replay"]["mean_p_stay_v1"]):
        raise RuntimeError("replay check table disagrees with bundle provenance")

    anchors = chk[chk["source_bucket"] == "stage_b_anchor"]
    exact = anchors[anchors["replay_p_stay_obs"] >= 1.0]
    exact = exact[exact["p_stay_v2_excl_replay"] > 0.5]
    x_ex = float(exact["p_stay_v2_excl_replay"].min())
    # Set above the K=0 band, where the right half of the panel is empty; the
    # old (0.60, 2.34) anchor straddled the rule between the last two rows.
    axC.annotate(
        "exact-balance anchors\nretain abstention",
        xy=(x_ex, 2.80), xytext=(0.70, 1.72),
        ha="center", va="top", fontsize=FS_TINY, color=MUTED, linespacing=1.3,
        arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=LW_HAIR,
                        shrinkA=2.0, shrinkB=2.0),
    )

    # Header band above the first row carries the series key and the group
    # means on separate lines: side by side they overlapped at any page width.
    y_key, y_sum = -1.42, -0.92
    axC.set_xlim(-0.02, 1.02)
    axC.set_ylim(3.62, -1.62)
    axC.set_yticks([])
    axC.spines["left"].set_visible(False)
    axC.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axC.set_xlabel("stay probability $p_{\\mathrm{stay}}$")
    trim_spines(axC, y=False)

    kx = 0.0
    for _col, lab, color, _dy in series:
        axC.plot([kx], [y_key], marker="o", color=color, ms=MS_SERIES,
                 linestyle="none", clip_on=False)
        axC.text(kx + 0.030, y_key, lab, ha="left", va="center",
                 fontsize=FS_TINY, color=INK, clip_on=False)
        kx += 0.030 + _text_mm(fig, lab, FS_TINY) / C_AX_W + 0.055

    # Right-aligned on the plot box; its budget is the plot box plus the label
    # gutter, because at this height the gutter is empty.
    axC.text(1.0, y_sum,
             _wrap_mm(fig,
                      f"collective fields ($n$={n_coll}): "
                      f"v1 {m_v1:.3f}, v2 {m_v2:.3f}, observed {m_obs:.3f}",
                      C_AX_W + C_AX_X - M_SIDE, FS_TINY),
             ha="right", va="center", fontsize=FS_TINY, color=MUTED,
             linespacing=1.35, clip_on=False)

    # ------------------------------------------------------------------
    # d  v2 descriptor-based regime gate
    # ------------------------------------------------------------------
    D_X, D_W = COL2_X, 81.0
    axD = mm_panel(fig, D_X, A_ROW2, D_W, A_H2)
    axD.set_xlim(0, D_W)
    axD.set_ylim(A_H2, 0)             # 1 unit == 1 mm, y grows downward
    panel_label_at(fig, COL2_X, A_ROW2 - 2.4, "d",
                   "v2 regime gate (descriptors only)")

    def _box(x0, y0, w, h, face, edge, text, fs=FS_TINY, weight="normal",
             color=INK, ls="solid"):
        axD.add_patch(
            mpatches.FancyBboxPatch(
                (x0, y0), w, h,
                boxstyle="round,pad=0,rounding_size=1.0",
                facecolor=face, edgecolor=edge, lw=LW_THIN, linestyle=ls,
                mutation_aspect=1.0, zorder=1,
            )
        )
        axD.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center",
                 fontsize=fs, color=color, linespacing=1.35, zorder=2,
                 fontweight=weight)

    def _arrow(x0, y0, x1, y1):
        axD.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color="#777777",
                                     lw=LW_THIN, mutation_scale=5,
                                     shrinkA=0, shrinkB=0), zorder=1)

    names = ", ".join(n.replace("abs_z1", "$|z_1|$").replace("_", " ")
                      for n in hyp["stay_feature_names"])
    _box(1.0, 0.5, D_W - 2.0, 6.4, "#F5F5F5", "#8C8C8C",
         f"fixed field descriptors: {names}")
    _arrow(D_W / 2, 6.9, D_W / 2, 9.4)

    _box(9.0, 9.6, D_W - 18.0, 7.6, "#EDEDED", "#5A5A5A",
         "regime test\n"
         f"antipodal balance $\\geq$ {hyp['balance_min']:g}  and  "
         f"$|z_1| \\leq$ {hyp['abs_z1_max']:g}",
         weight="bold")

    _arrow(D_W * 0.30, 17.3, D_W * 0.24, 21.4)
    _arrow(D_W * 0.70, 17.3, D_W * 0.76, 21.4)
    axD.text(D_W * 0.26, 19.6, "yes", ha="right", va="center",
             fontsize=FS_TINY, color=MUTED)
    axD.text(D_W * 0.74, 19.6, "no", ha="left", va="center",
             fontsize=FS_TINY, color=MUTED)

    half = (D_W - 5.0) / 2.0
    _box(1.0, 21.6, half, 10.6, "#EAF1F7", MOM_L,
         "abstention expert\n"
         f"kernel length scale {hyp['length_scale_stay_abstain']:g}")
    _box(1.0 + half + 3.0, 21.6, half, 10.6, "#EAF1F7", MOM,
         "active expert\n"
         f"length scale {hyp['length_scale_stay_active']:g}, "
         f"$k$={hyp['k_neighbors_active']:g}\n"
         "Dirichlet $\\alpha$="
         f"({hyp['alpha_active_stay'][0]:g}, {hyp['alpha_active_stay'][1]:g})")

    _arrow(D_W * 0.24, 32.4, D_W / 2, 35.2)
    _arrow(D_W * 0.76, 32.4, D_W / 2, 35.2)
    _box(D_W * 0.22, 35.4, D_W * 0.56, 5.8, "white", MOM,
         "$\\widehat{p}_{\\mathrm{stay}}$  (holdout gap "
         f"{prov['moments_stay']['moments_stay_gap']:.3f})",
         color=MOM, weight="bold")

    _box(1.0, 43.0, D_W - 2.0, 6.2, "#FBEDED", "#C99A9A",
         "not used as inputs:  coupling $K$  ·  time $t$  ·  "
         "source or panel label", color="#8A3B3A", ls=(0, (2.2, 1.4)))

    return save_fig(fig, "figS11a_moments_surrogate_development")


# ===========================================================================
# S11b — prospective repair and risk calibration
# ===========================================================================
def _page_b() -> Path:
    runs = pd.read_csv(SC2 / "combined_endpoints_per_run.csv")
    det = pd.read_csv(S3A / "pilot_prospective_detail.csv")

    # Integrity check against main Figure 6c: the plotted errors must be the
    # absolute differences between surrogate and LLM endpoints, not any
    # surrogate-vs-surrogate delta.
    for e_col, s_col, l_col in (("eA_v1", "v1_activity", "llm_activity"),
                                ("eA_v2", "v2_activity", "llm_activity"),
                                ("er1_v1", "v1_final_r1", "llm_final_r1"),
                                ("estay_v1", "v1_pred_stay", "obs_stay_teacher")):
        ref = (runs[s_col] - runs[l_col]).abs()
        if not np.allclose(runs[e_col], ref, atol=1e-12):
            raise RuntimeError(f"{e_col} is not |{s_col} - {l_col}|")

    cells = (runs[["panel", "n_agents", "coupling"]]
             .drop_duplicates()
             .sort_values(["panel", "n_agents", "coupling"],
                          ascending=[True, False, True])
             .reset_index(drop=True))

    # y positions: a small gap between the two acquisition panels
    ys, gap_after = [], None
    y = 0.0
    for i, row in cells.iterrows():
        if i and row["panel"] != cells.loc[i - 1, "panel"]:
            # last row of the block that is ending, not the next free slot:
            # taking ``y`` here pushed the separator rule a whole row down, so
            # it was drawn *under* the first v0.2b cell instead of above it.
            gap_after = ys[-1]
            y += 0.9
        ys.append(y)
        y += 1.0
    cells["y"] = ys
    y_max = max(ys)

    fig = new_figure(H_B)

    metrics = (
        ("e", "eA_v1", "eA_v2", "Activity error",
         "$|\\,$activity$_{\\mathrm{surrogate}}-$activity$_{\\mathrm{LLM}}|$"),
        ("f", "estay_v1", "estay_v2", "Stay-probability error",
         "$|\\,\\widehat{p}_{\\mathrm{stay}}-p_{\\mathrm{stay}}^{\\mathrm{obs}}|$"),
        ("g", "er1_v1", "er1_v2", "Final $r_1$ error",
         "$|\\,r_1(T)_{\\mathrm{surrogate}}-r_1(T)_{\\mathrm{LLM}}|$"),
    )

    summary = {}
    for pi, (letter, c1, c2, title, xlab) in enumerate(metrics):
        ax = mm_axes(fig, B_AX_X[pi], B_ROW1, B_AX_W, B_H1)
        panel_label_at(fig, B_AX_X[pi] + B_LET_DX, B_ROW1 - 2.4, letter, title)

        hi = float(max(runs[c1].max(), runs[c2].max()))
        xhi = hi * 1.18
        for _, cell in cells.iterrows():
            sub = runs[(runs["panel"] == cell["panel"])
                       & (runs["n_agents"] == cell["n_agents"])
                       & (runs["coupling"] == cell["coupling"])]
            v1 = sub[c1].to_numpy(dtype=float)
            v2 = sub[c2].to_numpy(dtype=float)
            m1, m2 = float(v1.mean()), float(v2.mean())
            yy = float(cell["y"])
            better = m2 < m1
            ax.annotate(
                "", xy=(m2, yy), xytext=(m1, yy),
                arrowprops=dict(arrowstyle="-|>",
                                color=PASS_GREEN if better else STOP_RED,
                                lw=LW_THIN, mutation_scale=4,
                                shrinkA=0.6, shrinkB=0.6), zorder=2,
            )
            ax.scatter(v1, np.full(v1.size, yy - 0.20), s=MS_POINT ** 2,
                       c=MOM_L, linewidths=0.0, zorder=3)
            ax.scatter(v2, np.full(v2.size, yy + 0.20), s=MS_POINT ** 2,
                       c=MOM, linewidths=0.0, zorder=3)
            ax.plot([m1], [yy], marker="o", mfc="white", mec=MOM_L,
                    ms=MS_SERIES, mew=LW_LINE, linestyle="none", zorder=4)
            ax.plot([m2], [yy], marker="o", mfc=MOM, mec=MOM,
                    ms=MS_SERIES, mew=LW_LINE, linestyle="none", zorder=4)

        if gap_after is not None:
            ax.axhline(gap_after + 0.95, color=RULE, lw=LW_HAIR, zorder=0)

        ax.set_xlim(0.0, xhi)
        ax.set_ylim(y_max + 0.85, B_HEAD_TOP)
        ax.set_xlabel(xlab, fontsize=FS_SMALL)
        ax.tick_params(axis="x", labelsize=FS_TICK)
        if pi == 0:
            ax.set_yticks(cells["y"].to_numpy())
            ax.set_yticklabels(
                [_cell_label(r["n_agents"], r["coupling"])
                 for _, r in cells.iterrows()], fontsize=FS_TICK)
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
        else:
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
        trim_spines(ax, y=False)

        d1, d2 = float(runs[c1].mean()), float(runs[c2].mean())
        summary[letter] = (d1, d2, int(len(runs)))
        n_worse = 0
        for _, cell in cells.iterrows():
            sub = runs[(runs["panel"] == cell["panel"])
                       & (runs["n_agents"] == cell["n_agents"])
                       & (runs["coupling"] == cell["coupling"])]
            if sub[c2].mean() >= sub[c1].mean():
                n_worse += 1
        # Two header lines, not one: set side by side these two strings are
        # wider than the 44 mm panel and ran into each other.
        ax.text(0.0, B_HEAD_ALL,
                f"all runs: v1 {d1:.3f} → v2 {d2:.3f}",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)
        ax.text(0.0, B_HEAD_CELLS,
                f"{len(cells) - n_worse}/{len(cells)} cells improved",
                ha="left", va="baseline", fontsize=FS_TINY,
                color=PASS_GREEN if n_worse == 0 else INK)

        # v1 -> v2 key, on the free right-hand end of the second header line
        if pi == len(metrics) - 1:
            for dx, (face, edge, lab) in enumerate((("white", MOM_L, "v1"),
                                                    (MOM, MOM, "v2"))):
                lw_mm = _text_mm(fig, lab, FS_TINY) + 3.4
                x0 = xhi * (1.0 - (2 - dx) * lw_mm / B_AX_W)
                ax.plot([x0], [B_HEAD_CELLS + 0.16], marker="o", mfc=face,
                        mec=edge, ms=MS_SERIES, mew=LW_LINE, linestyle="none",
                        clip_on=False)
                ax.text(x0 + xhi * 1.6 / B_AX_W, B_HEAD_CELLS, lab,
                        ha="left", va="baseline", fontsize=FS_TINY,
                        color=edge if lab == "v2" else MOM_L,
                        fontweight="bold")

    # group labels + shared row-axis caption, placed once over panel e
    def _y_mm(y: float) -> float:
        """Row coordinate -> mm from the top of the page, for the e-g row."""
        span = (y_max + 0.85) - B_HEAD_TOP
        return B_ROW1 + B_H1 * (y - B_HEAD_TOP) / span

    lab_a = cells.loc[cells["panel"] == cells["panel"].iloc[0]]
    # heading for the row-label column, on its own line above the first row
    fig_text_mm(fig, B_AX_X[0] - 2.0, _y_mm(B_HEAD_NK),
                "$N$, $K$", fontsize=FS_TINY, color=MUTED, ha="right",
                va="baseline")
    fig_text_mm(fig, B_AX_X[0], _y_mm(B_GROUP_DY),
                f"v0.2a: $N$=17, $K>0$ (the {len(lab_a)} cells behind "
                "Supplementary Fig. S22c)",
                fontsize=FS_TINY, color=MUTED, ha="left", va="baseline")
    fig_text_mm(fig, B_AX_X[0], _y_mm(gap_after + 0.95 + 0.46),
                "v0.2b: new $N$, sign($K$), $T$=100",
                fontsize=FS_TINY, color=MUTED, ha="left", va="baseline")

    # ------------------------------------------------------------------
    # h  frozen applicability risk vs observed response error
    # ------------------------------------------------------------------
    axH = mm_axes(fig, B_AX_X[0], B_ROW2, B_AX_H_W, B_H2)
    panel_label_at(fig, B_AX_X[0] + B_LET_DX, B_ROW2 - 2.4, "h",
                   "Fixed applicability risk vs observed response error")

    rho = {}
    for rep in REP_ORDER:
        sub = det[det["representation"] == rep]
        r = sub["R_pre"].to_numpy(dtype=float)
        e = sub["e_tv"].to_numpy(dtype=float)
        rho[rep] = _spearman(r, e)
        axH.scatter(r, e, s=MS_POINT ** 2, c=REP[rep], linewidths=0.0,
                    alpha=0.75, zorder=2 if rep != "moments_m1_m3" else 3)
        order = np.argsort(r)
        r_s, e_s = r[order], e[order]
        thirds = np.array_split(np.arange(r_s.size), 3)
        bx = [float(np.median(r_s[t])) for t in thirds]
        by = [float(np.median(e_s[t])) for t in thirds]
        axH.plot(bx, by, color=REP[rep], lw=LW_TRACE if rep == "moments_m1_m3"
                 else LW_LINE, marker="o", ms=MS_SERIES, zorder=4,
                 alpha=1.0 if rep == "moments_m1_m3" else 0.55)

    axH.set_xlabel("fixed predictive risk $R$ (pre-acquisition)")
    axH.set_ylabel("observed response error $e_{\\mathrm{TV}}$")
    axH.set_ylim(-0.03, 1.03)
    axH.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    # The x spine used to be trimmed to the 0.2 tick, which left the lowest
    # risk bin marker (R = 0.185) floating off the end of the axis. Run the
    # spine to the data instead.
    r_all = det["R_pre"].to_numpy(dtype=float)
    x_lo, x_hi = float(r_all.min()) - 0.015, float(r_all.max()) + 0.015
    axH.set_xlim(x_lo, x_hi)
    trim_spines(axH, x=False)
    axH.spines["bottom"].set_bounds(x_lo, x_hi)
    n_fields = int((det["representation"] == "moments_m1_m3").sum())
    axH.text(0.99, 0.985,
             f"Stage 3A pilot, {n_fields} fixed fields per encoding; "
             "risk fixed before acquisition",
             transform=axH.transAxes, ha="right", va="top",
             fontsize=FS_TINY, color=MUTED)

    # applicability-risk diagnostic table to the right of panel h.
    # This is descriptive: no branch decision turned on it, so no pass/fail
    # verdict is drawn and the word "gate" is not used.
    T_X = B_AX_X[0] + B_AX_H_W + 8.0
    fig_text_mm(fig, T_X, B_ROW2 + 3.2,
                "Applicability-risk diagnostic", fontsize=FS_SMALL, color=INK,
                ha="left", va="baseline", fontweight="bold")
    fig_text_mm(fig, T_X, B_ROW2 + 7.6,
                "rank correlation of risk with error", fontsize=FS_TINY,
                color=MUTED, ha="left", va="baseline")
    rule_mm(fig, T_X, T_X + B_TAB_W, B_ROW2 + 9.6)
    for i, rep in enumerate(REP_ORDER):
        yy = B_ROW2 + 16.4 + 10.0 * i
        fig_text_mm(fig, T_X, yy, REP_SHORT[rep], fontsize=FS_SMALL,
                    color=REP[rep], ha="left", va="baseline",
                    fontweight="bold")
        fig_text_mm(fig, T_X + 24.0, yy,
                    "$\\rho$=" + _fmt_signed(rho[rep]),
                    fontsize=FS_SMALL, color=INK, ha="right", va="baseline")
    fig_text_mm(fig, T_X, B_ROW2 + 48.0,
                _wrap_mm(fig,
                         "Descriptive only; not used for branch decisions. "
                         "No interval, permutation null or minimum effect "
                         "size is attached to these correlations.",
                         B_TAB_W, FS_TINY),
                fontsize=FS_TINY, color=MUTED, ha="left", va="top",
                linespacing=1.4)

    return save_fig(fig, "figS11b_moments_prospective_repair")


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    first = _page_a()
    _page_b()
    return first
