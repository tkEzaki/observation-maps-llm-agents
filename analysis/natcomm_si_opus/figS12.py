"""Supplementary Figure S12 — Centers: in-distribution compressibility
without collective transport.

Nine panels (a-i) carrying the complete evidence for stopping the centers
collective-surrogate branch. Every printed number is read from the frozen
artifacts under ``analysis/centers_branch/``; nothing is hard-coded.

Panel g repeats the local-support decomposition that main Figure 6d shows
(active feature gap / locally non-compressible / supported) and is read from
the same 36-row diagnosis table, so the two figures cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
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
    MS_MEAN,
    MS_POINT,
    MUTED,
    PASS_GREEN,
    REP,
    REP_LIGHT,
    ROOT,
    RULE,
    STOP_RED,
    annotate_cells,
    apply_style,
    blank,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label_at,
    save_fig,
    trim_spines,
)

CB = ROOT / "analysis" / "centers_branch"

CENT = REP["centers_24_standard"]
CENT_L = REP_LIGHT["centers_24_standard"]
CENT_D = "#8C6200"          # dark end of the centers ramp (this figure only)

# Diagnosis-bucket colours: identical hexes to main Figure 6d/e so the two
# figures read as one decomposition (style.py is shared and must not be edited).
BUCKET_COLOR = {
    "neighbors_active_feature_gap": "#E8A13C",
    "local_noncompressibility": "#C0392B",
    "supported": "#4C9A6E",
}
BUCKET_LABEL = {
    "neighbors_active_feature_gap": "active feature gap",
    "local_noncompressibility": "locally non-compressible",
    "supported": "supported",
}
BUCKET_ORDER = ["neighbors_active_feature_gap", "local_noncompressibility",
                "supported"]

# Reader-facing short names. Display strings only; every value is read from
# the frozen artifacts.
SCHEME_LAB = {
    "leave_one_profile_out": "profile",
    "sparse_realization_holdout": "sparse",
    "offset_group_holdout": "offset",
    "acquisition_block_holdout": "block",
    "source_family_holdout": "source",
}
MODEL_LAB = {
    "pooled_global": "pooled global",
    "peer_specific_global": "peer global",
    "peer_specific_circular_emd": "circular-EMD",
    "peer_specific_canonical_emd": "canonical-EMD",
    "hierarchical_emd_kernel": "hier. EMD",
    "peer_specific_activity_direction": "activity-dir.",
    "descriptor_collapse_mixture": "descr. mixture",
    "pooled_kernel_nn": "kernel-NN",
    "peer16_activity_direction": "activity-dir.",
    "peer16_canonical_emd_plus_discrete": "canon.+discrete",
    "peer16_canonical_emd": "canonical-EMD",
    "peer16_circular_emd": "circular-EMD",
    "peer16_discrete_emd": "discrete-EMD",
    "peer16_global": "peer-16 global",
}
LEVEL_LAB = {
    "row": "rows",
    "physical_field_cluster": "field clusters",
    "profile_family_cluster": "profile families",
}
GATE_LAB = {
    "cluster_bootstrap_oof_positive":
        r"grouped-OOF $\Delta$LL cluster CI $>$ 0",
    "peer16_oof_beats_global":
        "peer-16 OOF beats global",
    "peer16_cluster_bootstrap_positive":
        r"peer-16 cluster $\Delta$LL CI $>$ 0",
    "cent3_dev_beats_global":
        "structured model beats global\non the transfer panel",
    "stay_collapse_reproduced":
        "stay collapse reproduced",
}

# --- canvas plan (mm from the top-left) -------------------------------------
H_MM = 169.5
LET_X = (4.0, 62.0, 120.0)          # panel-letter columns
AX_X = (20.0, 78.0, 136.0)          # plot-box columns
AX_W = 40.0

R1_LET, R1_TOP = 11.4, 13.8
R2_LET, R2_TOP = 58.4, 60.8
R3_LET, R3_TOP = 113.4, 115.8

E_STEP = 1.0          # panel e: group pitch
E_BARW = 0.20         # panel e: bar width — narrow enough to leave a clear
#                       gutter between the two groups for the key


def _trinomial(stay: float, a0: float, activity: float) -> tuple[float, float, float]:
    """(p-, p0, p+) from the frozen (p_stay, a0, A) channel triple."""
    return (activity - a0) / 2.0, stay, (activity + a0) / 2.0


def _stack(ax, x: float, probs, width: float = 0.24) -> None:
    bottom = 0.0
    for v, col in zip(probs, (ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"])):
        ax.bar([x], [v], bottom=[bottom], width=width, color=col,
               edgecolor="white", linewidth=0.5)
        bottom += v


def _key_ax(ax, x: float, y: float, color: str, label: str, dx: float) -> None:
    """Swatch + label inside a panel, in that panel's data coordinates."""
    ax.plot([x], [y], "s", color=color, ms=MS_POINT, clip_on=False, zorder=5)
    ax.text(x + dx, y, label, ha="left", va="center", fontsize=FS_TINY,
            color=INK, clip_on=False, zorder=5)


def _key_mm(fig, x_mm: float, y_mm: float, color: str, label: str) -> None:
    """Swatch + label placed in mm from the top-left of the canvas."""
    fig.text(x_mm / 180.0, (H_MM - y_mm) / H_MM, "■", color=color,
             fontsize=FS_TINY, ha="left", va="baseline")
    fig.text((x_mm + 2.3) / 180.0, (H_MM - y_mm) / H_MM, label, color=INK,
             fontsize=FS_TINY, ha="left", va="baseline")


def build(_out_dir: Path | None = None) -> Path:
    apply_style()

    dec = json.loads((CB / "cent4_final_offline_decision.json").read_text())
    oof = json.loads((CB / "cent4_revision_oof.json").read_text())
    bake = json.loads((CB / "cent4b_bakeoff_summary.json").read_text())
    sup = pd.read_csv(CB / "cent4_final_local_support.csv")
    det = pd.read_csv(CB / "cent3_prospective_detail.csv")

    fig = new_figure(H_MM)

    # ==================================================================
    # a  grouped OOF: structured vs peer-specific global log loss
    # ==================================================================
    by_sm = oof["by_scheme_model"]
    schemes = list(dict.fromkeys(k.split("::")[0] for k in by_sm))
    models = list(dict.fromkeys(k.split("::")[1] for k in by_sm))
    M = np.array([[by_sm[f"{s}::{m}"]["metrics"]["log_loss"] for s in schemes]
                  for m in models])

    axA = mm_axes(fig, AX_X[0], R1_TOP, AX_W, 32.0)
    panel_label_at(fig, LET_X[0], R1_LET, "a", "Grouped out-of-fold log loss")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cent", ["#FFFFFF", CENT_L, CENT, CENT_D])
    norm = mcolors.Normalize(vmin=float(M.min()), vmax=1.30)
    cmap.set_over(CENT_D)
    axA.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    annotate_cells(axA, M, fmt="{:.2f}", cmap=cmap, norm=norm, fontsize=FS_TINY)
    axA.set_xticks(range(len(schemes)))
    axA.set_xticklabels([SCHEME_LAB[s] for s in schemes], fontsize=FS_TICK)
    axA.set_yticks(range(len(models)))
    axA.set_yticklabels([MODEL_LAB[m] for m in models], fontsize=FS_SMALL)
    for t, m in zip(axA.get_yticklabels(), models):
        t.set_color(MUTED if m.endswith("global") else INK)
    axA.tick_params(length=0, pad=1.6)
    for s in axA.spines.values():
        s.set_visible(False)
    for j in range(1, len(schemes)):
        axA.axvline(j - 0.5, color="white", lw=LW_LINE)
    for i in range(1, len(models)):
        axA.axhline(i - 0.5, color="white", lw=LW_LINE)
    axA.axhline(models.index("peer_specific_global") + 0.5, color=INK,
                lw=LW_HAIR)
    axA.set_xlabel("grouped holdout scheme", fontsize=FS_BODY, labelpad=2.4)

    # ==================================================================
    # b  cluster-bootstrap improvement over the peer-specific global model
    # ==================================================================
    axB = mm_axes(fig, AX_X[1], R1_TOP, AX_W, 32.0)
    panel_label_at(fig, LET_X[1], R1_LET, "b",
                   r"Cluster-bootstrap $\Delta$ log loss")
    cbs = dec["cluster_bootstrap"]
    b_models = sorted(cbs, key=lambda m: -cbs[m]["row_bootstrap_mean"])
    tone = dict(zip(b_models, (CENT_D, CENT, CENT_L)))
    shape = dict(zip(b_models, ("o", "s", "D")))
    levels = ["row", "physical_field_cluster", "profile_family_cluster"]
    n_units = {"row": int(oof["delta_vs_peer_global"][b_models[0]]["n_units"])}
    for lev in levels[1:]:
        n_units[lev] = int(cbs[b_models[0]][lev]["n_clusters"])

    step = len(b_models) + 0.9
    centres = []
    for gi, lev in enumerate(levels):
        centres.append(gi * step + (len(b_models) - 1) / 2.0)
        for mi, m in enumerate(b_models):
            if lev == "row":
                blk = oof["delta_vs_peer_global"][m]
                val = float(blk["point_delta_ll"])
            else:
                blk = cbs[m][lev]
                val = float(blk["mean_delta_ll"])
            lo, hi = float(blk["ci_low"]), float(blk["ci_high"])
            y = gi * step + mi
            axB.errorbar([val], [y], xerr=[[max(val - lo, 0.0)],
                                           [max(hi - val, 0.0)]],
                         fmt=shape[m], color=tone[m], markeredgecolor=CENT_D,
                         markeredgewidth=0.3, ms=MS_POINT, elinewidth=LW_LINE,
                         capsize=CAPSIZE, capthick=LW_LINE, clip_on=False,
                         zorder=3)
    axB.axvline(0.0, color="#B0B0B0", lw=LW_HAIR, zorder=0)
    axB.set_yticks(centres)
    axB.set_yticklabels([f"{LEVEL_LAB[lev]}\n($n$={n_units[lev]})"
                         for lev in levels], fontsize=FS_TINY, linespacing=1.3)
    axB.tick_params(axis="y", length=0)
    axB.set_ylim(2 * step + len(b_models) - 0.4, -0.9)
    axB.set_xlim(-0.02, 0.52)
    axB.set_xticks([0.0, 0.2, 0.4])
    axB.set_xlabel(r"$\Delta$ log loss vs peer-specific global"
                   "\n(95% bootstrap CI; resampling unit at left)",
                   fontsize=FS_SMALL, linespacing=1.35)
    axB.spines["left"].set_visible(False)
    trim_spines(axB, y=False)
    for mi, m in enumerate(b_models):
        axB.plot([0.345], [-0.75 + mi * 0.62], shape[m], color=tone[m],
                 markeredgecolor=CENT_D, markeredgewidth=0.3, ms=MS_POINT,
                 clip_on=False, zorder=5)
        axB.text(0.365, -0.75 + mi * 0.62, MODEL_LAB[m], ha="left",
                 va="center", fontsize=FS_TINY, color=INK, zorder=5)

    # ==================================================================
    # c  where in-distribution prediction succeeds
    # ==================================================================
    axC = mm_axes(fig, AX_X[2], R1_TOP, AX_W, 32.0)
    panel_label_at(fig, LET_X[2], R1_LET, "c",
                   "Signed-action and activity components")
    c_models = ["peer_specific_global"] + b_models
    c_tone = {"peer_specific_global": "#9A9A9A", b_models[0]: CENT_D,
              b_models[1]: CENT, b_models[2]: CENT_L}
    metrics = [("a0_sign_acc", r"$a_0$ sign accuracy"),
               ("abs_e_a0", r"signed bias $\overline{|e_{a_0}|}$"),
               ("abs_e_A", r"activity $\overline{|e_A|}$")]
    bar_h = 0.19
    for gi, (key, _lab) in enumerate(metrics):
        for mi, m in enumerate(c_models):
            v = float(oof["delta_vs_peer_global"][m]["model_metrics"][key])
            y = gi + (mi - 1.5) * bar_h
            axC.barh([y], [v], height=bar_h * 0.86, color=c_tone[m],
                     edgecolor="none")
            axC.text(v + 0.012, y, f"{v:.2f}", va="center", ha="left",
                     fontsize=FS_TINY, color=INK)
    axC.set_yticks(range(len(metrics)))
    axC.set_yticklabels([lab for _, lab in metrics], fontsize=FS_SMALL)
    axC.tick_params(axis="y", length=0)
    axC.set_ylim(len(metrics) - 0.42, -0.58)
    # a hair of headroom so the 1.0 tick label does not sit on the canvas
    # edge (the panel box already ends at the right-hand ink limit)
    axC.set_xlim(0, 1.06)
    axC.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axC.set_xlabel(f"value on grouped-OOF rows ($n$={n_units['row']})",
                   fontsize=FS_SMALL)
    axC.spines["left"].set_visible(False)
    trim_spines(axC, y=False)
    for mi, m in enumerate(c_models):
        _key_ax(axC, 0.50, 1.60 + mi * 0.25, c_tone[m], MODEL_LAB[m], dx=0.035)

    # ==================================================================
    # d  collective-like development transfer: predicted vs observed
    # ==================================================================
    axD = mm_axes(fig, AX_X[0], R2_TOP, AX_W, 36.0)
    panel_label_at(fig, LET_X[0], R2_LET, "d", "Collective-like transfer panel")
    axD.plot([0, 1], [0, 1], color="#B0B0B0", lw=LW_HAIR, zorder=0)
    for chan, col, lab in (("p_minus", ACTION["p_minus"], ACTION_LABELS[0]),
                           ("p_stay", ACTION["p_zero"], ACTION_LABELS[1]),
                           ("p_plus", ACTION["p_plus"], ACTION_LABELS[2])):
        axD.plot(det[f"{chan}_pre"], det[f"{chan}_obs"], "o", color=col,
                 ms=MS_POINT, markeredgecolor="#5A5A5A", markeredgewidth=0.25,
                 linestyle="none", zorder=2, label=lab)
    axD.set_xlim(-0.03, 1.03)
    axD.set_ylim(-0.04, 1.24)
    axD.set_xticks([0, 0.5, 1.0])
    axD.set_yticks([0, 0.5, 1.0])
    axD.set_xlabel("fixed pre-acquisition prediction", fontsize=FS_SMALL)
    axD.set_ylabel("observed response share", fontsize=FS_SMALL)
    axD.legend(frameon=False, loc="upper center", ncol=3, fontsize=FS_TINY,
               handlelength=0.8, columnspacing=0.7, handletextpad=0.3,
               borderaxespad=0.0)
    trim_spines(axD)
    e_tv = float(det["e_tv"].mean())
    a0_err = float(det["a0_abs_err"].mean())
    glob = next(r for r in bake["ranking_by_log_loss"]
                if r["model"] == "global_empirical")
    best_struct = min((r for r in bake["ranking_by_log_loss"]
                       if r["model"] != "global_empirical"),
                      key=lambda r: r["log_loss"])
    fig_text_mm(
        fig, AX_X[0], R2_TOP + 42.2,
        f"{len(det)} fixed fields · mean $d_{{TV}}$={e_tv:.2f} ·\n"
        f"mean $|e_{{a_0}}|$={a0_err:.2f}. Bake-off log loss: unstructured\n"
        f"global {glob['log_loss']:.3f} $<$ best structured "
        f"{best_struct['log_loss']:.3f}.",
        fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.4)

    # ==================================================================
    # e  finite-peer abstention mismatch on the +/- unimodal controls
    # ==================================================================
    axE = mm_axes(fig, AX_X[1], R2_TOP, AX_W, 30.0)
    panel_label_at(fig, LET_X[1], R2_LET, "e", "Finite-peer abstention mismatch")
    ctrl_ids = [c for c, v in dec["unimodal_control_diagnosis"].items()
                if v.startswith("neighbors")]
    hat_model = "peer16_activity_direction"
    for fi, cid in enumerate(ctrl_ids):
        row = sup.loc[sup["field_id"] == cid].iloc[0]
        hat = dec["phenotype_peer16_models"][hat_model][cid]
        trip = [
            _trinomial(float(row["obs_stay"]), float(row["obs_a0"]),
                       float(row["obs_A"])),
            _trinomial(float(hat["hat_stay"]), float(hat["hat_a0"]),
                       float(hat["hat_A"])),
            _trinomial(float(row["neighbor_mean_stay"]),
                       float(row["neighbor_mean_a0"]),
                       float(row["neighbor_mean_activity"])),
        ]
        for si, (probs, sub) in enumerate(zip(trip, ("obs.", "model", "nbrs"))):
            x = fi * E_STEP + (si - 1) * 0.28
            _stack(axE, x, probs, width=E_BARW)
            axE.text(x, -0.03, sub, ha="center", va="top", fontsize=FS_TINY,
                     color=INK)
    axE.set_xlim(-0.52, E_STEP + 0.52)
    axE.set_ylim(0, 1.0)
    axE.set_yticks([0, 0.5, 1.0])
    axE.set_xticks([0, E_STEP])
    axE.set_xticklabels(["$+$unimodal\n(polar)", "sign-reversed\nanchor"],
                        fontsize=FS_TINY, linespacing=1.3)
    axE.tick_params(axis="x", length=0, pad=7.0)
    axE.set_ylabel("response share", fontsize=FS_SMALL)
    trim_spines(axE, x=False)
    # key as a swatch column in the gutter between the two field groups,
    # clear of every bar, ordered as the stack is (p_+ on top, p_- at the foot)
    key_x = E_STEP / 2.0 - 0.06
    for lab, col, yy in zip(reversed(ACTION_LABELS),
                            (ACTION["p_plus"], ACTION["p_zero"],
                             ACTION["p_minus"]),
                            (0.80, 0.62, 0.44)):
        axE.plot([key_x], [yy], "s", color=col, ms=MS_POINT,
                 markeredgecolor=RULE, markeredgewidth=LW_HAIR, zorder=5)
        axE.text(key_x + 0.045, yy, lab, ha="left", va="center",
                 fontsize=FS_TINY, color=INK, zorder=5)
    hold = dec["stay_collapse_family_holdout"]
    fig_text_mm(
        fig, AX_X[1], R2_TOP + 37.8,
        f"model = {MODEL_LAB[hat_model]}; nbrs = mean of the "
        f"$k$={int(sup['k_used'].iloc[0])}\n"
        "nearest training fields. Stay-collapse family\n"
        f"hold-out ($n$={hold['n_holdout']}): observed mean "
        f"$p_0$={hold['obs_mean_stay']:.3f}, fraction\n"
        f"with $p_0\\geq$0.7 = {hold['frac_obs_stay_ge_0_7']:.2f}; "
        "all model families predict $p_0\\approx$ 0.",
        fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.4)

    # ==================================================================
    # f  nearest-neighbour response panel, all 36 diagnosis fields
    # ==================================================================
    srt = sup.sort_values("nearest_emd", ascending=True).reset_index(drop=True)
    cols_f = [("nearest_emd", "NN distance\n(EMD)"),
              ("neighbor_response_mean_tv", "neighbour\ndispersion $d_{TV}$"),
              ("obs_stay", "observed\n$p_0$")]
    strip_w, strip_gap = 9.5, 5.75      # gutter wide enough to keep the
    #                                     inner 0 / max ticks apart
    panel_label_at(fig, LET_X[2], R2_LET, "f",
                   "Nearest-neighbour response panel")
    ys = np.arange(len(srt))
    colors = [BUCKET_COLOR[d] for d in srt["diagnosis"]]
    for ci, (col, lab) in enumerate(cols_f):
        ax = mm_axes(fig, AX_X[2] + ci * (strip_w + strip_gap), R2_TOP + 4.4,
                     strip_w, 33.6)
        vals = srt[col].to_numpy(dtype=float)
        ax.barh(ys, vals, height=0.78, color=colors, edgecolor="none")
        ax.set_ylim(len(srt) - 0.5, -0.5)
        vmax = float(np.nanmax(vals))
        ax.set_xlim(0, vmax * 1.04)
        ax.set_xticks([0, vmax])
        ax.set_xticklabels(["0", f"{vmax:.2f}"], fontsize=FS_TINY)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=1.4, pad=1.2)
        ax.set_title(lab, fontsize=FS_TINY, color=INK, pad=2.0,
                     linespacing=1.3)
        trim_spines(ax, y=False)
        if ci == 0:
            ax.set_ylabel(f"{len(srt)} fields, sorted by NN distance",
                          fontsize=FS_TINY, labelpad=2.0)
    for bi, key in enumerate(BUCKET_ORDER[:2]):
        _key_mm(fig, AX_X[2], R2_TOP + 42.0 + bi * 2.6, BUCKET_COLOR[key],
                BUCKET_LABEL[key])

    # ==================================================================
    # g  diagnostic decomposition (same table as main Fig. 6d)
    # ==================================================================
    axG = mm_axes(fig, AX_X[0], R3_TOP, AX_W, 20.0)
    panel_label_at(fig, LET_X[0], R3_LET, "g", "Local-support decomposition")
    counts = srt["diagnosis"].value_counts()
    n_tot = int(len(srt))
    gys = np.arange(len(BUCKET_ORDER), dtype=float)
    for gy, key in zip(gys, BUCKET_ORDER):
        c = int(counts.get(key, 0))
        v = c / n_tot
        if c:
            axG.barh([gy], [v], height=0.42, color=BUCKET_COLOR[key])
        else:
            # a real zero, drawn as a zero-length bar at the origin so it
            # cannot read as a category whose mark went missing
            axG.plot([0.0, 0.0], [gy - 0.21, gy + 0.21],
                     color=BUCKET_COLOR[key], lw=LW_LINE,
                     solid_capstyle="butt", zorder=3)
        axG.text(0.004, gy - 0.42, BUCKET_LABEL[key], va="baseline", ha="left",
                 fontsize=FS_TINY,
                 color=BUCKET_COLOR[key] if c else MUTED)
        axG.text(v + 0.014, gy, f"{v:.0%} ($n$={c})" if c
                 else f"{v:.0%} ($n$={c}) · none", va="center", ha="left",
                 fontsize=FS_TINY, color=INK if c else MUTED)
    axG.set_xlim(0, 1)
    axG.set_ylim(2.62, -0.72)
    axG.set_yticks([])
    axG.set_xticks([0, 0.5, 1.0])
    axG.set_xticklabels(["0", "50%", "100%"])
    axG.set_xlabel(f"share of local diagnoses ($n$={n_tot})", fontsize=FS_SMALL)
    axG.spines["left"].set_visible(False)
    trim_spines(axG, y=False)

    # ==================================================================
    # h  peer-16-only robustness (OOF, then transfer)
    # ==================================================================
    panel_label_at(fig, LET_X[1], R3_LET, "h", "Peer-16-only robustness")
    p16 = dec["peer16_only"]
    rank16 = sorted(p16["acquisition_block_ranking"], key=lambda r: r["log_loss"])
    axH1 = mm_axes(fig, AX_X[1], R3_TOP, AX_W, 17.0)
    yy = np.arange(len(rank16), dtype=float)
    for y, r in zip(yy, rank16):
        base = r["model"].endswith("global")
        axH1.plot([0.70, r["log_loss"]], [y, y], color=RULE, lw=LW_THIN,
                  zorder=1)
        axH1.plot([r["log_loss"]], [y], "o", color=MUTED if base else CENT,
                  ms=MS_MEAN, markerfacecolor="white" if base else CENT,
                  clip_on=False, zorder=3)
        axH1.text(r["log_loss"] + 0.016, y, f"{r['log_loss']:.2f}", va="center",
                  ha="left", fontsize=FS_TINY, color=INK)
    axH1.set_yticks(yy)
    axH1.set_yticklabels([MODEL_LAB[r["model"]] for r in rank16],
                         fontsize=FS_TINY)
    axH1.tick_params(axis="y", length=0)
    axH1.set_ylim(len(rank16) - 0.4, -0.6)
    axH1.set_xlim(0.70, 1.24)
    axH1.set_xticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    axH1.set_xlabel(f"in-distribution OOF log loss ($n$={int(p16['n_train'])})",
                    fontsize=FS_SMALL)
    axH1.spines["left"].set_visible(False)
    trim_spines(axH1, y=False)

    dev = sorted((r for r in dec["cent3_peer16_development_ranking"]
                  if r["model"].startswith("peer16_")),
                 key=lambda r: r["log_loss"])
    axH2 = mm_axes(fig, AX_X[1], R3_TOP + 24.2, AX_W, 13.0)
    yy2 = np.arange(len(dev), dtype=float)
    for y, r in zip(yy2, dev):
        base = r["model"].endswith("global")
        axH2.plot([1.0, r["log_loss"]], [y, y], color=RULE, lw=LW_THIN, zorder=1)
        axH2.plot([r["log_loss"]], [y], "o", color=MUTED if base else CENT,
                  ms=MS_MEAN, markerfacecolor="white" if base else CENT,
                  clip_on=False, zorder=3)
        axH2.text(r["log_loss"] * 1.06, y, f"{r['log_loss']:.2f}", va="center",
                  ha="left", fontsize=FS_TINY, color=INK)
    axH2.set_xscale("log")
    axH2.set_yticks(yy2)
    axH2.set_yticklabels([MODEL_LAB[r["model"]] for r in dev], fontsize=FS_TINY)
    axH2.tick_params(axis="y", length=0)
    axH2.set_ylim(len(dev) - 0.4, -0.6)
    axH2.set_xlim(0.95, 16.0)
    axH2.set_xticks([1, 2, 5, 10])
    axH2.set_xticklabels(["1", "2", "5", "10"], fontsize=FS_TICK)
    axH2.set_xlabel(f"transfer log loss, peer-16 fields ($n$={int(dev[0]['n'])})",
                    fontsize=FS_SMALL)
    axH2.spines["left"].set_visible(False)
    trim_spines(axH2, y=False)

    # ==================================================================
    # i  preregistered stop decision
    # ==================================================================
    I_H = 48.5
    axI = mm_axes(fig, AX_X[2], R3_TOP, AX_W, I_H)
    blank(axI)
    axI.set_xlim(0, AX_W)
    axI.set_ylim(I_H, 0)                     # 1 unit == 1 mm, y downwards
    panel_label_at(fig, LET_X[2], R3_LET, "i", "Preregistered stop decision")
    y = 0.0
    for key, ok in dec["gates"].items():
        lab = GATE_LAB[key]
        h = 5.2 if "\n" in lab else 3.4
        axI.add_patch(mpatches.FancyBboxPatch(
            (0.0, y), AX_W - 6.4, h,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor="#EEF5F0" if ok else "#FAEAEA", edgecolor="none",
            mutation_aspect=1.0))
        axI.text(1.2, y + h / 2.0, lab, ha="left", va="center",
                 fontsize=FS_TINY, color=INK, linespacing=1.25)
        axI.text(AX_W - 5.8, y + h / 2.0, "PASS" if ok else "FAIL", ha="left",
                 va="center", fontsize=FS_TINY, fontweight="bold",
                 color=PASS_GREEN if ok else STOP_RED)
        y += h + 0.6
    axI.annotate("", xy=((AX_W - 6.4) / 2.0, y + 1.7),
                 xytext=((AX_W - 6.4) / 2.0, y - 0.3),
                 arrowprops=dict(arrowstyle="-|>", color="#777777",
                                 lw=LW_LINE, mutation_scale=6,
                                 shrinkA=0, shrinkB=0))
    y += 2.8
    verdicts = [("confirmatory pilot",
                 "GO" if dec["confirmatory_pilot_go"] else "NO-GO"),
                ("peer-16-only pilot",
                 "GO" if dec["peer16_only_pilot_go"] else "NO-GO"),
                ("collective bundle freeze", dec["freeze"]),
                ("centers Stage C", dec["stage_c"]),
                ("paid authorization", dec["paid_authorization"])]
    for lab, val in verdicts:
        axI.text(1.2, y, lab, ha="left", va="top", fontsize=FS_TINY, color=INK)
        axI.text(AX_W - 5.8, y, val, ha="left", va="top", fontsize=FS_TINY,
                 fontweight="bold", color=STOP_RED)
        y += 2.6
    y += 0.7
    axI.add_patch(mpatches.FancyBboxPatch(
        (0.0, y), AX_W, I_H - y - 0.3,
        boxstyle="round,pad=0,rounding_size=1.0",
        facecolor="#FAEAEA", edgecolor=STOP_RED, lw=LW_LINE,
        mutation_aspect=1.0))
    axI.text(AX_W / 2.0, y + 1.3, "STOP", ha="center", va="top",
             fontsize=FS_BODY, fontweight="bold", color=STOP_RED)
    axI.text(AX_W / 2.0, y + 4.2,
             "microscopic archive; no confirmatory\ncollective acquisition",
             ha="center", va="top", fontsize=FS_TINY, color=INK,
             linespacing=1.3)

    return save_fig(fig, "figS12_centers_stop_rule")


if __name__ == "__main__":  # pragma: no cover
    print(build())
