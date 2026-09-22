"""Supplementary Figure S17 — Complete three-family microscopic operator replication.

The complete version of main Figure 4f. Main Fig. 4f carries the compact
M–C / M–I / C–I forest by family plus the scope statement; this figure adds the
coverage that panel omits:

* the frozen preregistered replication gates for all three families (a);
* the validity audit by target encoding, which is the gate main Fig. 4f cannot
  show (b);
* the response trinomials per family and target encoding (c);
* the full 48-field pairwise sensitivity strips, one per family, in one shared
  field order (d, e, f);
* the descriptive cross-family field correspondence (g);
* the pairwise forest **with** the within-block noise floor, the between/within
  ratio and the permutation p per family (h).

Every printed statistic is read from the frozen artifacts
``analysis/matched_rep_collective/r1_primary/decision.json`` and
``analysis/natcomm_figures/data/r1_fieldwise_mean_tv.csv``. The per-field and
per-cell quantities that those artifacts do not store are recomputed from the
three frozen traces and then verified against everything the artifacts *do*
store (``_verify``): the 9 pairwise means, the 3 family means, the 144 fieldwise
means, the 27 cells of each family's 3×3 activity matrix and the 9 invalid
rates must all reproduce to floating-point equality, or nothing is drawn.

Panels dropped, and why
-----------------------
The panel plan opens S17 with three resampling-calibration panels (detection
rate, CI width and between/within ratio at n = 8, 12, 16, "for Claude and
Gemini", documenting why n = 16 was frozen). The only artifact behind them,
``analysis/matched_rep_collective/r1_downsample/downsample_report.json``, is
annotated as an offline **GPT-replay** subsample: no Claude or Gemini
resampling calibration exists, and the report recommends n = 8 rather than the
n = 16 the study froze. A GPT-only subsample cannot stand in for the two
replication families, so those three panels are not built. The remaining panels
are lettered consecutively from **a**; the omission belongs in the caption (the
in-figure footnote that used to state it has been removed).

The freeze itself is on the record rather than unexplained:
``docs/MATCHED_REP_COLLECTIVE_R1_PROTOCOL.md`` ("n_response choice") and
``analysis/matched_rep_collective/r1_downsample/stop_log.json`` both give the
reason as CI stability - n = 8 and n = 12 already reached detection_rate = 1.0
with a noise ratio above 2, and n = 16 was frozen for narrower intervals. The
caption should state that, and state that it was not a Claude- or
Gemini-specific calibration.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from analysis.matched_rep_collective.run_replay_primary_inference import (
    N_PERM,
    PAIRS,
    TARGETS,
)
from analysis.natcomm_si_opus.style import (
    CAPSIZE,
    DATA_DIR,
    FAMILY_EDGE,
    FAMILY_LABEL,
    FAMILY_MARKER,
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
    REP_ABBR,
    REP_ORDER,
    ROOT,
    RULE,
    STOP_RED,
    ACTION,
    ACTION_LABELS,
    apply_style,
    cbar_mm,
    despine,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label,
    rep_tick_colors,
    save_fig,
    stacked_trinomial,
    trim_spines,
)

PRIMARY = ROOT / "analysis" / "matched_rep_collective" / "r1_primary"
FIELDS_JSON = ROOT / "analysis" / "matched_rep_collective" / "replay_fields_v0_1.json"

FAMS = ("gpt", "claude", "gemini")
PAIR_LABEL = {
    ("moments_m1_m3", "centers_24_standard"): "M–C",
    ("moments_m1_m3", "intervals_24_decimal6"): "M–I",
    ("centers_24_standard", "intervals_24_decimal6"): "C–I",
}
PAIR_KEY = {p: f"{p[0]}__vs__{p[1]}" for p in PAIRS}

# Reading order for the field strips: transient -> onset -> ordered, then the
# two off-trajectory strata. Frozen field membership, not a re-selection.
STRATA = (
    "early_transient",
    "pre_onset",
    "post_onset",
    "ordered_state",
    "high_r2",
    "negative_K",
)
STRATUM_LABEL = {
    "early_transient": "early transient",
    "pre_onset": "pre-onset",
    "post_onset": "post-onset",
    "ordered_state": "ordered state",
    "high_r2": r"high $r_2$",
    "negative_K": r"negative $K$",
}

GATE_LABEL = {
    "global_target_effect_p_lt_0.05": r"Target-encoding effect, $p<0.05$",
    "between_exceeds_within": "Between-encoding > within-block",
    "between_minus_within_positive": r"Between $-$ within $>0$, $p<0.05$",
    "at_least_2_of_3_pairs_exceed_noise": r"$\geq$2 of 3 pairs above noise",
    "invalid_rate_ok": "Valid rate and target skew in gate",
}

S_POINT = MS_POINT ** 2
_DASH = (0, (2.2, 1.4))
_TOL = 1e-9

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm-wide page)
# ---------------------------------------------------------------------------
H_MM = 192.0

ROW1_TOP = 10.0
A_L, A_W, A_H = 18.0, 80.0, 36.0
B_L, B_W, B_H = 112.0, 62.0, 36.0

ROW2_TOP = 56.0
C_LEFTS, C_W, C_H = (18.0, 50.0, 82.0), 28.0, 28.0
NOTE_L, NOTE_W = 112.0, 64.0

STRIP_L, STRIP_W, STRIP_H = 18.0, 138.0, 11.0
STRIP_TOPS = (94.0, 110.0, 126.0)
CB_L, CB_W = 160.0, 2.8

ROW4_TOP = 151.0
G_LEFTS, G_W, G_H = (18.0, 47.0, 76.0), 23.0, 22.0
H_L, H_W, H_H = 112.0, 62.0, 30.0
H_GROUP = 5.6                    # rows per family block in h; the slack above
                                 # 3 data rows is the separator's own gap
H_SEP_LIFT = 1.3                 # rows the separator sits above the next family
                                 # heading, so the rule clears its cap height


# ---------------------------------------------------------------------------
# Loading + verification
# ---------------------------------------------------------------------------
def _decision() -> dict:
    return json.loads((PRIMARY / "decision.json").read_text(encoding="utf-8"))


def _fields() -> list[dict]:
    return json.loads(FIELDS_JSON.read_text(encoding="utf-8"))["fields"]


def _trace(run_dir: str) -> list[dict]:
    path = ROOT / run_dir / "trace.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))


def _family_stats(rows: list[dict], field_ids: list[str],
                  field_source: dict[str, str]) -> dict:
    """Per-field trinomials, pairwise TV, target trinomials and validity."""
    by_ft: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(3))
    n_by_t: dict[str, int] = {t: 0 for t in TARGETS}
    bad_by_t: dict[str, int] = {t: 0 for t in TARGETS}

    for r in rows:
        t = r["target_representation"]
        n_by_t[t] += 1
        if not r.get("valid"):
            bad_by_t[t] += 1
            continue
        idx = {-1: 0, 0: 1, 1: 2}[int(r["action_value"])]
        by_ft[(r["field_id"], t)][idx] += 1

    p_field = {
        (f, t): by_ft[(f, t)] / by_ft[(f, t)].sum()
        for f in field_ids for t in TARGETS
    }
    pair_tv = {
        pair: np.array([_tv(p_field[(f, pair[0])], p_field[(f, pair[1])])
                        for f in field_ids])
        for pair in PAIRS
    }
    field_mean = np.mean(np.vstack([pair_tv[p] for p in PAIRS]), axis=0)
    # The freeze averages the per-field trinomial over fields (it does not pool
    # raw counts), so the same convention is used here; the two differ only
    # where invalid responses make the per-field n unequal.
    src_of = field_source
    trinomial = {
        t: np.mean(np.vstack([p_field[(f, t)] for f in field_ids]), axis=0)
        for t in TARGETS
    }
    cell_A = {
        (s, t): float(np.mean([p_field[(f, t)][[0, 2]].sum()
                               for f in field_ids if src_of[f] == s]))
        for s in TARGETS for t in TARGETS
    }
    return {
        "pair_tv": pair_tv,
        "field_mean": field_mean,
        "trinomial": trinomial,
        "cell_A": cell_A,
        "n_by_target": n_by_t,
        "invalid_by_target": bad_by_t,
    }


def _verify(dec: dict, stats: dict, field_ids: list[str]) -> None:
    """Nothing is drawn unless the recomputation reproduces the freeze."""
    csv = pd.read_csv(DATA_DIR / "r1_fieldwise_mean_tv.csv")
    for fam in FAMS:
        frozen = dec["families"][fam]["primary_target"]
        st = stats[fam]

        for pair in PAIRS:
            got = float(st["pair_tv"][pair].mean())
            want = frozen["pairwise_bootstrap"][PAIR_KEY[pair]]["mean"]
            if abs(got - want) > _TOL:
                raise ValueError(f"{fam} {PAIR_KEY[pair]}: {got} != frozen {want}")

        got = float(np.mean([st["pair_tv"][p].mean() for p in PAIRS]))
        want = frozen["mean_pairwise_TV"]
        if abs(got - want) > _TOL:
            raise ValueError(f"{fam} mean pairwise TV: {got} != frozen {want}")

        sub = csv[csv["family"] == fam].set_index("field_id")
        ref = sub.loc[field_ids, "mean_pairwise_TV"].to_numpy()
        if np.max(np.abs(ref - st["field_mean"])) > _TOL:
            raise ValueError(f"{fam}: fieldwise means disagree with the CSV export")
        if fam == "gpt":
            for pair, col in zip(PAIRS, ("dTV_MC", "dTV_MI", "dTV_CI")):
                if np.max(np.abs(sub.loc[field_ids, col].to_numpy()
                                 - st["pair_tv"][pair])) > _TOL:
                    raise ValueError(f"gpt {col} disagrees with the CSV export")

        mat = dec["families"][fam]["matrix_3x3_A"]
        for s in TARGETS:
            for t in TARGETS:
                if abs(st["cell_A"][(s, t)] - mat[s][t]) > _TOL:
                    raise ValueError(f"{fam} A[{s}][{t}] disagrees with the freeze")
        for t in TARGETS:
            drawn = float(st["trinomial"][t][[0, 2]].sum())
            want = float(np.mean([mat[s][t] for s in TARGETS]))
            if abs(drawn - want) > _TOL:
                raise ValueError(f"{fam} activity for {t} disagrees with the freeze")

        audit = dec["families"][fam]["acquisition"]["invalid_audit"]
        for t in TARGETS:
            got = st["invalid_by_target"][t] / st["n_by_target"][t]
            if abs(got - audit["invalid_rate_by_target"][t]) > _TOL:
                raise ValueError(f"{fam} invalid rate for {t} disagrees with the freeze")

    for a, b in (("gpt", "claude"), ("gpt", "gemini"), ("claude", "gemini")):
        r = float(np.corrcoef(stats[a]["field_mean"], stats[b]["field_mean"])[0, 1])
        want = dec["cross_model_secondary"][f"{a}__vs__{b}"]["pearson_fieldwise_mean_TV"]
        if abs(r - want) > _TOL:
            raise ValueError(f"{a}-{b} fieldwise correlation: {r} != frozen {want}")


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _pfmt(p: float) -> str:
    return f"{p:.4f}"


# ---------------------------------------------------------------------------
# a — frozen replication gates
# ---------------------------------------------------------------------------
def _panel_a(fig, dec: dict) -> None:
    ax = mm_panel(fig, A_L, ROW1_TOP, A_W, A_H)
    ax.set_xlim(0, A_W)
    ax.set_ylim(A_H, 0)
    panel_label(ax, "a", "Prespecified replication gates")

    cx = {"gpt": 49.0, "claude": 60.5, "gemini": 72.0}
    cell_w, cell_h = 10.4, 3.6

    for fam in FAMS:
        acq = dec["families"][fam]["acquisition"]
        ax.text(cx[fam], 2.0, FAMILY_LABEL[fam], ha="center", va="center",
                fontsize=FS_SMALL, fontweight="bold", color=FAMILY_EDGE[fam])
        ax.text(cx[fam], 5.0, f"{_thousands(acq['n_trace'])} calls", ha="center",
                va="center", fontsize=FS_TINY, color=MUTED)
    ax.text(0.0, 2.0, "Preregistered R1 gate", ha="left", va="center",
            fontsize=FS_SMALL, fontweight="bold", color=INK)
    ax.text(0.0, 5.0, r"$n=16$ responses per field $\times$ encoding", ha="left",
            va="center", fontsize=FS_TINY, color=MUTED)
    ax.plot([0, A_W], [6.9, 6.9], color=RULE, lw=LW_THIN, clip_on=False)

    gates = list(dec["families"]["gpt"]["replication_gates"].keys())
    for k, gate in enumerate(gates):
        yc = 9.6 + k * 4.3
        ax.text(0.0, yc, GATE_LABEL[gate], ha="left", va="center",
                fontsize=FS_TINY, color=INK)
        for fam in FAMS:
            ok = bool(dec["families"][fam]["replication_gates"][gate])
            col = PASS_GREEN if ok else STOP_RED
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx[fam] - cell_w / 2, yc - cell_h / 2), cell_w, cell_h,
                boxstyle="round,pad=0,rounding_size=0.7", facecolor=col,
                alpha=0.12, edgecolor=col, lw=LW_THIN))
            ax.text(cx[fam], yc, "pass" if ok else "fail", ha="center",
                    va="center", fontsize=FS_TINY, fontweight="bold", color=col)

    y_pass = 9.6 + len(gates) * 4.3 + 0.9
    ax.plot([0, A_W], [y_pass - 3.0, y_pass - 3.0], color=RULE, lw=LW_THIN,
            clip_on=False)
    ax.text(0.0, y_pass, "Replication verdict", ha="left", va="center",
            fontsize=FS_SMALL, fontweight="bold", color=INK)
    for fam in FAMS:
        ok = bool(dec["families"][fam]["replication_pass"])
        ax.text(cx[fam], y_pass, "PASS" if ok else "STOP", ha="center",
                va="center", fontsize=FS_SMALL, fontweight="bold",
                color=PASS_GREEN if ok else STOP_RED)


# ---------------------------------------------------------------------------
# b — validity by target encoding
# ---------------------------------------------------------------------------
def _panel_b(fig, dec: dict, stats: dict) -> None:
    ax = mm_axes(fig, B_L, ROW1_TOP, B_W, B_H)
    panel_label(ax, "b", "Validity by target encoding")

    skew_gate = 0.10
    ticks, labels, colours = [], [], []
    slot = 0.0
    for fam in FAMS:
        audit = dec["families"][fam]["acquisition"]["invalid_audit"]
        # One colour grammar per panel: the bars and their ticks carry the
        # representation colours, so the family headings are set in INK.
        ax.text(0.0, slot, FAMILY_LABEL[fam], ha="left", va="center",
                fontsize=FS_SMALL, fontweight="bold", color=INK)
        ax.text(0.030, slot, f"valid {audit['valid_rate'] * 100:.1f}%", ha="left",
                va="center", fontsize=FS_TINY, color=MUTED)
        slot += 1.0
        for rep in REP_ORDER:
            rate = audit["invalid_rate_by_target"][rep]
            n_bad = stats[fam]["invalid_by_target"][rep]
            n_tot = stats[fam]["n_by_target"][rep]
            ax.barh([slot], [rate], height=0.66, color=REP[rep], edgecolor="none")
            ax.text(rate + 0.0035, slot,
                    f"{n_bad} / {_thousands(n_tot)}", ha="left",
                    va="center", fontsize=FS_TINY, color=INK)
            ticks.append(slot)
            labels.append(REP_ABBR[rep])
            colours.append(REP[rep])
            slot += 1.0
        slot += 0.4

    ax.axvline(skew_gate, color=STOP_RED, lw=LW_HAIR, ls=_DASH, zorder=1)
    ax.text(skew_gate - 0.004, -0.9, "per-encoding skew gate", ha="right",
            va="center", fontsize=FS_TINY, color=STOP_RED)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=FS_TINY)
    for tick, col in zip(ax.get_yticklabels(), colours):
        tick.set_color(col)
    ax.tick_params(axis="y", length=0, pad=1.4)
    ax.set_ylim(slot - 1.0, -1.6)
    ax.set_xlim(0, 0.118)
    ax.set_xticks([0, 0.05, 0.10])
    ax.set_xlabel("invalid response rate")
    despine(ax)
    trim_spines(ax, y=False)


# ---------------------------------------------------------------------------
# c — pooled response trinomials
# ---------------------------------------------------------------------------
def _panel_c(fig, stats: dict) -> None:
    for fi, fam in enumerate(FAMS):
        ax = mm_axes(fig, C_LEFTS[fi], ROW2_TOP, C_W, C_H)
        stacked_trinomial(ax, stats[fam]["trinomial"], show_legend=False)
        ax.set_xticklabels([REP_ABBR[r] for r in REP_ORDER], fontsize=FS_TICK)
        rep_tick_colors(ax)
        ax.set_xlim(-0.62, 2.62)
        ax.tick_params(axis="x", length=0, pad=2.6)
        despine(ax)
        if fi == 0:
            ax.set_ylabel("response fraction")
            panel_label(ax, "c", "Response trinomials by target encoding",
                        dy_mm=6.2)
        else:
            ax.set_yticklabels([])
        fig_text_mm(fig, C_LEFTS[fi] + C_W / 2.0, ROW2_TOP - 1.8,
                    FAMILY_LABEL[fam], ha="center", va="baseline",
                    fontsize=FS_BODY, fontweight="bold", color=FAMILY_EDGE[fam])


def _panel_c_key(fig) -> None:
    ax = mm_panel(fig, NOTE_L, ROW2_TOP, NOTE_W, C_H)
    y = 0.98
    for col, lab, name in zip(
        (ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"]),
        ACTION_LABELS,
        ("retard", "stay", "advance"),
    ):
        ax.add_patch(mpatches.Rectangle((0.0, y - 0.055), 0.075, 0.055,
                                        transform=ax.transAxes, facecolor=col,
                                        edgecolor="white", lw=LW_THIN))
        ax.text(0.095, y - 0.028, f"{lab}  {name}", ha="left", va="center",
                fontsize=FS_TINY, color=INK)
        y -= 0.085
    lines = (
        r"$A=p_-+p_+$ activity,  $a_0=p_+-p_-$ signed action",
        "mom = moments $m_1,m_3$   (M)",
        "cen = centers, 24 bins   (C)",
        "int = intervals, 24 bins, 6 dp   (I)",
        "encoding pairs in d–h: M–C, M–I, C–I",
    )
    y -= 0.045
    for line in lines:
        ax.text(0.0, y, line, ha="left", va="center", fontsize=FS_TINY,
                color=MUTED)
        y -= 0.088


# ---------------------------------------------------------------------------
# d, e, f — fieldwise sensitivity strips
# ---------------------------------------------------------------------------
def _panel_strip(fig, fam: str, letter: str, top: float, stats: dict,
                 order: np.ndarray, bounds: list[int], last: bool):
    ax = mm_axes(fig, STRIP_L, top, STRIP_W, STRIP_H)
    mat = np.vstack(
        [stats[fam]["pair_tv"][p][order] for p in PAIRS]
        + [stats[fam]["field_mean"][order]]
    )
    im = ax.imshow(mat, cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=LW_HAIR)
    ax.tick_params(which="minor", length=0)
    for b in bounds[1:-1]:
        ax.axvline(b - 0.5, color="white", lw=LW_LINE * 2)
    ax.axhline(2.5, color="white", lw=LW_LINE * 2)
    ax.axhline(2.5, color=INK, lw=LW_HAIR)

    ax.set_yticks(range(4))
    ax.set_yticklabels([PAIR_LABEL[p] for p in PAIRS] + ["mean"],
                       fontsize=FS_TINY)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=1.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    panel_label(ax, letter, f"{FAMILY_LABEL[fam]} fieldwise sensitivity",
                title_color=FAMILY_EDGE[fam])

    if last:
        for k in range(len(bounds) - 1):
            xc = (bounds[k] + bounds[k + 1] - 1) / 2.0
            ax.text(xc, -0.14, STRATUM_LABEL[STRATA[k]],
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=FS_TINY, color=MUTED)
        ax.text(0.5, -0.52, "48 physical fields, 8 per stratum, identical order "
                            "in the three strips",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=FS_TINY, color=MUTED)
    return im


# ---------------------------------------------------------------------------
# g — cross-family field correspondence
# ---------------------------------------------------------------------------
def _panel_g(fig, dec: dict, stats: dict) -> None:
    combos = (("gpt", "claude"), ("gpt", "gemini"), ("claude", "gemini"))
    for i, (a, b) in enumerate(combos):
        ax = mm_axes(fig, G_LEFTS[i], ROW4_TOP, G_W, G_H)
        ax.plot([0, 1], [0, 1], color=RULE, lw=LW_HAIR, ls=_DASH, zorder=1)
        ax.scatter(stats[a]["field_mean"], stats[b]["field_mean"], s=S_POINT,
                   facecolors="#5B6770", edgecolors="none", alpha=0.55, zorder=3)
        r = dec["cross_model_secondary"][f"{a}__vs__{b}"]["pearson_fieldwise_mean_TV"]
        ax.text(0.03, 0.97, f"$r$ = {r:.2f}", transform=ax.transAxes, ha="left",
                va="top", fontsize=FS_SMALL, color=INK)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel(rf"{FAMILY_LABEL[a]} mean $d_{{\mathrm{{TV}}}}$",
                      fontsize=FS_TINY, labelpad=1.2)
        ax.set_ylabel(rf"{FAMILY_LABEL[b]} mean $d_{{\mathrm{{TV}}}}$",
                      fontsize=FS_TINY, labelpad=1.2)
        ax.xaxis.label.set_color(FAMILY_EDGE[a])
        ax.yaxis.label.set_color(FAMILY_EDGE[b])
        ax.tick_params(labelsize=FS_TINY)
        despine(ax)
        trim_spines(ax)
        if i == 0:
            panel_label(ax, "g", "Cross-family field correspondence")


# ---------------------------------------------------------------------------
# h — pairwise forest with the noise floor
# ---------------------------------------------------------------------------
def _panel_h(fig, dec: dict) -> None:
    ax = mm_axes(fig, H_L, ROW4_TOP, H_W, H_H)
    panel_label(ax, "h", r"Pairwise $d_{\mathrm{TV}}$ and noise floor")

    row, grp = 1.0, H_GROUP
    ticks, labels = [], []
    for gi, fam in enumerate(FAMS):
        tgt = dec["families"][fam]["primary_target"]
        nf = tgt["noise_floor"]
        y0 = gi * grp
        lo, hi = nf["within_boot"]["ci95"]
        ax.fill_betweenx([y0 + 0.45, y0 + 3 * row + 0.45], lo, hi,
                         color="#D8D8D8", lw=0, zorder=0)
        ax.plot([nf["mean_within_target_block_TV"]] * 2,
                [y0 + 0.45, y0 + 3 * row + 0.45], color="#8C8C8C",
                lw=LW_HAIR, zorder=1)
        ax.text(0.0, y0, FAMILY_LABEL[fam], transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=FS_SMALL, fontweight="bold",
                color=FAMILY_EDGE[fam], clip_on=False)
        ax.text(0.24, y0,
                f"between/within {nf['ratio']:.2f}×  ·  "
                f"$p$ = {_pfmt(tgt['p'])}",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=FS_TINY, color=MUTED, clip_on=False)
        for pi, pair in enumerate(PAIRS):
            yy = y0 + (pi + 1) * row
            boot = tgt["pairwise_bootstrap"][PAIR_KEY[pair]]
            m, lo95, hi95 = boot["mean"], boot["ci95"][0], boot["ci95"][1]
            ax.errorbar([m], [yy], xerr=[[m - lo95], [hi95 - m]],
                        fmt=FAMILY_MARKER[fam], color=FAMILY_EDGE[fam],
                        ecolor=FAMILY_EDGE[fam], markerfacecolor="white",
                        markeredgewidth=0.9, ms=MS_SERIES, capsize=CAPSIZE,
                        elinewidth=LW_LINE, capthick=0.8, zorder=4)
            ticks.append(yy)
            labels.append(PAIR_LABEL[pair])
        if gi < len(FAMS) - 1:
            # The rule gets a band of its own: it sits clear of the foot of
            # this group's noise-floor patch and well above the cap height of
            # the bold family heading below it.
            ysep = y0 + grp - H_SEP_LIFT
            ax.plot([0.0, 1.0], [ysep, ysep], transform=ax.get_yaxis_transform(),
                    color=RULE, lw=LW_HAIR, zorder=0, clip_on=False)

    ax.text(0.0, -2.0, "grey band: within-block noise floor, 95% CI",
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=FS_TINY, color=MUTED, clip_on=False)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=FS_TINY)
    ax.tick_params(axis="y", length=0, pad=1.4)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-2.9, 2 * grp + 3 * row + 0.9)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.6)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel(r"mean $d_{\mathrm{TV}}$ (field-cluster 95% CI)")
    for gx in (0.2, 0.4):
        ax.axvline(gx, color="#EDEDED", lw=LW_HAIR, zorder=0)
    despine(ax)


# ---------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    dec = _decision()
    fields = _fields()

    by_stratum = defaultdict(list)
    for f in fields:
        by_stratum[f["stratum"]].append(f["field_id"])
    if set(by_stratum) != set(STRATA):
        raise ValueError(f"unexpected strata: {sorted(by_stratum)}")
    field_ids = [fid for s in STRATA for fid in sorted(by_stratum[s])]
    bounds, acc = [0], 0
    for s in STRATA:
        acc += len(by_stratum[s])
        bounds.append(acc)

    raw_ids = [f["field_id"] for f in fields]
    field_source = {f["field_id"]: f["source_representation"] for f in fields}
    order = np.array([raw_ids.index(fid) for fid in field_ids])
    stats = {
        fam: _family_stats(_trace(dec["families"][fam]["run_dir"]), raw_ids,
                           field_source)
        for fam in FAMS
    }
    _verify(dec, stats, raw_ids)

    fig = new_figure(H_MM)
    _panel_a(fig, dec)
    _panel_b(fig, dec, stats)
    _panel_c(fig, stats)
    _panel_c_key(fig)
    im = None
    for letter, fam, top in zip("def", FAMS, STRIP_TOPS):
        im = _panel_strip(fig, fam, letter, top, stats, order, bounds,
                          last=(fam == FAMS[-1]))
    cb = cbar_mm(fig, im, CB_L, STRIP_TOPS[0], CB_W,
                 STRIP_TOPS[-1] + STRIP_H - STRIP_TOPS[0], ticks=[0, 0.5, 1.0])
    cb.set_label(r"$d_{\mathrm{TV}}$", fontsize=FS_BODY, labelpad=1.6)
    _panel_g(fig, dec, stats)
    _panel_h(fig, dec)

    return save_fig(fig, "figS17_three_family_replication")
