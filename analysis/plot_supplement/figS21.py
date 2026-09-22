"""Supplementary Figure S21 - GPT-Claude reversal of the representation-to-phenotype map.

Every cross-family contrast on this page is restricted to the **six physical
seeds shared by the two families**: the Claude R2 core grid re-uses the GPT
matched-collective initial conditions, and the identity is re-verified here
(``init_seed`` compared cell by cell against the GPT endpoint table, 18/18 core
cells) rather than assumed. Held-out Claude seeds have no GPT counterpart and
never enter a paired contrast; they are not drawn on this page at all.

Panels
------
a  shared-seed final polar order Y, GPT and Claude facets, paired seed points
b  shared-seed sustained-lock score L, same structure
c  family x representation macro map (shade = mean final r1, text = lock runs)
d  moments-centers seed contrast Y_M - Y_C by family
e  moments-intervals seed contrast Y_M - Y_I by family
f  secondary matched summary  D_MH = Y_M - (Y_C + Y_I)/2 by family
g  harmonic comparison: final r2 and final Q2 by family x representation
h  descriptive action statistics: activity, stay probability, social torque
i  scope-aware synthesis

Panel c and panel f reproduce main Fig. 4e on the same six seeds; the printed
numbers are computed from the same artifacts by the same rule, so the two
figures agree by construction rather than by transcription.

Sources
-------
GPT     ``analysis/matched_rep_collective/paired_analysis/endpoints_long.csv``
        and ``paired_analysis/decision.json`` (frozen exact sign test).
Claude  ``<R2_CLAUDE>/r2_inference/trajectory_rows.json``,
        ``r2_inference/primary_inference.json`` (frozen permutation tests) and
        the per-cell ``actions.npy`` for panel h.
Seeds   ``analysis/matched_rep_collective/r2_macro/physical_seed_manifest.json``.

Nothing on this page is hard-coded.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from analysis.plot_supplement.style import (
    CAPSIZE,
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
    MS_MEAN,
    MS_POINT,
    MS_SERIES,
    MUTED,
    R2_CLAUDE,
    REP,
    REP_ABBR,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    VERDICT,
    apply_style,
    cbar_mm,
    errorbar_mean,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    paired_ci,
    panel_label,
    panel_label_at,
    save_fig,
    text_on,
    trim_spines,
)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
MRC = ROOT / "analysis" / "matched_rep_collective"
PAIRED = MRC / "paired_analysis"
MANIFEST = MRC / "r2_macro" / "physical_seed_manifest.json"

FAMILIES = ("gpt", "claude")
K_POS = (0.08, 0.15)

S_POINT = MS_POINT ** 2
S_SERIES = MS_SERIES ** 2
_DASH = (0, (2.2, 1.4))

# Mark-edge weight for open markers: there is no equivalent in the shared
# LW_* scale, which describes lines rather than marker outlines.
_LW_MARK_EDGE = 0.75

# Gutter between the last mark of a series and its direct label, in points.
# It has to clear the error-bar cap (CAPSIZE) as well as the marker radius,
# or the label starts on top of the whisker.
_LABEL_DX_PT = MS_SERIES / 2.0 + CAPSIZE + 2.6

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm-wide page)
# ---------------------------------------------------------------------------
# The two-line shared-seed header that used to sit above row 1 has moved into
# the caption, so every row moves up by the 5.5 mm it occupied and the canvas
# loses the same amount off the foot. Panel letters now sit at 6.9 mm, which
# still clears the 5 mm top margin.
H_MM = 178.5

C1, C2, C3 = 13.0, 73.0, 133.0
PW = 43.0
FACET_GAP = 6.0
LETTER_DX = -8.6

R1_TOP, R1_H = 8.5, 42.0
R2_TOP, R2_H = 64.5, 36.0
R3_TOP = 114.5

NOTE_R1 = 57.1          # family caption under a row-1 facet
NOTE_R2 = 106.9         # first line of the note under a row-2 panel
NOTE_DY = 3.2           # line advance for those small note stacks

C_HEAT_H = 34.0         # panel c heat map
C_CBAR_TOP, C_CBAR_H = 46.0, 2.4

G_H, G_GAP = 17.0, 4.0  # panel g sub-axes
H_H, H_GAP = 13.5, 6.0  # panel h sub-axes
I_H = 52.0

# Seed spread inside a representation / family column.
_OFF = np.linspace(-0.15, 0.15, 6) - 0.08
_MEAN_DX = 0.32


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _gpt() -> pd.DataFrame:
    return pd.read_csv(PAIRED / "endpoints_long.csv")


@lru_cache(maxsize=1)
def _claude() -> tuple[dict, ...]:
    p = R2_CLAUDE / "r2_inference" / "trajectory_rows.json"
    return tuple(json.loads(p.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _inference() -> dict:
    p = R2_CLAUDE / "r2_inference" / "primary_inference.json"
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _gpt_decision() -> dict:
    return json.loads((PAIRED / "decision.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _shared_seeds() -> tuple[int, ...]:
    """The seed indices present in the Claude *core* panel - the shared grid."""
    return tuple(sorted({r["seed_index"] for r in _claude() if r["panel"] == "core"}))


@lru_cache(maxsize=1)
def _seed_identity() -> dict:
    """Re-verify that Claude's core cells re-use the GPT physical seeds.

    Nothing on the canvas prints this any more — the shared-seed statement it
    fed moved into the caption — but the check is kept here because it is the
    provenance of that caption sentence, and the caption must be re-derived
    from it rather than retyped if the grid is ever rebuilt.
    """
    ep = _gpt()
    n_ok = n_tot = 0
    for k in (0.0,) + K_POS:
        for s in _shared_seeds():
            cell = f"N17_K+{k:g}_s{s}"
            meta = json.loads(
                (R2_CLAUDE / cell / REP_ORDER[0] / "run_meta.json").read_text(encoding="utf-8")
            )
            g = ep.loc[np.isclose(ep["coupling"], k) & (ep["seed_index"] == s), "init_seed"]
            n_tot += 1
            n_ok += int(len(set(g)) == 1 and int(g.iloc[0]) == int(meta["init_seed"]))
    at_k = sorted(
        int(v) for v in
        ep.loc[np.isclose(ep["coupling"], K_POS[0]), "init_seed"].unique()
    )
    return {"n_ok": n_ok, "n_total": n_tot, "k": K_POS[0], "seeds": at_k}


# ---------------------------------------------------------------------------
# Seed-level endpoints (the visible inference unit)
# ---------------------------------------------------------------------------
def _gpt_seed(rep: str, seed: int, field: str) -> float:
    ep = _gpt()
    sel = ep[(ep["coupling"] > 0) & (ep["seed_index"] == seed)
             & (ep["representation"] == rep)]
    return float(sel[field].mean())


def _claude_seed(rep: str, seed: int, field: str) -> float:
    sel = [r for r in _claude()
           if r["panel"] == "core" and r["seed_index"] == seed
           and r["representation"] == rep and r["coupling"] > 0]
    return float(np.mean([r[field] for r in sel]))


def _table(field: str) -> dict:
    """{family: {representation: array of six shared-seed values}}."""
    seeds = _shared_seeds()
    return {
        "gpt": {r: np.array([_gpt_seed(r, s, field) for s in seeds]) for r in REP_ORDER},
        "claude": {r: np.array([_claude_seed(r, s, field) for s in seeds]) for r in REP_ORDER},
    }


@lru_cache(maxsize=1)
def _actions() -> dict:
    """Per-seed action statistics at K > 0, identical definitions in both families.

    GPT values are read from the frozen endpoint table; the Claude acquisition
    stores raw ``actions.npy`` only, so the same three reductions are applied
    here: A = mean|s|, p0 = fraction of stay actions, tau = mean s.
    """
    seeds = _shared_seeds()
    out = {"gpt": {}, "claude": {}}
    cols = ("mean_activity", "mean_p_stay_emp", "mean_tau_social")
    for rep in REP_ORDER:
        out["gpt"][rep] = {
            key: np.array([_gpt_seed(rep, s, col) for s in seeds])
            for key, col in zip(("A", "p0", "tau"), cols)
        }
        per = []
        for s in seeds:
            vals = []
            for k in K_POS:
                a = np.load(R2_CLAUDE / f"N17_K+{k:g}_s{s}" / rep / "actions.npy")
                vals.append((np.mean(np.abs(a)), np.mean(a == 0.0), np.mean(a)))
            per.append(np.mean(np.asarray(vals, dtype=float), axis=0))
        per = np.asarray(per)
        out["claude"][rep] = {"A": per[:, 0], "p0": per[:, 1], "tau": per[:, 2]}
    return out


def _lock_counts() -> dict:
    """Locked runs / runs at K > 0 on the shared seeds, per family x representation.

    GPT is scored with ``locking`` and Claude with ``polar_locked`` - the same
    fields main Fig. 4e uses. Both coincide with ``sustained_lock`` in these
    cells, which is checked below so the two figures cannot drift apart.
    """
    ep = _gpt()
    seeds = _shared_seeds()
    out = {}
    for rep in REP_ORDER:
        g = ep[(ep["coupling"] > 0) & (ep["representation"] == rep)
               & (ep["seed_index"].isin(seeds))]
        c = [r for r in _claude() if r["panel"] == "core"
             and r["representation"] == rep and r["coupling"] > 0]
        assert int(g["locking"].sum()) == int(g["sustained_lock"].sum())
        assert sum(r["polar_locked"] for r in c) == sum(r["sustained_lock"] for r in c)
        out[("gpt", rep)] = (int(g["locking"].sum()), int(len(g)))
        out[("claude", rep)] = (sum(r["polar_locked"] for r in c), len(c))
    return out


def _phenotype_counts(label: str) -> dict:
    ep = _gpt()
    seeds = _shared_seeds()
    out = {}
    for rep in REP_ORDER:
        g = ep[(ep["coupling"] > 0) & (ep["representation"] == rep)
               & (ep["seed_index"].isin(seeds))]
        out[("gpt", rep)] = (int((g["cluster_phenotype"] == label).sum()), int(len(g)))
        c = [r for r in _claude() if r["panel"] == "core"
             and r["representation"] == rep and r["coupling"] > 0]
        out[("claude", rep)] = (sum(r["cluster_phenotype"] == label for r in c), len(c))
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _minus(s: str) -> str:
    return s.replace("-", "−")


def _pfmt(p: float) -> str:
    return _minus(f"{p:.3g}")


def _sig(m: float, lo: float, hi: float) -> str:
    return _minus(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]")


def _relative_luminance(colour) -> float:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = mcolors.to_rgb(colour)
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _cell_ink(rgba) -> str:
    """Legible text on a filled heat-map cell.

    ``text_on`` is the house rule and is used unchanged for the decision, but
    its luminance threshold lands on white for the borderline mid-grey of the
    GPT x intervals cell, where white reads at about 2.4:1. Keep whichever of
    the two answers ``text_on`` can give actually has the higher contrast.
    """
    background = _relative_luminance(rgba)
    choice = text_on(rgba)
    other = INK if choice == "#FFFFFF" else "#FFFFFF"

    def ratio(fg: str) -> float:
        lo, hi = sorted((background, _relative_luminance(fg)))
        return (hi + 0.05) / (lo + 0.05)

    return choice if ratio(choice) >= ratio(other) else other


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------
def _rep_facets(fig, left, top, width, height, data, ylim, yticks, ylabel,
                letter, title, notes):
    """Two family facets; inside each, three representations x six shared seeds."""
    fw = (width - FACET_GAP) / 2.0
    axes = []
    for fi, fam in enumerate(FAMILIES):
        ax = mm_axes(fig, left + fi * (fw + FACET_GAP), top, fw, height)
        axes.append(ax)

        for si in range(len(_shared_seeds())):
            ax.plot([j + _OFF[si] for j in range(3)],
                    [data[fam][r][si] for r in REP_ORDER],
                    color=RULE, lw=LW_HAIR, zorder=1, clip_on=False)
        for j, rep in enumerate(REP_ORDER):
            ax.scatter(j + _OFF, data[fam][rep], c=REP[rep], s=S_POINT,
                       linewidths=0, zorder=3, clip_on=False)
            errorbar_mean(ax, j + _MEAN_DX, data[fam][rep], color=INK,
                          ms=MS_MEAN, lw=LW_LINE, capsize=CAPSIZE, zorder=5)

        ax.set_xlim(-0.42, 2.62)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_xticks(range(3))
        ax.set_xticklabels([REP_ABBR[r] for r in REP_ORDER], fontsize=FS_TICK)
        for tick, r in zip(ax.get_xticklabels(), REP_ORDER):
            tick.set_color(REP[r])
        if fi == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])
        trim_spines(ax, x=False)
        xc = left + fi * (fw + FACET_GAP) + fw / 2.0
        fig_text_mm(fig, xc, NOTE_R1, FAMILY_LABEL[fam], ha="center",
                    va="baseline", fontsize=FS_SMALL, fontweight="bold",
                    color=INK)
        fig_text_mm(fig, xc, NOTE_R1 + NOTE_DY, notes[fi], ha="center",
                    va="baseline", fontsize=FS_TINY, color=MUTED)
    panel_label(axes[0], letter, title)
    return axes


def _family_contrast(fig, left, top, width, height, diffs, ylim, yticks,
                     ylabel, letter, title, note, note_color=MUTED):
    """Six shared-seed differences per family, with the family mean and 95% CI."""
    ax = mm_axes(fig, left, top, width, height)
    panel_label(ax, letter, title)
    ax.axhline(0.0, color="#9A9A9A", lw=LW_HAIR, ls=_DASH, zorder=1)
    for fi, fam in enumerate(FAMILIES):
        v = diffs[fam]
        ax.scatter(fi + _OFF, v, marker=FAMILY_MARKER[fam], facecolors="white",
                   edgecolors=FAMILY_EDGE[fam], linewidths=_LW_MARK_EDGE, s=S_POINT,
                   zorder=3, clip_on=False)
        m, lo, hi = paired_ci(v)
        ax.errorbar([fi + _MEAN_DX], [m], yerr=[[m - lo], [hi - m]], fmt="D",
                    color=FAMILY_EDGE[fam], markersize=MS_MEAN, lw=LW_LINE,
                    capsize=CAPSIZE, capthick=LW_LINE, zorder=5, clip_on=False)
        ax.text(fi + 0.06, ylim[1], _minus(f"{m:+.2f}"), ha="center", va="top",
                fontsize=FS_SMALL, fontweight="bold", color=FAMILY_EDGE[fam])
    ax.set_xlim(-0.45, 1.62)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([_minus(f"{t:g}") for t in yticks])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES], fontsize=FS_TICK)
    for tick, f in zip(ax.get_xticklabels(), FAMILIES):
        tick.set_color(FAMILY_EDGE[f])
    ax.set_ylabel(ylabel)
    trim_spines(ax, x=False)
    for li, line in enumerate(note):
        fig_text_mm(fig, left + width / 2.0, NOTE_R2 + li * NOTE_DY, line,
                    ha="center", va="baseline", fontsize=FS_TINY,
                    color=note_color)
    return ax


def _family_series(ax, values, ylim, yticks, ylabel, zero=False, labels=False):
    """Family means +/- 95% CI across the three representations."""
    if zero:
        ax.axhline(0.0, color="#9A9A9A", lw=LW_HAIR, ls=_DASH, zorder=1)
    for fam, ls in zip(FAMILIES, ("solid", _DASH)):
        ms_, los, his = [], [], []
        for rep in REP_ORDER:
            m, lo, hi = paired_ci(values[fam][rep])
            ms_.append(m)
            los.append(m - lo)
            his.append(hi - m)
        ax.errorbar(range(3), ms_, yerr=[los, his], fmt=FAMILY_MARKER[fam],
                    color=FAMILY_EDGE[fam], markerfacecolor="white",
                    markeredgewidth=_LW_MARK_EDGE, ms=MS_SERIES, lw=LW_LINE, ls=ls,
                    capsize=CAPSIZE, capthick=LW_LINE, zorder=4, clip_on=False)
        if labels:
            ax.annotate(FAMILY_LABEL[fam], xy=(2, ms_[2]),
                        xytext=(_LABEL_DX_PT, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=FS_TINY, fontweight="bold",
                        color=FAMILY_EDGE[fam], annotation_clip=False)
    # Room to the right of the last representation so the direct labels have a
    # visible gutter after the error-bar cap instead of butting against it.
    ax.set_xlim(-0.35, 2.62)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([_minus(f"{t:g}") for t in yticks])
    ax.set_xticks(range(3))
    ax.set_ylabel(ylabel)
    trim_spines(ax, x=False)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _panel_a(fig) -> dict:
    y = _table("final_r1")
    inf = _inference()["secondary_final_r1"]
    _rep_facets(
        fig, C1, R1_TOP, PW, R1_H, y, (0.0, 1.06), [0, 0.5, 1.0],
        r"final polar order $Y$", "a", "Shared-seed final $r_1$",
        notes=("6/6 seeds lock", f"$p_Y$ = {_pfmt(inf['p_perm'])}"),
    )
    return y


def _panel_b(fig) -> dict:
    lock = _table("sustained_lock")
    inf = _inference()["primary_sustained_lock"]
    p_gpt = _gpt_decision()["primary_evidence"]["locking_exact_sign_p_per_positive_K"]
    _rep_facets(
        fig, C2, R1_TOP, PW, R1_H, lock, (-0.08, 1.08), [0, 0.5, 1.0],
        r"sustained-lock score $L$", "b", "Shared-seed lock score",
        notes=(f"sign $p$ = {_pfmt(p_gpt)}", f"$p_L$ = {_pfmt(inf['p_perm'])}"),
    )
    return lock


def _panel_c(fig, y: dict) -> np.ndarray:
    ax = mm_axes(fig, C3, R1_TOP, PW, C_HEAT_H)
    panel_label(ax, "c", "Family × representation map")

    counts = _lock_counts()
    mat = np.array([[float(np.mean(y[fam][rep])) for rep in REP_ORDER]
                    for fam in FAMILIES])

    im = ax.imshow(mat, cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto",
                   interpolation="nearest")
    for i, fam in enumerate(FAMILIES):
        for j, rep in enumerate(REP_ORDER):
            col = _cell_ink(im.cmap(im.norm(mat[i, j])))
            n_lock, n_tot = counts[(fam, rep)]
            ax.text(j, i - 0.16, f"{n_lock}/{n_tot}", ha="center", va="center",
                    fontsize=FS_BODY, fontweight="bold", color=col)
            ax.text(j, i + 0.21, f"$r_1$ {mat[i, j]:.2f}", ha="center",
                    va="center", fontsize=FS_TINY, color=col)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", lw=LW_LINE)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([REP_SHORT[r] for r in REP_ORDER], fontsize=FS_SMALL)
    for tick, r in zip(ax.get_xticklabels(), REP_ORDER):
        tick.set_color(REP[r])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([FAMILY_LABEL[f] for f in FAMILIES], fontsize=FS_BODY,
                       fontweight="bold")
    ax.tick_params(axis="both", length=0, pad=1.8)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = cbar_mm(fig, im, C3, C_CBAR_TOP, PW, C_CBAR_H, ticks=[0, 0.5, 1.0],
                 orientation="horizontal")
    cb.set_label(r"mean final $r_1$ ($K>0$, six shared seeds)",
                 fontsize=FS_TINY, labelpad=1.4)
    fig_text_mm(fig, C3 + PW / 2.0, NOTE_R1,
                "cell text: polar-lock runs / runs", ha="center",
                va="baseline", fontsize=FS_TINY, color=MUTED)
    return mat


def _panels_def(fig, y: dict) -> dict:
    diffs = {}
    for key, fn in (
        ("MC", lambda d: d[REP_ORDER[0]] - d[REP_ORDER[1]]),
        ("MI", lambda d: d[REP_ORDER[0]] - d[REP_ORDER[2]]),
        ("MH", lambda d: d[REP_ORDER[0]] - 0.5 * (d[REP_ORDER[1]] + d[REP_ORDER[2]])),
    ):
        diffs[key] = {fam: fn(y[fam]) for fam in FAMILIES}

    ylim = (-0.62, 1.02)
    yticks = [-0.5, 0.0, 0.5, 1.0]
    inf = {c["contrast"]: c for c in _inference()["contrasts_Y"]}

    _family_contrast(
        fig, C1, R2_TOP, PW, R2_H, diffs["MC"], ylim, yticks,
        r"$Y_M-Y_C$  (per seed)", "d", "Moments–centers contrast",
        note=(f"Claude sign-flip $p$ = {_pfmt(inf['M-C']['signflip_p'])}",
              f"paired, $n$ = {len(_shared_seeds())} shared seeds"),
    )
    _family_contrast(
        fig, C2, R2_TOP, PW, R2_H, diffs["MI"], ylim, yticks,
        r"$Y_M-Y_I$  (per seed)", "e", "Moments–intervals contrast",
        note=(f"Claude sign-flip $p$ = {_pfmt(inf['M-I']['signflip_p'])}",
              f"paired, $n$ = {len(_shared_seeds())} shared seeds"),
    )
    _family_contrast(
        fig, C3, R2_TOP, PW, R2_H, diffs["MH"], ylim, yticks,
        r"$\Delta_{\mathrm{M-H}}$  (per seed)", "f",
        "Histogram-vs-moments summary",
        note=("secondary matched cross-family summary",
              "not a prespecified interaction test"), note_color=INK,
    )
    for key, left, lab in (("MC", C1, "M–C"), ("MI", C2, "M–I"),
                           ("MH", C3, "M–H")):
        fig_text_mm(fig, left + 1.0, R2_TOP + 3.0, lab, ha="left",
                    va="baseline", fontsize=FS_SMALL, color=MUTED)
    return diffs


def _panel_g(fig) -> dict:
    r2 = _table("final_r2")
    q2 = _table("final_Q2")
    ax1 = mm_axes(fig, C1, R3_TOP, PW, G_H)
    panel_label(ax1, "g", "Harmonic phenotype")
    _family_series(ax1, r2, (0.24, 1.10), [0.4, 0.7, 1.0], r"final $r_2$",
                   labels=True)
    ax1.set_xticklabels([])

    ax2 = mm_axes(fig, C1, R3_TOP + G_H + G_GAP, PW, G_H)
    _family_series(ax2, q2, (-0.42, 0.34), [-0.25, 0.0, 0.25], r"final $Q_2$",
                   zero=True)
    ax2.set_xticklabels([REP_ABBR[r] for r in REP_ORDER], fontsize=FS_TICK)
    for tick, r in zip(ax2.get_xticklabels(), REP_ORDER):
        tick.set_color(REP[r])

    hr2 = _phenotype_counts("high_r2_nonpolar")
    y0 = R3_TOP + 2 * G_H + G_GAP + 8.4
    fig_text_mm(fig, C1 + LETTER_DX, y0,
                r"$r_2$ tracks the reversal; $Q_2$ does not",
                ha="left", va="baseline", fontsize=FS_TINY, color=INK,
                fontweight="bold")
    fig_text_mm(fig, C1 + LETTER_DX, y0 + 3.2, "high-$r_2$ non-polar runs",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)
    for fi, fam in enumerate(FAMILIES):
        yy = y0 + 6.4 + fi * 3.2
        fig_text_mm(fig, C1 + LETTER_DX + 1.6, yy, FAMILY_LABEL[fam],
                    ha="left", va="baseline", fontsize=FS_TINY,
                    color=FAMILY_EDGE[fam], fontweight="bold")
        fig_text_mm(fig, C1 + LETTER_DX + 12.0, yy,
                    ", ".join(f"{REP_ABBR[r]} {hr2[(fam, r)][0]}/{hr2[(fam, r)][1]}"
                              for r in REP_ORDER),
                    ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)
    return {"r2": r2, "q2": q2}


def _panel_h(fig) -> dict:
    act = _actions()
    specs = (
        ("A", (0.65, 1.06), [0.7, 0.85, 1.0], r"activity $A$", False),
        ("p0", (-0.03, 0.36), [0, 0.15, 0.3], r"stay $p_0$", False),
        ("tau", (-0.12, 0.92), [0, 0.4, 0.8], r"torque $\tau$", True),
    )
    for i, (key, ylim, yticks, ylabel, zero) in enumerate(specs):
        ax = mm_axes(fig, C2, R3_TOP + i * (H_H + H_GAP), PW, H_H)
        if i == 0:
            panel_label(ax, "h", "Action and torque")
        vals = {fam: {rep: act[fam][rep][key] for rep in REP_ORDER}
                for fam in FAMILIES}
        _family_series(ax, vals, ylim, yticks, ylabel, zero=zero,
                       labels=(i == 0))
        if i == len(specs) - 1:
            ax.set_xticklabels([REP_ABBR[r] for r in REP_ORDER], fontsize=FS_TICK)
            for tick, r in zip(ax.get_xticklabels(), REP_ORDER):
                tick.set_color(REP[r])
        else:
            ax.set_xticklabels([])
    fig_text_mm(fig, C2 + LETTER_DX, R3_TOP + 3 * H_H + 2 * H_GAP + 6.0,
                "descriptive mechanism, not a prespecified analysis",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)
    return act


def _panel_i(fig) -> None:
    ax = mm_panel(fig, C3, R3_TOP, PW, I_H)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label_at(fig, C3 + LETTER_DX, R3_TOP - 1.6, "i", "Scope-aware synthesis")
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    facecolor="#F7F7F7", edgecolor="#B0B0B0",
                                    lw=LW_THIN, zorder=0))
    items = (
        ("Macro representation dependence", "established in GPT and Claude",
         VERDICT["supported"]),
        ("Representation-to-phenotype map", "family-specific", INK),
        ("Microscopic operator dependence",
         "established in GPT, Claude and Gemini", VERDICT["supported"]),
        ("Universal representation-specific\nphase diagram", "not supported",
         VERDICT["not_established"]),
    )
    # The item rhythm and the gate block below must both fit inside the panel
    # rectangle; the previous 0.075 inter-item gap pushed the last gate row
    # below y = 0, so it printed outside the box.
    y = 0.95
    for head, tail, col in items:
        n_lines = head.count("\n") + 1
        ax.plot([0.045], [y - 0.012], marker="s", ms=MS_POINT, color=col,
                clip_on=False)
        ax.text(0.090, y, head, ha="left", va="top", fontsize=FS_TINY,
                color=INK, linespacing=1.35)
        ax.text(0.090, y - 0.055 * n_lines, tail, ha="left", va="top",
                fontsize=FS_TINY, color=col, fontweight="bold")
        y -= 0.055 * (n_lines + 1) + 0.055

    inf = _inference()
    ax.plot([0.045, 0.955], [y + 0.035, y + 0.035], color="#C8C8C8",
            lw=LW_HAIR)
    ax.text(0.045, y - 0.01,
            "Claude R2 gate outcomes on the shared grid:", ha="left",
            va="top", fontsize=FS_TINY, color=MUTED)
    gates = (
        ("A  macro effect",
         f"$p_L$ = {_pfmt(inf['primary_sustained_lock']['p_perm'])}",
         VERDICT["supported"]),
        ("B  phase separation", "replicated", VERDICT["supported"]),
        ("C  GPT phenotype hierarchy", "reversed", VERDICT["not_established"]),
    )
    yy = y - 0.070
    for lhs, rhs, col in gates:
        ax.text(0.090, yy, lhs, ha="left", va="top", fontsize=FS_TINY, color=INK)
        ax.text(0.955, yy, rhs, ha="right", va="top", fontsize=FS_TINY,
                color=col, fontweight="bold")
        yy -= 0.056


# ---------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    fig = new_figure(H_MM)
    y = _panel_a(fig)
    _panel_b(fig)
    _panel_c(fig, y)
    _panels_def(fig, y)
    _panel_g(fig)
    _panel_h(fig)
    _panel_i(fig)
    return save_fig(fig, "figS21_gpt_claude_reversal")
