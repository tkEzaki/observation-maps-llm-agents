"""Supplementary Figure S16 — Replay robustness, secondary source effects and
interaction tests.

Nine panels (a–i) reporting every prespecified and secondary inference of the
matched 3×3 replay without elevating the source or the source × target
interaction beyond their frozen status. Main Figure 3e–g carries the headline
versions of panels a, d and the verdict; this figure must never contradict
them, so every observed statistic and every printed p is read from
``analysis/matched_rep_collective/replay_primary/decision.json`` (or the two
companion artifacts) rather than recomputed for display.

Two of the requested nulls were never stored:

* the block-conditioned target null (panel b) and
* the multinomial-deviance null (panel c), plus the two blocked secondary
  nulls (panels e, f).

Those are re-permuted inside this module from the frozen trace, with the
permutation count printed in the panel. The re-permutation is validated
against the freeze before anything is drawn: the observed statistics must
reproduce bit-identically and the recomputed target p must reproduce the
frozen 0.0002 — ``_verify_against_freeze`` raises otherwise. The two secondary
nulls are Monte-Carlo objects with an independent seed, so their tail masses
land a little above the frozen p values; the frozen p is the one printed, and
the re-permuted value is shown beside it in muted type so the panel cannot be
read as a re-analysis.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.natcomm_si_opus.style import (
    ACCENT_RED,
    CAPSIZE,
    DATA_DIR,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    LW_TRACE,
    MS_MEAN,
    MUTED,
    REP,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    VERDICT,
    annotate_cells,
    apply_style,
    cbar_mm,
    despine,
    mm_axes,
    new_figure,
    panel_label,
    save_fig,
    swarm_x,
    trim_spines,
)

PRIMARY = ROOT / "analysis" / "matched_rep_collective" / "replay_primary"
RUN_DIR = (
    ROOT / "runs" / "matched_rep_collective_replay"
    / "matched-rep-collective-3x3-replay-v0.1_1e976b79b71d" / "20260724T114515Z"
)
FIELDS_PATH = ROOT / "analysis" / "matched_rep_collective" / "replay_fields_v0_1.json"

TARGETS = tuple(REP_ORDER)
PAIRS = (("MC", 0, 1), ("MI", 0, 2), ("CI", 1, 2))
PAIR_LABEL = {"MC": "M–C", "MI": "M–I", "CI": "C–I"}
PAIR_COLOR = "#5B6770"          # same neutral ink main Fig. 3d uses for pairs
NULL_FILL = "#C9CDD2"
TAIL_FILL = "#E8B4B3"

N_PERM = 5000                    # matches the frozen inference
PERM_SEED = 20260725
CHUNK = 250

STRATA = (
    "early_transient", "pre_onset", "post_onset",
    "ordered_state", "negative_K", "high_r2",
)
STRATUM_LABEL = {
    "early_transient": "early transient",
    "pre_onset": "pre-onset",
    "post_onset": "post-onset",
    "ordered_state": "ordered state",
    "negative_K": "negative $K$",
    "high_r2": "high $r_2$",
}

# --- canvas plan (mm from top-left) ----------------------------------------
H_MM = 178.0

COL_L = (13.0, 71.0, 129.0)
COL_W = 47.0
LETTER_DY = 1.6

ROW_TOP = (10.0, 62.0, 120.0)
ROW_H = (36.0, 42.0, 42.0)      # row 1 is a narrow spike; it needs less height
HM_W, HM_H = 38.0, 25.5         # panel i heat map + its own colour bar
CB_GAP, CB_W = 1.6, 2.4

# Panel d headroom: the bootstrap means sit at 1.30, their CIs at 1.16, and the
# resample note gets a band of its own above both — it must clear the 1.0 tick
# label and the highest field point rather than share their baseline.
D_TOP = 1.62
D_NOTE_Y = 1.50

# Panels a and b share one axis. Both presented-label nulls live between 0.068
# and 0.115 while the observed statistic is 0.344, so a LINEAR axis wide enough
# to hold the observed rule gave the whole null 13% of the panel — 6 mm holding
# 50 bars 0.13 mm wide, which the white bar edges then painted out, so the null
# read as empty space. A log axis — the one panel c already uses for the same
# reason — gives the null about 27% of the width at 34 bins, so its shape is
# legible while the observed rule and its p stay inside the panel.
NULL_TV_XLIM = (0.060, 0.44)
NULL_TV_TICKS = (0.07, 0.1, 0.2, 0.4)
NULL_TV_TICKLABELS = ("0.07", "0.1", "0.2", "0.4")
NULL_TV_BINS = 34
NULL_TV_TEXT_X = 0.118


# ---------------------------------------------------------------------------
# Frozen trace -> per-field action counts
# ---------------------------------------------------------------------------
def _load_counts() -> tuple[np.ndarray, list[dict]]:
    """(n_field, 2 blocks, 3 targets, 16 samples) action indices + field meta."""
    fields = json.loads(FIELDS_PATH.read_text(encoding="utf-8"))["fields"]
    f_ix = {f["field_id"]: i for i, f in enumerate(fields)}
    t_ix = {t: i for i, t in enumerate(TARGETS)}
    a_ix = {-1: 0, 0: 1, 1: 2}

    bucket: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    with (RUN_DIR / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("valid"):
                continue
            bucket[(
                f_ix[r["field_id"]],
                t_ix[r["target_representation"]],
                int(r["acquisition_block"]),
            )].append(a_ix[int(r["action_value"])])

    n_rep = min(len(v) for v in bucket.values())
    acts = np.zeros((len(fields), 2, 3, n_rep), dtype=np.int8)
    for (f, t, b), vals in bucket.items():
        acts[f, b, t, :] = vals
    return acts, fields


def _counts(order: np.ndarray) -> np.ndarray:
    """(..., n_field, 3 targets, n) action indices -> (..., n_field, 3, 3)."""
    return np.eye(3)[order].sum(axis=-2)


def _mean_pairwise_tv(counts: np.ndarray) -> np.ndarray:
    p = counts / counts.sum(axis=-1, keepdims=True)
    tv = [0.5 * np.abs(p[..., i, :] - p[..., j, :]).sum(-1) for _, i, j in PAIRS]
    return np.stack(tv, axis=-1).mean(axis=(-1, -2))


def _deviance(counts: np.ndarray) -> np.ndarray:
    """Summed likelihood-ratio deviance, separate targets vs pooled trinomial."""
    pooled = counts.sum(axis=-2, keepdims=True)
    p_pool = np.broadcast_to(pooled / pooled.sum(-1, keepdims=True), counts.shape)
    p_row = counts / counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll_sat = np.where(counts > 0, counts * np.log(p_row), 0.0).sum((-1, -2))
        ll_nul = np.where(counts > 0, counts * np.log(p_pool), 0.0).sum((-1, -2))
    return (2.0 * (ll_sat - ll_nul)).sum(-1)


def _p_one_sided(null: np.ndarray, obs: float) -> float:
    return float((np.sum(null >= obs) + 1) / (null.size + 1))


def _target_nulls(acts: np.ndarray, rng: np.random.Generator) -> dict:
    """Re-permute target labels, unrestricted and conditioned on block."""
    n_f, _, _, n_rep = acts.shape
    wide = acts.reshape(n_f, 2 * 3 * n_rep)               # block-major pool
    per_block = acts.reshape(n_f, 2, 3 * n_rep)           # within-block pool
    tv_free, dev_free, tv_block = [], [], []
    for start in range(0, N_PERM, CHUNK):
        m = min(CHUNK, N_PERM - start)
        order = np.argsort(rng.random((m, n_f, wide.shape[1])), axis=-1)
        free = np.take_along_axis(
            np.broadcast_to(wide, (m,) + wide.shape), order, axis=-1
        ).reshape(m, n_f, 3, 2 * n_rep)
        c_free = _counts(free)
        tv_free.append(_mean_pairwise_tv(c_free))
        dev_free.append(_deviance(c_free))

        order_b = np.argsort(rng.random((m, n_f, 2, per_block.shape[2])), axis=-1)
        blocked = np.take_along_axis(
            np.broadcast_to(per_block, (m,) + per_block.shape), order_b, axis=-1
        ).reshape(m, n_f, 2, 3, n_rep)
        tv_block.append(_mean_pairwise_tv(
            _counts(blocked.transpose(0, 1, 3, 2, 4).reshape(m, n_f, 3, 2 * n_rep))
        ))
    return {
        "tv_free": np.concatenate(tv_free),
        "deviance": np.concatenate(dev_free),
        "tv_block": np.concatenate(tv_block),
    }


def _blocked_source_nulls(field_a0, field_dA, meta, rng) -> dict:
    """Permute source within (stratum, K) cells; size-1 cells fall to stratum."""
    src = np.array([meta["source_ix"]]).reshape(-1)
    cells: dict[tuple, list[int]] = defaultdict(list)
    strata: dict[str, list[int]] = defaultdict(list)
    for i, (st, k) in enumerate(zip(meta["stratum"], meta["K"])):
        cells[(st, k)].append(i)
        strata[st].append(i)

    def ssd(values: np.ndarray, labels: np.ndarray) -> float:
        m = np.array([values[labels == s].mean() for s in range(3)])
        return float(((m - m.mean()) ** 2).sum())

    null_a0 = np.empty(N_PERM)
    null_dA = np.empty(N_PERM)
    for it in range(N_PERM):
        lab = src.copy()
        used: set[int] = set()
        for idxs in cells.values():
            if len(idxs) >= 2:
                pick = src[idxs].copy()
                rng.shuffle(pick)
                lab[idxs] = pick
                used.update(idxs)
        left: dict[str, list[int]] = defaultdict(list)
        for i, st in enumerate(meta["stratum"]):
            if i not in used:
                left[st].append(i)
        for idxs in left.values():
            if len(idxs) >= 2:
                pick = src[idxs].copy()
                rng.shuffle(pick)
                lab[idxs] = pick
        null_a0[it] = ssd(field_a0, lab)
        null_dA[it] = ssd(field_dA, lab)
    return {"a0": null_a0, "dA": null_dA, "ssd": ssd, "src": src}


def _verify_against_freeze(observed: dict, recomputed_p: dict, dec: dict) -> None:
    """Refuse to draw if the in-module recomputation left the freeze."""
    tgt = dec["primary_global_tests"]["1_target_main_effect"]
    src = dec["primary_global_tests"]["2_source_main_effect"]
    ixn = dec["primary_global_tests"]["3_source_x_target_interaction"]
    checks = [
        ("mean pairwise TV", observed["tv"], tgt["observed_mean_pairwise_TV"]),
        ("deviance", observed["deviance"],
         tgt["secondary_multinomial_deviance"]["observed"]),
        ("SSD mean a0", observed["ssd_a0"], src["observed_SSD_mean_a0"]),
        ("SSD mean dA", observed["ssd_dA"], ixn["observed_SSD_mean_deltaA_MI"]),
    ]
    for name, got, want in checks:
        if not np.isclose(got, want, rtol=0, atol=1e-9):
            raise RuntimeError(
                f"S16: recomputed {name} = {got!r} does not reproduce the frozen "
                f"{want!r}; refusing to plot a value that is not in the freeze."
            )
    frozen_p = tgt["p_one_sided"]
    for name in ("tv_free", "tv_block", "deviance"):
        if not np.isclose(recomputed_p[name], frozen_p, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"S16: re-permuted target p ({name}) = {recomputed_p[name]!r} does "
                f"not reproduce the frozen {frozen_p!r}; stopping."
            )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _null_axes(ax, null, obs, bins, xlabel, *, log=False, tail=False,
               edge="white"):
    """Grey permutation null + red observed rule, in the main Fig. 3f grammar.

    The rejection tail is drawn on the SAME bin edges as the null. Passing an
    integer bin count to two separate hist() calls would let matplotlib derive
    the edges from each call's own data range, so the tail would be binned
    finely over its narrow span and its bars would neither align with nor be
    comparable in width to the null underneath it. Resolving the edges once and
    reusing them keeps the tail an exact subset of the histogram it sits in.

    ``edge`` exists because the white 0.2 pt separator that reads well on a
    ~1 mm bar *erases* a bar much narrower than that: the stroke is 0.07 mm
    wide and is centred on the bar outline, so once the bars fall below about
    0.3 mm the white ink covers the grey fill and the histogram disappears.
    Panels whose null occupies a narrow slice of the axis pass ``edge=None``
    and rely on the bin edges alone.
    """
    edges = np.histogram_bin_edges(null, bins=bins)
    ax.hist(null, bins=edges, color=NULL_FILL, edgecolor=edge, linewidth=0.2)
    if tail:
        keep = null[null >= obs]
        if keep.size:
            ax.hist(keep, bins=edges, color=TAIL_FILL, edgecolor=edge,
                    linewidth=0.2)
    ax.axvline(float(np.mean(null)), color="#666666", lw=LW_THIN,
               ls=(0, (2.5, 1.8)))
    ax.axvline(obs, color=ACCENT_RED, lw=LW_TRACE, zorder=4)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    despine(ax)
    ax.set_ylim(0, ax.get_ylim()[1])
    return ax


def _source_swarm(ax, values, src_ix, ylabel):
    """16 field values per source, coloured by representation, with the mean."""
    for k, rep in enumerate(TARGETS):
        vals = values[src_ix == k]
        ax.scatter(swarm_x(vals, k, 0.24), vals, s=4.4, c=REP[rep], alpha=0.55,
                   linewidths=0, zorder=2)
    ax.axhline(0.0, color=RULE_ZERO, lw=LW_HAIR, zorder=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([REP_SHORT[r] for r in TARGETS])
    for tick, r in zip(ax.get_xticklabels(), TARGETS):
        tick.set_color(REP[r])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel(ylabel)
    despine(ax)
    return ax


RULE_ZERO = "#9A9A9A"


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    dec = json.loads((PRIMARY / "decision.json").read_text(encoding="utf-8"))
    contrasts = json.loads(
        (PRIMARY / "interaction_contrasts.json").read_text(encoding="utf-8"))
    field_tv = json.loads(
        (PRIMARY / "field_pairwise_tv.json").read_text(encoding="utf-8"))
    stored_null = np.load(DATA_DIR / "replay_null_tv.npy")
    stored_meta = json.loads(
        (DATA_DIR / "replay_null_meta.json").read_text(encoding="utf-8"))

    tgt = dec["primary_global_tests"]["1_target_main_effect"]
    src_test = dec["primary_global_tests"]["2_source_main_effect"]
    ixn_test = dec["primary_global_tests"]["3_source_x_target_interaction"]

    # ---- recomputation from the frozen trace -------------------------------
    acts, fields = _load_counts()
    n_f, _, _, n_rep = acts.shape
    obs_counts = _counts(acts.transpose(0, 2, 1, 3).reshape(n_f, 3, 2 * n_rep))
    p_obs = obs_counts / obs_counts.sum(-1, keepdims=True)
    field_a0 = (-p_obs[..., 0] + p_obs[..., 2]).mean(1)         # target-averaged
    activity = p_obs[..., 0] + p_obs[..., 2]
    field_dA = activity[:, 0] - activity[:, 2]                  # ΔA^{M−I}

    meta = {
        "stratum": [f["stratum"] for f in fields],
        "K": [float(f["K"]) for f in fields],
        "source_ix": np.array(
            [TARGETS.index(f["source_representation"]) for f in fields]),
    }
    rng = np.random.default_rng(PERM_SEED)
    tnull = _target_nulls(acts, rng)
    snull = _blocked_source_nulls(field_a0, field_dA, meta, rng)
    src_ix = snull["src"]

    obs = {
        "tv": float(_mean_pairwise_tv(obs_counts)),
        "deviance": float(_deviance(obs_counts)),
        "ssd_a0": snull["ssd"](field_a0, src_ix),
        "ssd_dA": snull["ssd"](field_dA, src_ix),
    }
    p_re = {
        "tv_free": _p_one_sided(tnull["tv_free"], obs["tv"]),
        "tv_block": _p_one_sided(tnull["tv_block"], obs["tv"]),
        "deviance": _p_one_sided(tnull["deviance"], obs["deviance"]),
        "source": _p_one_sided(snull["a0"], obs["ssd_a0"]),
        "interaction": _p_one_sided(snull["dA"], obs["ssd_dA"]),
    }
    _verify_against_freeze(obs, p_re, dec)

    p_target = tgt["p_one_sided"]
    p_source = src_test["p_one_sided"]
    p_ixn = ixn_test["p_one_sided"]

    fig = new_figure(H_MM)

    # ---- a  primary target-label permutation null --------------------------
    axA = mm_axes(fig, COL_L[0], ROW_TOP[0], COL_W, ROW_H[0])
    panel_label(axA, "a", "Primary presented-label null")
    _null_axes(axA, stored_null, stored_meta["observed_mean_pairwise_TV"],
               NULL_TV_BINS, r"mean pairwise $d_{\mathrm{TV}}$ (log scale)",
               log=True, edge=None)
    axA.set_xlim(*NULL_TV_XLIM)
    axA.set_xticks(list(NULL_TV_TICKS))
    axA.set_xticklabels(list(NULL_TV_TICKLABELS))
    axA.minorticks_off()
    trim_spines(axA, x=False)
    topA = axA.get_ylim()[1]
    axA.text(NULL_TV_TEXT_X, topA * 0.99,
             f"observed = {stored_meta['observed_mean_pairwise_TV']:.3f}\n"
             f"$p$ = {p_target:.4f}\n({N_PERM:,} permutations)",
             fontsize=FS_TINY, color=ACCENT_RED, ha="left", va="top",
             linespacing=1.5)
    axA.text(NULL_TV_TEXT_X, topA * 0.60,
             "stored 5,000-draw null;\n"
             f"null mean = {tgt['null_mean_TV']:.3f}",
             fontsize=FS_TINY, color=MUTED, ha="left", va="top",
             linespacing=1.4)

    # ---- b  block-conditioned target permutation ---------------------------
    axB = mm_axes(fig, COL_L[1], ROW_TOP[0], COL_W, ROW_H[0])
    panel_label(axB, "b", "Block-conditioned null")
    _null_axes(axB, tnull["tv_block"], obs["tv"], NULL_TV_BINS,
               r"mean pairwise $d_{\mathrm{TV}}$ (log scale)",
               log=True, edge=None)
    axB.set_xlim(*NULL_TV_XLIM)
    axB.set_xticks(list(NULL_TV_TICKS))
    axB.set_xticklabels(list(NULL_TV_TICKLABELS))
    axB.minorticks_off()
    trim_spines(axB, x=False)
    topB = axB.get_ylim()[1]
    axB.text(NULL_TV_TEXT_X, topB * 0.99,
             f"observed = {obs['tv']:.3f}\n"
             f"$p$ = {p_re['tv_block']:.4f}\n({N_PERM:,} permutations)",
             fontsize=FS_TINY, color=ACCENT_RED, ha="left", va="top",
             linespacing=1.5)
    axB.text(NULL_TV_TEXT_X, topB * 0.60,
             "presented labels permuted\nwithin field × acquisition\nblock; "
             f"null mean = {float(np.mean(tnull['tv_block'])):.3f}",
             fontsize=FS_TINY, color=MUTED, ha="left", va="top",
             linespacing=1.4)

    # ---- c  multinomial deviance -------------------------------------------
    axC = mm_axes(fig, COL_L[2], ROW_TOP[0], COL_W, ROW_H[0])
    panel_label(axC, "c", "Multinomial deviance")
    dev_obs = tgt["secondary_multinomial_deviance"]["observed"]
    bins_c = np.logspace(np.log10(80.0), np.log10(2600.0), 44)
    _null_axes(axC, tnull["deviance"], dev_obs, bins_c, "deviance", log=True)
    axC.set_xlim(80, 2600)
    axC.set_xticks([100, 300, 1000])
    axC.set_xticklabels(["100", "300", "1,000"])
    axC.minorticks_off()
    trim_spines(axC, x=False)
    topC = axC.get_ylim()[1]
    axC.text(300, topC * 0.99,
             f"observed = {dev_obs:,.0f}\n"
             f"$p$ = {p_re['deviance']:.4f}\n({N_PERM:,} permutations)",
             fontsize=FS_TINY, color=ACCENT_RED, ha="left", va="top",
             linespacing=1.5)
    axC.text(300, topC * 0.66,
             "trinomial response,\n48 fields; null mean = "
             f"{float(np.mean(tnull['deviance'])):.0f}",
             fontsize=FS_TINY, color=MUTED, ha="left", va="top",
             linespacing=1.4)

    # ---- d  field-cluster bootstrap ----------------------------------------
    axD = mm_axes(fig, COL_L[0], ROW_TOP[1], COL_W, ROW_H[1])
    panel_label(axD, "d", "Field-cluster bootstrap")
    boot_keys = {
        "MC": "moments_m1_m3__vs__centers_24_standard",
        "MI": "moments_m1_m3__vs__intervals_24_decimal6",
        "CI": "centers_24_standard__vs__intervals_24_decimal6",
    }
    for i, (tag, _, _) in enumerate(PAIRS):
        vals = np.array([r[f"dTV_{tag}"] for r in field_tv], dtype=float)
        boot = tgt["pairwise_bootstrap"][boot_keys[tag]]
        axD.scatter(swarm_x(vals, i, 0.24), vals, s=4.0, c=PAIR_COLOR, alpha=0.42,
                    linewidths=0, zorder=2)
        m, lo, hi = boot["mean"], boot["ci95"][0], boot["ci95"][1]
        axD.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="D", color=INK,
                     ms=MS_MEAN, capsize=CAPSIZE, elinewidth=LW_LINE,
                     capthick=LW_LINE, zorder=4)
        axD.text(i, 1.30, f"{m:.3f}", ha="center", va="center",
                 fontsize=FS_SMALL, color=INK, fontweight="bold")
        axD.text(i, 1.16, f"[{lo:.3f}, {hi:.3f}]", ha="center", va="center",
                 fontsize=FS_TINY, color=MUTED)
    axD.set_xticks([0, 1, 2])
    axD.set_xticklabels([PAIR_LABEL[t] for t, _, _ in PAIRS])
    axD.set_xlim(-0.55, 2.55)
    axD.set_ylim(0, D_TOP)
    axD.set_yticks([0, 0.5, 1.0])
    axD.set_ylabel(r"$d_{\mathrm{TV}}$")
    axD.set_xlabel("presented pair")
    despine(axD)
    trim_spines(axD, x=False)
    axD.text(-0.45, D_NOTE_Y, f"{boot['n_fields']} fields, {N_PERM:,} resamples",
             fontsize=FS_TINY, color=MUTED, ha="left", va="center")

    # ---- e  source main-effect null ----------------------------------------
    axE = mm_axes(fig, COL_L[1], ROW_TOP[1], COL_W, ROW_H[1])
    panel_label(axE, "e", "Source main-effect null")
    _null_axes(axE, snull["a0"], obs["ssd_a0"], 46,
               r"SSD of source-mean $a_0$", tail=True)
    axE.set_xlim(0, 0.22)
    axE.set_xticks([0, 0.1, 0.2])
    trim_spines(axE, x=False)
    topE = axE.get_ylim()[1]
    axE.text(0.098, topE * 0.99,
             f"observed = {obs['ssd_a0']:.3f}\n$p$ = {p_source:.3f}",
             fontsize=FS_TINY, color=ACCENT_RED, ha="left", va="top",
             linespacing=1.5, fontweight="bold")
    # Panels e and f are reported at equal status: neither omnibus source test
    # met the prespecified threshold, and the caption states that neither is
    # established. Labelling e "suggested" and f "not established" would put a
    # presentation asymmetry on two results that the analysis treats alike, so
    # both carry the same verdict word.
    axE.text(0.098, topE * 0.66, "not established",
             fontsize=FS_SMALL, color=VERDICT["not_established"], ha="left",
             va="top", fontweight="bold")
    axE.text(0.098, topE * 0.40,
             f"{N_PERM:,} stratum × $K$ blocked\npermutations; re-permuted\n"
             f"$p$ = {p_re['source']:.3f}",
             fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.4)

    # ---- f  source × target interaction null -------------------------------
    axF = mm_axes(fig, COL_L[2], ROW_TOP[1], COL_W, ROW_H[1])
    panel_label(axF, "f", "Source × presented null")
    _null_axes(axF, snull["dA"], obs["ssd_dA"], 46,
               r"SSD of source-mean $\Delta A^{M-I}$", tail=True)
    axF.set_xlim(0, 0.036)
    axF.set_xticks([0, 0.01, 0.02, 0.03])
    trim_spines(axF, x=False)
    topF = axF.get_ylim()[1]
    axF.text(0.0158, topF * 0.99,
             f"observed = {obs['ssd_dA']:.4f}\n$p$ = {p_ixn:.3f}",
             fontsize=FS_TINY, color=ACCENT_RED, ha="left", va="top",
             linespacing=1.5, fontweight="bold")
    axF.text(0.0158, topF * 0.66, "not established",
             fontsize=FS_SMALL, color=VERDICT["not_established"], ha="left",
             va="top", fontweight="bold")
    axF.text(0.0158, topF * 0.50,
             f"{N_PERM:,} stratum × $K$ blocked\npermutations; re-permuted\n"
             f"$p$ = {p_re['interaction']:.3f}",
             fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.4)

    # ---- g  source-specific mean signed action -----------------------------
    axG = mm_axes(fig, COL_L[0], ROW_TOP[2], COL_W, ROW_H[2])
    panel_label(axG, "g", "Signed action by source")
    _source_swarm(axG, field_a0, src_ix, r"field mean $a_0$ (presented-averaged)")
    axG.set_ylim(-1.12, 1.42)
    axG.set_yticks([-1, -0.5, 0, 0.5, 1])
    axG.set_xlabel("source encoding")
    trim_spines(axG, x=False)
    for k, rep in enumerate(TARGETS):
        m = src_test["source_means"][rep]["mean_a0"]
        axG.plot([k - 0.30, k + 0.30], [m, m], color=INK, lw=LW_TRACE,
                 solid_capstyle="butt", zorder=5)
        axG.text(k, 1.16, f"{m:+.3f}".replace("-", "−"), ha="center",
                 va="center", fontsize=FS_SMALL, color=INK, fontweight="bold")
    ctr = src_test["contrasts_a0"]["intervals_minus_moments"]
    axG.text(-0.50, 1.36,
             ("intervals $-$ moments $\\Delta a_0$ = "
              f"{ctr['diff']:.3f} [{ctr['ci95'][0]:.3f}, {ctr['ci95'][1]:.3f}]"),
             fontsize=FS_TINY, color=MUTED, ha="left", va="center")

    # ---- h  target contrast by source --------------------------------------
    axH = mm_axes(fig, COL_L[1], ROW_TOP[2], COL_W, ROW_H[2])
    panel_label(axH, "h", r"$\Delta A^{M-I}$ by source")
    dA_by_field = np.array([r["dA_MI"] for r in contrasts], dtype=float)
    src_of_field = np.array(
        [TARGETS.index(r["source"]) for r in contrasts], dtype=int)
    _source_swarm(axH, dA_by_field, src_of_field,
                  r"$\Delta A^{M-I}$ per field")
    axH.set_ylim(-0.62, 1.42)
    axH.set_yticks([-0.5, 0, 0.5, 1.0])
    axH.set_xlabel("source encoding")
    trim_spines(axH, x=False)
    for k, rep in enumerate(TARGETS):
        b = ixn_test["deltaA_MI_by_source"][rep]
        m, lo, hi = b["mean"], b["ci95"][0], b["ci95"][1]
        axH.errorbar([k], [m], yerr=[[m - lo], [hi - m]], fmt="D", color=INK,
                     ms=MS_MEAN, capsize=CAPSIZE, elinewidth=LW_LINE,
                     capthick=LW_LINE, zorder=5)
        axH.text(k, 1.16, f"{m:.3f}", ha="center", va="center",
                 fontsize=FS_SMALL, color=INK, fontweight="bold")
    axH.text(-0.50, 1.36,
             f"effect sizes only; interaction $p$ = {p_ixn:.3f}, not established",
             fontsize=FS_TINY, color=MUTED, ha="left", va="center")

    # ---- i  stratum-specific target effect ---------------------------------
    axI = mm_axes(fig, COL_L[2], ROW_TOP[2], HM_W, HM_H)
    panel_label(axI, "i", "Target effect by stratum")
    grid = np.array([
        [np.mean([r[f"dTV_{tag}"] for r in field_tv if r["stratum"] == st])
         for st in STRATA]
        for tag, _, _ in PAIRS
    ])
    n_per_stratum = min(
        sum(1 for r in field_tv if r["stratum"] == st) for st in STRATA)
    im = axI.imshow(grid, cmap="Reds", vmin=0.0, vmax=0.8, aspect="auto",
                    interpolation="nearest")
    annotate_cells(axI, grid, fmt="{:.2f}", cmap=im.cmap, norm=im.norm,
                   fontsize=FS_TINY)
    axI.set_xticks(np.arange(-0.5, len(STRATA), 1), minor=True)
    axI.set_yticks(np.arange(-0.5, len(PAIRS), 1), minor=True)
    axI.grid(which="minor", color="white", lw=0.6)
    axI.tick_params(which="both", length=0, pad=1.4)
    for sp in axI.spines.values():
        sp.set_visible(False)
    axI.set_xticks(range(len(STRATA)))
    axI.set_yticks(range(len(PAIRS)))
    axI.set_xticklabels([STRATUM_LABEL[s] for s in STRATA], fontsize=FS_TINY,
                        rotation=30, ha="right", rotation_mode="anchor")
    axI.set_yticklabels([PAIR_LABEL[t] for t, _, _ in PAIRS], fontsize=FS_TICK)
    axI.set_ylabel("presented pair", labelpad=1.4)
    cbar_mm(fig, im, COL_L[2] + HM_W + CB_GAP, ROW_TOP[2], CB_W, HM_H,
            ticks=[0, 0.4, 0.8])
    fig.text((COL_L[2] + 1.0) / 180.0, (H_MM - (ROW_TOP[2] + HM_H + 13.0)) / H_MM,
             f"mean $d_{{\\mathrm{{TV}}}}$, {n_per_stratum} fields per stratum;\n"
             "secondary effect-size decomposition,\nnot a prespecified test",
             fontsize=FS_TINY, color=MUTED, ha="left", va="top", linespacing=1.4)

    return save_fig(fig, "figS16_replay_robustness")
