"""Supplementary Figure S15 — Complete fieldwise replay response panel.

Six lettered pages under one figure number, one page per replay stratum::

    S15a  early transient   figS15a_replay_fields_early_transient
    S15b  pre-onset         figS15b_replay_fields_pre_onset
    S15c  post-onset        figS15c_replay_fields_post_onset
    S15d  ordered state     figS15d_replay_fields_ordered_state
    S15e  negative K        figS15e_replay_fields_negative_k
    S15f  high r2           figS15f_replay_fields_high_r2

Every page carries the eight frozen physical fields of its stratum as eight
small multiples. Each small multiple is the compact trinomial triplet for one
field: the empirical (p-, p0, p+) stack under the moments, centers and
intervals *target* encodings, with the three pairwise total-variation
distances drawn as brackets underneath. This is the complete empirical
evidence behind main Fig. 3 — nothing is pooled, nothing is averaged across
fields, and no stratum-level summary is drawn on these pages.

The trinomial responses are **recomputed** here by a single streaming pass
over the frozen replay trace::

    runs/matched_rep_collective_replay/<session>/<stamp>/trace.jsonl

4,608 records = 48 fields x 3 targets x 32 responses. The build asserts the
144 x 32 shape and then checks that the d_TV derived from the recomputed
trinomials reproduces every value in
``analysis/matched_rep_collective/replay_primary/field_pairwise_tv.json``
exactly; a disagreement raises rather than silently plotting private numbers.
Field metadata (physical hash, K, r1, r2) is read from the frozen selection
record and the stratum predicates are parsed out of the lock document, so
nothing on the canvas is hard-coded.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analysis.plot_supplement.style import (
    ACTION,
    ACTION_LABELS,
    FS_BODY,
    FS_LETTER,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_THIN,
    MUTED,
    REP,
    REP_ABBR,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    RUNS,
    apply_style,
    despine,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    rep_tick_colors,
    rule_mm,
    save_fig,
    stacked_trinomial,
    trim_spines,
)

MRC = ROOT / "analysis" / "matched_rep_collective"
FIELDS_JSON = MRC / "replay_fields_v0_1.json"
TV_JSON = MRC / "replay_primary" / "field_pairwise_tv.json"
LOCK_DOC = ROOT / "docs" / "MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md"
REPLAY_ROOT = RUNS / "matched_rep_collective_replay"
TRACE_STAMP = "20260724T114515Z"

N_REPLAYS = 32           # responses per (field, target); asserted, not assumed
HASH_SHOW = 6            # visible hex characters of the physical sha256
TV_TOL = 1e-9            # the frozen d_TV are exact multiples of 1/32

STRATA = [
    "early_transient",
    "pre_onset",
    "post_onset",
    "ordered_state",
    "negative_K",
    "high_r2",
]
STRATUM_NAME = {
    "early_transient": "early transient",
    "pre_onset": "pre-onset",
    "post_onset": "post-onset",
    "ordered_state": "ordered state",
    "negative_K": "negative $K$",
    "high_r2": "high $r_2$",
}
PAGE_TAG = dict(zip(STRATA, "abcdef"))

# (column in field_pairwise_tv.json, first target, second target, label)
PAIRS = (
    ("dTV_MC", 0, 1, "M–C"),
    ("dTV_MI", 0, 2, "M–I"),
    ("dTV_CI", 1, 2, "C–I"),
)

# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm-wide page)
# ---------------------------------------------------------------------------
H_MM = 128.0

N_COL = 4
GRID_LEFT, CELL_W, COL_PITCH = 12.4, 33.0, 42.6   # four field columns
ROW_TOP = (27.0, 78.0)                            # two field rows
AX_H = 27.0                                       # trinomial stack height
BAR_W = 0.70                                      # bar width, in x data units
ID_DY = 5.6                                       # field id baseline above ax
META_DY = 1.9                                     # K / r1 / r2 baseline
BR_GAP, BR_H = 4.4, 10.4                          # d_TV bracket strip

TITLE_Y, TITLE_X2, SUB_Y = 8.8, 34.0, 13.9
HEAD_RULE_Y = 17.4
KEY_LEFT, KEY_PITCH, KEY_SW, KEY_SH = 128.0, 16.0, 3.2, 2.2

NOTE_1_Y = 123.0

# Bracket strip internals (mm inside the strip, top-down)
BR_Y1, BR_Y2 = 2.1, 6.5      # near / far bracket rules
BR_TICK = 0.9                # upstand at each bracket end
BR_TEXT_DY = 2.7             # value baseline below its rule
BR_XPAD = 0.56               # trinomial x limit padding, in bar units


# ---------------------------------------------------------------------------
# Frozen artifacts
# ---------------------------------------------------------------------------
def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_path() -> Path:
    hits = sorted(REPLAY_ROOT.glob(f"*/{TRACE_STAMP}/trace.jsonl"))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one replay trace under {REPLAY_ROOT} at "
            f"{TRACE_STAMP}, found {len(hits)}"
        )
    return hits[0]


def _pretty_predicate(raw: str) -> str:
    """Lock-doc LaTeX -> matplotlib mathtext, every clause kept.

    The predicate is interpolated into a footnote sentence that ends in a full
    stop, so it must never be elided mid-clause: a trailing ellipsis followed
    by that full stop reads as an unfinished sentence.
    """
    s = raw.replace("\\(", "$").replace("\\)", "$")
    s = s.replace("\\lvert", "|").replace("\\rvert", "|")
    s = s.replace("\\ge", "\\geq").replace("\\le", "\\leq")
    clauses = [c.strip() for c in s.split(";") if c.strip()]
    return "; ".join(clauses)


def _lock_predicates() -> dict[str, str]:
    """{stratum: selection predicate} parsed from the frozen lock document."""
    out: dict[str, str] = {}
    for line in LOCK_DOC.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in STRATA:
            out[cells[0]] = _pretty_predicate(cells[2])
    missing = [s for s in STRATA if s not in out]
    if missing:
        raise ValueError(f"lock document has no predicate for {missing}")
    return out


def _read_trace() -> tuple[dict, dict, str]:
    """Group the frozen replay trace into (field, target) trinomials.

    Returns ``(probs, action_words, model)`` where ``probs`` maps
    ``(field_id, target)`` to the (p-, p0, p+) triple recomputed from that
    cell's 32 frozen responses.
    """
    tally: dict[tuple[str, str], Counter] = defaultdict(Counter)
    words: dict[int, Counter] = defaultdict(Counter)
    models: Counter = Counter()
    invalid = 0
    with _trace_path().open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            tally[(rec["field_id"], rec["target_representation"])][
                rec["action_value"]] += 1
            words[rec["action_value"]][rec["action_label"]] += 1
            models[rec["model"]] += 1
            invalid += 0 if rec["valid"] else 1

    if invalid:
        raise ValueError(f"{invalid} invalid replay records in the frozen trace")
    sizes = {sum(c.values()) for c in tally.values()}
    if sizes != {N_REPLAYS}:
        raise ValueError(f"replay group sizes {sorted(sizes)} != {{{N_REPLAYS}}}")
    if set(words) != {-1, 0, 1}:
        raise ValueError(f"unexpected action values {sorted(words)}")
    if len(models) != 1:
        raise ValueError(f"more than one model in the replay trace: {dict(models)}")

    probs = {
        key: tuple(c[v] / N_REPLAYS for v in (-1, 0, 1))
        for key, c in tally.items()
    }
    action_words = {v: words[v].most_common(1)[0][0] for v in (-1, 0, 1)}
    return probs, action_words, models.most_common(1)[0][0]


def _verify(probs: dict, tv_rows: list[dict]) -> None:
    """Recomputed trinomials must reproduce the frozen pairwise d_TV exactly."""
    n_expected = len(tv_rows) * len(REP_ORDER)
    if len(probs) != n_expected:
        raise ValueError(
            f"{len(probs)} (field, target) groups in the trace, expected "
            f"{len(tv_rows)} x {len(REP_ORDER)} = {n_expected}"
        )
    bad = []
    for row in tv_rows:
        p = {r: np.asarray(probs[(row["field_id"], r)]) for r in REP_ORDER}
        for col, i, j, _lab in PAIRS:
            got = 0.5 * float(np.abs(p[REP_ORDER[i]] - p[REP_ORDER[j]]).sum())
            if abs(got - float(row[col])) > TV_TOL:
                bad.append((row["field_id"], col, got, float(row[col])))
    if bad:
        raise ValueError(
            "recomputed trinomials do not reproduce the frozen pairwise d_TV; "
            f"{len(bad)} disagreements, first: {bad[0]}"
        )


# ---------------------------------------------------------------------------
# One small multiple = one frozen physical field
# ---------------------------------------------------------------------------
def _bracket(ax, x0: float, x1: float, y: float, text: str) -> None:
    """A span rule with upstands at both ends and its value beneath."""
    ax.plot([x0, x1], [y, y], color=MUTED, lw=LW_HAIR, solid_capstyle="butt")
    for x in (x0, x1):
        ax.plot([x, x], [y, y - BR_TICK], color=MUTED, lw=LW_HAIR,
                solid_capstyle="butt")
    ax.text((x0 + x1) / 2.0, y + BR_TEXT_DY, text, ha="center", va="baseline",
            fontsize=FS_TINY, color=INK)


def _tv_strip(fig, left: float, top: float, row: dict) -> None:
    """The three pairwise d_TV of one field, positioned under their own bars."""
    ax = mm_panel(fig, left, top, CELL_W, BR_H)
    ax.set_xlim(0.0, CELL_W)
    ax.set_ylim(BR_H, 0.0)          # mm, top-down
    span = 2.0 + 2.0 * BR_XPAD
    xc = [(i + BR_XPAD) / span * CELL_W for i in range(3)]
    for col, i, j, lab in PAIRS:
        y = BR_Y2 if (j - i) == 2 else BR_Y1
        _bracket(ax, xc[i], xc[j], y, f"{lab} {float(row[col]):.2f}")


def _cell(fig, left: float, top: float, meta: dict, row: dict,
          probs: dict, show_y: bool) -> None:
    """Field header, trinomial triplet across the three target encodings."""
    short = meta["field_id"].rsplit("_", 1)[-1]
    src = meta["source_representation"]
    fig_text_mm(fig, left, top - ID_DY, short, ha="left", va="baseline",
                fontsize=FS_BODY, fontweight="bold", color=INK)
    fig_text_mm(fig, left + CELL_W, top - ID_DY, f"source: {REP_SHORT[src]}",
                ha="right", va="baseline", fontsize=FS_TINY, color=REP[src])
    fig_text_mm(fig, left, top - META_DY, meta["physical_hash"][:HASH_SHOW],
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)
    fig_text_mm(
        fig, left + CELL_W, top - META_DY,
        rf"$K={meta['K']:g},\ r_1={meta['r1']:.2f},\ r_2={meta['r2']:.2f}$",
        ha="right", va="baseline", fontsize=FS_TINY, color=MUTED,
    )

    ax = mm_axes(fig, left, top, CELL_W, AX_H)
    stacked_trinomial(ax, {r: probs[(meta["field_id"], r)] for r in REP_ORDER},
                      show_legend=False, annotate=False, bar_width=BAR_W)
    ax.set_xlim(-BR_XPAD, 2.0 + BR_XPAD)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticklabels([REP_ABBR[r] for r in REP_ORDER], fontsize=FS_TICK)
    rep_tick_colors(ax)
    if show_y:
        ax.set_ylabel("response probability", fontsize=FS_SMALL)
    else:
        ax.set_yticklabels([])
    despine(ax)
    trim_spines(ax, x=False)

    _tv_strip(fig, left, top + AX_H + BR_GAP, row)


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------
def _key(fig, action_words: dict[int, str]) -> None:
    """The one shared trinomial key per page, in the header band."""
    cols = (ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"])
    for k, (lab, col, val) in enumerate(zip(ACTION_LABELS, cols, (-1, 0, 1))):
        x = KEY_LEFT + k * KEY_PITCH
        axk = mm_axes(fig, x, TITLE_Y - KEY_SH, KEY_SW, KEY_SH)
        axk.set_xticks([])
        axk.set_yticks([])
        for sp in axk.spines.values():
            sp.set_color(RULE)
            sp.set_linewidth(LW_HAIR)
        axk.set_facecolor(col)
        fig_text_mm(fig, x + KEY_SW + 0.9, TITLE_Y, f"{lab} {action_words[val]}",
                    ha="left", va="baseline", fontsize=FS_TINY, color=INK)


def _header(fig, stratum: str, n_fields: int) -> None:
    fig_text_mm(fig, 4.0, TITLE_Y, STRATUM_NAME[stratum], ha="left",
                va="baseline", fontsize=FS_LETTER, fontweight="bold", color=INK)
    fig_text_mm(fig, TITLE_X2, TITLE_Y,
                f"stratum: all {n_fields} locked physical fields, "
                "three target encodings each",
                ha="left", va="baseline", fontsize=FS_BODY, color=INK)
    fig_text_mm(
        fig, 4.0, SUB_Y,
        f"Stacks: empirical trinomial over {N_REPLAYS} replays per "
        "(field, target); ticks mom / cen / int = moments / centers / "
        r"intervals target. Brackets: pairwise $d_{\mathrm{TV}}$.",
        ha="left", va="baseline", fontsize=FS_SMALL, color=MUTED,
    )
    rule_mm(fig, 4.0, 176.0, HEAD_RULE_Y, color=RULE, lw=LW_THIN)


def _page(stratum: str, metas: list[dict], rows: dict, probs: dict,
          action_words: dict[int, str], model: str, predicate: str) -> Path:
    tag = f"S15{PAGE_TAG[stratum]}"
    order = sorted(metas, key=lambda m: (REP_ORDER.index(m["source_representation"]),
                                         m["field_id"]))
    fig = new_figure(H_MM)
    _header(fig, stratum, len(order))
    _key(fig, action_words)

    for k, meta in enumerate(order):
        r, c = divmod(k, N_COL)
        _cell(fig, GRID_LEFT + c * COL_PITCH, ROW_TOP[r], meta,
              rows[meta["field_id"]], probs, show_y=(c == 0))

    fig_text_mm(
        fig, 4.0, NOTE_1_Y,
        f"Locked selection predicate: {predicate}.",
        ha="left", va="baseline", fontsize=FS_SMALL, color=MUTED,
    )
    return save_fig(fig, f"figS15{PAGE_TAG[stratum]}_replay_fields_{stratum.lower()}")


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    tv_rows = _load_json(TV_JSON)
    fields = _load_json(FIELDS_JSON)["fields"]
    predicates = _lock_predicates()
    probs, action_words, model = _read_trace()
    _verify(probs, tv_rows)

    rows = {r["field_id"]: r for r in tv_rows}
    meta_by_stratum: dict[str, list[dict]] = defaultdict(list)
    for meta in fields:
        if meta["field_id"] not in rows:
            raise ValueError(f"{meta['field_id']} has no frozen pairwise d_TV row")
        meta_by_stratum[meta["stratum"]].append(meta)
    if set(meta_by_stratum) != set(STRATA):
        raise ValueError(f"strata {sorted(meta_by_stratum)} != {sorted(STRATA)}")
    for st, group in meta_by_stratum.items():
        if len(group) > len(ROW_TOP) * N_COL:
            raise ValueError(f"stratum {st} has {len(group)} fields, grid holds "
                             f"{len(ROW_TOP) * N_COL}")

    paths = [
        _page(st, meta_by_stratum[st], rows, probs, action_words, model,
              predicates[st])
        for st in STRATA
    ]
    return paths[0]


if __name__ == "__main__":
    print(build())
