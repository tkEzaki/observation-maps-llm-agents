"""Supplementary Figure S18 - complete same-task-information observation-map control.

Two pages.

* **S18a** - how the three prompt variants are constructed. Every line of every
  variant is regenerated from the frozen 48-field panel through the production
  builder (``circlemap.observation_map_controls``) and checked against the
  ``prompt_sha256`` the acquisition runner stored for each of the 2,304 calls
  before any of it is printed. Panels d-f audit what is held fixed (the six
  moment values) and what the padding condition actually changes (length, context
  volume and the position of the task-relevant block - three things at once).
* **S18b** - the complete response evidence behind main Fig. 5: fieldwise
  pairwise total variation for all 48 fields and all three variant pairs, the
  within-variant block-noise floor, the full per-field action trinomials, the
  actual 5,000-draw variant-label permutation null, the stratum breakdown and
  three prespecified quantile examples.

Nothing on either page is hard-coded: the trinomials, the pairwise distances
and the block noise are recomputed from ``trace.jsonl`` and cross-checked
against the frozen CSV and ``decision.json``; a disagreement raises.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from analysis.natcomm_si_opus.style import (
    ACTION,
    ACTION_LABELS,
    CAPSIZE,
    DATA_DIR,
    FS_BODY,
    FS_LETTER,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    FS_TITLE,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    LW_TRACE,
    MS_MEAN,
    MS_POINT,
    MUTED,
    OMAP,
    OMAP_MID,
    PASS_GREEN,
    ROOT,
    RUNS,
    apply_style,
    cbar_mm,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label,
    panel_label_at,
    rule_mm,
    save_fig,
    swarm_x,
    trim_spines,
)

from circlemap.observation import RelativePhaseHistogram
from circlemap.observation_map_controls import (
    CONTROL_VARIANTS,
    LENGTH_MATCH_TARGET_CHARS,
    build_control_prompt,
)

PRIMARY = ROOT / "analysis" / "matched_rep_collective" / "omap_primary"
FIELDS_JSON = ROOT / "analysis" / "matched_rep_collective" / "replay_fields_v0_1.json"
OMAP_RUN = (
    RUNS / "matched_rep_collective_omap"
    / "matched-rep-omap-control-v0.1_gpt-5.4-mini" / "20260724T161234Z"
)

PT_MM = 25.4 / 72.0
LS_MONO = 1.30          # leading of the verbatim prompt blocks
LS_PARA = 1.42          # leading of the wrapped prose notes
X_LEFT = 4.4            # outer ink margin == panel_label letter column
X_RIGHT = 176.0         # right-hand ink limit

VARIANTS = list(CONTROL_VARIANTS)
V_SHORT = {
    "moments_original": "original",
    "moments_reformatted": "reformatted",
    "moments_length_matched": "length-matched",
}
V_TICK = {
    "moments_original": "orig",
    "moments_reformatted": "ref",
    "moments_length_matched": "pad",
}
ACT_ORDER = ("retard", "stay", "advance")     # matches ACTION p-, p0, p+
ACT_COLORS = (ACTION["p_minus"], ACTION["p_zero"], ACTION["p_plus"])

PAIRS = (
    ("dTV_orig_reformat", "original–reformatted", OMAP["moments_reformatted"]),
    ("dTV_orig_pad", "original–padded", OMAP["moments_length_matched"]),
    ("dTV_reformat_pad", "reformatted–padded", OMAP_MID),
)

STRATA = ("early_transient", "pre_onset", "post_onset",
          "ordered_state", "negative_K", "high_r2")
STRATUM_LABEL = {
    "early_transient": "early\ntransient",
    "pre_onset": "pre-\nonset",
    "post_onset": "post-\nonset",
    "ordered_state": "ordered\nstate",
    "negative_K": "negative\n$K$",
    "high_r2": "high\n$r_2$",
}

TV_CMAP = LinearSegmentedColormap.from_list(
    "omap_tv", ["#FFFFFF", OMAP["moments_reformatted"], OMAP["moments_length_matched"]]
)
DTV = r"$d_{\mathrm{TV}}$"


# ---------------------------------------------------------------------------
# Text helpers (measured, so nothing can silently run off a 180 mm canvas)
# ---------------------------------------------------------------------------
def _pitch(fontsize: float, lead: float = LS_MONO) -> float:
    return fontsize * lead * PT_MM


def _text_w_mm(fig, s: str, fontsize: float, **kw) -> float:
    art = fig.text(0.0, 0.0, s, fontsize=fontsize, **kw)
    bb = art.get_window_extent(renderer=fig.canvas.get_renderer())
    art.remove()
    return float(bb.width) / float(fig.dpi) * 25.4


def _wrap(fig, text: str, max_mm: float, fontsize: float, **kw) -> list[str]:
    """Word-wrap ``text`` to a measured width; display-only, no characters added."""
    out: list[str] = []
    cur = ""
    for word in text.split(" "):
        trial = word if not cur else f"{cur} {word}"
        if not cur or _text_w_mm(fig, trial, fontsize, **kw) <= max_mm:
            cur = trial
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def _para(fig, x_mm: float, y_mm: float, text: str, max_mm: float,
          fontsize: float = FS_TINY, color: str = MUTED) -> float:
    """Draw a wrapped prose note; return the baseline just past its last line."""
    lines = _wrap(fig, text, max_mm, fontsize)
    pitch = _pitch(fontsize, LS_PARA)
    for i, line in enumerate(lines):
        fig_text_mm(fig, x_mm, y_mm + i * pitch, line, ha="left", va="baseline",
                    fontsize=fontsize, color=color)
    return y_mm + len(lines) * pitch


def _overflow(fig, tag: str) -> None:
    """Warn if any text leaves the canvas or the right-hand ink limit."""
    W, H = fig.get_size_inches()
    W, H = W * 25.4, H * 25.4
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    for ax in fig.axes:
        artists.extend(ax.texts)
    for art in artists:
        try:
            bb = art.get_window_extent(renderer=renderer)
        except Exception:      # pragma: no cover - unrenderable artist
            continue
        x1 = bb.x1 / fig.dpi * 25.4
        y_bottom = H - bb.y0 / fig.dpi * 25.4
        label = art.get_text()[:60].replace("\n", " ")
        if x1 > X_RIGHT + 1.6 or y_bottom > H - 0.4:
            print(f"[figS18:{tag}] overflow x1={x1:.1f} bottom={y_bottom:.1f} "
                  f"(limits {X_RIGHT:.1f}, {H - 0.4:.1f}): {label!r}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Data recovery + verification
# ---------------------------------------------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _trinomial(records) -> np.ndarray:
    counts = collections.Counter(r["action_label"] for r in records)
    n = len(records)
    return np.array([counts[a] / n for a in ACT_ORDER], dtype=float)


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def _load() -> dict:
    """Regenerate the prompts, verify every SHA-256, recompute every statistic."""
    fields = json.loads(FIELDS_JSON.read_text(encoding="utf-8"))["fields"]
    by_id = {f["field_id"]: f for f in fields}
    edges = np.linspace(-np.pi, np.pi, 25, dtype=float)

    prompts: dict[tuple[str, str], str] = {}
    for f in fields:
        hist = RelativePhaseHistogram(
            edges=edges.copy(),
            fractions=np.asarray(f["physical_histogram_24"], dtype=float),
            peer_count=int(f["peer_count"]),
        )
        for v in VARIANTS:
            prompts[(f["field_id"], v)] = build_control_prompt(v, hist)

    records = [json.loads(line) for line in
               (OMAP_RUN / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line]
    n_verified = 0
    for r in records:
        text = prompts[(r["field_id"], r["target_representation"])]
        if _sha(text) != r["prompt_sha256"]:
            raise RuntimeError(
                f"regenerated {r['field_id']}/{r['target_representation']} prompt does "
                f"not reproduce the stored prompt_sha256; refusing to print it"
            )
        n_verified += 1

    # per-field / per-variant and per-block trinomials
    g: dict[tuple[str, str], list] = collections.defaultdict(list)
    gb: dict[tuple[str, str, int], list] = collections.defaultdict(list)
    for r in records:
        g[(r["field_id"], r["target_representation"])].append(r)
        gb[(r["field_id"], r["target_representation"], r["acquisition_block"])].append(r)

    field_ids = [f["field_id"] for f in fields]
    tri = {k: _trinomial(v) for k, v in g.items()}
    block_noise = {
        (fid, v): _tv(_trinomial(gb[(fid, v, 0)]), _trinomial(gb[(fid, v, 1)]))
        for fid in field_ids for v in VARIANTS
    }

    # cross-check against the frozen CSV - bit-identical or raise
    csv = pd.read_csv(DATA_DIR / "omap_field_pairwise_tv.csv")
    csv = csv.set_index("field_id").loc[field_ids].reset_index()
    for _, row in csv.iterrows():
        fid = row["field_id"]
        po, pr, pp = (tri[(fid, v)] for v in VARIANTS)
        got = (_tv(po, pr), _tv(po, pp), _tv(pr, pp))
        want = (row["dTV_orig_reformat"], row["dTV_orig_pad"], row["dTV_reformat_pad"])
        if max(abs(a - b) for a, b in zip(got, want)) > 1e-12:
            raise RuntimeError(f"{fid}: recomputed pairwise TV disagrees with the frozen CSV")
        for stored, live in zip(("p_orig", "p_reformat", "p_pad"), (po, pr, pp)):
            arr = np.asarray(ast.literal_eval(row[stored]), dtype=float)
            if np.abs(arr - live).max() > 1e-12:
                raise RuntimeError(f"{fid}: recomputed {stored} disagrees with the frozen CSV")

    dec = json.loads((PRIMARY / "decision.json").read_text(encoding="utf-8"))
    bn_mean = float(np.mean(list(block_noise.values())))
    if abs(bn_mean - dec["same_info_noise_floor"]["mean_within_variant_block_TV"]) > 1e-12:
        raise RuntimeError("recomputed block noise disagrees with decision.json")

    null_tv = np.load(DATA_DIR / "omap_null_tv.npy")
    null_meta = json.loads((DATA_DIR / "omap_null_meta.json").read_text(encoding="utf-8"))
    if int(null_meta["n_perm"]) != int(null_tv.size):
        raise RuntimeError("omap_null_tv.npy length disagrees with omap_null_meta.json")
    if abs(null_meta["observed_mean_pairwise_TV"]
           - dec["same_info_global_test"]["observed_mean_pairwise_TV"]) > 1e-12:
        raise RuntimeError("null metadata disagrees with decision.json")

    # value-identity audit: the six printed moment components, per field
    val_re = re.compile(r"[+-]\d\.\d{6}")
    identity = np.zeros((6, len(field_ids)), dtype=bool)
    spans: dict[str, set] = {v: set() for v in VARIANTS}
    lengths: dict[str, set] = {v: set() for v in VARIANTS}
    for j, fid in enumerate(field_ids):
        vals = {}
        for v in VARIANTS:
            text = prompts[(fid, v)]
            hits = list(val_re.finditer(text))
            if len(hits) != 6:
                raise RuntimeError(f"{fid}/{v}: expected 6 moment values, found {len(hits)}")
            vals[v] = [h.group(0) for h in hits]
            spans[v].add((hits[0].start(), hits[-1].end(), len(text)))
            lengths[v].add(len(text))
        for k in range(6):
            identity[k, j] = vals[VARIANTS[0]][k] == vals[VARIANTS[1]][k] == vals[VARIANTS[2]][k]
    for v in VARIANTS:
        if len(spans[v]) != 1:
            raise RuntimeError(f"{v}: the task-relevant span is not constant across fields")

    # per-variant character and token counts (constants, not distributions)
    chars = {v: sorted(lengths[v]) for v in VARIANTS}
    tokens: dict[str, set] = {v: set() for v in VARIANTS}
    for r in records:
        if int(r.get("attempts", 1)) == 1 and r.get("input_tokens"):
            tokens[r["target_representation"]].add(int(r["input_tokens"]))
    tokens = {v: sorted(tokens[v]) for v in VARIANTS}

    order = sorted(range(len(field_ids)),
                   key=lambda i: (STRATA.index(by_id[field_ids[i]]["stratum"]), i))

    return {
        "fields": fields,
        "field_ids": field_ids,
        "stratum": [by_id[f]["stratum"] for f in field_ids],
        "order": order,
        "prompts": prompts,
        "n_verified": n_verified,
        "n_records": len(records),
        "n_valid": sum(1 for r in records if r["valid"]),
        "tri": tri,
        "block_noise": block_noise,
        "csv": csv,
        "dec": dec,
        "null_tv": null_tv,
        "null_meta": null_meta,
        "identity": identity,
        "spans": {v: next(iter(spans[v])) for v in VARIANTS},
        "chars": chars,
        "tokens": tokens,
        "model": records[0]["model"],
    }


# ---------------------------------------------------------------------------
# Page S18a - prompt construction and lengths
# ---------------------------------------------------------------------------
H_A = 222.0

MONO_NUM_X = 8.5        # right edge of the line-number gutter
MONO_TXT_X = 11.5       # left edge of the verbatim column
MONO_W = X_RIGHT - MONO_TXT_X


def _mono_rows(fig, rows, y_top: float, fontsize: float = FS_TINY) -> float:
    """Draw numbered verbatim prompt lines; return the y just past the block.

    ``rows`` are ``(number, text, colour, weight)``. A row whose number is
    ``None`` is a display-only band (an elision statement), never prompt text.
    """
    pitch = _pitch(fontsize)
    y = y_top
    for num, text, color, weight in rows:
        if num is None:
            fig_text_mm(fig, MONO_TXT_X, y, text, ha="left", va="top",
                        fontsize=fontsize, color=color, style="italic")
            y += pitch
            continue
        pieces = _wrap(fig, text, MONO_W, fontsize, family="monospace")
        for i, piece in enumerate(pieces):
            if i == 0:
                fig_text_mm(fig, MONO_NUM_X, y, num, ha="right", va="top",
                            fontsize=fontsize, color=MUTED, family="monospace")
            fig_text_mm(fig, MONO_TXT_X + (0.0 if i == 0 else 2.2), y, piece,
                        ha="left", va="top", fontsize=fontsize, color=color,
                        family="monospace", fontweight=weight)
            y += pitch
    return y


def _sha_line(fig, y: float, d: dict, variant: str, extra: str) -> None:
    fid = d["field_ids"][0]
    text = d["prompts"][(fid, variant)]
    fig_text_mm(fig, MONO_TXT_X, y,
                f"{fid} · sha256 {_sha(text)[:12]}… · {len(text)} chars · "
                f"{len(text.splitlines())} lines · {extra}",
                ha="left", va="baseline", fontsize=FS_TINY, color=MUTED,
                family="monospace")


def _page_a(d: dict) -> Path:
    fig = new_figure(H_A)

    fid = d["field_ids"][0]
    orig = d["prompts"][(fid, "moments_original")].splitlines()
    ref = d["prompts"][(fid, "moments_reformatted")].splitlines()
    pad = d["prompts"][(fid, "moments_length_matched")].splitlines()

    # the shared prompt contract, asserted rather than assumed
    shared = [0] + list(range(2, 9))
    for i in shared:
        if not (orig[i] == ref[i] == pad[i]):
            raise RuntimeError(f"prompt line {i + 1} is not shared across the three variants")
    if pad[:len(orig)] != orig:
        raise RuntimeError("the padded prompt does not begin with the complete original prompt")

    C_ORIG = OMAP["moments_original"]
    C_REF = OMAP["moments_reformatted"]
    C_PAD = OMAP["moments_length_matched"]

    # ---- a  full original prompt ----------------------------------------
    panel_label_at(fig, X_LEFT, 7.0, "a",
                   "Full original prompt (16 lines, verbatim)", title_color=C_ORIG)
    _sha_line(fig, 10.6, d, "moments_original", "task-relevant block = lines 11–16")
    rows = []
    for i, line in enumerate(orig):
        payload = i >= 10
        rows.append((str(i + 1), line, INK if payload else "#3A3A3A",
                     "bold" if payload else "normal"))
    y = _mono_rows(fig, rows, 12.8)
    y = _para(fig, X_LEFT, y + 2.6,
              "Lines 1 and 3–9 (the action contract) are byte-identical in all three variants; "
              "only line 2 and the payload block below it change. The six signed six-decimal "
              "moment components in lines 11–16 are the whole of the task-relevant information.",
              X_RIGHT - X_LEFT)
    rule_mm(fig, X_LEFT, X_RIGHT, y + 1.2)

    # ---- b  full reformatted prompt --------------------------------------
    y0 = y + 6.4
    panel_label_at(fig, X_LEFT, y0, "b",
                   "Full reformatted prompt (15 lines; the 7 that differ from a)",
                   title_color=C_REF)
    _sha_line(fig, y0 + 3.6, d, "moments_reformatted", "task-relevant block = lines 11–15")
    rows = [(None, "lines 1, 3–9 byte-identical to a (not reprinted)", MUTED, "normal")]
    rows.append(("2", ref[1], "#3A3A3A", "normal"))
    for i in range(9, len(ref)):
        rows.append((str(i + 1), ref[i], INK if i >= 12 else "#3A3A3A",
                     "bold" if i >= 12 else "normal"))
    y = _mono_rows(fig, rows, y0 + 5.8)
    y = _para(fig, X_LEFT, y + 2.6,
              "The same six values in the same order, retyped as a markdown table: one row per "
              "harmonic, cos before sin. Nothing numerical is added or removed; the prompt grows "
              f"by {len(d['prompts'][(fid, 'moments_reformatted')]) - len(d['prompts'][(fid, 'moments_original')])} "
              "characters of table scaffolding and the wording of line 2.",
              X_RIGHT - X_LEFT)
    rule_mm(fig, X_LEFT, X_RIGHT, y + 1.2)

    # ---- c  full length-matched prompt -----------------------------------
    y0 = y + 6.4
    panel_label_at(fig, X_LEFT, y0, "c",
                   "Full length-matched prompt (24 lines; values and appended padding)",
                   title_color=C_PAD)
    _sha_line(fig, y0 + 3.6, d, "moments_length_matched",
              "task-relevant block = lines 11–16")
    if len(set(pad[17:23])) != 1 or pad[17] != pad[16]:
        raise RuntimeError("padding lines 18–23 are not byte-identical repeats of line 17")
    rows = [(None, "lines 1–10 byte-identical to a (not reprinted)", MUTED, "normal")]
    for i in range(10, 16):
        rows.append((str(i + 1), pad[i], INK, "bold"))
    rows.append(("17", pad[16], C_PAD, "normal"))
    rows.append((None, "lines 18–23: six further byte-identical copies of line 17",
                 MUTED, "normal"))
    rows.append((str(len(pad)), pad[-1], C_PAD, "normal"))
    y = _mono_rows(fig, rows, y0 + 5.8)
    n_pad_lines = len(pad) - len(orig)
    n_pad_chars = (len(d["prompts"][(fid, "moments_length_matched")])
                   - len(d["prompts"][(fid, "moments_original")]))
    y = _para(fig, X_LEFT, y + 2.6,
              f"The padded prompt is the complete original prompt followed by {n_pad_chars} "
              f"characters of neutral filler ({n_pad_lines} lines; the last is cut mid-sentence "
              f"by the deterministic {LENGTH_MATCH_TARGET_CHARS}-character target). The filler "
              "carries no phase information, but it is neither empty nor inert: it is eight "
              "further instruction-shaped sentences appended after the values.",
              X_RIGHT - X_LEFT)
    rule_mm(fig, X_LEFT, X_RIGHT, y + 1.2)

    # ---- d, e, f  audits --------------------------------------------------
    row_letter = y + 5.6
    row_top = row_letter + 4.4
    row_h = 22.0
    D_L, D_W = 13.0, 42.0
    E_L, E_W = 66.0, 42.0
    F_L, F_W = 120.0, 56.0

    # d  value identity audit
    axD = mm_axes(fig, D_L, row_top, D_W, row_h)
    panel_label(axD, "d", "Value identity audit")
    ident = d["identity"]
    axD.imshow(ident.astype(float), cmap=LinearSegmentedColormap.from_list(
        "ident", ["#F4C7C3", "#DFF0E2"]), vmin=0.0, vmax=1.0,
        aspect="auto", interpolation="nearest",
        extent=(-0.5, ident.shape[1] - 0.5, ident.shape[0] - 0.5, -0.5))
    for k in range(1, ident.shape[0]):
        axD.axhline(k - 0.5, color="white", lw=LW_HAIR)
    for j in range(1, ident.shape[1]):
        axD.axvline(j - 0.5, color="white", lw=0.12)
    axD.set_yticks(range(6))
    axD.set_yticklabels([r"$m_1\cos$", r"$m_1\sin$", r"$m_2\cos$",
                         r"$m_2\sin$", r"$m_3\cos$", r"$m_3\sin$"], fontsize=FS_TINY)
    axD.set_xticks([0, 15, 31, 47])
    axD.set_xticklabels(["1", "16", "32", "48"], fontsize=FS_TICK)
    axD.set_xlabel("field", labelpad=1.0)
    axD.tick_params(length=1.4, width=0.45, pad=1.2)
    for s in axD.spines.values():
        s.set_linewidth(0.45)
        s.set_color("#8C8C8C")
        s.set_visible(True)
    n_ok, n_tot = int(ident.sum()), int(ident.size)
    note_y = row_top + row_h + 9.6
    fig_text_mm(fig, D_L, note_y, f"{n_ok}/{n_tot} components identical",
                ha="left", va="baseline", fontsize=FS_TINY, color=PASS_GREEN,
                fontweight="bold")
    _para(fig, D_L, note_y + 2.9,
          "48 fields × 6 moment components, compared as strings; a mismatching cell "
          "would print red.", D_W)

    # e  character and token counts (constants, drawn as point values)
    axE = mm_axes(fig, E_L, row_top, E_W, row_h)
    panel_label(axE, "e", "Prompt size per variant")
    xs = np.arange(3)
    ch = [d["chars"][v][0] for v in VARIANTS]
    tk = [d["tokens"][v][0] for v in VARIANTS]
    axE.bar(xs, ch, width=0.56, color=[OMAP[v] for v in VARIANTS],
            edgecolor="white", linewidth=LW_HAIR)
    for x, c, t in zip(range(3), ch, tk):
        axE.text(x, c + 215.0, f"{c} chars", ha="center", va="bottom",
                 fontsize=FS_TINY, color=INK, fontweight="bold")
        axE.text(x, c + 50.0, f"{t} tokens", ha="center", va="bottom",
                 fontsize=FS_TINY, color=MUTED)
    axE.set_xticks(xs)
    axE.set_xticklabels(["original", "reformatted", "length-\nmatched"],
                        fontsize=FS_TINY, linespacing=1.2)
    axE.tick_params(axis="x", length=0, pad=1.4)
    # Equal side margins wide enough that the "chars / tokens" stack over the
    # first bar clears the y tick column; otherwise "200 tokens" butts against
    # the "1000" tick label and the two read as one line.
    axE.set_xlim(-0.66, 2.66)
    axE.set_ylabel("characters")
    axE.set_ylim(0, 2020)
    axE.set_yticks([0, 500, 1000, 1500])
    trim_spines(axE, x=False)
    fig_text_mm(fig, E_L, note_y, "exact constants, not distributions",
                ha="left", va="baseline", fontsize=FS_TINY, color=INK,
                fontweight="bold")
    _para(fig, E_L, note_y + 2.9,
          "Fixed-width numbers make every field give the same length in a given "
          "variant, so neither count has any spread over the 48 fields.", E_W)

    # f  position of the task-relevant values
    axF = mm_axes(fig, F_L, row_top, F_W, row_h)
    panel_label(axF, "f", "Position of the task-relevant values")
    for i, v in enumerate(VARIANTS):
        lo, hi, total = d["spans"][v]
        y = 2 - i
        axF.barh([y], [total], height=0.46, color="#ECECEC",
                 edgecolor="#9A9A9A", linewidth=LW_HAIR, zorder=1)
        axF.barh([y], [hi - lo], left=lo, height=0.46, color=OMAP[v],
                 edgecolor="none", zorder=2)
        if total > hi:
            axF.barh([y], [total - hi], left=hi, height=0.46, color=OMAP[v],
                     alpha=0.20, edgecolor="none", zorder=2)
            # A hatch takes its colour from the *edge*, so the padding segment
            # drew as a plain tint while the note promised hatching. Overlay a
            # hairline hatch in the variant colour instead of restating it.
            pad_bar = axF.barh([y], [total - hi], left=hi, height=0.46,
                               facecolor="none", edgecolor=OMAP[v],
                               hatch="///", linewidth=LW_HAIR, zorder=3)
            for patch in pad_bar.patches:
                if hasattr(patch, "set_hatch_linewidth"):
                    patch.set_hatch_linewidth(LW_HAIR)
        axF.text(total + 30.0, y, f"{lo}–{hi} of {total}", va="center", ha="left",
                 fontsize=FS_TINY, color=INK)
    axF.set_yticks([2, 1, 0])
    axF.set_yticklabels([V_SHORT[v] for v in VARIANTS], fontsize=FS_TINY)
    axF.set_ylim(-0.55, 2.55)
    axF.tick_params(axis="y", length=0, pad=1.4)
    axF.set_xlim(0, 2160)
    axF.set_xticks([0, 500, 1000, 1500])
    axF.set_xlabel("character index in the user message", labelpad=1.0)
    trim_spines(axF, y=False)
    lo_p, hi_p, tot_p = d["spans"]["moments_length_matched"]
    fig_text_mm(fig, F_L, note_y, "solid = the six values; hatched = padding",
                ha="left", va="baseline", fontsize=FS_TINY, color=INK,
                fontweight="bold")
    _para(fig, F_L, note_y + 2.9,
          f"The block ends the message in the first two variants; in the padded one it "
          f"closes at character {hi_p} of {tot_p}, with eight further sentences after it.",
          F_W)

    y_claim = note_y + 14.0
    fig_text_mm(fig, X_LEFT, y_claim, "Claim boundary", ha="left", va="baseline",
                fontsize=FS_TITLE, fontweight="bold", color=INK)
    _para(fig, 27.0, y_claim,
          "The padded condition is not a length-only intervention. Relative to the original it "
          f"changes prompt length ({d['chars']['moments_original'][0]} → "
          f"{d['chars']['moments_length_matched'][0]} characters), context volume "
          f"({len(orig)} → {len(pad)} lines of instruction-shaped text) and the position of the "
          "task-relevant block (message-final → mid-message) simultaneously. Any difference it "
          "produces is attributable to that bundle, not to prompt length alone.",
          X_RIGHT - 27.0, color=INK)

    _overflow(fig, "S18a")
    return save_fig(fig, "figS18a_omap_prompt_construction")


# ---------------------------------------------------------------------------
# Page S18b - full response analysis
# ---------------------------------------------------------------------------
# Shortened from 186.0 when the closing "Reading" statement moved to the
# caption: that block spanned 12.4 mm, so the page loses exactly that and
# keeps its original foot margin. Panel geometry above is untouched.
H_B = 173.6

G_L, G_TOP, G_W, G_H = 22.0, 9.0, 142.0, 13.5
R2_TOP, R2_H = 48.0, 30.0
R2_LETTER = R2_TOP - 1.6
H_L, H_W = 13.0, 40.0
I_L, I_W = 78.0, 98.0
R2_NOTE = 86.0
R3_TOP, R3_H = 100.0, 34.0
J_L, J_W = 13.0, 40.0
K_L, K_W = 62.0, 52.0
L_L, L_W = 126.0, 48.0
R3_NOTE = 149.0
K_KEY_DY = 3.0          # line advance of the variant-pair key under panel k


def _key_swatch(fig, x_mm: float, y_mm: float, colour: str, h_mm: float) -> None:
    """A small filled square sitting on the baseline of a key label."""
    fig.add_artist(mpatches.Rectangle(
        (x_mm / 180.0, (h_mm - y_mm + 0.15) / h_mm), 2.0 / 180.0, 1.8 / h_mm,
        transform=fig.transFigure, facecolor=colour, edgecolor="#8C8C8C",
        lw=LW_HAIR, clip_on=False))


def _stratum_axis(ax, strata_sorted, y_frac: float, n: int) -> None:
    """Bracket the ordered field axis into its six replay strata."""
    tr = ax.get_xaxis_transform()
    start = 0
    for k in range(1, n + 1):
        if k == n or strata_sorted[k] != strata_sorted[start]:
            a, b = start - 0.42, k - 1 + 0.42
            ax.plot([a, b], [y_frac, y_frac], transform=tr, color=MUTED,
                    lw=LW_HAIR, clip_on=False, solid_capstyle="butt")
            for xe in (a, b):
                ax.plot([xe, xe], [y_frac, y_frac + 0.045], transform=tr, color=MUTED,
                        lw=LW_HAIR, clip_on=False, solid_capstyle="butt")
            ax.text((a + b) / 2.0, y_frac - 0.05, STRATUM_LABEL[strata_sorted[start]],
                    transform=tr, ha="center", va="top", fontsize=FS_TINY,
                    color=INK, linespacing=1.2, clip_on=False)
            start = k


def _page_b(d: dict) -> Path:
    fig = new_figure(H_B)

    order = d["order"]
    fids = [d["field_ids"][i] for i in order]
    strata_sorted = [d["stratum"][i] for i in order]
    csv = d["csv"]
    n = len(fids)

    # ---- g  fieldwise pairwise TV ----------------------------------------
    axG = mm_axes(fig, G_L, G_TOP, G_W, G_H)
    panel_label_at(fig, X_LEFT, G_TOP - 1.6, "g",
                   "Fieldwise pairwise total variation, all 48 fields × 3 variant pairs")
    mat = np.vstack([csv[col].to_numpy()[order] for col, _, _ in PAIRS])
    im = axG.imshow(mat, cmap=TV_CMAP, vmin=0.0, vmax=1.0, aspect="auto",
                    interpolation="nearest",
                    extent=(-0.5, n - 0.5, mat.shape[0] - 0.5, -0.5))
    for k in range(1, mat.shape[0]):
        axG.axhline(k - 0.5, color="white", lw=LW_THIN)
    axG.set_yticks(range(3))
    axG.set_yticklabels([lab for _, lab, _ in PAIRS], fontsize=FS_TINY)
    axG.set_xticks([])
    axG.tick_params(axis="y", length=0, pad=1.4)
    for sp in axG.spines.values():
        sp.set_linewidth(0.45)
        sp.set_color("#8C8C8C")
        sp.set_visible(True)
    _stratum_axis(axG, strata_sorted, -0.10, n)
    cbar_mm(fig, im, G_L + G_W + 3.0, G_TOP, 2.4, G_H, ticks=[0, 0.5, 1.0])
    fig_text_mm(fig, G_L + G_W + 2.6, G_TOP - 1.6, DTV, ha="left", va="baseline",
                fontsize=FS_BODY, color=INK)
    med = {col: float(csv[col].median()) for col, _, _ in PAIRS}
    mean = {col: float(csv[col].mean()) for col, _, _ in PAIRS}
    zero = {col: int((csv[col] == 0).sum()) for col, _, _ in PAIRS}
    txt = "; ".join(
        f"{lab}: median {med[col]:.4f}, mean {mean[col]:.4f}, {zero[col]}/48 exactly zero"
        for col, lab, _ in PAIRS)
    _para(fig, X_LEFT, 35.0,
          txt + ". Fields are ordered by replay stratum (8 per stratum), not by field index; "
          "the same order is used in i.", X_RIGHT - X_LEFT)

    # ---- h  within-variant block noise -----------------------------------
    axH = mm_axes(fig, H_L, R2_TOP, H_W, R2_H)
    panel_label(axH, "h", "Within-variant block noise")
    for i, v in enumerate(VARIANTS):
        vals = np.array([d["block_noise"][(f, v)] for f in d["field_ids"]])
        axH.scatter(swarm_x(vals, i, width=0.17), vals, s=MS_POINT ** 2,
                    c=OMAP[v], alpha=0.60, zorder=2, linewidths=0)
        m = float(vals.mean())
        axH.plot([i - 0.26, i + 0.26], [m, m], color=INK, lw=LW_LINE, zorder=4)
        axH.text(i, 1.075, f"{m:.3f}", ha="center", va="center", fontsize=FS_TINY,
                 fontweight="bold", color=INK)
    bn_all = float(np.mean(list(d["block_noise"].values())))
    axH.axhline(bn_all, color=MUTED, lw=LW_HAIR, ls=(0, (3, 2)), zorder=1)
    axH.set_xticks(range(3))
    axH.set_xticklabels(["original", "reformatted", "length-\nmatched"],
                        fontsize=FS_TINY, linespacing=1.2)
    axH.tick_params(axis="x", length=0, pad=1.4)
    axH.set_xlim(-0.55, 2.55)
    # Most fields sit exactly at zero. With the limit at 0 that dense row was
    # sliced in half by the axis, so the range runs a little below zero and the
    # bottom spine goes; the trimmed left spine still starts at 0.
    axH.set_ylim(-0.055, 1.14)
    axH.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axH.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    axH.set_ylabel(DTV + " (block 1 vs block 2)")
    trim_spines(axH, x=False)
    axH.spines["bottom"].set_visible(False)
    _para(fig, H_L, R2_NOTE,
          f"Block 1 vs block 2 within one variant, per field: the noise floor the pairwise "
          f"distances in g must beat. Pooled mean {bn_all:.4f} (dashed).", I_L - H_L - 5.0)

    # ---- i  full action matrices -----------------------------------------
    panel_label_at(fig, I_L - 8.6, R2_LETTER, "i",
                   "Empirical action trinomials, every field × variant")
    strip_h, strip_gap = 7.4, 2.4
    for iv, v in enumerate(VARIANTS):
        top = R2_TOP + 2.0 + iv * (strip_h + strip_gap)
        ax = mm_axes(fig, I_L, top, I_W, strip_h)
        probs = np.vstack([d["tri"][(f, v)] for f in fids])
        bottoms = np.zeros(n)
        for k, col in enumerate(ACT_COLORS):
            ax.bar(np.arange(n), probs[:, k], bottom=bottoms, width=0.86,
                   color=col, linewidth=0)
            bottoms = bottoms + probs[:, k]
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"], fontsize=FS_TINY)
        ax.tick_params(axis="y", length=1.2, width=0.45, pad=1.0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(-0.035, 0.5, V_SHORT[v], ha="right", va="center", fontsize=FS_TINY,
                color=INK, fontweight="bold", transform=ax.transAxes, clip_on=False)
        if iv == len(VARIANTS) - 1:
            _stratum_axis(ax, strata_sorted, -0.18, n)
    key_y = R2_LETTER
    for k, (col, lab) in enumerate(zip(ACT_COLORS, ACTION_LABELS)):
        x = I_L + I_W - 28.0 + k * 9.6
        fig.add_artist(mpatches.Rectangle(
            (x / 180.0, (H_B - key_y - 1.7) / H_B), 2.0 / 180.0, 2.0 / H_B,
            transform=fig.transFigure, facecolor=col, edgecolor="#8C8C8C",
            lw=LW_HAIR, clip_on=False))
        fig_text_mm(fig, x + 2.8, key_y, lab, ha="left", va="baseline",
                    fontsize=FS_TINY, color=INK)
    _para(fig, I_L, R2_NOTE,
          "Each bar is one field: the 16 responses split into retard / stay / advance. "
          "Rows are the three variants; the same 48 fields in the same order as g.", I_W)

    # ---- j  permutation null ---------------------------------------------
    axJ = mm_axes(fig, J_L, R3_TOP, J_W, R3_H)
    panel_label(axJ, "j", "Variant-label permutation null")
    meta = d["null_meta"]
    axJ.hist(d["null_tv"], bins=40, color="#C9CDD2", edgecolor="white",
             linewidth=0.2, density=True)
    obs = float(meta["observed_mean_pairwise_TV"])
    axJ.axvline(obs, color=OMAP["moments_length_matched"], lw=LW_TRACE)
    axJ.axvline(float(meta["null_mean"]), color="#666666", lw=LW_THIN, ls=(0, (3, 2)))
    axJ.axvline(float(meta["null_p95"]), color="#666666", lw=LW_HAIR, ls=(0, (1, 1.6)))
    # The observed line used to land past the last tick, hard against the right
    # edge, where it read as a right spine; carry the axis past it instead.
    axJ.set_xlim(0.050, 0.400)
    axJ.set_xticks([0.1, 0.2, 0.3, 0.4])
    axJ.set_xlabel("mean pairwise " + DTV, labelpad=1.0)
    axJ.set_ylabel("density")
    # Headroom above the tallest bar so the line labels sit clear of both the
    # histogram and the dashed reference lines they name.
    hist_top = axJ.get_ylim()[1]
    y_ticks = [t for t in axJ.get_yticks() if 0.0 <= t <= hist_top]
    axJ.set_ylim(0, hist_top * 1.34)
    axJ.set_yticks(y_ticks)
    top_y = axJ.get_ylim()[1]
    # The two reference lines are 0.015 apart, so neither can carry a label
    # beside it without lying across the other line or the tallest bar. Set the
    # labels out in the empty middle of the panel with hairline leaders, above
    # every bar.
    for label, x_line, y_frac in (("null mean", float(meta["null_mean"]), 0.95),
                                  ("p95", float(meta["null_p95"]), 0.80)):
        axJ.annotate(label, xy=(x_line, top_y * y_frac),
                     xytext=(0.215, top_y * y_frac), ha="right", va="center",
                     fontsize=FS_TINY, color="#666666",
                     arrowprops=dict(arrowstyle="-", color="#666666",
                                     lw=LW_HAIR, shrinkA=1.4, shrinkB=0.0))
    # The observed block sits beside its own line at mid height: at the top it
    # would sit line-for-line with the two reference labels above.
    axJ.text(obs - 0.010, top_y * 0.62,
             f"observed\n{obs:.4f}\n$p$ = {meta['p_one_sided']:.4f}",
             fontsize=FS_TINY, color=OMAP["moments_length_matched"], ha="right",
             va="top", linespacing=1.45, fontweight="bold")
    trim_spines(axJ)
    _para(fig, J_L, R3_NOTE,
          f"All {int(meta['n_perm']):,} field-blocked permutations of the variant label are "
          f"plotted.", J_W + 5.0)

    # ---- k  effect by replay stratum -------------------------------------
    axK = mm_axes(fig, K_L, R3_TOP, K_W, R3_H)
    panel_label(axK, "k", "Effect by replay stratum")
    xs = np.arange(len(STRATA))
    width = 0.26
    for pi, (col, lab, colour) in enumerate(PAIRS):
        means, points = [], []
        for st in STRATA:
            idx = [i for i in range(n) if strata_sorted[i] == st]
            vals = csv[col].to_numpy()[order][idx]
            means.append(float(vals.mean()))
            points.append(vals)
        axK.bar(xs + (pi - 1) * width, means, width=width * 0.9, color=colour,
                edgecolor="white", linewidth=LW_HAIR, zorder=2)
        for x, vals in zip(xs, points):
            axK.scatter(np.full(vals.size, x + (pi - 1) * width), vals,
                        s=(MS_POINT * 0.66) ** 2, c=INK, alpha=0.35, zorder=3,
                        linewidths=0)
    axK.set_xticks(xs)
    axK.set_xticklabels([STRATUM_LABEL[st] for st in STRATA], fontsize=FS_TINY,
                        linespacing=1.2)
    axK.tick_params(axis="x", length=0, pad=1.4)
    axK.set_xlim(-0.62, len(STRATA) - 0.38)
    # Every stratum has fields at exactly zero; with the limit at 0 those dots
    # were bisected by the axis. Same repair as h.
    axK.set_ylim(-0.055, 1.06)
    axK.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axK.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    axK.set_ylabel(DTV)
    trim_spines(axK, x=False)
    axK.spines["bottom"].set_visible(False)
    # A real key: g identifies the three variant pairs by row text, not colour,
    # so "colours as in g" gave the reader nothing to match the bars against.
    for pi, (_, lab, colour) in enumerate(PAIRS):
        y_key = R3_NOTE + pi * K_KEY_DY
        _key_swatch(fig, K_L, y_key, colour, H_B)
        fig_text_mm(fig, K_L + 3.0, y_key, lab, ha="left", va="baseline",
                    fontsize=FS_TINY, color=INK)
    _para(fig, K_L, R3_NOTE + len(PAIRS) * K_KEY_DY + 0.6,
          "Bars are stratum means over the 8 fields of that stratum; dots are those "
          "8 fields. Every stratum shows the padded condition separating.", K_W)

    # ---- l  prespecified quantile examples --------------------------------
    axL = mm_axes(fig, L_L, R3_TOP, L_W, R3_H)
    panel_label(axL, "l", "Quantile-selected examples")
    ref_col = "dTV_orig_pad"
    vals = csv[ref_col].to_numpy()
    picks = []
    for q in (0.25, 0.50, 0.75):
        target = float(np.quantile(vals, q))
        j = int(np.lexsort((np.arange(vals.size), np.abs(vals - target)))[0])
        picks.append((q, j))
    for gi, (q, j) in enumerate(picks):
        x0 = gi * 4.0
        tris = [d["tri"][(csv["field_id"].iloc[j], v)] for v in VARIANTS]
        bottoms = np.zeros(3)
        for k, col in enumerate(ACT_COLORS):
            vv = [t[k] for t in tris]
            axL.bar(np.arange(3) + x0, vv, bottom=bottoms, color=col, width=0.74,
                    edgecolor="white", linewidth=LW_HAIR)
            bottoms = bottoms + np.array(vv)
    axL.set_ylim(0, 1.34)
    axL.set_yticks([0, 0.5, 1.0])
    axL.set_ylabel("probability")
    axL.set_xlim(-0.8, 10.8)
    axL.set_xticks([x + o for x in (0, 4, 8) for o in (0, 1, 2)])
    axL.set_xticklabels([V_TICK[v] for _ in picks for v in VARIANTS], fontsize=FS_TINY)
    axL.tick_params(axis="x", length=0, pad=1.2)
    trim_spines(axL, x=False)
    tr = axL.get_xaxis_transform()
    for gi, (q, j) in enumerate(picks):
        a, b = gi * 4.0 - 0.44, gi * 4.0 + 2.44
        axL.plot([a, b], [-0.155, -0.155], transform=tr, color=MUTED, lw=LW_HAIR,
                 clip_on=False, solid_capstyle="butt")
        for xe in (a, b):
            axL.plot([xe, xe], [-0.155, -0.125], transform=tr, color=MUTED,
                     lw=LW_HAIR, clip_on=False, solid_capstyle="butt")
        row = csv.iloc[j]
        axL.text((a + b) / 2.0, -0.205,
                 f"$Q_{{{int(q * 100)}}}$  {row['field_id'].replace('mrc_replay_', '')}",
                 transform=tr, ha="center", va="top", fontsize=FS_TINY, color=INK,
                 clip_on=False)
        for li, (tag, col) in enumerate((("o–r", "dTV_orig_reformat"),
                                         ("o–p", "dTV_orig_pad"),
                                         ("r–p", "dTV_reformat_pad"))):
            axL.text(gi * 4.0 + 1.0, 1.30 - li * 0.088,
                     f"{tag} {row[col]:.3f}", ha="center", va="top",
                     fontsize=FS_TINY, color=INK,
                     fontweight="bold" if col == "dTV_orig_pad" else "normal")
    _para(fig, L_L, R3_NOTE,
          "Fields nearest the 25th, 50th and 75th percentile of the original–padded contrast "
          "(ties broken by field index). o–r, o–p and r–p are that field's own distances "
          "for the three variant pairs.", L_W + 2.0)

    # The "Reading" statement that used to close this page now lives in the
    # caption; the canvas was shortened by exactly the height it occupied.

    _overflow(fig, "S18b")
    return save_fig(fig, "figS18b_omap_response_analysis")


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    d = _load()
    first = _page_a(d)
    _page_b(d)
    return first


if __name__ == "__main__":      # pragma: no cover
    print(build())
