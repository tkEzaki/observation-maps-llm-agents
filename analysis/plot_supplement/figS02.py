"""Supplementary Figure S2 - complete observation maps, prompt contracts and
serialisation metadata.

Main Figure 2b shows a five-line excerpt of each observation map and points
here for the complete serialisation. This module therefore has to print the
*whole* prompt, byte for byte, and prove that what it prints is what the model
was actually sent.

How the text is recovered
-------------------------
The production traces store ``"prompt": null`` and keep only a
``prompt_sha256``. Every prompt in the study is nevertheless exactly
reproducible: ``circlemap.representations.build_representation_prompt_from_histogram``
is a pure function of the 24-bin relative-phase histogram and the frozen
representation spec. This module regenerates each prompt it prints and
**asserts the SHA-256 against the frozen artifact before drawing it**. A
mismatch raises rather than printing an unverified prompt.

The three frozen examples are the three stimulus-manifold records main Fig. 2
now quotes: ``offset_index`` 0 of the unimodal kappa = 9 family, one physical
field observed through the three maps. The selection rule and the
physical-identity guard mirror
``analysis/plot_main/fig2_micro._shared_field_records`` - exact
equality across the three records on ``profile``, ``concentration``,
``offset_index`` and ``offset_radians``, plus a derived physical id over that
whole tuple - so the text printed here reproduces the main figure exactly where
the two overlap. Panel a therefore draws that one field, and b, c and d are its
three complete serialised observations.

Layout
------
Two pages, sanctioned by the SI plan's own "use two pages if necessary":

* **S2a** - a (the frozen example field under three observation maps), b/c/d (the
  three complete payloads of that field) under the prompt contract they share.
* **S2b** - e (character and token counts), f (retained-feature inventory).

Data honesty
------------
* Character counts and provider token counts are **exact constants** of the
  encoder, not distributions: the serialiser uses fixed-width ``%+.6f`` /
  ``%.6f`` fields, so every prompt of a given map has the same length. Panel e
  draws them as point values under a "representation-level constants" heading.
* The number of occupied bins is a **different kind of quantity** - a property
  of the physical field, not of the map - so it gets its own heading, its own
  axis and its own units in panel e, and is never plotted alongside the length
  constants.
* The apparent Gemini token spread is ``attempts x`` the per-call constant
  (the Google client accumulates usage across retries). It is verified as such
  record by record and reported in a separate retry-audit element, not in the
  token display.

What the figure verifies but does not print
-------------------------------------------
Every prompt drawn here is still regenerated from the stored 24-bin field and
matched against its frozen SHA-256 before it is drawn, and the
one-physical-field guard still compares a derived hash across the three
records. Those are build-time guards only: hashes, hash prefixes and the
derived physical id are no longer printed on the page. A reader cannot check a
truncated hex string by eye, so printing it only crowds out the numbers that
can be read.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

from analysis.plot_supplement.style import (
    FS_BODY,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    FS_TITLE,
    GRAY_BOX,
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
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    RUNS,
    STOP_RED,
    apply_style,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label_at,
    rule_mm,
    save_fig,
)
from circlemap.observation import (
    DEFAULT_N_BINS,
    RelativePhaseHistogram,
    relative_phase_histogram,
)
from circlemap.representations import build_representation_prompt_from_histogram
from circlemap.stimuli import build_stimulus_histogram

# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------
MRC = ROOT / "analysis" / "matched_rep_collective"
REPLAY_FIELDS = json.loads((MRC / "replay_fields_v0_1.json").read_text(encoding="utf-8"))
REPLAY_TARGETS = json.loads(
    (MRC / "replay_target_prompts_v0_1.json").read_text(encoding="utf-8")
)
FROZEN_PROFILE = "unimodal_k9"
# Not a free choice: the lowest offset index carried by all three maps in the
# first stimulus acquisition, which is the offset main Fig. 2 quotes in a/b and
# conditions its panel c response curves on.
FROZEN_OFFSET_INDEX = 0
# The quantities that pin the physical field. ``offset_index`` alone is a
# label, so the guard compares the whole tuple exactly and then hashes it.
PHYSICAL_KEYS = ("profile", "concentration", "offset_index", "offset_radians")
PHYSICAL_ID_CHARS = 6
EDGES = np.linspace(-np.pi, np.pi, DEFAULT_N_BINS + 1)
BIN_W = 2.0 * np.pi / DEFAULT_N_BINS

_BIN_CENTER_RE = re.compile(r"bin_\d+ center ([+-]\d+\.\d+): (\d+\.\d+)")

# GPT collective sessions, from the frozen replay-field lock.
GPT_SESSIONS = REPLAY_FIELDS["sessions"]
R2_ROOT = (
    RUNS / "stage_c" / "r2_claude_macro"
    / "matched-rep-collective-r2-claude-v0.1_ff2ef27f0dfe" / "20260725T050234Z"
)
R1_ROOT = RUNS / "matched_rep_collective_r1"

PT_MM = 25.4 / 72.0          # 1 pt in mm
LS_MONO = 1.30               # linespacing of the prompt-contract block
PAY_LS = 1.20                # tighter leading for the 24-line payload columns

# Page canvas heights (mm from the top-left; width is always 180 mm)
H_A = 211.0


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pitch(fontsize: float) -> float:
    """Baseline-to-baseline distance in mm for a monospace block."""
    return fontsize * LS_MONO * PT_MM


def _overflow(fig, tag: str) -> None:
    """Warn if any drawn text leaves the canvas or the right-hand ink limit.

    A serialised line that runs off the page is the one failure mode this
    figure cannot afford, so the check is part of the build rather than a
    thing to notice by eye.
    """
    import sys

    W, H = fig.get_size_inches()
    W, H = W * 25.4, H * 25.4
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    for ax in fig.axes:
        artists.extend(ax.texts)
    for art in artists:
        try:
            bb = art.get_window_extent(renderer=renderer)
        except Exception:  # pragma: no cover - unrenderable artist
            continue
        x1 = bb.x1 / fig.dpi * 25.4
        y_bottom = H - bb.y0 / fig.dpi * 25.4
        label = art.get_text()[:56].replace("\n", " ")
        if x1 > X_RIGHT + 1.4 or y_bottom > H - 0.6:
            print(f"[figS02:{tag}] overflow x1={x1:.1f} bottom={y_bottom:.1f} "
                  f"(limits {X_RIGHT:.1f}, {H - 0.6:.1f}): {label!r}", file=sys.stderr)


def _text_w_mm(fig, s: str, fontsize: float, **kw) -> float:
    """Rendered width of a string in mm on this canvas."""
    art = fig.text(0.0, 0.0, s, fontsize=fontsize, **kw)
    bb = art.get_window_extent(renderer=fig.canvas.get_renderer())
    art.remove()
    return float(bb.width) / float(fig.dpi) * 25.4


def _para(fig, x_mm: float, y_mm: float, text: str, max_mm: float,
          fontsize: float = FS_TINY, color: str = MUTED, lead: float = 1.42) -> float:
    """Draw a measured, word-wrapped paragraph; return the next free baseline.

    Every full-width note on these two pages goes through this, so no line can
    silently run off the 180 mm canvas.
    """
    words = text.split(" ")
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if not cur or _text_w_mm(fig, trial, fontsize) <= max_mm:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    pitch = fontsize * lead * PT_MM
    for i, line in enumerate(lines):
        fig_text_mm(fig, x_mm, y_mm + i * pitch, line, ha="left", va="baseline",
                    fontsize=fontsize, color=color)
    return y_mm + len(lines) * pitch


# ---------------------------------------------------------------------------
# Prompt recovery + verification
# ---------------------------------------------------------------------------
def _canonical_field_string(record: dict) -> str:
    """Canonical, type-normalised rendering of the physical tuple.

    Byte-identical to ``fig2_micro._canonical_field_string`` so the short
    physical id stamped on this page is the same string main Fig. 2a stamps on
    its own. ``repr`` of a float is the shortest round-tripping form, so the
    canonical string is exact (no tolerance) and platform independent.
    """
    return "|".join((
        f"profile={record['profile']}",
        f"concentration={float(record['concentration'])!r}",
        f"offset_index={int(record['offset_index'])}",
        f"offset_radians={float(record['offset_radians'])!r}",
    ))


def _physical_id(record: dict) -> str:
    """sha256 of :func:`_canonical_field_string` - a derived physical hash.

    ``stimuli.jsonl`` stores a ``prompt_sha256``, which differs between
    representations of the *same* field and therefore cannot be compared across
    them, but no hash of the field itself, so one is derived here.
    """
    return _sha(_canonical_field_string(record))


def _centers_bins(text: str) -> tuple[np.ndarray, np.ndarray]:
    """(bin centre, mass) exactly as ``centers_24_standard`` prints them."""
    found = _BIN_CENTER_RE.findall(text)
    if len(found) != DEFAULT_N_BINS:
        raise RuntimeError(
            f"centers_24_standard: expected {DEFAULT_N_BINS} bin centres, "
            f"found {len(found)}"
        )
    return (np.array([float(t) for t, _ in found]),
            np.array([float(m) for _, m in found]))


def _frozen_examples() -> tuple[dict, dict]:
    """The three serialised observations of ONE frozen physical field.

    Selection rule, identical to ``fig2_micro._shared_field_records``:

    1. walk ``runs/stimulus_manifold/**/stimuli.jsonl`` in sorted path order
       and take the first acquisition carrying all three representations for
       ``unimodal_k9``;
    2. within it take the **lowest ``offset_index`` present for all three** -
       offset 0, the first offset of the sweep and the one Fig. 2c conditions
       its response curves on.

    An earlier build of this module took the first record per representation in
    file order, which matched on ``profile`` only and silently paired three
    *different* rotations (offsets 30 / 4 / 29). The guards below make that
    failure mode loud rather than silent:

    * exact equality of ``profile``, ``concentration``, ``offset_index`` and
      ``offset_radians`` across the three records (two rotations differ in the
      last bits of the offset, so exact equality is the right test);
    * a derived ``physical_id`` over that whole tuple, recomputed per record;
    * a single ``RelativePhaseHistogram`` rebuilt from that one offset, from
      which all three payloads are regenerated and each matched against its own
      stored ``prompt_sha256`` and stored text before anything is drawn.
    """
    recs: dict | None = None
    for path in sorted((RUNS / "stimulus_manifold").rglob("stimuli.jsonl")):
        by_rep: dict[str, dict[int, dict]] = {rep: {} for rep in REP_ORDER}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                rep = obj.get("representation")
                if rep in by_rep and obj.get("profile") == FROZEN_PROFILE:
                    by_rep[rep][int(obj["offset_index"])] = obj
        shared = set.intersection(*(set(v) for v in by_rep.values()))
        if shared:
            recs = {rep: by_rep[rep][min(shared)] for rep in REP_ORDER}
            break
    if recs is None:
        raise RuntimeError(
            f"no stimuli.jsonl carries all of {REP_ORDER} for profile "
            f"{FROZEN_PROFILE!r}"
        )

    # (1) the generating parameters, compared exactly
    for key in PHYSICAL_KEYS:
        seen = {rep: recs[rep][key] for rep in REP_ORDER}
        if len(set(seen.values())) != 1:
            raise RuntimeError(
                "panel a must show ONE physical field, but the three selected "
                f"records diverge on {key!r}: "
                + ", ".join(f"{rep}={seen[rep]!r}" for rep in REP_ORDER)
            )
    # (2) a derived physical id over that whole tuple
    ids = {rep: _physical_id(recs[rep]) for rep in REP_ORDER}
    if len(set(ids.values())) != 1:
        raise RuntimeError(
            "panel a must show ONE physical field, but the derived physical_id "
            "diverges across representations: "
            + ", ".join(f"{rep}={ids[rep][:PHYSICAL_ID_CHARS]}" for rep in REP_ORDER)
        )
    offset_index = int(recs[REP_ORDER[0]]["offset_index"])
    if offset_index != FROZEN_OFFSET_INDEX:
        raise RuntimeError(
            f"the shared offset index is {offset_index}, not the "
            f"{FROZEN_OFFSET_INDEX} main Fig. 2 quotes"
        )

    # (3) one histogram, three serialisations, each hash-matched before drawing
    mu = float(recs[REP_ORDER[0]]["offset_radians"])
    hist = build_stimulus_histogram(FROZEN_PROFILE, mu)
    out = {}
    for rep, obj in recs.items():
        text = build_representation_prompt_from_histogram(rep, hist)
        if _sha(text) != obj["prompt_sha256"]:
            raise RuntimeError(
                f"regenerated {rep} prompt does not reproduce the stored "
                f"prompt_sha256 ({obj['prompt_sha256']}); refusing to print it"
            )
        if text != obj["serialized_prompt"]:
            raise RuntimeError(f"regenerated {rep} prompt differs from the stored text")
        out[rep] = {
            "record": obj,
            "text": text,
            "lines": text.splitlines(),
            "hist": hist,
            "sha": obj["prompt_sha256"],
        }

    # (4) the mass panel a draws is read back out of the centers payload, so
    #     the figure plots the numbers that were sent rather than a re-derivation
    theta, mass = _centers_bins(out["centers_24_standard"]["text"])
    centres = (EDGES[:-1] + EDGES[1:]) / 2.0
    if float(np.abs(theta - centres).max()) > 1e-6:
        raise RuntimeError("the serialised bin centres are not the canonical grid")
    if float(np.abs(mass - hist.fractions).max()) > 1e-6:
        raise RuntimeError("the serialised masses are not the regenerated field")

    field = {
        "record": recs[REP_ORDER[0]],
        "hist": hist,
        "theta": theta,
        "mass": mass,
        "mu": mu,
        "kappa": float(recs[REP_ORDER[0]]["concentration"]),
        "peer_count": int(hist.peer_count),
        "offset_index": offset_index,
        "physical_id": ids[REP_ORDER[0]],
        "n_printed_nonzero": int((mass > 0).sum()),
    }
    return out, field


def _verify_replay_prompts() -> dict:
    """Regenerate all 144 frozen replay prompts and check every SHA-256."""
    by_id = {f["field_id"]: f for f in REPLAY_FIELDS["fields"]}
    chars: dict[str, set] = {r: set() for r in REP_ORDER}
    n_ok = 0
    nonzero_bins = []
    for row in REPLAY_TARGETS["prompts"]:
        field = by_id[row["field_id"]]
        fracs = np.asarray(field["physical_histogram_24"], dtype=float)
        hist = RelativePhaseHistogram(
            edges=EDGES.copy(), fractions=fracs, peer_count=int(field["peer_count"])
        )
        text = build_representation_prompt_from_histogram(row["target_representation"], hist)
        if _sha(text) != row["prompt_sha256"] or len(text) != row["prompt_chars"]:
            raise RuntimeError(
                f"replay prompt {row['field_id']}/{row['target_representation']} "
                "does not reproduce its frozen sha256; refusing to print it"
            )
        n_ok += 1
        chars[row["target_representation"]].add(len(text))
    for field in REPLAY_FIELDS["fields"]:
        fracs = np.asarray(field["physical_histogram_24"], dtype=float)
        # occupancy as the *message* reports it: a bin counts as occupied when
        # its six-decimal printed mass is non-zero, the same criterion the
        # frozen example field in panel a is measured by
        nonzero_bins.append(int((np.round(fracs, 6) > 0).sum()))
    return {
        "n_verified": n_ok,
        "chars": {r: sorted(chars[r]) for r in REP_ORDER},
        "nonzero_bins": np.asarray(nonzero_bins),
    }


def _verify_collective(pairs, times=(0,)) -> dict:
    """Regenerate collective prompts from the stored phase snapshots.

    ``pairs`` is an iterable of ``(representation, trace_path)``. The prompts
    themselves were never stored, so each one is rebuilt from ``phases.npy``
    through the production observation operator and checked against the
    ``prompt_sha256`` the runner recorded.
    """
    chars: dict[str, set] = {r: set() for r in REP_ORDER}
    tokens: dict[str, set] = {r: set() for r in REP_ORDER}
    usage: Counter = Counter()
    n_ok = 0
    for rep, trace in pairs:
        phases = np.load(trace.parent / "phases.npy")
        with trace.open(encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                tok = obj.get("input_tokens")
                attempts = int(obj.get("attempts", 1))
                usage[(rep, attempts, int(tok) if tok else None)] += 1
                if tok and attempts == 1:
                    tokens[rep].add(int(tok))
                if obj["t"] not in times:
                    continue
                hist = relative_phase_histogram(phases[obj["t"]], obj["agent"])
                text = build_representation_prompt_from_histogram(rep, hist)
                if _sha(text) != obj["prompt_sha256"]:
                    raise RuntimeError(
                        f"{trace}: regenerated prompt does not match prompt_sha256"
                    )
                n_ok += 1
                chars[rep].add(len(text))
    return {
        "n_verified": n_ok,
        "chars": {r: sorted(chars[r]) for r in REP_ORDER},
        "tokens": {r: sorted(tokens[r]) for r in REP_ORDER},
        "usage": usage,
    }


def _gpt_pairs():
    for rep, session in GPT_SESSIONS.items():
        for trace in sorted((ROOT / session).glob("N17_*/trace.jsonl")):
            yield rep, trace


def _r2_pairs():
    for trace in sorted(R2_ROOT.glob("N17_*/*/trace.jsonl")):
        yield trace.parent.name, trace


def _r1_tokens() -> dict:
    """Per-call input tokens for the two non-OpenAI providers (attempts == 1)."""
    out: dict[str, dict] = {}
    for session in sorted(R1_ROOT.glob("*")):
        traces = sorted(session.rglob("trace.jsonl"))
        if not traces:
            continue
        cfg = json.loads((traces[0].parent / "resolved_config.json").read_text(encoding="utf-8"))
        per: dict[str, set] = {r: set() for r in REP_ORDER}
        usage: Counter = Counter()
        total = 0
        for trace in traces:
            with trace.open(encoding="utf-8") as handle:
                for line in handle:
                    obj = json.loads(line)
                    total += 1
                    rep = obj["target_representation"]
                    tok = obj.get("input_tokens")
                    attempts = int(obj.get("attempts", 1))
                    usage[(rep, attempts, int(tok) if tok else None)] += 1
                    if tok and attempts == 1:
                        per[rep].add(int(tok))
        out[cfg["backend"]] = {
            "model": cfg["model"],
            "tokens": {r: sorted(per[r]) for r in REP_ORDER},
            "usage": usage,
            "n": total,
        }
    return out


def _audit_usage(*usages: Counter) -> dict:
    """Per-call token constants and a record-by-record retry accounting audit.

    ``usages`` are ``Counter`` objects keyed by ``(representation, attempts,
    input_tokens or None)``. The per-call constant of a map is taken from the
    single-attempt calls (and has to *be* a single value, or this raises). Each
    multi-attempt record is then classified against that constant: either the
    provider reports the constant unchanged, or it reports ``attempts x`` the
    constant (usage accumulated across retries), or it is something else - in
    which case the "attempts x constant" reading would be wrong and the build
    stops rather than printing it.
    """
    merged: Counter = Counter()
    for usage in usages:
        merged.update(usage)
    const: dict[str, set] = {}
    for (rep, attempts, tok), n in merged.items():
        if attempts == 1 and tok:
            const.setdefault(rep, set()).add(tok)
    for rep, values in const.items():
        if len(values) != 1:
            raise RuntimeError(
                f"{rep} does not have one per-call token constant: {sorted(values)}"
            )
    constants = {rep: next(iter(values)) for rep, values in const.items()}

    n_calls = int(sum(merged.values()))
    n_multi = 0
    n_multi_scaled = 0
    n_multi_flat = 0
    max_attempts = 1
    for (rep, attempts, tok), n in merged.items():
        max_attempts = max(max_attempts, attempts)
        if attempts == 1:
            continue
        n_multi += n
        if tok is None:
            continue
        if tok == attempts * constants[rep]:
            n_multi_scaled += n
        elif tok == constants[rep]:
            n_multi_flat += n
        else:
            raise RuntimeError(
                f"{rep}: a {attempts}-attempt call reports {tok} input tokens, "
                f"which is neither the constant {constants[rep]} nor "
                f"{attempts} x it; the retry-accounting reading is wrong"
            )
    return {
        "constants": constants,
        "n_calls": n_calls,
        "n_multi": n_multi,
        "n_multi_scaled": n_multi_scaled,
        "n_multi_flat": n_multi_flat,
        "max_attempts": max_attempts,
    }


def _stimulus_survey() -> dict:
    """Stage-B character counts (from the stored text) and OpenAI token counts.

    The stored serialisations are re-hashed against their own ``prompt_sha256``
    before being counted, so the character counts are as trustworthy as the
    regenerated ones.
    """
    chars: dict[str, set] = {r: set() for r in REP_ORDER}
    per_profile: dict[tuple[str, str], int] = {}
    n = 0
    for path in sorted((RUNS / "stimulus_manifold").rglob("stimuli.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                rep = obj.get("representation")
                text = obj.get("serialized_prompt")
                if rep not in chars or not text:
                    continue
                if _sha(text) != obj["prompt_sha256"]:
                    raise RuntimeError(f"{path}: stored prompt fails its own sha256")
                chars[rep].add(len(text))
                per_profile[(rep, obj["profile"])] = len(text)
                n += 1
    tokens: dict[tuple[str, int], set] = {}
    for path in sorted((RUNS / "stimulus_manifold").rglob("trace.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                key = (obj.get("representation"), obj.get("profile"))
                tok = obj.get("input_tokens")
                if key not in per_profile or not tok:
                    continue
                if int(obj.get("attempts", 1)) != 1:
                    continue
                tokens.setdefault((key[0], per_profile[key]), set()).add(int(tok))
    return {
        "n_verified": n,
        "chars": {r: sorted(chars[r]) for r in REP_ORDER},
        "tokens": {k: sorted(v) for k, v in tokens.items()},
    }


# ---------------------------------------------------------------------------
# Page S2a - the frozen field and the three complete serialisations
# ---------------------------------------------------------------------------
A_LETTER_Y = 7.0
A_TOP = 11.5
A_H = 17.5
A_POLAR = (11.0, 11.0, 22.0, 22.0)
A_DENS = (50.0, A_TOP, 50.0, A_H)
A_HIST = (116.0, A_TOP, 56.0, A_H)
A_CAP_Y = 38.6         # sub-caption, line 1: what the view is
A_CAP2_Y = 41.4        # sub-caption, line 2: how it was obtained
A_NOTE_Y = 45.0        # conventions + field-identity paragraph

P_HEAD_Y = 60.0        # "prompt contract" heading baseline
P_TOP = 66.5           # first preamble line (va="top")
P_NUM_X = 13.0         # right edge of the line-number gutter
P_TAG_X = 14.5         # map tag
P_TXT_X = 26.5         # monospace column

X_RIGHT = 176.0        # right-hand ink limit of the canvas
COL_X = (12.0, 60.0, 118.0)
COL_LETTER_Y = 113.3
COL_NOTE_Y = 117.3
COL_TOP = 120.3

PI_TICKS = ([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi],
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])


def _draw_page_a(examples: dict, field: dict, replay: dict) -> Path:
    fig = new_figure(H_A)

    reps = REP_ORDER

    # ---- a  the one frozen example field, in three views ----------------
    theta, mass = field["theta"], field["mass"]
    panel_label_at(fig, 4.0, A_LETTER_Y, "a",
                   "Stored example physical field: one physical field, "
                   "three observation maps")
    fig_text_mm(fig, X_RIGHT, A_LETTER_Y,
                f"unimodal $\\kappa=${field['kappa']:.0f} \u00b7 offset index "
                f"{field['offset_index']} \u00b7 {field['peer_count']} peers",
                ha="right", va="baseline", fontsize=FS_SMALL, color=MUTED)

    # view 1 - the serialised 24-bin mass laid out on the circle. The raw peer
    # phases are not stored anywhere in the stimulus manifold, so this is the
    # circular view the data actually supports; no peer positions are invented.
    axP = mm_axes(fig, A_POLAR[0], A_POLAR[1], A_POLAR[2], A_POLAR[3], projection="polar")
    axP.bar(theta, mass, width=BIN_W, bottom=0.0, align="center",
            facecolor=GRAY_BOX["facecolor"], edgecolor=INK, linewidth=LW_HAIR)
    rmax = float(mass.max())
    axP.plot([0.0], [rmax * 1.04], marker="o", color=INK, ms=MS_POINT)
    axP.text(0.42, rmax * 0.80, "focal", ha="center", va="center",
             fontsize=FS_TINY, color=INK)
    axP.set_ylim(0.0, rmax * 1.1)
    axP.set_yticks([])
    axP.set_xticks([np.pi / 2, np.pi, 3 * np.pi / 2])
    axP.set_xticklabels([r"$\pi/2$", r"$\pm\pi$", r"$-\pi/2$"], fontsize=FS_TINY)
    axP.tick_params(pad=-1.0)
    axP.grid(color=RULE, lw=LW_HAIR)
    axP.spines["polar"].set_linewidth(LW_HAIR)
    axP.spines["polar"].set_color(RULE)

    # view 2 - the continuous density the field is defined by. unimodal_k9 is a
    # von Mises law discretised at the bin centres, so rho is exact from
    # (profile, concentration, offset_radians) and the markers show the
    # midpoint rule that turns it into the 24 masses of views 1 and 3.
    axD = mm_axes(fig, *A_DENS)
    phi = np.linspace(-np.pi, np.pi, 721)
    dens = np.exp(field["kappa"] * np.cos(phi - field["mu"]))
    area = np.trapezoid(dens, phi) if hasattr(np, "trapezoid") else np.trapz(dens, phi)
    dens = dens / area
    dens_at_bin = np.exp(field["kappa"] * np.cos(theta - field["mu"])) / area
    axD.fill_between(phi, 0.0, dens, facecolor=GRAY_BOX["facecolor"], linewidth=0)
    axD.plot(phi, dens, color=INK, lw=LW_LINE)
    axD.plot(theta, dens_at_bin, linestyle="none", marker="o", color=MUTED,
             ms=MS_POINT)
    axD.set_xlim(-np.pi, np.pi)
    axD.set_xticks(PI_TICKS[0])
    axD.set_xticklabels(PI_TICKS[1])
    axD.set_xlabel(r"relative phase $\varphi$ (rad)")
    axD.set_ylabel(r"density $\rho(\varphi)$")
    axD.set_ylim(bottom=0.0)

    # view 3 - the same 24 numbers as a mass histogram, parsed back out of the
    # centers payload printed in c
    axH = mm_axes(fig, *A_HIST)
    axH.step(np.concatenate([EDGES[:1], theta, EDGES[-1:]]),
             np.concatenate([mass[:1], mass, mass[-1:]]),
             where="mid", color=INK, lw=LW_LINE)
    axH.set_xlim(-np.pi, np.pi)
    axH.set_xticks(PI_TICKS[0])
    axH.set_xticklabels(PI_TICKS[1])
    axH.set_xlabel(r"bin centre (rad)")
    axH.set_ylabel("bin mass")
    axH.set_ylim(bottom=0.0)

    captions = (
        (A_POLAR[0], A_POLAR[2],
         "circular view of the 24 bin masses",
         "raw peer phases are not stored"),
        (A_DENS[0], A_DENS[2],
         "generating von Mises density",
         r"$\rho\propto e^{\kappa\cos(\varphi-\mu)}$, dots: $\rho$ at the 24 bin centres"),
        (A_HIST[0], A_HIST[2],
         "24-bin mass histogram",
         "the same 24 numbers, parsed back out of c"),
    )
    for x, w, line1, line2 in captions:
        fig_text_mm(fig, x + w / 2.0, A_CAP_Y, line1, ha="center", va="baseline",
                    fontsize=FS_TINY, color=INK)
        fig_text_mm(fig, x + w / 2.0, A_CAP2_Y, line2, ha="center", va="baseline",
                    fontsize=FS_TINY, color=MUTED)
    y_after = _para(
        fig, 12.0, A_NOTE_Y,
        f"One field, not three. {field['peer_count']} synthetic peers; the focal agent is "
        f"excluded and held at $\\varphi=0$; peer phases are taken relative to it and wrapped "
        f"to $[-\\pi,\\pi)$, then binned into 24 equal bins of width $2\\pi/24$, bin_00 "
        f"opening at $-\\pi$. b, c and d are the three complete serialised observations of "
        f"this same field: the record main Fig. 2 quotes in its panel a; the three "
        f"records agree exactly on profile, concentration, offset index and offset angle "
        f"($\\mu={field['mu']:.6f}$ rad).",
        X_RIGHT - 12.0)
    rule_mm(fig, 4.0, X_RIGHT, y_after + 1.4)

    # ---- shared prompt contract ------------------------------------------
    head = examples[reps[0]]["lines"][0]
    body = examples[reps[0]]["lines"][2:9]
    for rep in reps[1:]:
        if examples[rep]["lines"][0] != head or examples[rep]["lines"][2:9] != body:
            raise RuntimeError("the shared prompt contract is not identical across maps")

    fig_text_mm(fig, 4.0, P_HEAD_Y, "Prompt contract", ha="left", va="baseline",
                fontsize=FS_TITLE, fontweight="bold", color=INK)
    _para(fig, 33.0, P_HEAD_Y,
          "lines 1 and 3–9 are byte-identical in b, c and d; only lines 2 and 10 differ, "
          "so all three are given. The user message is exactly these ten lines followed by "
          "the map's payload column below.",
          X_RIGHT - 33.0)

    pitch = _pitch(FS_SMALL)
    rows: list[tuple[str, str | None, str]] = [("1", None, head)]
    rows += [("2", r, examples[r]["lines"][1]) for r in reps]
    rows += [(str(3 + i), None, ln) for i, ln in enumerate(body)]
    rows += [("10", r, examples[r]["lines"][9]) for r in reps]

    mono_texts = []
    for i, (num, rep, text) in enumerate(rows):
        y = P_TOP + i * pitch
        fig_text_mm(fig, P_NUM_X, y, num, ha="right", va="top",
                    fontsize=FS_TINY, color=MUTED, family="monospace")
        if rep is not None:
            fig_text_mm(fig, P_TAG_X, y, REP_SHORT[rep], ha="left", va="top",
                        fontsize=FS_TINY, color=REP[rep], fontweight="bold")
        mono_texts.append(
            fig_text_mm(fig, P_TXT_X, y, text, ha="left", va="top",
                        fontsize=FS_SMALL, color=INK if rep is None else REP[rep],
                        family="monospace")
        )
    n_pre = len(rows)
    rule_mm(fig, 4.0, X_RIGHT, P_TOP + n_pre * pitch + 2.2)

    # ---- b, c, d  the complete payloads ----------------------------------
    titles = {
        "moments_m1_m3": "Full moments observation",
        "centers_24_standard": "Full centers observation",
        "intervals_24_decimal6": "Full intervals observation",
    }
    notes = {
        "moments_m1_m3":
            "lines 11–16, 6 numeric values. Harmonics $m=1,2,3$ in order; within each "
            "harmonic the real (cos) component precedes the imaginary (sin) one. Signed, six "
            "decimals, fixed width. No per-bin mass is sent.",
        "centers_24_standard":
            "lines 11–34, 48 numeric values. 24 bin centres ascending from "
            "$-\\pi+\\pi/24$, standard origin (no half-bin shift). Centre signed, mass "
            "unsigned, both six decimals.",
        "intervals_24_decimal6":
            "lines 11–34, 72 numeric values. Half-open $[\\,\\mathrm{lo},\\mathrm{hi})$: "
            "the lower edge belongs to the bin, the upper edge does not. bin_00 (bold) opens "
            "at $-\\pi$; bin_23 (bold) closes at $+\\pi$, which wraps back onto $-\\pi$.",
    }
    col_w = {"moments_m1_m3": 46.0, "centers_24_standard": 56.0,
             "intervals_24_decimal6": X_RIGHT - COL_X[2]}
    pay_pitch = FS_BODY * PAY_LS * PT_MM
    payload_texts = []
    zero_counts = {}
    for k, rep in enumerate(reps):
        panel_label_at(fig, COL_X[k], COL_LETTER_Y, "bcd"[k], titles[rep],
                       title_color=REP[rep])
        lines = examples[rep]["lines"][10:]
        zero = sum(1 for ln in lines if ln.endswith(": 0.000000"))
        zero_counts[rep] = zero
        fig_text_mm(fig, COL_X[k], COL_NOTE_Y,
                    f"{len(examples[rep]['text'])} characters",
                    ha="left", va="baseline", fontsize=FS_TINY, color=MUTED,
                    family="monospace")
        for i, ln in enumerate(lines):
            faint = ln.endswith(": 0.000000")
            edge = rep == "intervals_24_decimal6" and i in (0, len(lines) - 1)
            payload_texts.append(
                fig_text_mm(fig, COL_X[k], COL_TOP + i * pay_pitch, ln,
                            ha="left", va="top", fontsize=FS_BODY,
                            color=MUTED if faint else INK, family="monospace",
                            fontweight="bold" if edge else "normal")
            )
        y_note = COL_TOP + len(lines) * pay_pitch + 3.2
        body = notes[rep]
        if zero:
            body += (f" {zero} of the 24 bins print 0.000000 (grey): mass below "
                     f"$5\\times10^{{-7}}$, not an empty bin.")
        _para(fig, COL_X[k], y_note, body, col_w[rep])

    _overflow(fig, "S2a")
    return save_fig(fig, "figS02a_observation_maps")


# ---------------------------------------------------------------------------
# Small vector glyphs (Arial has no U+2713/U+2717, so they are drawn)
# ---------------------------------------------------------------------------
def _check(ax, x: float, y: float, s: float, color: str) -> None:
    ax.plot([x - 0.42 * s, x - 0.10 * s, x + 0.46 * s],
            [y + 0.02 * s, y + 0.38 * s, y - 0.40 * s],
            color=color, lw=LW_LINE, solid_capstyle="round",
            solid_joinstyle="round", clip_on=False)


def _cross(ax, x: float, y: float, s: float, color: str) -> None:
    for dx in (-1.0, 1.0):
        ax.plot([x - 0.36 * s * dx, x + 0.36 * s * dx],
                [y - 0.36 * s, y + 0.36 * s],
                color=color, lw=LW_LINE, solid_capstyle="round", clip_on=False)


def _ring(ax, x: float, y: float, s: float, color: str) -> None:
    ax.add_patch(mpatches.Circle((x, y), 0.34 * s, facecolor="none",
                                 edgecolor=color, lw=LW_THIN, clip_on=False))


MARK = {"yes": _check, "no": _cross, "via": _ring}
MARK_COLOR = {"yes": PASS_GREEN, "no": STOP_RED, "via": MUTED}


# ---------------------------------------------------------------------------
# Page S2b - counts and retained features
# ---------------------------------------------------------------------------
H_B = 140.0
# Panel e is split into two independent blocks with their own headings, axes
# and units, because it carries two different kinds of quantity: constants of
# the observation map (left/right of the upper block) and a descriptor of the
# physical field (lower block). A third, smaller element audits the provider
# retry accounting that makes one provider's token counts *look* variable.
E_HEAD1_Y = 11.8
# The dot plot carries four rows, two of them two-line labels, so it is given a
# 6.5 mm row pitch rather than the 4.5 mm four rows would otherwise get.
E_DOT = (26.0, 15.2, 46.0, 26.0)
E_TAB = (86.0, 13.6, 90.0, 24.0)
E_HEAD2_Y = 52.5
E_OCC = (26.0, 55.5, 46.0, 14.0)
E_RET = (86.0, 55.0, 90.0, 22.0)
E_PROSE_Y = 80.0

F_LETTER_Y = 89.5
F_BOX = (12.0, 92.5, 162.0, 43.0)

# The two intervals rows are not two experiments: the encoder writes a shorter
# line 2 when the field carries exactly 240 peers (the dense stimulus families
# and panel a's example) and a longer one for every other peer count (the
# finite-peer stimulus families, the replay fields and the collective runs).
# See ``circlemap.representations.build_representation_prompt_from_histogram``.
CHAR_FORMS = (
    ("moments_m1_m3", None, "moments"),
    ("centers_24_standard", None, "centers"),
    ("intervals_24_decimal6", "dense", "intervals,\n240-peer fields"),
    ("intervals_24_decimal6", "general", "intervals,\nother peer counts"),
)
N_VALUES = {"moments_m1_m3": 6, "centers_24_standard": 48, "intervals_24_decimal6": 72}
N_LINES = {"moments_m1_m3": 16, "centers_24_standard": 34, "intervals_24_decimal6": 34}

PROVIDERS = (
    ("GPT-5.4-mini", "OpenAI"),
    ("Claude Haiku 4.5", "Anthropic"),
    ("Gemini 3.5 Flash", "Google"),
)


def _blank_mm_axes(fig, box):
    """A frame whose data units are millimetres, measured down from its top."""
    ax = mm_axes(fig, *box)
    ax.set_xlim(0.0, box[2])
    ax.set_ylim(box[3], 0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_visible(False)
    return ax


def _sub_head(fig, y_mm: float, text: str, note: str, x_mm: float = 4.0) -> None:
    """A bold sub-heading with a muted qualifier on one line, and a rule.

    The qualifier has to stay on a single line: the rule is drawn immediately
    beneath the heading, so a wrapped second line would fall through it and
    into the block below. The width is measured rather than trusted.
    """
    fig_text_mm(fig, x_mm, y_mm, text, ha="left", va="baseline",
                fontsize=FS_SMALL, fontweight="bold", color=INK)
    x_note = x_mm + _text_w_mm(fig, text, FS_SMALL, fontweight="bold") + 2.4
    room = X_RIGHT - x_note
    have = _text_w_mm(fig, note, FS_TINY)
    if have > room:
        raise RuntimeError(
            f"sub-heading note for {text!r} needs {have:.1f} mm but only "
            f"{room:.1f} mm is free; shorten it rather than letting it wrap"
        )
    fig_text_mm(fig, x_note, y_mm, note, ha="left", va="baseline",
                fontsize=FS_TINY, color=MUTED)
    rule_mm(fig, x_mm, X_RIGHT, y_mm + 1.6)


def _draw_page_b(examples, field, replay, stim, gpt, r2, r1) -> Path:
    fig = new_figure(H_B)

    # ---- e  two different quantities, kept apart -------------------------
    panel_label_at(fig, 4.0, 7.0, "e",
                   "Message length and field occupancy are different quantities")

    char_union = {r: sorted(set(stim["chars"][r]) | set(replay["chars"][r])
                            | set(gpt["chars"][r]) | set(r2["chars"][r]))
                  for r in REP_ORDER}
    for r in ("moments_m1_m3", "centers_24_standard"):
        if len(char_union[r]) != 1:
            raise RuntimeError(f"{r} character count is not a single value: {char_union[r]}")
    if len(char_union["intervals_24_decimal6"]) != 2:
        raise RuntimeError(
            "intervals character counts are not the expected two encoder forms: "
            f"{char_union['intervals_24_decimal6']}"
        )
    dense_chars, general_chars = char_union["intervals_24_decimal6"]

    # --- e, upper block: constants of the observation map -----------------
    _sub_head(
        fig, E_HEAD1_Y, "Representation-level constants",
        "one value per map, set by the fixed-width $\\pm$d.dddddd serialiser "
        "and the tokenizer. Point values, not distributions.")

    dot_vals, dot_cols, dot_labels = [], [], []
    for rep, form, label in CHAR_FORMS:
        if form is None:
            dot_vals.append(char_union[rep][0])
            dot_cols.append(REP[rep])
        else:
            dot_vals.append(dense_chars if form == "dense" else general_chars)
            dot_cols.append(REP_LIGHT[rep] if form == "dense" else REP[rep])
        dot_labels.append(label)

    axE = mm_axes(fig, *E_DOT)
    ypos = np.arange(len(dot_vals))
    for y, v, c in zip(ypos, dot_vals, dot_cols):
        axE.plot([0.0, v], [y, y], color=RULE, lw=LW_HAIR, zorder=1)
        axE.plot([v], [y], marker="o", color=c, ms=MS_MEAN,
                 linestyle="none", zorder=3)
        axE.text(v + 55.0, y, f"{v}", va="center", ha="left",
                 fontsize=FS_TINY, color=INK)
    axE.set_yticks(ypos)
    axE.set_yticklabels(dot_labels, fontsize=FS_TICK)
    axE.set_ylim(len(dot_vals) - 0.5, -0.5)
    axE.set_xlim(0, max(dot_vals) * 1.24)
    axE.set_xticks([0, 500, 1000, 1500])
    axE.set_xlabel("characters in the user message")
    axE.tick_params(axis="y", length=0)

    axT = _blank_mm_axes(fig, E_TAB)

    def _tok(store, rep):
        vals = store["tokens"][rep]
        if len(vals) != 1:
            raise RuntimeError(f"expected one per-call token value for {rep}, got {vals}")
        return vals[0]

    audits = {
        "OpenAI": _audit_usage(gpt["usage"]),
        "Anthropic": _audit_usage(r1["anthropic"]["usage"], r2["usage"]),
        "Google": _audit_usage(r1["google"]["usage"]),
    }
    # the R2 acquisition is the same provider and model as R1, so its constants
    # have to agree with R1's; disagreement would break the "one value" claim
    for rep in REP_ORDER:
        if _tok(r1["anthropic"], rep) != audits["Anthropic"]["constants"][rep]:
            raise RuntimeError(f"Anthropic token constant for {rep} is not shared by R1 and R2")

    col_x = (62.0, 75.0, 90.0)
    for x, rep in zip(col_x, REP_ORDER):
        axT.text(x, 2.0, REP_SHORT[rep], ha="right", va="baseline",
                 fontsize=FS_TINY, color=REP[rep], fontweight="bold")
    axT.plot([0.0, E_TAB[2]], [3.2, 3.2], color=RULE, lw=LW_HAIR, clip_on=False)

    struct_rows = [
        ("Numeric values in the message", [N_VALUES[r] for r in REP_ORDER]),
        ("Lines in the message", [N_LINES[r] for r in REP_ORDER]),
    ]
    for i, (label, vals) in enumerate(struct_rows):
        y = 6.0 + i * 3.2
        axT.text(0.0, y, label, ha="left", va="baseline", fontsize=FS_TINY, color=INK)
        for x, v in zip(col_x, vals):
            axT.text(x, y, f"{v}", ha="right", va="baseline", fontsize=FS_TINY,
                     color=INK, family="monospace")
    y0 = 6.0 + len(struct_rows) * 3.2 + 1.0
    axT.text(0.0, y0, "Input tokens per call", ha="left", va="baseline",
             fontsize=FS_TINY, fontweight="bold", color=INK)
    for i, (model, vendor) in enumerate(PROVIDERS):
        y = y0 + 3.2 + i * 3.2
        axT.text(2.4, y, model, ha="left", va="baseline", fontsize=FS_TINY, color=INK)
        axT.text(31.0, y, f"({vendor})", ha="left", va="baseline",
                 fontsize=FS_TINY, color=MUTED)
        for x, rep in zip(col_x, REP_ORDER):
            axT.text(x, y, f"{audits[vendor]['constants'][rep]}", ha="right",
                     va="baseline", fontsize=FS_TINY, color=INK, family="monospace")
    dense_tok = stim["tokens"].get(("intervals_24_decimal6", dense_chars), [])
    y_foot = y0 + 3.2 + len(PROVIDERS) * 3.2 + 0.4
    dense_note = (f"; the {dense_chars}-character 240-peer form costs {dense_tok[0]}"
                  if len(dense_tok) == 1 else "")
    axT.text(0.0, y_foot,
             f"Tokens are for the {general_chars}-character intervals form"
             f"{dense_note}.",
             ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)
    axT.text(0.0, y_foot + 2.9,
             "Line 2 names the 24 circular intervals unless the field has "
             "240 peers.",
             ha="left", va="baseline", fontsize=FS_TINY, color=MUTED)

    # --- e, lower block: a property of the physical field -----------------
    nz = replay["nonzero_bins"]
    _sub_head(
        fig, E_HEAD2_Y, "Field-dependent quantity",
        "how sparse the physical field is, counted in bins, not characters. It "
        "does not change message length: both bin encodings print all 24 rows.")

    axO = mm_axes(fig, *E_OCC)
    counts = np.bincount(nz, minlength=DEFAULT_N_BINS + 1)
    axO.bar(np.arange(counts.size), counts, width=0.82,
            facecolor=GRAY_BOX["facecolor"], edgecolor=INK, linewidth=LW_HAIR)
    axO.set_xlim(0.0, DEFAULT_N_BINS + 0.8)
    axO.set_xticks([0, 6, 12, 18, 24])
    axO.set_xlabel("bins with non-zero printed mass (of 24)")
    axO.set_ylabel(f"replay fields\n(n = {nz.size})")
    # The example field of panel a is a stimulus-manifold field, not one of the
    # replay fields counted here, so its 16 occupied bins land on an empty
    # column: the arrow marks where it sits on the axis, not a bar height. The
    # label is set to the right of the arrow so that it clears the bars.
    n_occ = field["n_printed_nonzero"]
    axO.annotate(
        f"example field of a,\nnot one of these {nz.size}",
        xy=(n_occ, 0.0), xytext=(n_occ + 0.9, float(counts.max()) * 0.52),
        ha="left", va="bottom", fontsize=FS_TINY, color=INK, linespacing=1.25,
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=LW_THIN,
                        shrinkA=0.0, shrinkB=1.0))

    # --- e, audit element: the retry artefact, kept out of the token display
    axR = _blank_mm_axes(fig, E_RET)
    axR.text(0.0, 2.6, "Retry audit", ha="left", va="baseline",
             fontsize=FS_TINY, fontweight="bold", color=INK)
    axR.text(13.0, 2.6, "why one provider's counts look variable; these calls "
             "are excluded above", ha="left", va="baseline",
             fontsize=FS_TINY, color=MUTED)
    head_x = (30.0, 42.0)
    axR.text(head_x[0], 5.8, "calls", ha="right", va="baseline",
             fontsize=FS_TINY, color=MUTED)
    axR.text(head_x[1], 5.8, "$\\geq2$ att.", ha="right", va="baseline",
             fontsize=FS_TINY, color=MUTED)
    axR.text(head_x[1] + 2.4, 5.8, "input_tokens on those calls", ha="left",
             va="baseline", fontsize=FS_TINY, color=MUTED)
    axR.plot([0.0, E_RET[2]], [7.0, 7.0], color=RULE, lw=LW_HAIR, clip_on=False)
    for i, (model, vendor) in enumerate(PROVIDERS):
        a = audits[vendor]
        y = 10.0 + i * 3.2
        axR.text(0.0, y, model, ha="left", va="baseline",
                 fontsize=FS_TINY, color=INK)
        axR.text(head_x[0], y, f"{a['n_calls']:,}".replace(",", "\u2009"),
                 ha="right", va="baseline", fontsize=FS_TINY, color=INK,
                 family="monospace")
        axR.text(head_x[1], y, f"{a['n_multi']}", ha="right", va="baseline",
                 fontsize=FS_TINY, color=INK, family="monospace")
        # A retried call either reports no input_tokens at all, or reports the
        # per-call constant unchanged, or reports attempts x that constant.
        # Both counted classes are quoted out of the same denominator, and the
        # silent calls are named rather than left to be inferred.
        n_silent = a["n_multi"] - a["n_multi_scaled"] - a["n_multi_flat"]
        if a["n_multi"] == 0:
            verdict = "no retried calls"
        elif a["n_multi_scaled"]:
            verdict = (f"attempts $\\times$ constant, "
                       f"{a['n_multi_scaled']} of {a['n_multi']}")
        else:
            verdict = (f"constant unchanged, "
                       f"{a['n_multi_flat']} of {a['n_multi']}")
        if n_silent:
            verdict += f"; {n_silent} report no count"
        axR.text(head_x[1] + 2.4, y, verdict, ha="left", va="baseline",
                 fontsize=FS_TINY, color=INK)
    n_dark = DEFAULT_N_BINS - n_occ
    _para(
        fig, 12.0, E_PROSE_Y,
        f"Across the {nz.size} stored replay fields "
        f"({REPLAY_FIELDS['fields'][0]['peer_count']} peers each) "
        f"{int(nz.min())}\u2013{int(nz.max())} of the 24 bins print non-zero mass (median "
        f"{int(np.median(nz))}). The example field of panel a is denser: {n_occ} of its 24 "
        f"bins print non-zero mass, and the remaining {n_dark} print 0.000000 only because "
        f"their mass falls below $5\\times10^{{-7}}$ and rounds to zero at six decimals; "
        f"every one of the 24 bins holds mass, none is empty. "
        f"Occupancy is what the field controls; the message length above is not.",
        X_RIGHT - 12.0)

    # ---- f  retained-feature inventory -----------------------------------
    panel_label_at(fig, 4.0, F_LETTER_Y, "f",
                   "Retained-feature inventory: the maps differ in content, not only in format")
    axF = _blank_mm_axes(fig, F_BOX)

    f_cols = (60.0, 72.0, 84.0)
    for x, rep in zip(f_cols, REP_ORDER):
        axF.text(x, 2.6, REP_SHORT[rep], ha="center", va="baseline",
                 fontsize=FS_SMALL, color=REP[rep], fontweight="bold")
    axF.text(98.0, 2.6, "what the difference does", ha="left", va="baseline",
             fontsize=FS_SMALL, color=MUTED)
    axF.plot([0.0, F_BOX[2]], [4.0, 4.0], color=RULE, lw=LW_HAIR, clip_on=False)

    f_rows = [
        ("First harmonic $m=1$ (polar order $r_1$, $\\psi_1$)",
         ("yes", "via", "via"),
         "printed outright by moments; a 24-term sum for the other two"),
        ("Second and third harmonics $m=2,3$",
         ("yes", "via", "via"),
         "the same three complex numbers, arrived at differently"),
        ("Harmonics $m\\geq4$",
         ("no", "via", "via"),
         "discarded by moments: 24 masses carry 12 harmonics, moments keep 3"),
        ("Per-bin mass, all 24 bins",
         ("no", "yes", "yes"),
         "local detail; identical numbers in centers and intervals"),
        ("Explicit bin-centre angle",
         ("no", "yes", "no"),
         "intervals leaves the centre implicit as the interval midpoint"),
        ("Explicit half-open bin edges",
         ("no", "no", "yes"),
         "only intervals states the boundary convention in the text"),
        ("Six-decimal fixed-width presentation",
         ("yes", "yes", "yes"),
         "identical numeric precision, so precision cannot explain the gap"),
        ("Peer count / absolute phase / agent identity",
         ("no", "no", "no"),
         "never serialised by any map"),
    ]
    pitch_f = 4.2
    for i, (label, marks, note) in enumerate(f_rows):
        y = 8.0 + i * pitch_f
        axF.text(0.0, y, label, ha="left", va="center", fontsize=FS_TINY, color=INK)
        for x, m in zip(f_cols, marks):
            MARK[m](axF, x, y, 2.2, MARK_COLOR[m])
        axF.text(98.0, y, note, ha="left", va="center", fontsize=FS_TINY, color=MUTED)
    y_leg = 8.0 + len(f_rows) * pitch_f + 1.0
    _check(axF, 1.2, y_leg, 2.2, PASS_GREEN)
    axF.text(3.4, y_leg, "printed explicitly", ha="left", va="center",
             fontsize=FS_TINY, color=MUTED)
    _ring(axF, 30.0, y_leg, 2.2, MUTED)
    axF.text(32.2, y_leg, "recoverable from what is printed", ha="left", va="center",
             fontsize=FS_TINY, color=MUTED)
    _cross(axF, 80.0, y_leg, 2.2, STOP_RED)
    axF.text(82.2, y_leg, "not present in the message", ha="left", va="center",
             fontsize=FS_TINY, color=MUTED)

    _overflow(fig, "S2b")
    return save_fig(fig, "figS02b_serialization_metadata")


# ---------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    examples, field = _frozen_examples()
    replay = _verify_replay_prompts()
    stim = _stimulus_survey()
    gpt = _verify_collective(list(_gpt_pairs()), times=(0, 50))
    r2 = _verify_collective(list(_r2_pairs()), times=(0,))
    r1 = _r1_tokens()

    page_a = _draw_page_a(examples, field, replay)
    _draw_page_b(examples, field, replay, stim, gpt, r2, r1)
    return page_a


if __name__ == "__main__":
    print(build())
