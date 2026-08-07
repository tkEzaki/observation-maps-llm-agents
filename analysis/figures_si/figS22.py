"""Supplementary Figure S22 - serialization-length control on the replay panel.

Four conditions, each a pure re-serialization of one of two content classes at
fixed numeric precision, with no filler text:

    condition          content class   form       chars   tokens
    moments_compact    moments3        compact      752      317
    moments_standard   moments3        standard     789      332
    centers_compact    histogram24     compact      971      405
    centers_standard   histogram24     standard    1429      588

The point of the page is that the length axis crosses the content axis:
``centers_compact`` sits 458 characters from its own content-class partner but
only 182 characters from ``moments_standard``. A prompt-length account predicts
responses group by length, a content account predicts they group by content
class, and the two predictions are crossed by construction.

Nothing on this page is hard-coded. Panels a-c are rebuilt live: the four
prompts are regenerated from one frozen replay field through the production
builder (``circlemap.serialization_length_controls.build_slc_prompt``), decoded
back to numbers, and checked against
``analysis/matched_rep_collective/slc_information_audit.json`` before a single
character is printed; a disagreement raises. Panels d-h are read from
``analysis/matched_rep_collective/slc_primary/decision.json``.

Panel e carries the length-immune contrast. One moments prompt is held fixed as
an anchor and compared with the two information-identical histogram prompts: the
one nearer to the anchor in characters and the one further away. Any monotone
increasing function of prompt length requires the further prompt to be at least
as distant in response, and in both anchors the nearer prompt is significantly
the more distant, paired within field. The two single-class reference contrasts
in panel h cannot make that argument, because their character gaps differ by an
order of magnitude; they are labelled confounded rather than read as evidence
about serialization form.

Two render modes
----------------
The acquisition behind ``decision.json`` is paid and may not exist yet. If the
artifact is absent the design panels (a-c) are drawn in full and the response
panels (d-h) are drawn as marked empty axes; a warning goes to stdout and the
build still succeeds. A single ``HAVE_RESULTS`` flag guards the difference, so
the layout is written once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

from analysis.figures_si.style import (
    CAPSIZE,
    FS_SMALL,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    MS_MEAN,
    MS_POINT,
    MS_SERIES,
    MUTED,
    PASS_GREEN,
    REP,
    ROOT,
    STOP_RED,
    VERDICT,
    apply_style,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label_at,
    rule_mm,
    save_fig,
    trim_spines,
)

from analysis.matched_rep_collective.analyze_slc_control import (
    ARMS,
    CONTENT_PARTNER,
    LENGTH_NEIGHBOUR,
    PAIRS,
    PIVOT,
    _pair_key,
)
from circlemap.costing import estimate_tokens_from_characters
from circlemap.observation import RelativePhaseHistogram
from circlemap.representations import build_representation_prompt_from_histogram
from circlemap.serialization_length_controls import (
    N_BINS,
    SLC_CONTENT,
    SLC_FORM,
    build_slc_prompt,
    decode_centers,
    decode_moments,
)

MRC = ROOT / "analysis" / "matched_rep_collective"
AUDIT_PATH = MRC / "slc_information_audit.json"
FIELDS_PATH = MRC / "replay_fields_v0_1.json"
DECISION_PATH = MRC / "slc_primary" / "decision.json"

#: The primary encodings the two standard conditions must reproduce byte for byte.
PRIMARY_OF_CLASS = {"moments3": "moments_m1_m3", "histogram24": "centers_24_standard"}

#: Ordered by prompt length, which is the order panel b draws.
BY_LENGTH = ("moments_compact", "moments_standard", "centers_compact", "centers_standard")

CLASS_COLOR = {"moments3": REP["moments_m1_m3"], "histogram24": REP["centers_24_standard"]}
CLASS_LABEL = {"moments3": "moments3 content", "histogram24": "histogram24 content"}
SHORT = {
    "moments_standard": "mom std",
    "moments_compact": "mom cmp",
    "centers_standard": "cen std",
    "centers_compact": "cen cmp",
}
#: Panels d, f and g colour a pair by whether it holds the content class fixed.
SAME_CONTENT = "#009E73"
DIFF_CONTENT = "#CC79A7"

DTV = r"$d_{\mathrm{TV}}$"
PT_MM = 25.4 / 72.0
LS_MONO = 1.30
CODE_INK = "#3A3A3A"

X_LEFT = 4.4
X_RIGHT = 176.0
H_MM = 225.0

# --- canvas plan (mm from the top-left) ------------------------------------
A_LETTER_Y = 7.0
A_NOTE_Y = 10.9
A_TOP = 14.6
#: The two columns are sized to their payloads rather than split evenly: the
#: wider right-hand column fits the 24 masses in three display lines instead of
#: four, which is 2.7 mm of page for no loss of printed character.
A_COL_X = (X_LEFT, 87.2)
A_COL_W = (78.8, 88.8)
A_RULE_GAP = 2.4          # the rule follows the taller column, it is not fixed
#: Bin lines printed verbatim before the elision marker in centers_standard.
A_BINS_SHOWN = 2

B_LETTER_Y = 66.6
B_L, B_TOP, B_W, B_H = 15.0, 69.0, 152.0, 19.0

R3_LETTER_Y = 98.6
C_L, C_TOP, C_W, C_H = 34.0, 101.0, 36.0, 16.0
D_L, D_TOP, D_W, D_H = 92.0, 101.0, 84.0, 17.0
D_KEY_Y = 126.4

# e: the anchored monotonicity test. Two axes, one letter: the crossing on the
# left, the paired excess it implies on the right.
R4_LETTER_Y = 131.4
E_TOP, E_H = 133.8, 22.0
EA_L, EA_W = 22.0, 82.0
EB_L, EB_W = 126.0, 50.0
#: Anchor-group geometry for panel e, in the crossing panel's own x units.
E_PITCH = 2.0             # distance between the two anchor groups
E_FAR_DX = 0.66           # further-in-length prompt, right of the nearer one
E_EXC_DX = 0.14           # excess bracket, right of the further prompt

R5_LETTER_Y = 167.4
F_L, F_TOP, F_W, F_H = 34.0, 169.8, 66.0, 17.5
G_L, G_W, G_GAP = 118.0, 25.0, 9.0
G_TOP, G_H = 169.8, 17.5
G_NOTE_Y = 196.6

# the strip is the lowest ink on the page: its box sits 2.6 mm off the trim,
# which check_figures requires to be at least 2.0 mm
H_LETTER_Y = 200.5
H_L, H_TOP, H_W, H_H = 13.0, 202.9, 163.0, 19.5

# Panel h anchors, in axes fractions of the H_H-tall box. The item pitch is
# derived from the type size rather than guessed, and _gate_rows asserts that
# the last row still sits inside the box, which is how figS21's panel i
# overflowed before its spacing was pinned.
H_X_MARK = 0.022
H_X_LABEL = 0.045
H_X_STAT = 0.400
H_X_RIGHT = 0.978
H_Y_TOP = 0.930
H_ROW_LEAD = 1.52         # verdict rows are set tighter than running text


# ---------------------------------------------------------------------------
# Measured text helpers - nothing may silently run off a 180 mm canvas
# ---------------------------------------------------------------------------
def _text_w_mm(fig, s: str, fontsize: float, **kw) -> float:
    art = fig.text(0.0, 0.0, s, fontsize=fontsize, **kw)
    bb = art.get_window_extent(renderer=fig.canvas.get_renderer())
    art.remove()
    return float(bb.width) / float(fig.dpi) * 25.4


def _pitch(fontsize: float, lead: float = LS_MONO) -> float:
    return fontsize * lead * PT_MM


def _mono_adv(fig, fontsize: float) -> float:
    """Advance width of one monospace character, in mm."""
    return _text_w_mm(fig, "M" * 40, fontsize, family="monospace") / 40.0


_ATOM_RE = re.compile(r"\S+\s*|\s+")


def _mono_flow(fig, x_mm: float, y_mm: float, w_mm: float, segments, fontsize: float):
    """Lay coloured segments out as a fixed-width column; return the next y.

    ``segments`` are ``(text, colour)``; a newline forces a break. Wrapping is
    display-only: the column breaks between whole tokens where it can and
    falls back to a terminal-style character break for a token longer than the
    column, so no character is ever added, removed or reordered.
    """
    adv = _mono_adv(fig, fontsize)
    n_cols = max(1, int(w_mm // adv))
    pitch = _pitch(fontsize)
    col, y = 0, y_mm

    def _put(chunk: str, colour: str) -> None:
        nonlocal col
        fig_text_mm(fig, x_mm + col * adv, y, chunk, ha="left", va="top",
                    fontsize=fontsize, color=colour, family="monospace")
        col += len(chunk)

    for text, colour in segments:
        for k, line in enumerate(text.split("\n")):
            if k:
                col, y = 0, y + pitch
            atoms = _ATOM_RE.findall(line) or ([line] if line else [])
            for atom in atoms:
                if len(atom) <= n_cols and col + len(atom) > n_cols:
                    col, y = 0, y + pitch
                    if not atom.strip():
                        continue          # never open a line with white space
                while atom:
                    take = min(len(atom), n_cols - col)
                    if take <= 0:
                        col, y = 0, y + pitch
                        continue
                    _put(atom[:take], colour)
                    atom = atom[take:]
    return y + pitch


def _overflow(fig, tag: str) -> None:
    """Report any text that leaves the canvas or the right-hand ink limit."""
    W, H = (v * 25.4 for v in fig.get_size_inches())
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    for ax in fig.axes:
        artists.extend(ax.texts)
    n_bad = 0
    for art in artists:
        try:
            bb = art.get_window_extent(renderer=renderer)
        except Exception:      # pragma: no cover - unrenderable artist
            continue
        x0 = bb.x0 / fig.dpi * 25.4
        x1 = bb.x1 / fig.dpi * 25.4
        y_bottom = H - bb.y0 / fig.dpi * 25.4
        y_top = H - bb.y1 / fig.dpi * 25.4
        if x1 > X_RIGHT + 1.6 or x0 < 1.6 or y_bottom > H - 1.6 or y_top < 1.0:
            n_bad += 1
            print(f"[figS22:{tag}] overflow x=[{x0:.1f},{x1:.1f}] "
                  f"y=[{y_top:.1f},{y_bottom:.1f}]: "
                  f"{art.get_text()[:56]!r}", file=sys.stderr)
    if n_bad:
        print(f"[figS22:{tag}] {n_bad} overflowing text artist(s)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Design side: rebuild the prompts and audit them before printing anything
# ---------------------------------------------------------------------------
_MOMENT_RE = re.compile(r"[+-]\d\.\d{6}")
_BIN_RE = re.compile(r"^(bin_\d+ center )([+-]\d\.\d{6})(: )(\d\.\d{6})$")


def _sha_file(path: Path) -> str:
    """Newline-normalized digest of a text artifact.

    Python's text writer translates \\n to \\r\\n on Windows, so the raw bytes of
    a JSON artifact, and therefore its digest, depend on the platform that wrote
    it. Normalizing to LF before hashing makes the integrity check survive an
    artifact and a decision produced on different platforms, or a checkout that
    rewrote line endings, while still catching any change to the content.
    """
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()


def _histogram(field: dict) -> RelativePhaseHistogram:
    return RelativePhaseHistogram(
        edges=np.linspace(-np.pi, np.pi, N_BINS + 1),
        fractions=np.asarray(field["physical_histogram_24"], dtype=float),
        peer_count=int(field.get("peer_count", 16)),
    )


def _load_design() -> dict:
    """Rebuild the four prompts and verify every number this page will print."""
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if not audit.get("pass"):
        raise RuntimeError("slc_information_audit.json does not report pass; refusing to draw")
    if int(audit["n_fields"]) != 48:
        raise RuntimeError(f"the audit covers {audit['n_fields']} fields, expected the 48-field panel")

    fields = json.loads(FIELDS_PATH.read_text(encoding="utf-8"))["fields"]
    field = fields[0]
    hist = _histogram(field)
    prompts = {v: build_slc_prompt(v, hist) for v in ARMS}

    chars, tokens = {}, {}
    for v in ARMS:
        row = audit["prompt_chars"][v]
        if not (row["min"] == row["max"] == int(row["median"])):
            raise RuntimeError(f"{v}: prompt length is not constant over the frozen panel")
        if len(prompts[v]) != int(row["median"]):
            raise RuntimeError(
                f"{v}: the rebuilt prompt is {len(prompts[v])} characters, the audit "
                f"records {int(row['median'])}; refusing to print an unverified prompt"
            )
        tok = audit["prompt_tokens"][v]
        if not (tok["min"] == tok["max"] == int(tok["median"])):
            raise RuntimeError(f"{v}: prompt token count is not constant over the frozen panel")
        if estimate_tokens_from_characters(prompts[v]) != int(tok["median"]):
            raise RuntimeError(f"{v}: the rebuilt prompt does not reproduce the audited token count")
        if audit["variants"][v] != {"content": SLC_CONTENT[v], "form": SLC_FORM[v]}:
            raise RuntimeError(f"{v}: the audit disagrees with the builder about content or form")
        chars[v] = int(row["median"])
        tokens[v] = int(tok["median"])

    # the two standard conditions are the primary encodings, byte for byte
    for cls, primary in PRIMARY_OF_CLASS.items():
        variant = f"{'moments' if cls == 'moments3' else 'centers'}_standard"
        if prompts[variant] != build_representation_prompt_from_histogram(primary, hist):
            raise RuntimeError(f"{variant} is not byte-identical to the primary {primary} prompt")
    if not all(audit["standard_arms_byte_identical_to_primary"].values()):
        raise RuntimeError("the audit does not carry byte identity for both standard conditions")

    # information equivalence, recomputed live and matched to the audit
    eq = audit["information_equivalence"]
    gap_mom = float(np.max(np.abs(
        decode_moments("moments_standard", prompts["moments_standard"])
        - decode_moments("moments_compact", prompts["moments_compact"]))))
    c_std, m_std = decode_centers("centers_standard", prompts["centers_standard"])
    c_cmp, m_cmp = decode_centers("centers_compact", prompts["centers_compact"])
    gap_mass = float(np.max(np.abs(m_std - m_cmp)))
    gap_centre = float(np.max(np.abs(c_std - c_cmp)))
    if gap_mom > eq["max_abs_moment_component_gap"]:
        raise RuntimeError("the rebuilt moments exceed the audited component gap")
    if gap_mass > eq["max_abs_bin_mass_gap"]:
        raise RuntimeError("the rebuilt bin masses exceed the audited mass gap")
    if gap_centre > eq["max_abs_bin_centre_gap_print_precision"] * (1.0 + 1e-9):
        raise RuntimeError("the rebuilt bin centres exceed the audited print-precision gap")
    if eq["max_abs_moment_component_gap"] != 0.0 or eq["max_abs_bin_mass_gap"] != 0.0:
        raise RuntimeError("the audit no longer reports exact numeric equivalence")

    # the crossing itself
    cross = audit["length_axis_crosses_content_axis"]
    own = abs(chars["centers_compact"] - chars["centers_standard"])
    other = abs(chars["centers_compact"] - chars["moments_standard"])
    if own != int(cross["chars_centers_compact_to_centers_standard"]):
        raise RuntimeError("the within-class character gap disagrees with the audit")
    if other != int(cross["chars_centers_compact_to_moments_standard"]):
        raise RuntimeError("the cross-class character gap disagrees with the audit")
    if not (cross["crossed"] and other < own):
        raise RuntimeError("the length axis no longer crosses the content axis")

    # the shared instruction contract, asserted rather than assumed
    lines = {v: prompts[v].splitlines() for v in ARMS}
    for i in [0] + list(range(2, 9)):
        if len({lines[v][i] for v in ARMS}) != 1:
            raise RuntimeError(f"instruction line {i + 1} is not shared by all four conditions")
    if len({lines[v][1] for v in ARMS}) != 2:
        raise RuntimeError("line 2 does not split the four conditions into exactly two content classes")

    return {
        "audit": audit,
        "audit_sha256": _sha_file(AUDIT_PATH),
        "field": field,
        "prompts": prompts,
        "lines": lines,
        "chars": chars,
        "tokens": tokens,
        "cross": cross,
        "eq": eq,
        "n_instruction_lines": 9,
    }


def _load_decision(path: Path, design: dict) -> dict | None:
    """Read the acquisition verdict, or return None if it does not exist yet."""
    if not path.exists():
        return None
    dec = json.loads(path.read_text(encoding="utf-8"))
    for v in ARMS:
        if int(dec["arms"][v]["prompt_chars"]) != design["chars"][v]:
            raise RuntimeError(f"{v}: decision.json prompt_chars disagrees with the audit")
        if dec["arms"][v]["content_class"] != SLC_CONTENT[v]:
            raise RuntimeError(f"{v}: decision.json content class disagrees with the builder")
    if dec["hashes"]["information_audit_sha256"] != design["audit_sha256"]:
        raise RuntimeError(
            "decision.json was written against a different slc_information_audit.json; "
            "re-run analyze_slc_control.py before drawing this page"
        )
    keys = {_pair_key(a, b) for a, b in PAIRS}
    if set(dec["pairwise_mean_TV"]) != keys or set(dec["pairwise_bootstrap"]) != keys:
        raise RuntimeError("decision.json does not carry all six condition pairs")
    _check_anchored(dec, design)
    return dec


def _check_anchored(dec: dict, design: dict) -> None:
    """Verify the length-immune contrast panel e is about to draw.

    Panel e claims something specific: the two prompts compared against an
    anchor carry identical information and differ only in serialization, one is
    nearer to the anchor in characters than the other, and the excess drawn as
    the gap between the two means is that gap. All four are checked here, so a
    changed analysis script cannot leave the figure asserting the old geometry.
    """
    sec = dec["secondary"]
    if "reference_contrast_caveat" not in sec:
        raise RuntimeError(
            "decision.json carries no secondary.reference_contrast_caveat; the two "
            "single-class reference contrasts must not be drawn unqualified"
        )
    amt = sec.get("anchored_monotonicity_test")
    if amt is None:
        raise RuntimeError(
            "decision.json carries no secondary.anchored_monotonicity_test; re-run "
            "analysis/matched_rep_collective/analyze_slc_control.py before drawing"
        )
    chars = design["chars"]
    for anchor, row in amt["anchors"].items():
        near, far = row["nearer_in_length"], row["further_in_length"]
        if SLC_CONTENT[near] != SLC_CONTENT[far]:
            raise RuntimeError(
                f"{anchor}: the two compared prompts are not one content class, so "
                "they are not information-identical"
            )
        if SLC_CONTENT[anchor] == SLC_CONTENT[near]:
            raise RuntimeError(f"{anchor}: the anchor shares a content class with {near}")
        for target, key in ((near, "nearer_char_gap"), (far, "further_char_gap")):
            if abs(chars[anchor] - chars[target]) != int(row[key]):
                raise RuntimeError(
                    f"{anchor}: decision.json records {row[key]} characters to "
                    f"{target}, the audited prompts differ by "
                    f"{abs(chars[anchor] - chars[target])}"
                )
        if int(row["nearer_char_gap"]) >= int(row["further_char_gap"]):
            raise RuntimeError(
                f"{anchor}: the prompt named nearer in length is not the nearer one"
            )
        for target, value in ((near, row["nearer_mean_TV"]), (far, row["further_mean_TV"])):
            key = _pair_key(*sorted((anchor, target), key=ARMS.index))
            if abs(float(dec["pairwise_mean_TV"][key]) - float(value)) > 1e-12:
                raise RuntimeError(
                    f"{anchor}: the anchored mean TV to {target} disagrees with "
                    f"pairwise_mean_TV[{key}]"
                )
        excess = float(row["paired_excess_near_minus_far"]["mean"])
        drawn = float(row["nearer_mean_TV"]) - float(row["further_mean_TV"])
        if abs(excess - drawn) > 1e-9:
            raise RuntimeError(
                f"{anchor}: the paired excess ({excess:.6f}) is not the gap between "
                f"the two means panel e draws it as ({drawn:.6f})"
            )


# ---------------------------------------------------------------------------
# a  prompt construction
# ---------------------------------------------------------------------------
def _moment_segments(text: str, colour: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    i = 0
    for m in _MOMENT_RE.finditer(text):
        if m.start() > i:
            out.append((text[i:m.start()], CODE_INK))
        out.append((m.group(0), colour))
        i = m.end()
    if i < len(text):
        out.append((text[i:], CODE_INK))
    return out


def _mass_segments(text: str, colour: str) -> list[tuple[str, str]]:
    """One coloured token per mass; the separating comma rides with its value
    so the column never breaks a number in half."""
    values = text.split(",")
    return [(v + ("," if k + 1 < len(values) else ""), colour)
            for k, v in enumerate(values)]


def _bin_segments(line: str, colour: str) -> list[tuple[str, str]]:
    m = _BIN_RE.match(line)
    if m is None:
        raise RuntimeError(f"unexpected centers_standard line: {line!r}")
    head, centre, sep, mass = m.groups()
    return [(head, CODE_INK), (centre, MUTED), (sep, CODE_INK), (mass, colour)]


def _entries(design: dict, variant: str) -> list[tuple[str, object]]:
    """Display entries for one condition: verbatim payload, elisions marked."""
    lines = design["lines"][variant][design["n_instruction_lines"]:]
    colour = CLASS_COLOR[SLC_CONTENT[variant]]
    header, body = lines[0], lines[1:]
    out: list[tuple[str, object]] = [("code", [(header + "\n", CODE_INK)])]
    if variant == "moments_compact":
        out.append(("code", _moment_segments(body[0], colour)))
    elif variant == "moments_standard":
        segs: list[tuple[str, str]] = []
        for k, line in enumerate(body):
            if k:
                segs.append(("\n", CODE_INK))
            segs.extend(_moment_segments(line, colour))
        out.append(("code", segs))
    elif variant == "centers_compact":
        out.append(("code", _mass_segments(body[0], colour)))
    else:
        keep, tail = body[:A_BINS_SHOWN], body[-1]
        segs = []
        for k, line in enumerate(keep):
            if k:
                segs.append(("\n", CODE_INK))
            segs.extend(_bin_segments(line, colour))
        out.append(("code", segs))
        n_elided = len(body) - A_BINS_SHOWN - 1
        out.append(("elide", f"bins {A_BINS_SHOWN:02d} to {len(body) - 2:02d}: "
                             f"{n_elided} further lines, same form"))
        out.append(("code", _bin_segments(tail, colour)))
    return out


def _condition_block(fig, x: float, y: float, w: float, design: dict, variant: str) -> float:
    colour = CLASS_COLOR[SLC_CONTENT[variant]]
    fig_text_mm(fig, x, y, variant, ha="left", va="top", fontsize=FS_SMALL,
                color=colour, fontweight="bold")
    fig_text_mm(fig, x + w, y,
                f"{design['chars'][variant]} chars",
                ha="right", va="top", fontsize=FS_TINY, color=MUTED)
    y += _pitch(FS_SMALL, 1.60)
    for kind, payload in _entries(design, variant):
        if kind == "elide":
            fig_text_mm(fig, x, y, payload, ha="left", va="top", fontsize=FS_TINY,
                        color=MUTED, style="italic")
            y += _pitch(FS_TINY)
        else:
            y = _mono_flow(fig, x, y, w, payload, FS_TINY)
    return y


def _panel_a(fig, design: dict) -> None:
    panel_label_at(fig, X_LEFT, A_LETTER_Y, "a", "Prompt construction, one replay field")
    fid = design["field"]["field_id"]
    fig_text_mm(fig, X_LEFT, A_NOTE_Y,
                f"{fid} · replay fields sha256 {_sha_file(FIELDS_PATH)[:12]}… · "
                f"{design['n_instruction_lines']}-line instruction block identical in all "
                "four conditions, not printed",
                ha="left", va="top", fontsize=FS_TINY, color=MUTED, family="monospace")

    pairs = (("moments_compact", "moments_standard", "identical six values, identical precision"),
             ("centers_compact", "centers_standard", f"identical {N_BINS} masses, identical precision"))
    y_end = A_TOP
    for (cmp_v, std_v, note), x, w in zip(pairs, A_COL_X, A_COL_W):
        colour = CLASS_COLOR[SLC_CONTENT[cmp_v]]
        y = _condition_block(fig, x, A_TOP, w, design, cmp_v) + 0.9
        fig_text_mm(fig, x + 1.4, y, "↕ " + note, ha="left", va="top",
                    fontsize=FS_TINY, color=colour, fontweight="bold")
        y += _pitch(FS_TINY, 1.75)
        y_end = max(y_end, _condition_block(fig, x, y, w, design, std_v))
    y_rule = y_end + A_RULE_GAP
    if y_rule > B_LETTER_Y - 2.5:
        raise RuntimeError(
            f"panel a runs to {y_end:.1f} mm and its rule would land at {y_rule:.1f} mm, "
            f"inside panel b's letter at {B_LETTER_Y:.1f} mm"
        )
    rule_mm(fig, X_LEFT, X_RIGHT, y_rule)


# ---------------------------------------------------------------------------
# b  the crossing
# ---------------------------------------------------------------------------
def _bracket(ax, x0: float, x1: float, y: float, tick: float, colour: str) -> None:
    ax.plot([x0, x1], [y, y], color=colour, lw=LW_LINE, solid_capstyle="butt", zorder=4)
    for xe in (x0, x1):
        ax.plot([xe, xe], [y, y + tick], color=colour, lw=LW_LINE,
                solid_capstyle="butt", zorder=4)


#: Where each condition's stack of labels hangs relative to its own mark. The
#: two moments conditions are 37 characters apart, which is narrower than their
#: own labels, so they are anchored away from each other instead of centred.
B_LABEL_HA = {
    "moments_compact": ("right", -9.0),
    "moments_standard": ("left", 9.0),
    "centers_compact": ("center", 0.0),
    "centers_standard": ("center", 0.0),
}
B_Y_MARK = 0.34
B_Y_LABEL = 0.42
#: The condition name sits directly above its own two-line count stack. The
#: offset is measured from the type size, not guessed, so B_H can change without
#: the name dropping onto the "752 chars" line underneath it.
B_Y_NAME = B_Y_LABEL + (2.0 * _pitch(FS_TINY) + 0.5) / B_H


def _panel_b(fig, design: dict) -> None:
    ax = mm_axes(fig, B_L, B_TOP, B_W, B_H)
    panel_label_at(fig, X_LEFT, B_LETTER_Y, "b",
                   "The length axis crosses the content axis")
    chars, tokens, cross = design["chars"], design["tokens"], design["cross"]

    ax.set_xlim(630, 1530)
    ax.set_ylim(0, 1)
    ax.set_xticks([700, 900, 1100, 1300, 1500])
    ax.set_yticks([])
    ax.set_xlabel("prompt characters (constant over the 48 fields)")
    for name in ("left", "right", "top"):
        ax.spines[name].set_visible(False)
    trim_spines(ax, y=False)

    for v in BY_LENGTH:
        colour = CLASS_COLOR[SLC_CONTENT[v]]
        filled = SLC_FORM[v] == "standard"
        ha, dx = B_LABEL_HA[v]
        ax.plot([chars[v], chars[v]], [0.0, B_Y_MARK], color=colour, lw=LW_THIN, zorder=2)
        ax.plot([chars[v]], [B_Y_MARK], marker="o", ms=MS_MEAN,
                mfc=colour if filled else "white", mec=colour,
                mew=LW_LINE, zorder=5, clip_on=False)
        ax.text(chars[v] + dx, B_Y_LABEL, f"{chars[v]} chars",
                ha=ha, va="bottom", fontsize=FS_TINY, color=MUTED, linespacing=1.30)
        ax.text(chars[v] + dx, B_Y_NAME, v, ha=ha, va="bottom",
                fontsize=FS_TINY, color=colour, fontweight="bold")

    # 458 characters inside the histogram class, drawn above the marks
    x0, x1 = chars["centers_compact"], chars["centers_standard"]
    _bracket(ax, x0, x1, 0.855, -0.055, CLASS_COLOR["histogram24"])
    ax.text((x0 + x1) / 2.0, 0.885,
            f"{int(cross['chars_centers_compact_to_centers_standard'])} chars · same content",
            ha="center", va="bottom", fontsize=FS_TINY,
            color=CLASS_COLOR["histogram24"], fontweight="bold")

    # 182 characters across the content classes, drawn below the marks
    x0, x1 = chars["moments_standard"], chars["centers_compact"]
    _bracket(ax, x0, x1, 0.215, 0.055, INK)
    # the four stems run down to the axis, so this label carries its own white
    # ground rather than being cut in half by the two it sits between
    ax.text((x0 + x1) / 2.0, 0.175,
            f"{int(cross['chars_centers_compact_to_moments_standard'])} chars · "
            "different content",
            ha="center", va="top", fontsize=FS_TINY, color=INK, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=6)

    # a compact key, kept left of the upper bracket. Content classes and mark
    # fill share one line: on a 19 mm panel a second key line lands on the
    # condition names, which stand 15.6 mm up from the axis.
    for k, cls in enumerate(("moments3", "histogram24")):
        xk = 0.010 + k * 0.135
        ax.plot([xk], [0.965], marker="o", ms=MS_POINT, color=CLASS_COLOR[cls],
                transform=ax.transAxes, clip_on=False, zorder=6)
        ax.text(xk + 0.014, 0.965, CLASS_LABEL[cls], transform=ax.transAxes,
                ha="left", va="center", fontsize=FS_TINY, color=CLASS_COLOR[cls])
    ax.text(0.295, 0.965, "filled = standard, open = compact",
            transform=ax.transAxes, ha="left", va="center", fontsize=FS_TINY,
            color=MUTED)


# ---------------------------------------------------------------------------
# c  information equivalence
# ---------------------------------------------------------------------------
def _sci(value: float) -> str:
    """A number for print: exact zero stays 0.0, everything else goes to maths."""
    if value == 0.0:
        return "0.0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** exponent
    return rf"${mantissa:.1f}\times10^{{{exponent}}}$"


def _panel_c(fig, design: dict) -> None:
    ax = mm_axes(fig, C_L, C_TOP, C_W, C_H)
    panel_label_at(fig, X_LEFT, R3_LETTER_Y, "c", "Information equivalence")
    eq = design["eq"]
    # single-line row labels: the panel is 16 mm tall, and a two-line label per
    # row leaves the three rows touching
    rows = (
        ("moments3 components", eq["max_abs_moment_component_gap"], PASS_GREEN),
        (f"histogram24, {N_BINS} masses", eq["max_abs_bin_mass_gap"], PASS_GREEN),
        ("histogram24 bin centres", eq["max_abs_bin_centre_gap_print_precision"], MUTED),
    )
    top = 1.2e-6
    for i, (_, value, colour) in enumerate(rows):
        y = 2 - i
        ax.plot([0, top], [y, y], color="#E8E8E8", lw=LW_HAIR, zorder=1)
        ax.plot([value], [y], marker="D", ms=MS_MEAN, color=colour, zorder=3,
                clip_on=False)
        ax.text(value + 0.075 * top, y, _sci(value), ha="left", va="center",
                fontsize=FS_TINY, color=colour, fontweight="bold")
    ax.axvline(0.0, color="#B0B0B0", lw=LW_HAIR, zorder=2)
    ax.set_ylim(-0.62, 2.62)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS_TINY, linespacing=1.25)
    ax.tick_params(axis="y", length=0, pad=1.4)
    ax.set_xlim(-0.06e-6, top)
    ax.set_xticks([0, 5e-7, 1e-6])
    ax.set_xticklabels(["0", r"$5{\times}10^{-7}$", r"$10^{-6}$"], fontsize=FS_TINY)
    ax.set_xlabel("max |compact $-$ standard|", labelpad=1.0)
    trim_spines(ax, y=False)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------------------
# d  response distances
# ---------------------------------------------------------------------------
def _await(ax, label: str = "awaiting acquisition") -> None:
    """Mark a response panel whose acquisition has not been run yet."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(LW_HAIR)
        s.set_color("#C8C8C8")
        s.set_linestyle((0, (2.4, 1.8)))
    ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=FS_TINY, color=MUTED, style="italic")


def _pair_colour(a: str, b: str) -> str:
    return SAME_CONTENT if SLC_CONTENT[a] == SLC_CONTENT[b] else DIFF_CONTENT


def _panel_d(fig, dec: dict | None) -> None:
    ax = mm_axes(fig, D_L, D_TOP, D_W, D_H)
    panel_label_at(fig, D_L - 8.6, R3_LETTER_Y, "d", "Between-condition distances")
    if dec is None:
        _await(ax)
        return

    floor = dec["noise_floor"]
    lo, hi = floor["within_boot"]["ci95"]
    ax.axhspan(lo, hi, color="#DCDCDC", zorder=0, linewidth=0)
    ax.axhline(floor["mean_within_arm_between_block_TV"], color="#8C8C8C",
               lw=LW_HAIR, ls=(0, (3, 2)), zorder=1)

    tops = []
    for i, (a, b) in enumerate(PAIRS):
        key = _pair_key(a, b)
        boot = dec["pairwise_bootstrap"][key]
        m, clo, chi = boot["mean"], boot["ci95"][0], boot["ci95"][1]
        colour = _pair_colour(a, b)
        ax.errorbar([i], [m], yerr=[[m - clo], [chi - m]], fmt="o", color=colour,
                    ms=MS_SERIES, elinewidth=LW_LINE, capsize=CAPSIZE,
                    capthick=LW_LINE, zorder=4)
        ax.text(i, chi + 0.035, f"{m:.3f}", ha="center", va="bottom",
                fontsize=FS_TINY, color=INK)
        tops.append(chi)
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels([f"{SHORT[a]}\n{SHORT[b]}" for a, b in PAIRS],
                       fontsize=FS_TINY, linespacing=1.25)
    ax.tick_params(axis="x", length=0, pad=1.4)
    ax.set_xlim(-0.6, len(PAIRS) - 0.4)
    ax.set_ylim(0, max(max(tops) + 0.15, 0.45))
    # the short form of the label: spelled out at this type size it is taller
    # than the 17 mm panel and would climb into the panel letter's row
    ax.set_ylabel("mean " + DTV)
    trim_spines(ax, x=False)

    ax.text(0.0, 1.0, "shares content class", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=FS_TINY, color=SAME_CONTENT, fontweight="bold")
    ax.text(0.44, 1.0, "different content class", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=FS_TINY, color=DIFF_CONTENT, fontweight="bold")
    # the band's own label goes under the band, which is the only strip of the
    # panel no point or interval can reach; y is in data units so it tracks the
    # floor rather than a guessed fraction of whatever the y limit turns out to be
    ax.text(0.985, lo - 0.006,
            f"band = within-condition block noise {floor['mean_within_arm_between_block_TV']:.3f}",
            transform=ax.get_yaxis_transform(), ha="right", va="top",
            fontsize=FS_TINY, color=MUTED)


def _panel_d_key(fig) -> None:
    fig_text_mm(fig, D_L, D_KEY_Y,
                "mom = moments3, cen = histogram24; std / cmp = standard / compact form",
                ha="left", va="top", fontsize=FS_TINY, color=MUTED)


# ---------------------------------------------------------------------------
# e  anchored monotonicity: the length-immune contrast
# ---------------------------------------------------------------------------
def _vbracket(ax, x: float, y0: float, y1: float, tick: float, colour: str) -> None:
    """A vertical measure bracket with its ticks turned towards ``tick``."""
    ax.plot([x, x], [y0, y1], color=colour, lw=LW_HAIR, solid_capstyle="butt", zorder=4)
    for ye in (y0, y1):
        ax.plot([x, x + tick], [ye, ye], color=colour, lw=LW_HAIR,
                solid_capstyle="butt", zorder=4)


def _anchor_order(amt: dict) -> list[str]:
    """Anchors in condition order, so the page does not depend on dict order."""
    return sorted(amt["anchors"], key=ARMS.index)


def _panel_e_crossing(ax, dec: dict, design: dict, amt: dict) -> None:
    """Left half of panel e: the two distances, with their length gaps."""
    ticks: list[float] = []
    labels: list[str] = []
    lo_all: list[float] = []
    hi_all: list[float] = []

    for g, anchor in enumerate(_anchor_order(amt)):
        row = amt["anchors"][anchor]
        base = g * E_PITCH
        means = {}
        for dx, target, mean_key, gap_key in (
            (0.0, row["nearer_in_length"], "nearer_mean_TV", "nearer_char_gap"),
            (E_FAR_DX, row["further_in_length"], "further_mean_TV", "further_char_gap"),
        ):
            x = base + dx
            m = float(row[mean_key])
            means[dx] = m
            key = _pair_key(*sorted((anchor, target), key=ARMS.index))
            ci = dec["pairwise_bootstrap"][key]["ci95"]
            colour = CLASS_COLOR[SLC_CONTENT[target]]
            filled = SLC_FORM[target] == "standard"
            ax.errorbar([x], [m], yerr=[[m - ci[0]], [ci[1] - m]], fmt="o",
                        ms=MS_MEAN, mfc=colour if filled else "white", mec=colour,
                        mew=LW_LINE, ecolor=colour, elinewidth=LW_LINE,
                        capsize=CAPSIZE, capthick=LW_LINE, zorder=5)
            ax.text(x, ci[1] + 0.010, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=FS_TINY, color=colour, fontweight="bold")
            ticks.append(x)
            labels.append(f"to {SHORT[target]}\n{int(row[gap_key])} ch away")
            lo_all.append(float(ci[0]))
            hi_all.append(float(ci[1]))

        near_m, far_m = means[0.0], means[E_FAR_DX]
        x_exc = base + E_FAR_DX + E_EXC_DX
        # the level the further prompt has to reach under any monotone
        # increasing function of prompt length, and the distance it falls short
        ax.plot([base, x_exc], [near_m, near_m], color="#8C8C8C", lw=LW_HAIR,
                ls=(0, (2.4, 1.8)), zorder=2)
        ax.plot([base, base + E_FAR_DX], [near_m, far_m], color=INK, lw=LW_THIN,
                zorder=3)
        _vbracket(ax, x_exc, far_m, near_m, -0.05, STOP_RED)
        excess = float(row["paired_excess_near_minus_far"]["mean"])
        ax.text(x_exc + 0.07, (near_m + far_m) / 2.0, f"excess {excess:+.3f}",
                ha="left", va="center", fontsize=FS_TINY, color=STOP_RED,
                fontweight="bold")
        # the short condition name, as panel d's key defines it: the full name is
        # wide enough to reach the y tick labels at this panel width
        ax.text(base + E_FAR_DX / 2.0, 0.055,
                f"anchor {SHORT[anchor]}, {design['chars'][anchor]} ch",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=FS_TINY, color=CLASS_COLOR[SLC_CONTENT[anchor]],
                fontweight="bold")

    span = max(hi_all) - min(lo_all)
    ax.set_ylim(min(lo_all) - 0.42 * span, max(hi_all) + 0.17 * span)
    ax.set_xlim(-0.43, (len(_anchor_order(amt)) - 1) * E_PITCH
                + E_FAR_DX + E_EXC_DX + 0.72)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=FS_TINY, linespacing=1.25)
    ax.tick_params(axis="x", length=0, pad=1.4)
    ax.set_ylabel("mean " + DTV)
    trim_spines(ax, x=False)
    ax.text(0.0, 1.0, "dashed = the level a monotone length account requires",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_TINY,
            color=MUTED)


def _panel_e_excess(ax, dec: dict, amt: dict) -> None:
    """Right half of panel e: the paired excess, which must be at most zero."""
    order = _anchor_order(amt)
    lo_all, hi_all = [], []
    for g, anchor in enumerate(order):
        boot = amt["anchors"][anchor]["paired_excess_near_minus_far"]
        lo_all.append(float(boot["ci95"][0]))
        hi_all.append(float(boot["ci95"][1]))
    span = max(hi_all) - min(lo_all)
    x_lo = min(0.0, min(lo_all)) - 0.85 * span
    x_hi = max(hi_all) + 0.30 * span
    ax.set_xlim(x_lo, x_hi)
    # one row carries a marker with a line of text above and below it, so the
    # rows are given a unit each and the limits are set from the row count
    ax.set_ylim(-1.05, len(order) - 0.45)
    ax.axvspan(x_lo, 0.0, color="#ECECEC", zorder=0, linewidth=0)
    ax.axvline(0.0, color="#8C8C8C", lw=LW_HAIR, zorder=1)

    for g, anchor in enumerate(order):
        row = amt["anchors"][anchor]
        boot = row["paired_excess_near_minus_far"]
        st = row["exact_sign_test"]
        y = len(order) - 1 - g
        m, lo, hi = (float(boot["mean"]), float(boot["ci95"][0]), float(boot["ci95"][1]))
        colour = (VERDICT["supported"] if row["monotone_length_violated"]
                  else VERDICT["not_established"])
        ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="D", ms=MS_MEAN,
                    color=colour, elinewidth=LW_LINE, capsize=CAPSIZE,
                    capthick=LW_LINE, zorder=4)
        top = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        bottom = (f"sign test $p$ = {st['p_two_sided']:.3f}, "
                  f"{int(st['n_near_larger'])} of {int(st['n_discordant'])}")
        for text, dy, va in ((top, 0.21, "bottom"), (bottom, -0.21, "top")):
            w = _text_w_mm(ax.figure, text, FS_TINY)
            half = 0.5 * w / EB_W * (x_hi - x_lo)
            xt = min(max(m, x_lo + half + 0.004), x_hi - half - 0.004)
            ax.text(xt, y + dy, text, ha="center", va=va, fontsize=FS_TINY,
                    color=colour if va == "bottom" else MUTED,
                    fontweight="bold" if va == "bottom" else "normal")

    n_fields = int(amt["anchors"][order[0]]["paired_excess_near_minus_far"]["n_fields"])
    ax.text(x_lo + 0.03 * (x_hi - x_lo), -0.85,
            r"a monotone length account requires excess $\leq$ 0",
            ha="left", va="center", fontsize=FS_TINY, color=INK, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"anchor {SHORT[a]}" for a in reversed(order)],
                       fontsize=FS_TINY)
    ax.tick_params(axis="y", length=0, pad=1.4)
    ax.set_xlabel(f"paired excess over {n_fields} fields, nearer $-$ further",
                  labelpad=1.0)
    trim_spines(ax, y=False)
    ax.spines["left"].set_visible(False)


def _panel_e(fig, dec: dict | None, design: dict) -> None:
    ax_a = mm_axes(fig, EA_L, E_TOP, EA_W, E_H)
    ax_b = mm_axes(fig, EB_L, E_TOP, EB_W, E_H)
    panel_label_at(fig, X_LEFT, R4_LETTER_Y, "e",
                   "Anchored monotonicity: the prompt nearer in length is the "
                   "more distant in response")
    if dec is None:
        _await(ax_a)
        _await(ax_b, "")
        return
    amt = dec["secondary"]["anchored_monotonicity_test"]
    _panel_e_crossing(ax_a, dec, design, amt)
    _panel_e_excess(ax_b, dec, amt)


# ---------------------------------------------------------------------------
# f  the primary contrast
# ---------------------------------------------------------------------------
def _panel_f(fig, dec: dict | None, design: dict) -> None:
    ax = mm_axes(fig, F_L, F_TOP, F_W, F_H)
    panel_label_at(fig, X_LEFT, R5_LETTER_Y, "f", "Primary contrast")
    if dec is None:
        _await(ax)
        return

    pc = dec["primary_contrast"]
    k_len = pc["pair_different_content_182_chars"]
    k_con = pc["pair_same_content_458_chars"]
    if k_len != _pair_key(*sorted((PIVOT, LENGTH_NEIGHBOUR), key=ARMS.index)):
        raise RuntimeError("decision.json names an unexpected length-neighbour pair")
    if k_con != _pair_key(*sorted((PIVOT, CONTENT_PARTNER), key=ARMS.index)):
        raise RuntimeError("decision.json names an unexpected content-partner pair")

    gap_len = int(design["cross"]["chars_centers_compact_to_moments_standard"])
    gap_con = int(design["cross"]["chars_centers_compact_to_centers_standard"])
    rows = (
        (2, pc["mean_TV_different_content"], dec["pairwise_bootstrap"][k_len]["ci95"],
         DIFF_CONTENT, f"different content\n{gap_len} chars apart"),
        (1, pc["mean_TV_same_content"], dec["pairwise_bootstrap"][k_con]["ci95"],
         SAME_CONTENT, f"same content\n{gap_con} chars apart"),
        (0, pc["delta"], pc["delta_bootstrap"]["ci95"], INK,
         "$\\Delta$ = different $-$ same"),
    )
    lo = min(0.0, min(r[2][0] for r in rows))
    hi = max(r[2][1] for r in rows)
    span = max(hi - lo, 0.10)
    ax.set_xlim(lo - 0.10 * span, hi + 0.34 * span)
    for y, m, ci, colour, _ in rows:
        ax.errorbar([m], [y], xerr=[[m - ci[0]], [ci[1] - m]], fmt="D", color=colour,
                    ms=MS_MEAN, elinewidth=LW_LINE, capsize=CAPSIZE,
                    capthick=LW_LINE, zorder=4)
        x_lo, x_hi = ax.get_xlim()
        xt = min(max(m, x_lo + 0.16 * (x_hi - x_lo)), x_hi - 0.16 * (x_hi - x_lo))
        ax.text(xt, y + 0.30, f"{m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", ha="center",
                va="bottom", fontsize=FS_TINY, color=colour, fontweight="bold")
    ax.axvline(0.0, color="#B0B0B0", lw=LW_HAIR, zorder=1)
    ax.set_ylim(-0.75, 2.75)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[4] for r in rows], fontsize=FS_TINY, linespacing=1.25)
    ax.tick_params(axis="y", length=0, pad=1.4)
    ax.set_xlabel("mean " + DTV + " over fields", labelpad=1.0)
    trim_spines(ax, y=False)
    ax.spines["left"].set_visible(False)
    ax.text(0.985, 0.02,
            f"$p$ = {pc['p_one_sided']:.4f}, {int(pc['n_perm']):,} permutations",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_TINY,
            color=INK)


# ---------------------------------------------------------------------------
# g  competing predictors
# ---------------------------------------------------------------------------
def _panel_g(fig, dec: dict | None) -> None:
    ax1 = mm_axes(fig, G_L, G_TOP, G_W, G_H)
    ax2 = mm_axes(fig, G_L + G_W + G_GAP, G_TOP, G_W, G_H)
    panel_label_at(fig, G_L - 8.6, R5_LETTER_Y, "g", "Competing predictors")
    if dec is None:
        _await(ax1)
        _await(ax2, "")
        return

    pr = dec["secondary"]["predictor_rank_correlation"]
    rows = pr["rows"]
    tv = np.asarray([r["mean_TV"] for r in rows], dtype=float)
    mism = np.asarray([r["content_mismatch"] for r in rows], dtype=float)
    dchar = np.asarray([r["abs_prompt_char_diff"] for r in rows], dtype=float)
    colours = [SAME_CONTENT if r["content_mismatch"] == 0 else DIFF_CONTENT for r in rows]
    y_hi = float(tv.max()) * 1.36 + 0.02

    for k in range(len(rows)):
        ax1.plot([mism[k]], [tv[k]], marker="o", ms=MS_POINT * 1.35, mfc=colours[k],
                 mec="white", mew=0.3, zorder=3)
        ax2.plot([dchar[k]], [tv[k]], marker="o", ms=MS_POINT * 1.35, mfc=colours[k],
                 mec="white", mew=0.3, zorder=3)
    for k in (0, 1):
        sel = mism == k
        if sel.any():
            ax1.plot([k - 0.24, k + 0.24], [tv[sel].mean()] * 2, color=INK,
                     lw=LW_LINE, solid_capstyle="butt", zorder=4)

    ax1.set_xlim(-0.55, 1.55)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["same", "different"], fontsize=FS_TINY)
    ax1.set_xlabel("content class", labelpad=1.0)
    ax1.set_ylabel("mean " + DTV)
    ax2.set_xlim(-40, float(dchar.max()) * 1.12)
    ax2.set_xticks([0, 300, 600])
    ax2.set_xlabel("|$\\Delta$ prompt chars|", labelpad=1.0)
    ax2.set_yticklabels([])
    for ax in (ax1, ax2):
        ax.set_ylim(0, y_hi)
        trim_spines(ax)
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_ylim(0, y_hi)
    ax2.set_yticklabels([])

    rho_c = pr["spearman_rho_content_mismatch"]
    rho_l = pr["spearman_rho_abs_prompt_char_diff"]
    best = pr["better_predictor"]
    for ax, rho, key in ((ax1, rho_c, "content_mismatch"),
                         (ax2, rho_l, "abs_prompt_char_diff")):
        ax.text(0.5, 0.985, rf"$\rho$ = {rho:+.2f}".replace("-", "−"),
                transform=ax.transAxes, ha="center", va="top", fontsize=FS_TINY,
                color=INK, fontweight="bold" if best == key else "normal")
    winner = {"content_mismatch": "content class orders the six pairs",
              "abs_prompt_char_diff": "prompt length orders the six pairs",
              "tie": "neither predictor orders the six pairs"}[best]
    fig_text_mm(fig, G_L, G_NOTE_Y,
                f"{winner} (Spearman, {int(pr['n_pairs'])} pairs)",
                ha="left", va="top", fontsize=FS_TINY, color=INK, fontweight="bold")


# ---------------------------------------------------------------------------
# h  verdict strip
# ---------------------------------------------------------------------------
def _gate_rows(fig, ax, rows) -> None:
    """Draw the verdict rows and refuse to leave the box.

    figS21's panel i overflowed its own rectangle when the item pitch was
    guessed; here the pitch is derived from the type size and the last baseline
    is asserted to sit inside the box before anything else is drawn.
    """
    step = _pitch(FS_TINY, H_ROW_LEAD) / H_H        # one row, in axes fractions
    y_last = H_Y_TOP - step * (len(rows) + 0.5)     # +0.5 for the source note
    if y_last < 0.045:
        raise RuntimeError(
            f"panel h needs {len(rows)} rows at {step:.4f} of the box; the last one "
            f"would land at {y_last:.3f} and print outside the rectangle"
        )
    w_label = (H_X_STAT - H_X_LABEL) * H_W - 2.0
    w_stat = (H_X_RIGHT - H_X_STAT) * H_W
    y = H_Y_TOP
    for label, stat, verdict, colour in rows:
        if _text_w_mm(fig, label, FS_TINY) > w_label:
            raise RuntimeError(f"panel h label does not fit its column: {label!r}")
        if _text_w_mm(fig, stat, FS_TINY) + _text_w_mm(fig, verdict, FS_TINY) + 3.0 > w_stat:
            raise RuntimeError(f"panel h statistic and verdict collide: {stat!r} / {verdict!r}")
        ax.plot([H_X_MARK], [y], marker="s", ms=MS_POINT, color=colour, clip_on=False)
        ax.text(H_X_LABEL, y, label, ha="left", va="center", fontsize=FS_TINY, color=INK)
        ax.text(H_X_STAT, y, stat, ha="left", va="center", fontsize=FS_TINY, color=MUTED)
        ax.text(H_X_RIGHT, y, verdict, ha="right", va="center", fontsize=FS_TINY,
                color=colour, fontweight="bold")
        y -= step


def _panel_h(fig, dec: dict | None) -> None:
    ax = mm_panel(fig, H_L, H_TOP, H_W, H_H)
    panel_label_at(fig, X_LEFT, H_LETTER_Y, "h", "Verdicts")
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    facecolor="#F7F7F7", edgecolor="#B0B0B0",
                                    lw=LW_THIN, zorder=0))
    if dec is None:
        ax.text(0.5, 0.5, "awaiting acquisition", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS_TINY, color=MUTED, style="italic")
        return

    alpha = float(dec["inference_settings"]["alpha_primary"])
    glob = dec["secondary"]["global_any_arm_difference"]
    pc = dec["primary_contrast"]
    lay = dec["secondary"]["layout_only_reference_moments_class"]
    inf = dec["secondary"]["information_only_reference_histogram_class"]
    amt = dec["secondary"]["anchored_monotonicity_test"]
    anchors = _anchor_order(amt)

    verdict_word = {
        "PASS_content_over_length": ("content over length", VERDICT["supported"]),
        "LENGTH_SENSITIVE": ("length over content", STOP_RED),
        "LENGTH_REFUTED_DELTA_UNRESOLVED": (
            "directional, not established",
            VERDICT["not_established"],
        ),
        "UNRESOLVED": ("unresolved", VERDICT["not_established"]),
    }
    if dec["verdict"] not in verdict_word:
        raise RuntimeError(f"decision.json carries an unknown verdict: {dec['verdict']!r}")
    primary_word, primary_col = verdict_word[dec["verdict"]]
    glob_ok = float(glob["p_one_sided"]) < alpha

    # the anchored test speaks for every anchor at once, so it takes one row
    refuted = bool(amt["monotone_length_account_refuted"])
    excesses = " and ".join(
        f"{amt['anchors'][a]['paired_excess_near_minus_far']['mean']:+.3f}"
        for a in anchors)
    sign_ps = " and ".join(
        f"{amt['anchors'][a]['exact_sign_test']['p_two_sided']:.3f}" for a in anchors)
    # the two single-class references vary form and length together, so they are
    # marked confounded here and the gaps that confound them are printed below
    confounded = "confounded with length"

    rows = (
        ("Global any-condition difference",
         f"mean pairwise TV {glob['observed_mean_pairwise_TV']:.3f}, "
         f"p = {glob['p_one_sided']:.4f} ({int(glob['n_perm']):,} perms)",
         "established" if glob_ok else "not established",
         VERDICT["supported"] if glob_ok else VERDICT["not_established"]),
        ("Primary contrast, content minus length",
         f"Δ = {pc['delta']:.3f} [{pc['delta_bootstrap']['ci95'][0]:.3f}, "
         f"{pc['delta_bootstrap']['ci95'][1]:.3f}], p = {pc['p_one_sided']:.4f}",
         primary_word, primary_col),
        (f"Anchored monotonicity, {len(anchors)} anchors",
         f"paired excess {excesses}, sign test p = {sign_ps}",
         "inconsistent with a monotone\nlength account at both anchors" if refuted else "not refuted",
         VERDICT["supported"] if refuted else VERDICT["not_established"]),
        ("Layout-only reference, moments class",
         f"mean TV {lay['mean']:.3f} [{lay['ci95'][0]:.3f}, {lay['ci95'][1]:.3f}], "
         f"{lay['ratio_to_block_noise']:.1f}× block noise, {int(lay['prompt_char_gap'])} chars",
         confounded, MUTED),
        ("Information-only reference, histogram class",
         f"mean TV {inf['mean']:.3f} [{inf['ci95'][0]:.3f}, {inf['ci95'][1]:.3f}], "
         f"{inf['ratio_to_block_noise']:.1f}× block noise, {int(inf['prompt_char_gap'])} chars",
         confounded, MUTED),
    )
    _gate_rows(fig, ax, rows)
    step = _pitch(FS_TINY, H_ROW_LEAD) / H_H
    note = (f"{int(dec['acquisition']['n_valid']):,} valid responses over "
            f"{int(dec['acquisition']['n_fields_used'])} fields · alpha {alpha:g} · "
            f"the two reference rows differ by {int(lay['prompt_char_gap'])} and "
            f"{int(inf['prompt_char_gap'])} prompt characters, so form and length "
            "vary together in both")
    w_note = (H_X_RIGHT - H_X_LABEL) * H_W
    if _text_w_mm(fig, note, FS_TINY) > w_note:
        raise RuntimeError(f"panel h source note does not fit the box: {note!r}")
    ax.text(H_X_LABEL, H_Y_TOP - step * (len(rows) + 0.25), note,
            ha="left", va="center", fontsize=FS_TINY, color=MUTED)


# ---------------------------------------------------------------------------
def build(_out_dir: Path | None = None, decision_path: Path | None = None) -> Path:
    apply_style()
    design = _load_design()
    path = Path(decision_path or DECISION_PATH)
    dec = _load_decision(path, design)
    HAVE_RESULTS = dec is not None      # noqa: N806 - the one flag the layout branches on
    if not HAVE_RESULTS:
        print(f"[figS22] WARNING: {path} is absent; the response panels "
              "(d-h) are drawn as empty axes marked 'awaiting acquisition'. "
              "Re-run this module after analysis/matched_rep_collective/"
              "analyze_slc_control.py has been run.")

    fig = new_figure(H_MM)
    _panel_a(fig, design)
    _panel_b(fig, design)
    _panel_c(fig, design)
    _panel_d(fig, dec)
    _panel_d_key(fig)
    _panel_e(fig, dec, design)
    _panel_f(fig, dec, design)
    _panel_g(fig, dec)
    _panel_h(fig, dec)
    _overflow(fig, "results" if HAVE_RESULTS else "design-only")
    return save_fig(fig, "figS22_serialization_length")


def main(argv: list[str] | None = None) -> int:      # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, default=None,
                        help="override the path to slc_primary/decision.json")
    args = parser.parse_args(argv)
    print(build(decision_path=args.decision))
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
