"""Supplementary Figure S23 - serialization-binding ladder on the replay panel.

Four conditions, each a re-serialization of the *same* 24 bin masses at the same
precision, crossing two binary textual features: whether the prompt prints a bin
index next to each mass, and whether it prints the numeric bin centre.

                        index not printed        index printed
    centre not printed  centers_compact   971    centers_indexed_row  1055
    centre printed      centers_centered_row 1114 centers_standard    1429

The message of the page is narrow and is stated at exactly that strength: within
this tested ladder, response distance tracked WHICH textual binding feature was
changed more closely than the SIZE of the character-count gap. The page does not
claim that prompt length never matters, and it does not claim a universal
serialization mechanism. Panel f carries the evidence: inside each of the three
factorial cells the character gap varies by up to 7.8-fold while the paired
response difference stays inside an interval that covers zero.

The centre-printing result (panel e) is secondary and post hoc, and is labelled
so. The prespecified pivot contrast is reported in the verdict strip as
non-diagnostic rather than as a result: both a prompt-length account and a
centre-printing account predict the same sign for it, which is what
``primary_contrast.diagnosticity.diagnostic`` records.

Nothing on this page is hard-coded. Panels a-b are rebuilt live: the four
prompts are regenerated from one frozen replay field through the production
builder (``circlemap.serialization_binding_controls.build_sbc_prompt``), decoded
back to numbers, and checked against
``analysis/matched_rep_collective/sbc_information_audit.json`` before a single
character is printed; a disagreement raises. Panels c-f are read from
``analysis/matched_rep_collective/sbc_primary/decision.json``. Panel g needs a
per-condition block noise that ``decision.json`` does not carry (it records the
pooled floor only), so that one panel is recomputed from the run's
``trace.jsonl`` the way ``analyze_sbc_control.py`` finds it, and says so.

Two render modes
----------------
The acquisition behind ``decision.json`` is paid and may not exist yet. If the
artifact is absent the design panels (a-b) are drawn in full and the response
panels (c-g) plus the verdict strip are drawn as marked empty axes; a warning
goes to stdout and the build still succeeds. A single ``HAVE_RESULTS`` flag
guards the difference, so the layout is written once.
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

from analysis.plot_supplement.style import (
    CAPSIZE,
    FS_SMALL,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    MS_MEAN,
    MS_SERIES,
    MUTED,
    PASS_GREEN,
    REP,
    ROOT,
    RULE,
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

from analysis.matched_rep_collective.analyze_sbc_control import (
    CONDITIONS,
    LABELLED_ANCHOR,
    PAIRS,
    PIVOT,
    POSITIONAL_ANCHOR,
    _canon_pair_key,
    build_panel,
    find_run_dir,
)
from analysis.matched_rep_collective.run_replay_primary_inference import (
    _tv,
    cluster_bootstrap_mean,
)
from circlemap.costing import estimate_tokens_from_characters
from circlemap.observation import RelativePhaseHistogram
from circlemap.representations import build_representation_prompt_from_histogram
from circlemap.serialization_binding_controls import (
    N_BINS,
    SBC_BINDING,
    SBC_CONTENT,
    SBC_FORM,
    build_sbc_prompt,
    decode_centres,
    decode_masses,
)
from circlemap.serialization_length_controls import build_slc_prompt

MRC = ROOT / "analysis" / "matched_rep_collective"
AUDIT_PATH = MRC / "sbc_information_audit.json"
FIELDS_PATH = MRC / "replay_fields_v0_1.json"
DECISION_PATH = MRC / "sbc_primary" / "decision.json"

#: The two binary textual features the ladder crosses. These are declared here
#: and then checked against ``decision.json``'s own feature maps and against the
#: builder's binding labels, so the page can never draw a 2x2 the analysis did
#: not compute.
PRINTS_INDEX = {
    "centers_compact": False,
    "centers_indexed_row": True,
    "centers_centered_row": False,
    "centers_standard": True,
}
PRINTS_CENTRE = {
    "centers_compact": False,
    "centers_indexed_row": False,
    "centers_centered_row": True,
    "centers_standard": True,
}
#: The builder's binding label implied by the two features, asserted not assumed.
BINDING_OF_FEATURES = {
    (False, False): "positional",
    (True, False): "explicit_index",
    (False, True): "explicit_centre",
    (True, True): "explicit_index_and_centre",
}

#: The primary encoding ``centers_standard`` must reproduce byte for byte.
PRIMARY_ENCODING = "centers_24_standard"

CELL_ORDER = ("index_only", "centre_only", "both")
CELL_LABEL = {"index_only": "index only", "centre_only": "centre only", "both": "both"}

INDEX_COL = "#0072B2"     # the bin-index feature
CENTRE_COL = "#D55E00"    # the bin-centre feature
BOTH_COL = "#7059A8"      # both features change
NONE_COL = "#4D4D4D"      # neither feature printed
CELL_COL = {"index_only": INDEX_COL, "centre_only": CENTRE_COL, "both": BOTH_COL}

#: A condition is coloured by which features it prints, so the colour key of
#: panels a, c-g is one key rather than four.
COND_COL = {
    "centers_compact": NONE_COL,
    "centers_indexed_row": INDEX_COL,
    "centers_centered_row": CENTRE_COL,
    "centers_standard": BOTH_COL,
}
#: Compact tags for the pair tick labels of panels c and f. ``full`` is the
#: fully labelled condition, which is the anchor ``analyze_sbc_control.py`` names
#: LABELLED_ANCHOR; the tags are printed next to the full condition names in
#: panel a so the page defines its own abbreviations.
SHORT = {
    "centers_compact": "pos",
    "centers_indexed_row": "idx",
    "centers_centered_row": "ctr",
    "centers_standard": "full",
}

DTV = r"$d_{\mathrm{TV}}$"
PT_MM = 25.4 / 72.0
LS_MONO = 1.30
CODE_INK = "#6E6E6E"      # the serialization scaffolding: commas, colons, "center"
MASS_COL = "#1A1A1A"      # the 24 masses, identical in all four conditions

X_LEFT = 4.4
X_RIGHT = 176.0
H_MM = 225.0

# --- canvas plan (mm from the top-left) ------------------------------------
A_LETTER_Y = 7.0
A_NOTE_Y = 10.9
#: the 2x2: a row-feature gutter, then two feature columns
A_GUTTER_X, A_GUTTER_W = X_LEFT, 21.6
A_COL_X = (27.0, 103.0)
A_COL_W = (72.0, 72.0)
A_HEAD_Y = 15.2                  # the column-feature headings
A_ROW_TOP = (20.4, 37.9)         # cell tops, row 0 = centre not printed
A_ROW_H = (16.2, 18.8)           # row 1 holds the per-line condition, so it is taller
A_PAD = 1.3                      # inside a cell rectangle
A_MONO_LINES = 2                 # display lines of payload before the elision
A_BINS_SHOWN = 2                 # verbatim bin lines before the elision marker
A_RULE_GAP = 2.4

B_LETTER_Y = 62.6
B_L, B_TOP, B_W, B_H = 24.0, 65.0, 42.0, 13.0
#: The note runs under panels b and c but stops short of panel c's centred axis
#: label, which begins 111 mm across; a full-width line would print through it.
B_NOTE_Y = 85.8
B_NOTE_W = 104.0

C_L, C_TOP, C_W, C_H = 96.0, 65.0, 80.0, 13.0

R3_LETTER_Y = 93.0
D_L, D_TOP, D_W, D_H = 24.0, 96.5, 62.0, 15.5
E_L, E_TOP, E_W, E_H = 116.0, 96.5, 60.0, 15.5

# f: the key panel. Two axes, one letter: the six pairs grouped into their three
# cells on the left, the paired within-cell difference on the right.
R4_LETTER_Y = 122.5
F_TOP, F_H = 127.5, 25.0
FA_L, FA_W = 20.0, 88.0
FB_L, FB_W = 128.0, 48.0
F_PITCH = 1.7                    # distance between two cell groups
F_WIDE_DX = 0.60                 # the wide-gap pair, right of the narrow-gap one

R5_LETTER_Y = 164.0
G_NOTE_Y = 167.5
G_L, G_TOP, G_W, G_H = 34.0, 170.5, 142.0, 18.4

# the strip is the lowest ink on the page: its box sits 2.2 mm off the trim,
# which check_figures requires to be at least 2.0 mm. The box grew 1.4 mm
# downwards when the one-line source note became the three-line closing block
# (acquisition / verdict / scope); it cannot grow upwards without colliding
# with panel g's axis label.
V_LETTER_Y = 198.9
V_L, V_TOP, V_W, V_H = 13.0, 201.4, 163.0, 21.4

# Verdict-strip anchors, in axes fractions of the V_H-tall box. The item pitch is
# derived from the type size rather than guessed, and _gate_rows asserts that the
# last row still sits inside the box, which is how figS21's panel i overflowed
# before its spacing was pinned.
V_X_MARK = 0.022
V_X_LABEL = 0.045
V_X_STAT = 0.330
V_X_RIGHT = 0.978
V_Y_TOP = 0.930
V_ROW_LEAD = 1.44                # verdict rows are set tighter than running text

# The closing block under the gate rows: how the acquisition line, the overall
# verdict and the scope limit are spaced, in mm. They are separate lines rather
# than one middot list because they are answers to different questions - what
# was collected, what the prespecified threshold was, what the analysis decided,
# and how far that decision reaches - and a single run of middots let a reader
# take the scope limit for one more result.
V_SUMMARY_GAP = 3.5              # mm, last gate row -> the provenance line
V_SUMMARY_LEAD = 2.85            # mm between the three closing lines
V_X_THRESHOLD = 0.700            # left edge of the prespecified-threshold group


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
    display-only: the column breaks between whole tokens where it can and falls
    back to a terminal-style character break for a token longer than the column,
    so no character is ever added, removed or reordered.
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
            print(f"[figS23:{tag}] overflow x=[{x0:.1f},{x1:.1f}] "
                  f"y=[{y_top:.1f},{y_bottom:.1f}]: "
                  f"{art.get_text()[:56]!r}", file=sys.stderr)
    if n_bad:
        print(f"[figS23:{tag}] {n_bad} overflowing text artist(s)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Design side: rebuild the prompts and audit them before printing anything
# ---------------------------------------------------------------------------
_BIN_RE = re.compile(r"^(bin_\d+)( center )([+-]\d\.\d{6})(: )(\d\.\d{6})$")
_IDX_RE = re.compile(r"^(b\d{2})(=)(\d\.\d{6})$")
_CTR_RE = re.compile(r"^([+-]\d\.\d{6})(:)(\d\.\d{6})$")


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


def _cell_of(a: str, b: str) -> str:
    """Which factorial cell a condition pair falls in, from the feature maps."""
    dc = PRINTS_CENTRE[a] != PRINTS_CENTRE[b]
    di = PRINTS_INDEX[a] != PRINTS_INDEX[b]
    if dc and di:
        return "both"
    if dc:
        return "centre_only"
    if di:
        return "index_only"
    raise RuntimeError(f"{a} and {b} differ in neither feature; the 2x2 is degenerate")


def _load_design() -> dict:
    """Rebuild the four prompts and verify every number this page will print."""
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if not audit.get("pass"):
        raise RuntimeError("sbc_information_audit.json does not report pass; refusing to draw")
    if int(audit["n_fields"]) != 48:
        raise RuntimeError(
            f"the audit covers {audit['n_fields']} fields, expected the 48-field panel"
        )

    fields = json.loads(FIELDS_PATH.read_text(encoding="utf-8"))["fields"]
    field = fields[0]
    hist = _histogram(field)
    prompts = {v: build_sbc_prompt(v, hist) for v in CONDITIONS}

    # the two feature maps this page draws must agree with the builder's own
    # binding label, so a relabelled condition cannot leave the 2x2 stale
    for v in CONDITIONS:
        want = BINDING_OF_FEATURES[(PRINTS_INDEX[v], PRINTS_CENTRE[v])]
        if SBC_BINDING[v] != want:
            raise RuntimeError(
                f"{v}: the feature maps imply binding {want!r}, the builder says "
                f"{SBC_BINDING[v]!r}"
            )
    if len({(PRINTS_INDEX[v], PRINTS_CENTRE[v]) for v in CONDITIONS}) != 4:
        raise RuntimeError("the four conditions do not occupy four distinct 2x2 cells")

    chars, tokens = {}, {}
    for v in CONDITIONS:
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
        if audit["variants"][v] != {
            "content": SBC_CONTENT[v], "binding": SBC_BINDING[v], "form": SBC_FORM[v]
        }:
            raise RuntimeError(f"{v}: the audit disagrees with the builder about the condition")
        chars[v] = int(row["median"])
        tokens[v] = int(tok["median"])

    # the two inherited anchors, byte for byte
    if prompts["centers_standard"] != build_representation_prompt_from_histogram(
        PRIMARY_ENCODING, hist
    ):
        raise RuntimeError(
            f"centers_standard is not byte-identical to the primary {PRIMARY_ENCODING} prompt"
        )
    if prompts["centers_compact"] != build_slc_prompt("centers_compact", hist):
        raise RuntimeError(
            "centers_compact is not byte-identical to the serialization-length control arm"
        )
    if not all(audit["anchors_byte_identical"].values()):
        raise RuntimeError("the audit does not carry byte identity for both inherited anchors")

    # information identity, recomputed live and matched to the audit
    eq = audit["information_equivalence"]
    masses = {v: decode_masses(v, prompts[v]) for v in CONDITIONS}
    centres = {v: decode_centres(v, prompts[v]) for v in CONDITIONS}
    for v in CONDITIONS:
        if masses[v].shape != (N_BINS,) or centres[v].shape != (N_BINS,):
            raise RuntimeError(f"{v}: does not decode to {N_BINS} bins")
    ref = masses[CONDITIONS[0]]
    gap_mass = max(float(np.max(np.abs(masses[v] - ref))) for v in CONDITIONS)
    ref_c = centres[CONDITIONS[0]]
    gap_centre = max(float(np.max(np.abs(centres[v] - ref_c))) for v in CONDITIONS)
    if float(eq["max_abs_bin_mass_gap"]) != 0.0:
        raise RuntimeError("the audit no longer reports exact mass identity")
    if gap_mass > float(eq["max_abs_bin_mass_gap"]):
        raise RuntimeError(
            f"the rebuilt masses differ by {gap_mass:.3e}, the audit records "
            f"{eq['max_abs_bin_mass_gap']:.3e}"
        )
    if gap_centre > float(eq["max_abs_bin_centre_gap_print_precision"]) * (1.0 + 1e-9):
        raise RuntimeError("the rebuilt bin centres exceed the audited print-precision gap")

    # the crossing the ladder was built on
    cross = audit["length_axis_crosses_binding_axis"]
    if cross["pivot"] != PIVOT:
        raise RuntimeError("the audit pivots on a different condition than the analysis")
    for key, other in (
        ("chars_pivot_to_positional_anchor", POSITIONAL_ANCHOR),
        ("chars_pivot_to_labelled_anchor", LABELLED_ANCHOR),
    ):
        if abs(chars[PIVOT] - chars[other]) != int(cross[key]):
            raise RuntimeError(f"the audited {key} disagrees with the audited prompt lengths")
    if not cross["crossed"]:
        raise RuntimeError("the audit no longer reports the length axis crossing the binding axis")

    # the shared instruction contract, asserted rather than assumed
    lines = {v: prompts[v].splitlines() for v in CONDITIONS}
    n_instr = 9
    for i in range(n_instr):
        if len({lines[v][i] for v in CONDITIONS}) != 1:
            raise RuntimeError(f"instruction line {i + 1} is not shared by all four conditions")
    if len({lines[v][n_instr] for v in CONDITIONS}) != 4:
        raise RuntimeError("the four conditions do not carry four distinct payload headers")

    return {
        "audit": audit,
        "audit_sha256": _sha_file(AUDIT_PATH),
        "field": field,
        "prompts": prompts,
        "lines": lines,
        "chars": chars,
        "tokens": tokens,
        "eq": eq,
        "cross": cross,
        "n_instruction_lines": n_instr,
    }


def _load_decision(path: Path, design: dict) -> dict | None:
    """Read the acquisition verdict, or return None if it does not exist yet."""
    if not path.exists():
        return None
    dec = json.loads(path.read_text(encoding="utf-8"))
    for v in CONDITIONS:
        if int(dec["conditions"][v]["prompt_chars"]) != design["chars"][v]:
            raise RuntimeError(f"{v}: decision.json prompt_chars disagrees with the audit")
        if dec["conditions"][v]["binding"] != SBC_BINDING[v]:
            raise RuntimeError(f"{v}: decision.json binding disagrees with the builder")
    if dec["hashes"]["information_audit_sha256"] != design["audit_sha256"]:
        raise RuntimeError(
            "decision.json was written against a different sbc_information_audit.json; "
            "re-run analyze_sbc_control.py before drawing this page"
        )
    keys = {_canon_pair_key(a, b) for a, b in PAIRS}
    if len(keys) != 6:
        raise RuntimeError("the four conditions do not make six distinct pairs")
    if set(dec["pairwise_mean_TV"]) != keys or set(dec["pairwise_bootstrap"]) != keys:
        raise RuntimeError("decision.json does not carry all six condition pairs")
    _check_factorial(dec, design)
    return dec


def _check_factorial(dec: dict, design: dict) -> None:
    """Verify the 2x2 decomposition panels d-f are about to draw.

    Every claim on this page beyond the six raw distances rests on the factorial
    block, so its feature maps, its cell membership, its character gaps and the
    arithmetic linking its cells to its contrast are all checked here. A changed
    analysis script cannot leave the figure asserting the old design.
    """
    fac = dec["secondary"]["factorial_decomposition"]
    if fac.get("reframing") != "post hoc":
        raise RuntimeError(
            "secondary.factorial_decomposition no longer records itself as post hoc; "
            "panels d-f label it so and must not print an unqualified reframing"
        )
    feats = fac["features"]
    if feats["prints_bin_index"] != PRINTS_INDEX or feats["prints_bin_centre"] != PRINTS_CENTRE:
        raise RuntimeError("decision.json's feature maps disagree with this figure's 2x2")

    chars = design["chars"]
    pair_mean = dec["pairwise_mean_TV"]
    within = float(dec["noise_floor"]["mean_within_condition_between_block_TV"])
    cells = fac["cells"]
    if set(cells) != set(CELL_ORDER):
        raise RuntimeError(f"decision.json carries cells {sorted(cells)}, expected {CELL_ORDER}")

    # cell membership, derived from the feature maps rather than trusted
    expected: dict[str, set[str]] = {name: set() for name in CELL_ORDER}
    for a, b in PAIRS:
        expected[_cell_of(a, b)].add(_canon_pair_key(a, b))
    for name in CELL_ORDER:
        got = {row["pair"] for row in cells[name]["pairs"]}
        if got != expected[name]:
            raise RuntimeError(
                f"cell {name}: decision.json holds {sorted(got)}, the feature maps "
                f"give {sorted(expected[name])}"
            )
        for row in cells[name]["pairs"]:
            a, b = row["pair"].split("__vs__")
            if abs(chars[a] - chars[b]) != int(row["abs_prompt_char_diff"]):
                raise RuntimeError(f"{row['pair']}: the recorded character gap is not the audited one")
            if abs(float(row["mean_TV"]) - float(pair_mean[row["pair"]])) > 1e-12:
                raise RuntimeError(f"{row['pair']}: the cell mean TV disagrees with pairwise_mean_TV")
        ratio = float(cells[name]["cell_bootstrap"]["mean"]) / max(within, 1e-12)
        if abs(ratio - float(cells[name]["ratio_to_block_noise"])) > 1e-9:
            raise RuntimeError(f"cell {name}: ratio_to_block_noise is not mean / block noise")

    # the contrast panel e draws is the gap between two cell means it can see
    contrast = fac["centre_only_minus_index_only"]
    drawn = (float(cells["centre_only"]["cell_bootstrap"]["mean"])
             - float(cells["index_only"]["cell_bootstrap"]["mean"]))
    if abs(float(contrast["difference"]) - drawn) > 1e-9:
        raise RuntimeError(
            f"the centre-minus-index difference ({contrast['difference']:+.6f}) is not "
            f"the gap between the two cell means panel d draws ({drawn:+.6f})"
        )
    if abs(float(contrast["bootstrap"]["mean"]) - float(contrast["difference"])) > 1e-12:
        raise RuntimeError("the centre-minus-index bootstrap mean is not the observed difference")

    # panel f: the within-cell length gaps
    ins = fac["within_cell_length_insensitivity"]
    if set(ins) != set(CELL_ORDER):
        raise RuntimeError("within_cell_length_insensitivity does not cover the three cells")
    for name in CELL_ORDER:
        row = ins[name]
        narrow, wide = row["narrow_gap_pair"], row["wide_gap_pair"]
        if {narrow["pair"], wide["pair"]} != expected[name]:
            raise RuntimeError(f"cell {name}: the two compared pairs are not the cell's own pairs")
        for side in (narrow, wide):
            a, b = side["pair"].split("__vs__")
            if _cell_of(a, b) != name:
                raise RuntimeError(f"{side['pair']} is not an {name} change")
            if abs(chars[a] - chars[b]) != int(side["char_gap"]):
                raise RuntimeError(f"{side['pair']}: the recorded character gap is not the audited one")
            if abs(float(side["mean_TV"]) - float(pair_mean[side["pair"]])) > 1e-12:
                raise RuntimeError(f"{side['pair']}: mean_TV disagrees with pairwise_mean_TV")
        if int(narrow["char_gap"]) >= int(wide["char_gap"]):
            raise RuntimeError(f"cell {name}: the pair named narrow is not the narrower one")
        ratio = int(wide["char_gap"]) / max(int(narrow["char_gap"]), 1)
        if abs(ratio - float(row["char_gap_ratio"])) > 1e-9:
            raise RuntimeError(f"cell {name}: char_gap_ratio is not wide / narrow")
        covers = float(row["paired_difference_wide_minus_narrow"]["ci95"][0]) <= 0.0 <= float(
            row["paired_difference_wide_minus_narrow"]["ci95"][1]
        )
        if covers != bool(row["difference_ci_covers_zero"]):
            raise RuntimeError(f"cell {name}: difference_ci_covers_zero contradicts its own interval")
    if bool(fac["length_insensitive_in_every_cell"]) != all(
        bool(ins[n]["difference_ci_covers_zero"]) for n in CELL_ORDER
    ):
        raise RuntimeError("length_insensitive_in_every_cell contradicts the three cells")

    # the prespecified contrast must still carry its diagnosticity verdict
    diag = dec["primary_contrast"].get("diagnosticity")
    if diag is None:
        raise RuntimeError(
            "decision.json carries no primary_contrast.diagnosticity; the prespecified "
            "pivot contrast must not be drawn without it"
        )
    if not isinstance(diag["diagnostic"], bool):
        raise RuntimeError("primary_contrast.diagnosticity.diagnostic is not a boolean")
    if diag["diagnostic"] != (
        diag["length_account_predicts"] != diag["centre_printing_account_predicts"]
    ):
        raise RuntimeError("diagnosticity.diagnostic contradicts its own two predictions")


def _per_condition_noise(dec: dict) -> tuple[dict[str, dict], bool]:
    """Within-condition between-block distance per condition, and its provenance.

    ``decision.json`` records the pooled floor only, so if it grows a
    per-condition block that is used; otherwise the four numbers are recomputed
    from the run's ``trace.jsonl``, found the way ``analyze_sbc_control.py``
    finds it, and the pooled mean is checked against the recorded floor.
    """
    carried = dec["noise_floor"].get("per_condition")
    if carried is not None:
        if set(carried) != set(CONDITIONS):
            raise RuntimeError("noise_floor.per_condition does not cover the four conditions")
        return {c: carried[c] for c in CONDITIONS}, False

    run_dir = find_run_dir(ROOT / dec["run_dir"])
    if _sha_file(run_dir / "trace.jsonl") != dec["hashes"]["trace_sha256"]:
        raise RuntimeError(
            f"the trace at {run_dir} is not the one decision.json was written from; "
            "re-run analyze_sbc_control.py before drawing this page"
        )
    rows = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    fields = json.loads(FIELDS_PATH.read_text(encoding="utf-8"))["fields"]
    panel, dropped = build_panel(fields, rows)
    if len(panel) != int(dec["acquisition"]["n_fields_used"]):
        raise RuntimeError("the recomputed panel does not cover the fields decision.json used")
    if sorted(dropped) != sorted(dec["acquisition"]["fields_dropped_incomplete"]):
        raise RuntimeError("the recomputed panel drops a different set of fields")

    out: dict[str, dict] = {}
    pooled: list[float] = []
    for c in CONDITIONS:
        vals = []
        for fr in panel:
            p0 = np.asarray(fr["blocks"][f"{c}|b0"]["p"], dtype=float)
            p1 = np.asarray(fr["blocks"][f"{c}|b1"]["p"], dtype=float)
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                vals.append(_tv(p0, p1))
        arr = np.asarray(vals, dtype=float)
        pooled.extend(vals)
        out[c] = {**cluster_bootstrap_mean(arr), "n_cells": int(arr.size)}
    if len(pooled) != int(dec["noise_floor"]["n_cells"]):
        raise RuntimeError("the recomputed noise cells do not match the recorded cell count")
    if abs(float(np.mean(pooled))
           - float(dec["noise_floor"]["mean_within_condition_between_block_TV"])) > 1e-12:
        raise RuntimeError(
            "the recomputed pooled block noise does not reproduce the recorded floor"
        )
    return out, True


# ---------------------------------------------------------------------------
# a  the 2x2 of conditions
# ---------------------------------------------------------------------------
def _entry_groups(design: dict, variant: str) -> tuple[list[list[tuple[str, str]]], str]:
    """One coloured group per printed entry, plus the unit the elision counts.

    The group boundaries are where the display column may break, so a mass, a
    bin index and a bin centre are never split in half.
    """
    body = design["lines"][variant][design["n_instruction_lines"] + 1:]
    if variant == "centers_standard":
        groups = []
        for line in body:
            m = _BIN_RE.match(line)
            if m is None:
                raise RuntimeError(f"unexpected centers_standard line: {line!r}")
            idx, mid, centre, sep, mass = m.groups()
            groups.append([(idx, INDEX_COL), (mid, CODE_INK), (centre, CENTRE_COL),
                           (sep, CODE_INK), (mass, MASS_COL)])
        return groups, "lines"

    if len(body) != 1:
        raise RuntimeError(f"{variant}: expected a single payload row, got {len(body)} lines")
    groups = []
    entries = body[0].split(",")
    for k, entry in enumerate(entries):
        tail = "," if k + 1 < len(entries) else ""
        if variant == "centers_compact":
            parts = [(entry, MASS_COL)]
        elif variant == "centers_indexed_row":
            m = _IDX_RE.match(entry)
            if m is None:
                raise RuntimeError(f"unexpected centers_indexed_row entry: {entry!r}")
            parts = [(m.group(1), INDEX_COL), (m.group(2), CODE_INK), (m.group(3), MASS_COL)]
        else:
            m = _CTR_RE.match(entry)
            if m is None:
                raise RuntimeError(f"unexpected centers_centered_row entry: {entry!r}")
            parts = [(m.group(1), CENTRE_COL), (m.group(2), CODE_INK), (m.group(3), MASS_COL)]
        if tail:
            parts.append((tail, CODE_INK))
        groups.append(parts)
    return groups, "values"


def _pack(groups, n_cols: int, max_lines: int):
    """Pack whole groups into ``max_lines`` display lines; return segments, count."""
    segs: list[tuple[str, str]] = []
    col, line, shown = 0, 0, 0
    for group in groups:
        width = sum(len(t) for t, _ in group)
        if col and col + width > n_cols:
            if line + 1 >= max_lines:
                break
            segs.append(("\n", CODE_INK))
            col, line = 0, line + 1
        if width > n_cols:
            break                       # a single group wider than the column
        segs.extend(group)
        col += width
        shown += 1
    return segs, shown


def _cell(fig, x: float, y: float, w: float, h: float, design: dict, variant: str) -> None:
    """One quadrant of the 2x2: name, feature badges, verbatim payload excerpt."""
    W, H = (v * 25.4 for v in fig.get_size_inches())
    fig.add_artist(mpatches.Rectangle(
        ((x - A_PAD) / W, (H - y - h) / H), (w + 2 * A_PAD) / W, (h - 1.0) / H,
        transform=fig.transFigure, facecolor="#FBFBFB", edgecolor="#D0D0D0",
        lw=LW_HAIR, zorder=0))

    yy = y + A_PAD
    fig_text_mm(fig, x, yy, variant, ha="left", va="top", fontsize=FS_SMALL,
                color=COND_COL[variant], fontweight="bold")
    fig_text_mm(fig, x + _text_w_mm(fig, variant, FS_SMALL, fontweight="bold") + 1.4, yy,
                f"[{SHORT[variant]}]", ha="left", va="top", fontsize=FS_TINY,
                color=COND_COL[variant])
    fig_text_mm(fig, x + w, yy,
                f"{design['chars'][variant]} chars",
                ha="right", va="top", fontsize=FS_TINY, color=MUTED)
    yy += _pitch(FS_SMALL, 1.55)

    for dx, printed, word, colour in (
        (0.0, PRINTS_INDEX[variant], "index", INDEX_COL),
        (0.50 * w, PRINTS_CENTRE[variant], "centre", CENTRE_COL),
    ):
        fig_text_mm(fig, x + dx, yy,
                    f"{word} {'printed' if printed else 'not printed'}",
                    ha="left", va="top", fontsize=FS_TINY,
                    color=colour if printed else MUTED,
                    fontweight="bold" if printed else "normal")
    yy += _pitch(FS_TINY, 1.50)

    groups, unit = _entry_groups(design, variant)
    n_cols = max(1, int(w // _mono_adv(fig, FS_TINY)))
    if variant == "centers_standard":
        segs, shown = _pack(groups[:A_BINS_SHOWN], n_cols, A_BINS_SHOWN)
        yy = _mono_flow(fig, x, yy, w, segs, FS_TINY)
        fig_text_mm(fig, x, yy, f"{len(groups) - shown - 1} further {unit}, same form",
                    ha="left", va="top", fontsize=FS_TINY, color=MUTED, style="italic")
        yy += _pitch(FS_TINY, 1.35)
        yy = _mono_flow(fig, x, yy, w, groups[-1], FS_TINY)
    else:
        segs, shown = _pack(groups, n_cols, A_MONO_LINES)
        yy = _mono_flow(fig, x, yy, w, segs, FS_TINY)
        fig_text_mm(fig, x, yy, f"{len(groups) - shown} further {unit}, same form",
                    ha="left", va="top", fontsize=FS_TINY, color=MUTED, style="italic")
        yy += _pitch(FS_TINY, 1.35)
    if yy > y + h:
        raise RuntimeError(
            f"panel a cell {variant} runs to {yy:.1f} mm, past its {y + h:.1f} mm box"
        )


def _panel_a(fig, design: dict) -> None:
    panel_label_at(fig, X_LEFT, A_LETTER_Y, "a",
                   "Two textual binding features, crossed at fixed information")
    fid = design["field"]["field_id"]
    fig_text_mm(fig, X_LEFT, A_NOTE_Y,
                f"{fid} · replay fields sha256 {_sha_file(FIELDS_PATH)[:12]}… · "
                f"{design['n_instruction_lines']}-line instruction block identical in all "
                "four conditions, not printed",
                ha="left", va="top", fontsize=FS_TINY, color=MUTED, family="monospace")

    for k, (x, w) in enumerate(zip(A_COL_X, A_COL_W)):
        printed = bool(k)
        fig_text_mm(fig, x + 0.5 * w, A_HEAD_Y,
                    f"bin index {'printed' if printed else 'not printed'}",
                    ha="center", va="top", fontsize=FS_SMALL,
                    color=INDEX_COL if printed else MUTED,
                    fontweight="bold" if printed else "normal")
    for r, (top, h) in enumerate(zip(A_ROW_TOP, A_ROW_H)):
        printed = bool(r)
        fig_text_mm(fig, A_GUTTER_X + A_GUTTER_W - 2.0, top + 0.5 * h - 1.4,
                    f"bin centre\n{'printed' if printed else 'not printed'}",
                    ha="right", va="center", fontsize=FS_SMALL,
                    color=CENTRE_COL if printed else MUTED,
                    fontweight="bold" if printed else "normal", linespacing=1.30)
        for c, (x, w) in enumerate(zip(A_COL_X, A_COL_W)):
            variant = next(v for v in CONDITIONS
                           if PRINTS_INDEX[v] == bool(c) and PRINTS_CENTRE[v] == printed)
            _cell(fig, x, top, w, h, design, variant)

    y_rule = A_ROW_TOP[1] + A_ROW_H[1] + A_RULE_GAP
    if y_rule > B_LETTER_Y - 2.5:
        raise RuntimeError(
            f"panel a's rule would land at {y_rule:.1f} mm, inside panel b's letter "
            f"at {B_LETTER_Y:.1f} mm"
        )
    rule_mm(fig, X_LEFT, X_RIGHT, y_rule)


# ---------------------------------------------------------------------------
# b  information identity
# ---------------------------------------------------------------------------
def _sci(value: float) -> str:
    """A number for print: exact zero stays 0.0, everything else goes to maths."""
    if value == 0.0:
        return "0.0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** exponent
    return rf"${mantissa:.1f}\times10^{{{exponent}}}$"


def _panel_b(fig, design: dict) -> None:
    ax = mm_axes(fig, B_L, B_TOP, B_W, B_H)
    panel_label_at(fig, X_LEFT, B_LETTER_Y, "b", "Information identity")
    eq = design["eq"]
    rows = (
        (f"{N_BINS} bin masses", eq["max_abs_bin_mass_gap"], PASS_GREEN),
        ("bin centres", eq["max_abs_bin_centre_gap_print_precision"], MUTED),
    )
    top = 1.2e-6
    for i, (_, value, colour) in enumerate(rows):
        y = 1 - i
        ax.plot([0, top], [y, y], color="#E8E8E8", lw=LW_HAIR, zorder=1)
        ax.plot([value], [y], marker="D", ms=MS_MEAN, color=colour, zorder=3, clip_on=False)
        ax.text(value + 0.075 * top, y, _sci(value), ha="left", va="center",
                fontsize=FS_TINY, color=colour, fontweight="bold")
    ax.axvline(0.0, color="#B0B0B0", lw=LW_HAIR, zorder=2)
    ax.set_ylim(-0.70, 1.70)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS_TINY)
    ax.tick_params(axis="y", length=0, pad=1.4)
    ax.set_xlim(-0.06e-6, top)
    ax.set_xticks([0, 5e-7, 1e-6])
    ax.set_xticklabels(["0", r"$5{\times}10^{-7}$", r"$10^{-6}$"], fontsize=FS_TINY)
    ax.set_xlabel("max gap over the four conditions", labelpad=1.0)
    trim_spines(ax, y=False)
    ax.spines["left"].set_visible(False)


def _panel_b_note(fig, design: dict) -> None:
    note = (f"compact and indexed rows state the {N_BINS}-bin grid by rule, so their "
            "centre gap is print precision")
    if _text_w_mm(fig, note, FS_TINY) > B_NOTE_W:
        raise RuntimeError(f"panel b's note does not fit its column: {note!r}")
    fig_text_mm(fig, X_LEFT, B_NOTE_Y, note, ha="left", va="top", fontsize=FS_TINY,
                color=MUTED)


# ---------------------------------------------------------------------------
# c  the six pairwise distances, ordered by prompt character gap
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


def _pairs_by_gap(design: dict) -> list[tuple[str, str, int, str]]:
    """The six pairs ordered by prompt character gap, with cell membership."""
    chars = design["chars"]
    rows = [(a, b, abs(chars[a] - chars[b]), _cell_of(a, b)) for a, b in PAIRS]
    return sorted(rows, key=lambda r: (r[2], r[0], r[1]))


def _noise_band(ax, dec: dict) -> float:
    floor = dec["noise_floor"]
    lo, hi = floor["within_boot"]["ci95"]
    ax.axhspan(lo, hi, color="#DCDCDC", zorder=0, linewidth=0)
    ax.axhline(floor["mean_within_condition_between_block_TV"], color="#8C8C8C",
               lw=LW_HAIR, ls=(0, (3, 2)), zorder=1)
    return float(lo)


def _panel_c(fig, dec: dict | None, design: dict) -> None:
    ax = mm_axes(fig, C_L, C_TOP, C_W, C_H)
    panel_label_at(fig, C_L - 8.6, B_LETTER_Y, "c",
                   "All six pairwise response distances")
    if dec is None:
        _await(ax)
        return

    lo_band = _noise_band(ax, dec)
    rows = _pairs_by_gap(design)
    tops = []
    for i, (a, b, gap, cell) in enumerate(rows):
        boot = dec["pairwise_bootstrap"][_canon_pair_key(a, b)]
        m, clo, chi = float(boot["mean"]), float(boot["ci95"][0]), float(boot["ci95"][1])
        ax.errorbar([i], [m], yerr=[[m - clo], [chi - m]], fmt="o", color=CELL_COL[cell],
                    ms=MS_SERIES, elinewidth=LW_LINE, capsize=CAPSIZE,
                    capthick=LW_LINE, zorder=4)
        ax.text(i, chi + 0.012, f"{m:.3f}", ha="center", va="bottom",
                fontsize=FS_TINY, color=CELL_COL[cell])
        tops.append(chi)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{SHORT[a]}/{SHORT[b]}\n{gap} ch" for a, b, gap, _ in rows],
                       fontsize=FS_TINY, linespacing=1.25)
    ax.tick_params(axis="x", length=0, pad=1.4)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, max(tops) + 0.13)
    ax.set_yticks([0.0, 0.25, 0.50])
    ax.set_ylabel("mean " + DTV)
    ax.set_xlabel("condition pair, ordered by prompt character gap", labelpad=1.0)
    trim_spines(ax, x=False)

    x_key = 0.005
    for cell in CELL_ORDER:
        text = CELL_LABEL[cell]
        ax.text(x_key, 1.005, text, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=FS_TINY, color=CELL_COL[cell], fontweight="bold")
        x_key += (_text_w_mm(fig, text, FS_TINY, fontweight="bold") + 2.6) / C_W
    ax.text(x_key, 1.005, "= feature that differs", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=FS_TINY, color=MUTED)
    if x_key > 0.72:
        raise RuntimeError("panel c's key runs into the right-hand end of the panel")
    ax.text(0.985, lo_band - 0.006,
            "band = within-condition block noise "
            f"{dec['noise_floor']['mean_within_condition_between_block_TV']:.3f}",
            transform=ax.get_yaxis_transform(), ha="right", va="top",
            fontsize=FS_TINY, color=MUTED)


# ---------------------------------------------------------------------------
# d  the factorial cells
# ---------------------------------------------------------------------------
def _panel_d(fig, dec: dict | None) -> None:
    ax = mm_axes(fig, D_L, D_TOP, D_W, D_H)
    panel_label_at(fig, X_LEFT, R3_LETTER_Y, "d",
                   "Which feature changed (post hoc 2x2)")
    if dec is None:
        _await(ax)
        return

    cells = dec["secondary"]["factorial_decomposition"]["cells"]
    _noise_band(ax, dec)
    tops = []
    for i, name in enumerate(CELL_ORDER):
        boot = cells[name]["cell_bootstrap"]
        m, lo, hi = float(boot["mean"]), float(boot["ci95"][0]), float(boot["ci95"][1])
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=CELL_COL[name],
                    ms=MS_MEAN, elinewidth=LW_LINE, capsize=CAPSIZE, capthick=LW_LINE,
                    zorder=4)
        ax.text(i, hi + 0.014, f"{m:.3f}", ha="center", va="bottom", fontsize=FS_TINY,
                color=CELL_COL[name], fontweight="bold")
        ax.text(i, lo - 0.016, f"{cells[name]['ratio_to_block_noise']:.1f}×",
                ha="center", va="top", fontsize=FS_TINY, color=MUTED)
        tops.append(hi)
    ax.set_xticks(range(len(CELL_ORDER)))
    ax.set_xticklabels([CELL_LABEL[n] for n in CELL_ORDER], fontsize=FS_TINY)
    ax.tick_params(axis="x", length=0, pad=1.4)
    ax.set_xlim(-0.55, len(CELL_ORDER) - 0.45)
    ax.set_ylim(0, max(tops) + 0.10)
    ax.set_ylabel("mean " + DTV)
    ax.set_xlabel("feature that differs · × = block-noise multiple", labelpad=1.0)
    trim_spines(ax, x=False)


# ---------------------------------------------------------------------------
# e  the centre feature minus the index feature
# ---------------------------------------------------------------------------
def _panel_e(fig, dec: dict | None) -> None:
    ax = mm_axes(fig, E_L, E_TOP, E_W, E_H)
    panel_label_at(fig, E_L - 8.6, R3_LETTER_Y, "e",
                   "Centre minus index (secondary, post hoc)")
    if dec is None:
        _await(ax)
        return

    contrast = dec["secondary"]["factorial_decomposition"]["centre_only_minus_index_only"]
    boot = contrast["bootstrap"]
    st = contrast["exact_sign_test"]
    m, lo, hi = float(boot["mean"]), float(boot["ci95"][0]), float(boot["ci95"][1])
    span = max(hi - lo, 0.05)
    x_lo, x_hi = min(0.0, lo) - 0.55 * span, hi + 0.30 * span
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-1.15, 1.15)
    ax.axvspan(x_lo, 0.0, color="#ECECEC", zorder=0, linewidth=0)
    ax.axvline(0.0, color="#8C8C8C", lw=LW_HAIR, zorder=1)

    colour = VERDICT["suggested"] if contrast["ci_excludes_zero"] else VERDICT["not_established"]
    ax.errorbar([m], [0.14], xerr=[[m - lo], [hi - m]], fmt="D", ms=MS_MEAN, color=colour,
                elinewidth=LW_LINE, capsize=CAPSIZE, capthick=LW_LINE, zorder=4)
    top = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    bottom = (f"sign test $p$ = {st['p_two_sided']:.3f}, "
              f"{int(st['n_centre_larger'])} of {int(st['n_discordant'])} fields")
    for text, y, va, col, weight in ((top, 0.42, "bottom", colour, "bold"),
                                     (bottom, -0.16, "top", MUTED, "normal")):
        w = _text_w_mm(fig, text, FS_TINY)
        half = 0.5 * w / E_W * (x_hi - x_lo)
        xt = min(max(m, x_lo + half + 0.002), x_hi - half - 0.002)
        ax.text(xt, y, text, ha="center", va=va, fontsize=FS_TINY, color=col,
                fontweight=weight)
    ax.text(x_lo + 0.02 * (x_hi - x_lo), -0.92, "favours index", ha="left", va="center",
            fontsize=FS_TINY, color=MUTED)
    ax.text(x_hi - 0.02 * (x_hi - x_lo), -0.92, "favours centre", ha="right", va="center",
            fontsize=FS_TINY, color=CENTRE_COL)
    ax.set_yticks([])
    ax.set_xlabel("centre-only $-$ index-only mean " + DTV, labelpad=1.0)
    trim_spines(ax, y=False)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------------------
# f  THE KEY PANEL: within-cell length gaps
# ---------------------------------------------------------------------------
def _panel_f_pairs(ax, dec: dict, ins: dict) -> None:
    """Left half of panel f: the narrow-gap and wide-gap pair of every cell."""
    ticks: list[float] = []
    labels: list[str] = []
    lo_all, hi_all = [], []
    for g, name in enumerate(CELL_ORDER):
        row = ins[name]
        base = g * F_PITCH
        means = []
        for dx, side in ((0.0, row["narrow_gap_pair"]), (F_WIDE_DX, row["wide_gap_pair"])):
            boot = dec["pairwise_bootstrap"][side["pair"]]
            m, lo, hi = float(boot["mean"]), float(boot["ci95"][0]), float(boot["ci95"][1])
            a, b = side["pair"].split("__vs__")
            ax.errorbar([base + dx], [m], yerr=[[m - lo], [hi - m]], fmt="o",
                        ms=MS_MEAN, color=CELL_COL[name], elinewidth=LW_LINE,
                        capsize=CAPSIZE, capthick=LW_LINE, zorder=5)
            ax.text(base + dx, hi + 0.013, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=FS_TINY, color=CELL_COL[name], fontweight="bold")
            ticks.append(base + dx)
            labels.append(f"{SHORT[a]}/{SHORT[b]}\n{int(side['char_gap'])} ch")
            means.append(m)
            lo_all.append(lo)
            hi_all.append(hi)
        ax.plot([base, base + F_WIDE_DX], means, color=CELL_COL[name], lw=LW_THIN,
                zorder=3)

    span = max(hi_all) - min(lo_all)
    # the two-line group header sits above every interval AND above the value
    # labels that ride 0.013 over each interval's top, so the headroom is set
    # from the header's own leading rather than guessed
    y_lo, y_hi = min(lo_all) - 0.16 * span, max(hi_all) + 0.60 * span
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.52, (len(CELL_ORDER) - 1) * F_PITCH + F_WIDE_DX + 0.52)
    for g, name in enumerate(CELL_ORDER):
        base = g * F_PITCH + 0.5 * F_WIDE_DX
        ax.text(base, y_hi, CELL_LABEL[name] + " differs", ha="center", va="top",
                fontsize=FS_TINY, color=CELL_COL[name], fontweight="bold")
        ax.text(base, y_hi - 0.130 * span,
                f"character gap × {float(ins[name]['char_gap_ratio']):.1f}",
                ha="center", va="top", fontsize=FS_TINY, color=INK)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=FS_TINY, linespacing=1.25)
    ax.tick_params(axis="x", length=0, pad=1.4)
    ax.set_ylabel("mean " + DTV)
    ax.set_xlabel("narrow-gap and wide-gap pair within each cell", labelpad=1.0)
    trim_spines(ax, x=False)


def _panel_f_diff(ax, ins: dict) -> None:
    """Right half of panel f: the paired wide minus narrow difference per cell."""
    lo_all, hi_all = [], []
    for name in CELL_ORDER:
        ci = ins[name]["paired_difference_wide_minus_narrow"]["ci95"]
        lo_all.append(float(ci[0]))
        hi_all.append(float(ci[1]))
    span = max(hi_all) - min(lo_all)
    x_lo = min(0.0, min(lo_all)) - 0.62 * span
    x_hi = max(0.0, max(hi_all)) + 0.24 * span
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-1.10, len(CELL_ORDER) - 0.35)
    ax.axvline(0.0, color="#8C8C8C", lw=LW_HAIR, zorder=1)

    n_fields = None
    for g, name in enumerate(CELL_ORDER):
        boot = ins[name]["paired_difference_wide_minus_narrow"]
        m, lo, hi = float(boot["mean"]), float(boot["ci95"][0]), float(boot["ci95"][1])
        n_fields = int(boot["n_fields"])
        y = len(CELL_ORDER) - 1 - g
        covers = bool(ins[name]["difference_ci_covers_zero"])
        colour = VERDICT["not_established"] if covers else CELL_COL[name]
        ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="D", ms=MS_MEAN,
                    color=colour, elinewidth=LW_LINE, capsize=CAPSIZE,
                    capthick=LW_LINE, zorder=4)
        text = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        w = _text_w_mm(ax.figure, text, FS_TINY)
        half = 0.5 * w / FB_W * (x_hi - x_lo)
        xt = min(max(m, x_lo + half + 0.002), x_hi - half - 0.002)
        ax.text(xt, y + 0.24, text, ha="center", va="bottom", fontsize=FS_TINY,
                color=colour,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=6)
    n_cover = sum(1 for n in CELL_ORDER if ins[n]["difference_ci_covers_zero"])
    ax.text(x_lo + 0.03 * (x_hi - x_lo), -0.82,
            f"{n_cover} of {len(CELL_ORDER)} intervals cover zero",
            ha="left", va="center", fontsize=FS_TINY, color=INK, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=6)
    ax.set_yticks(range(len(CELL_ORDER)))
    ax.set_yticklabels([CELL_LABEL[n] for n in reversed(CELL_ORDER)], fontsize=FS_TINY)
    ax.tick_params(axis="y", length=0, pad=1.4)
    ax.set_xlabel(f"paired wide $-$ narrow over {n_fields} fields", labelpad=1.0)
    trim_spines(ax, y=False)
    ax.spines["left"].set_visible(False)


def _panel_f(fig, dec: dict | None) -> None:
    ax_a = mm_axes(fig, FA_L, F_TOP, FA_W, F_H)
    ax_b = mm_axes(fig, FB_L, F_TOP, FB_W, F_H)
    panel_label_at(fig, X_LEFT, R4_LETTER_Y, "f",
                   "Within a cell the character gap moves and the response does not")
    if dec is None:
        _await(ax_a)
        _await(ax_b, "")
        return
    ins = dec["secondary"]["factorial_decomposition"]["within_cell_length_insensitivity"]
    _panel_f_pairs(ax_a, dec, ins)
    _panel_f_diff(ax_b, ins)


# ---------------------------------------------------------------------------
# g  per-condition block noise
# ---------------------------------------------------------------------------
def _panel_g(fig, dec: dict | None) -> None:
    ax = mm_axes(fig, G_L, G_TOP, G_W, G_H)
    panel_label_at(fig, X_LEFT, R5_LETTER_Y, "g",
                   "Noise heterogeneity across the four conditions")
    if dec is None:
        _await(ax)
        fig_text_mm(fig, X_LEFT, G_NOTE_Y,
                    "within-condition between-block distance; awaiting acquisition",
                    ha="left", va="top", fontsize=FS_TINY, color=MUTED)
        return

    noise, recomputed = _per_condition_noise(dec)
    fig_text_mm(fig, X_LEFT, G_NOTE_Y,
                "within-condition between-block "
                + DTV
                + (", recomputed from the run's trace.jsonl because decision.json "
                   "carries the pooled floor only" if recomputed
                   else ", read from decision.json"),
                ha="left", va="top", fontsize=FS_TINY, color=MUTED)

    floor = dec["noise_floor"]
    pooled = float(floor["mean_within_condition_between_block_TV"])
    p_lo, p_hi = (float(v) for v in floor["within_boot"]["ci95"])
    ax.axvspan(p_lo, p_hi, color="#DCDCDC", zorder=0, linewidth=0)
    ax.axvline(pooled, color="#8C8C8C", lw=LW_HAIR, ls=(0, (3, 2)), zorder=1)

    order = list(CONDITIONS)
    lo_all, hi_all = [], []
    for g, cond in enumerate(order):
        row = noise[cond]
        m, lo, hi = float(row["mean"]), float(row["ci95"][0]), float(row["ci95"][1])
        lo_all.append(lo)
        hi_all.append(hi)
        y = len(order) - 1 - g
        ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="o", ms=MS_MEAN,
                    color=COND_COL[cond], elinewidth=LW_LINE, capsize=CAPSIZE,
                    capthick=LW_LINE, zorder=4)
        ax.text(hi + 0.004, y, f"{m:.3f}", ha="left", va="center", fontsize=FS_TINY,
                color=COND_COL[cond], fontweight="bold")
    x_hi = max(hi_all) + 0.030
    ax.set_xlim(min(0.05, min(lo_all) - 0.010), x_hi)
    ax.set_ylim(-0.55, len(order) - 0.30)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)), fontsize=FS_TINY)
    for tick, cond in zip(ax.get_yticklabels(), reversed(order)):
        tick.set_color(COND_COL[cond])
    ax.tick_params(axis="y", length=0, pad=1.4)
    ax.set_xlabel("within-condition between-block mean " + DTV, labelpad=1.0)
    ax.text(pooled, len(order) - 0.58, f"pooled floor {pooled:.3f}", ha="center",
            va="center", fontsize=FS_TINY, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=6)
    trim_spines(ax, y=False)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------------------
# verdict strip
# ---------------------------------------------------------------------------
def _gate_rows(fig, ax, rows) -> float:
    """Draw the verdict rows, refuse to leave the box, return the last baseline.

    figS21's panel i overflowed its own rectangle when the item pitch was
    guessed; here the pitch is derived from the type size and the closing block
    is asserted to sit inside the box before anything else is drawn.
    """
    step = _pitch(FS_TINY, V_ROW_LEAD) / V_H         # one row, in axes fractions
    y_last = V_Y_TOP - step * (len(rows) - 1)
    y_floor = y_last - (V_SUMMARY_GAP + 2 * V_SUMMARY_LEAD) / V_H
    if y_floor < 0.045:
        raise RuntimeError(
            f"the verdict strip needs {len(rows)} rows at {step:.4f} of the box plus "
            f"a three-line closing block; the last line would land at {y_floor:.3f} "
            f"and print outside the rectangle"
        )
    w_label = (V_X_STAT - V_X_LABEL) * V_W - 2.0
    w_stat = (V_X_RIGHT - V_X_STAT) * V_W
    y = V_Y_TOP
    for label, stat, verdict, colour in rows:
        if _text_w_mm(fig, label, FS_TINY) > w_label:
            raise RuntimeError(f"verdict-strip label does not fit its column: {label!r}")
        if _text_w_mm(fig, stat, FS_TINY) + _text_w_mm(fig, verdict, FS_TINY) + 3.0 > w_stat:
            raise RuntimeError(f"verdict-strip statistic and verdict collide: {stat!r} / {verdict!r}")
        ax.plot([V_X_MARK], [y], marker="s", ms=MS_SERIES, color=colour, clip_on=False)
        ax.text(V_X_LABEL, y, label, ha="left", va="center", fontsize=FS_TINY, color=INK)
        ax.text(V_X_STAT, y, stat, ha="left", va="center", fontsize=FS_TINY, color=MUTED)
        ax.text(V_X_RIGHT, y, verdict, ha="right", va="center", fontsize=FS_TINY,
                color=colour, fontweight="bold")
        y -= step
    return y_last


#: Every verdict ``analyze_sbc_control.py`` can emit, and how the strip words it.
VERDICT_WORD = {
    "CENTRE_PRINTING_OVER_LENGTH": ("centre printing over length", VERDICT["suggested"]),
    "PRIMARY_CONTRAST_NOT_DIAGNOSTIC": ("not diagnostic", VERDICT["not_established"]),
    "BINDING_OVER_LENGTH": ("binding over length", VERDICT["supported"]),
    "LENGTH_OVER_BINDING": ("length over binding", VERDICT["not_established"]),
    "UNRESOLVED": ("unresolved", VERDICT["not_established"]),
}


def _panel_verdicts(fig, dec: dict | None) -> None:
    ax = mm_panel(fig, V_L, V_TOP, V_W, V_H)
    fig_text_mm(fig, X_LEFT, V_LETTER_Y, "Verdicts", ha="left", va="baseline",
                fontsize=FS_SMALL, color=INK, fontweight="bold")
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                    facecolor="#F7F7F7", edgecolor="#B0B0B0",
                                    lw=LW_THIN, zorder=0))
    if dec is None:
        ax.text(0.5, 0.5, "awaiting acquisition", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS_TINY, color=MUTED, style="italic")
        return

    if dec["verdict"] not in VERDICT_WORD:
        raise RuntimeError(f"decision.json carries an unknown verdict: {dec['verdict']!r}")
    alpha = float(dec["inference_settings"]["alpha_primary"])
    glob = dec["secondary"]["global_any_condition_difference"]
    fac = dec["secondary"]["factorial_decomposition"]
    contrast = fac["centre_only_minus_index_only"]
    ins = fac["within_cell_length_insensitivity"]
    pc = dec["primary_contrast"]
    diag = pc["diagnosticity"]

    glob_ok = float(glob["p_one_sided"]) < alpha
    centre_resolved = bool(contrast["ci_excludes_zero"]) and float(contrast["difference"]) > 0.0
    insensitive = bool(fac["length_insensitive_in_every_cell"])
    ratios = ", ".join(f"{float(ins[n]['char_gap_ratio']):.1f}×" for n in CELL_ORDER)
    n_cover = sum(1 for n in CELL_ORDER if ins[n]["difference_ci_covers_zero"])

    rows = (
        ("Global any-condition difference",
         f"mean pairwise TV {glob['observed_mean_pairwise_TV']:.3f}, "
         f"p = {glob['p_one_sided']:.4f} ({int(glob['n_perm']):,} perms)",
         "established" if glob_ok else "not established",
         VERDICT["supported"] if glob_ok else VERDICT["not_established"]),
        ("Centre feature minus index feature",
         f"{contrast['difference']:+.3f} "
         f"[{contrast['bootstrap']['ci95'][0]:+.3f}, {contrast['bootstrap']['ci95'][1]:+.3f}], "
         f"sign test p = {contrast['exact_sign_test']['p_two_sided']:.3f}",
         "resolved, post hoc" if centre_resolved else "not resolved",
         VERDICT["suggested"] if centre_resolved else VERDICT["not_established"]),
        ("Within-cell length insensitivity",
         f"character gap {ratios}; {n_cover} of {len(CELL_ORDER)} paired CIs cover zero",
         "no resolvable length effect" if insensitive else "a cell moves with length",
         VERDICT["not_established"] if insensitive else VERDICT["supported"]),
        (f"Prespecified pivot contrast on {SHORT[PIVOT]}",
         f"{pc['delta']:+.3f} [{pc['delta_bootstrap']['ci95'][0]:+.3f}, "
         f"{pc['delta_bootstrap']['ci95'][1]:+.3f}]; length and centre accounts both "
         f"predict {diag['length_account_predicts']}",
         "diagnostic" if diag["diagnostic"] else "non-diagnostic",
         VERDICT["supported"] if diag["diagnostic"] else MUTED),
    )
    y_last = _gate_rows(fig, ax, rows)

    # ---- closing block ----------------------------------------------------
    # Four things used to share one middot-separated line: what was collected,
    # the prespecified threshold, the decision, and how far the decision may be
    # read. They are ranked here instead of listed. Provenance is set small and
    # grey on its own line; the decision is set large, in its verdict colour,
    # against a bold "Overall verdict" label; the scope limit sits alone on the
    # last line behind its own "Scope" label so it cannot be skimmed as one
    # more finding. The verdict word and every number still come straight from
    # decision.json - only their arrangement and weight change.
    word, colour = VERDICT_WORD[dec["verdict"]]
    y_facts = y_last - V_SUMMARY_GAP / V_H
    y_word = y_facts - V_SUMMARY_LEAD / V_H
    y_scope = y_word - V_SUMMARY_LEAD / V_H

    rule_y = y_last - (V_SUMMARY_GAP * 0.5) / V_H
    ax.plot([V_X_MARK, V_X_RIGHT], [rule_y, rule_y], color=RULE, lw=LW_HAIR,
            zorder=1, clip_on=False)

    # The three lines keep the label / content columns the gate rows above use,
    # so the eye reads one table rather than a paragraph that grew a rule.
    acq = (f"{int(dec['acquisition']['n_valid']):,} of "
           f"{int(dec['acquisition']['n_calls']):,} calls valid over "
           f"{int(dec['acquisition']['n_fields_used'])} fields")
    thresh = f"prespecified threshold   alpha {alpha:g}"
    scope = "read as a statement about these four serializations only"

    w_label = (V_X_STAT - V_X_LABEL) * V_W - 2.0
    for lab in ("acquisition", "Overall verdict", "scope"):
        size = FS_SMALL if lab == "Overall verdict" else FS_TINY
        if _text_w_mm(fig, lab, size, fontweight="bold") > w_label:
            raise RuntimeError(f"closing-block label does not fit its column: {lab!r}")
    w_acq = (V_X_THRESHOLD - V_X_STAT) * V_W - 2.0
    if _text_w_mm(fig, acq, FS_TINY) > w_acq:
        raise RuntimeError(f"the acquisition count does not fit its column: {acq!r}")
    w_tail = (V_X_RIGHT - V_X_THRESHOLD) * V_W
    if _text_w_mm(fig, thresh, FS_TINY) > w_tail:
        raise RuntimeError(f"the prespecified threshold does not fit: {thresh!r}")
    w_content = (V_X_RIGHT - V_X_STAT) * V_W
    if _text_w_mm(fig, word, FS_SMALL, fontweight="bold") > w_content:
        raise RuntimeError(f"the verdict word does not fit the box: {word!r}")
    if _text_w_mm(fig, scope, FS_TINY, style="italic") > w_content:
        raise RuntimeError(f"the scope note does not fit the box: {scope!r}")

    ax.text(V_X_LABEL, y_facts, "acquisition", ha="left", va="center",
            fontsize=FS_TINY, color=MUTED)
    ax.text(V_X_STAT, y_facts, acq, ha="left", va="center",
            fontsize=FS_TINY, color=MUTED)
    ax.text(V_X_RIGHT, y_facts, thresh, ha="right", va="center",
            fontsize=FS_TINY, color=MUTED)

    ax.plot([V_X_MARK], [y_word], marker="s", ms=MS_MEAN, color=colour,
            clip_on=False)
    ax.text(V_X_LABEL, y_word, "Overall verdict", ha="left", va="center",
            fontsize=FS_SMALL, color=INK, fontweight="bold")
    ax.text(V_X_STAT, y_word, word, ha="left", va="center",
            fontsize=FS_SMALL, color=colour, fontweight="bold")

    ax.text(V_X_LABEL, y_scope, "scope", ha="left", va="center",
            fontsize=FS_TINY, color=MUTED)
    ax.text(V_X_STAT, y_scope, scope, ha="left", va="center",
            fontsize=FS_TINY, color=MUTED, style="italic")


# ---------------------------------------------------------------------------
def build(_out_dir: Path | None = None, decision_path: Path | None = None) -> Path:
    apply_style()
    design = _load_design()
    path = Path(decision_path or DECISION_PATH)
    dec = _load_decision(path, design)
    HAVE_RESULTS = dec is not None      # noqa: N806 - the one flag the layout branches on
    if not HAVE_RESULTS:
        print(f"[figS23] WARNING: {path} is absent; the response panels "
              "(c-g) and the verdict strip are drawn as empty axes marked "
              "'awaiting acquisition'. Re-run this module after "
              "analysis/matched_rep_collective/analyze_sbc_control.py has been run.")

    fig = new_figure(H_MM)
    _panel_a(fig, design)
    _panel_b(fig, design)
    _panel_b_note(fig, design)
    _panel_c(fig, dec, design)
    _panel_d(fig, dec)
    _panel_e(fig, dec)
    _panel_f(fig, dec)
    _panel_g(fig, dec)
    _panel_verdicts(fig, dec)
    _overflow(fig, "results" if HAVE_RESULTS else "design-only")
    return save_fig(fig, "figS23_serialization_binding")


def main(argv: list[str] | None = None) -> int:      # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, default=None,
                        help="override the path to sbc_primary/decision.json")
    args = parser.parse_args(argv)
    print(build(decision_path=args.decision))
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
