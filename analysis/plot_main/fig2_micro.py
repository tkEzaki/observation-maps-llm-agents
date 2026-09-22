"""Figure 2 - Equivalent physical fields elicit different microscopic operators.

Geometry pass, plus one correctness repair. The canvas is a fixed 180 mm
board with every panel placed in millimetres from the top-left, the dead
bands between rows are gone, the prompt-excerpt boxes are fitted to their own
text and legend boxes are replaced by direct labels.

Round 2 restructuring: the former panels a (field drawing) and b (three
excerpt boxes) are merged into ONE wide panel a - "Same physical field,
three serialized observations" - reading left to right from the field to its
three encodings. The remaining panels re-letter to b-g, removing the f'
prime letter. Content, data, statistics and colours are unchanged.

Correctness repair (panel a): the figure claims *one physical field,
three observation maps*, so all three serialised observations must be of the
same rotation of the same profile. They are read directly from the Stage B
stimulus records at a single ``offset_index`` (see
:func:`_shared_field_records`), and panel a draws that record's generating
von Mises density - with kappa and mu read from the record itself - verified
in-build against the record's own encoded 24-bin field.

The guard behind that repair does not trust the ``offset_index`` label on its
own - an index is a name, and names can be re-used between dataset versions.
:func:`_shared_field_records` therefore checks the generating parameters
exactly, a derived :func:`_physical_id` over them, and finally the encoded
field itself, by reading the 24-bin mass back out of two independently
formatted serialisations and out of the moment summary
(:func:`_assert_one_encoded_field`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.plot_main.style import (
    ACTION_LABELS,
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
    MS_SERIES,
    MUTED,
    REP,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    annotate_cells,
    apply_style,
    cbar_mm,
    despine,
    direct_label,
    fig_text_mm,
    mm_axes,
    new_figure,
    panel_label,
    panel_label_at,
    save_fig,
    stacked_trinomial,
    trim_spines,
)

# --------------------------------------------------------------------------
# Layout board (mm from the top-left of a 180 mm canvas)
# --------------------------------------------------------------------------
H_MM = 125.9

COL_X = (12.6, 71.0, 129.4)   # shared left edge of every panel column
COL_W = 46.0
ROW_TOP = (11.8, 45.1, 86.6)  # shared top edge of every panel row
ROW_H = 31.0
LETTER_Y = (6.4, 43.5, 85.0)  # shared letter baseline per row

A_W = 19.0                    # schematic circle: compressed, it carries no data
A_AMP = 0.50                  # peak of the density lobe, above the unit circle
A_PAD = 0.16                  # breathing room around the circle-plus-lobe extent
# The drawn density is only allowed on the panel if integrating it over the
# serialised bin intervals reproduces the serialised bin masses to this
# tolerance - the pretty picture is pinned to the data it claims to show.
A_DENSITY_TOL = 0.02
B_LEFT = 34.0
B_RIGHT = COL_X[2] + COL_W    # excerpt strip ends flush with column 3
B_GAP = 3.2
B_TOP = 12.2                  # top edge of the excerpt boxes
B_H = 30.0                    # drawing surface for the excerpt strip
B_INSET = 1.3                 # uniform padding between excerpt text and its box
B_HEAD_DY = 1.2               # baseline of the representation name above a box
B_NOTE_DY = 1.7               # gap between a box and its full-length note

# The field drawn and quoted in panel a.
PROFILE = "unimodal_k9"
# Panel b (unimodal response) conditions its response curves on this same
# offset index, so panel a and panel b describe one rotation of one field.
PANEL_B_OFFSET = 0

_BIN_CENTER_RE = re.compile(r"bin_\d+ center ([+-]\d+\.\d+): (\d+\.\d+)")
_BIN_INTERVAL_RE = re.compile(r"bin_\d+ \[([+-]\d+\.\d+),([+-]\d+\.\d+)\): (\d+\.\d+)")
_MOMENT_RE = re.compile(r"moment_(\d)_(cos|sin): ([+-]\d+\.\d+)")

N_BINS = 24
# Both bin encoders print angles and masses at six decimals, so two
# serialisations of one field have to agree to that printed precision.
PRINTED_DP = 6
BIN_TOL = 10.0 ** -PRINTED_DP
# The moment encoder prints six decimals of a sum over 24 masses that were
# themselves rounded to six decimals: 24 * 5e-7 + 5e-7, rounded up.
MOMENT_TOL = 2.5e-5

# The quantities that actually pin the physical field. ``offset_index`` alone
# is only a *label* - its meaning can change between dataset versions - so the
# guard hashes the whole tuple and then verifies the encoded field itself.
PHYSICAL_KEYS = ("profile", "concentration", "offset_index", "offset_radians")
PHYSICAL_ID_CHARS = 6   # short form, annotated on panel a so a reader can trace it


def _canonical_field_string(record: dict) -> str:
    """Canonical, type-normalised rendering of the physical tuple.

    ``repr`` of a float is the shortest string that round-trips it, so the
    canonical string is exact (no ``isclose``, no rounding) and identical on
    every platform. Ints and floats are normalised so that ``9`` and ``9.0``
    cannot produce two different ids for one field.
    """
    return "|".join((
        f"profile={record['profile']}",
        f"concentration={float(record['concentration'])!r}",
        f"offset_index={int(record['offset_index'])}",
        f"offset_radians={float(record['offset_radians'])!r}",
    ))


def _physical_id(record: dict) -> str:
    """sha256 of :func:`_canonical_field_string` - a derived physical hash.

    ``stimuli.jsonl`` stores ``prompt_sha256`` (which differs between
    representations of the *same* field, so it cannot be compared across them)
    but no hash of the field, so one is derived here.
    """
    return hashlib.sha256(_canonical_field_string(record).encode("utf-8")).hexdigest()


def _centers_bins(text: str) -> tuple[np.ndarray, np.ndarray]:
    """(bin centre, mass) as ``centers_24_standard`` prints them."""
    found = _BIN_CENTER_RE.findall(text)
    assert len(found) == N_BINS, (
        f"centers_24_standard: expected {N_BINS} bin centres, found {len(found)}"
    )
    return (np.array([float(t) for t, _ in found]),
            np.array([float(m) for _, m in found]))


def _interval_bins(text: str) -> tuple[np.ndarray, np.ndarray]:
    """(bin midpoint, mass) as ``intervals_24_decimal6`` prints them."""
    found = _BIN_INTERVAL_RE.findall(text)
    assert len(found) == N_BINS, (
        f"intervals_24_decimal6: expected {N_BINS} bin intervals, found {len(found)}"
    )
    return (np.array([(float(lo) + float(hi)) / 2.0 for lo, hi, _ in found]),
            np.array([float(m) for _, _, m in found]))


def _serialised_moments(text: str) -> dict[tuple[int, str], float]:
    """The six numbers ``moments_m1_m3`` prints, keyed by (order, part)."""
    found = _MOMENT_RE.findall(text)
    assert len(found) == 6, (
        f"moments_m1_m3: expected 6 moment values, found {len(found)}"
    )
    return {(int(k), part): float(v) for k, part, v in found}


def _assert_one_encoded_field(recs: dict) -> None:
    """Physical cross-check on the prompt *content*, not on the labels.

    ``centers_24_standard`` and ``intervals_24_decimal6`` are two different
    printings of the same underlying 24-bin relative-phase mass, and
    ``moments_m1_m3`` is that same mass summarised by its first three circular
    moments. Recovering one from the other is real evidence that the three
    records describe one physical field, rather than three records that merely
    carry the same ``offset_index``.
    """
    th_c, m_c = _centers_bins(recs["centers_24_standard"]["serialized_prompt"])
    th_i, m_i = _interval_bins(recs["intervals_24_decimal6"]["serialized_prompt"])

    d_theta = np.abs(th_c - th_i)
    if float(d_theta.max()) > BIN_TOL:
        j = int(np.argmax(d_theta))
        raise AssertionError(
            "panel a: centers_24_standard and intervals_24_decimal6 do not "
            "bin the circle the same way, so they are not two observations of "
            f"one field - worst bin_{j:02d}: centre {th_c[j]:+.6f} rad vs "
            f"interval midpoint {th_i[j]:+.6f} rad "
            f"(max |dtheta| = {float(d_theta.max()):.3e} > {BIN_TOL:.1e})"
        )

    d_mass = np.abs(m_c - m_i)
    if float(d_mass.max()) > BIN_TOL:
        j = int(np.argmax(d_mass))
        raise AssertionError(
            "panel a: centers_24_standard and intervals_24_decimal6 "
            "serialise DIFFERENT 24-bin mass distributions, so they are not "
            "two observations of one field - worst bin_"
            f"{j:02d}: {m_c[j]:.6f} vs {m_i[j]:.6f} "
            f"(max |dmass| = {float(d_mass.max()):.3e} > {BIN_TOL:.1e}, the "
            f"precision both encoders print at, {PRINTED_DP} dp)"
        )

    serialised = _serialised_moments(recs["moments_m1_m3"]["serialized_prompt"])
    for k in (1, 2, 3):
        rebuilt = {"cos": float((m_c * np.cos(k * th_c)).sum()),
                   "sin": float((m_c * np.sin(k * th_c)).sum())}
        for part, value in rebuilt.items():
            gap = abs(value - serialised[(k, part)])
            if gap > MOMENT_TOL:
                raise AssertionError(
                    "panel a: moments_m1_m3 is not the moment summary of "
                    "the 24-bin mass drawn in panel a, so the three records "
                    f"are not one field - moment_{k}_{part} serialised "
                    f"{serialised[(k, part)]:+.6f}, reconstructed from the "
                    f"bins {value:+.6f} (|d| = {gap:.3e} > {MOMENT_TOL:.1e})"
                )


def _shared_field_records(profile: str = PROFILE) -> dict:
    """The three serialised observations of a *single* physical field.

    Selection rule (deterministic, stated so it cannot drift):

    1. Walk ``runs/stimulus_manifold/**/stimuli.jsonl`` in sorted path order
       and take the first acquisition that carries all three representations
       for ``profile`` - the same acquisition the Stage B exporter drew from.
    2. Within it, take the **lowest ``offset_index`` present for all three
       representations**. Not cherry-picked: it is the first offset of the
       sweep, and it is also the offset panel b's response curves are
       conditioned on (``PANEL_B_OFFSET``).

    The previous build took, per representation, the first record encountered
    in the file, which matched on ``profile`` only. That silently paired three
    *different* rotations (offset 30 / 4 / 29), so the panel titles "Same
    field" and "three observations of the same field" were false. The assert
    below makes that failure mode loud instead of silent.
    """
    for path in sorted((ROOT / "runs/stimulus_manifold").rglob("stimuli.jsonl")):
        by_rep: dict[str, dict[int, dict]] = {rep: {} for rep in REP_ORDER}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                o = json.loads(line)
                rep = o.get("representation")
                if rep in by_rep and o.get("profile") == profile:
                    by_rep[rep][int(o["offset_index"])] = o
        shared = set.intersection(*(set(v) for v in by_rep.values()))
        if not shared:
            continue
        recs = {rep: by_rep[rep][min(shared)] for rep in REP_ORDER}
        # Three-stage physical-identity guard. An index is a label, and a
        # label's meaning can change between dataset versions, so matching on
        # offset_index alone is not enough.
        # (1) the generating parameters, compared exactly - two different
        #     rotations differ in the last bits of offset_radians too, so
        #     exact equality is the right test, not isclose.
        for key in PHYSICAL_KEYS:
            seen = {rep: recs[rep][key] for rep in REP_ORDER}
            if len(set(seen.values())) != 1:
                raise AssertionError(
                    "panel a must quote ONE physical field, but the three "
                    f"selected records diverge on {key!r}: "
                    + ", ".join(f"{rep}={seen[rep]!r}" for rep in REP_ORDER)
                )
        # (2) a derived physical_id over that whole tuple, recomputed per record
        ids = {rep: _physical_id(recs[rep]) for rep in REP_ORDER}
        if len(set(ids.values())) != 1:
            raise AssertionError(
                "panel a must quote ONE physical field, but the derived "
                "physical_id diverges across representations: "
                + ", ".join(f"{rep}={ids[rep][:PHYSICAL_ID_CHARS]}"
                            for rep in REP_ORDER)
            )
        # (3) the field itself, read back out of the serialisations
        _assert_one_encoded_field(recs)
        return recs
    raise FileNotFoundError(
        f"no stimuli.jsonl carries all of {REP_ORDER} for profile {profile!r}"
    )


def _encoded_field(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """Bin centres (rad, relative to the focal phase) and mass, as serialised.

    The stimulus records carry the field only in its encoded form - the raw
    peer phases are not stored anywhere in ``stimuli.jsonl`` - so panel a
    verifies its drawn generating density against this 24-bin relative-phase
    mass, and overlays the mass itself as a stepped outline.
    """
    found = _BIN_CENTER_RE.findall(record["serialized_prompt"])
    assert len(found) == 24, f"expected 24 bin centres, found {len(found)}"
    theta = np.array([float(a) for a, _ in found])
    mass = np.array([float(b) for _, b in found])
    return theta, mass


def _mean_trinomial(curves: pd.DataFrame, profile: str, offset_index: int | None = 0) -> dict:
    sub = curves[curves["profile"] == profile]
    if offset_index is not None:
        sub = sub[sub["offset_index"] == offset_index]
    out = {}
    for rep in REP_ORDER:
        g = sub[sub["representation"] == rep]
        out[rep] = (
            float(g["p_retard"].mean()),
            float(g["p_stay"].mean()),
            float(g["p_advance"].mean()),
        )
    return out


def _from_atlas(atlas: pd.DataFrame, profile: str) -> dict:
    """Recover (p-, p0, p+) from activity and a0."""
    out = {}
    for rep_short, rep in (
        ("moments", "moments_m1_m3"),
        ("centers", "centers_24_standard"),
        ("intervals", "intervals_24_decimal6"),
    ):
        row = atlas[(atlas["representation"] == rep_short) & (atlas["profile"] == profile)].iloc[0]
        a = float(row["activity"])
        a0 = float(row["a0"])
        p_plus = float(np.clip(0.5 * (a + a0), 0, 1))
        p_minus = float(np.clip(0.5 * (a - a0), 0, 1))
        p0 = float(np.clip(1.0 - a, 0, 1))
        s = p_minus + p0 + p_plus
        out[rep] = (p_minus / s, p0 / s, p_plus / s)
    return out


def _excerpt_lines(text: str) -> str:
    """First five data rows of a serialisation, marked as an excerpt.

    Unchanged selection rule: the leading prose block is dropped, the first
    five data rows are kept, and a trailing ellipsis row is appended so the
    block reads as a window onto a longer serialisation rather than as the
    complete observation.
    """
    lines = [ln for ln in text.splitlines() if ln.strip() and ln.strip() != "…"]
    data_lines = [ln for ln in lines if any(k in ln for k in ("moment_", "bin_", "cos", "sin", "["))][:5]
    if not data_lines:
        data_lines = lines[3:8]
    return "\n".join([ln.strip() for ln in data_lines] + ["…"])


def _mm_bbox(fig, artist, ax):
    """Artist extent in the mm data coordinates of ``ax``."""
    r = fig.canvas.get_renderer()
    return artist.get_window_extent(renderer=r).transformed(ax.transData.inverted())


def _tight_trinomial_labels(ax, probs_by_rep, y: float = 1.01) -> None:
    """A / a0 annotations sitting just clear of the bar tops (same numbers)."""
    for i, rep in enumerate(REP_ORDER):
        p = probs_by_rep[rep]
        a = p[0] + p[2]
        a0 = p[2] - p[0]
        ax.text(i, y, f"$A$={a:.2f}\n$a_0$={a0:+.2f}", ha="center", va="bottom",
                fontsize=FS_TINY, linespacing=1.15, color=INK)


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    curves = pd.read_csv(ROOT / "analysis/complex_kernel_stimulus_manifold/offset_action_curves.csv")
    atlas = pd.read_csv(ROOT / "analysis/stage_b_offline_p3_p4/phenotype_atlas.csv")
    eps = pd.read_csv(ROOT / "analysis/complex_kernel_antipodal_weight/epsilon_trajectories.csv")
    field = _shared_field_records()
    offset_index = int(field[REP_ORDER[0]]["offset_index"])
    physical_id = _physical_id(field[REP_ORDER[0]])
    pid_short = physical_id[:PHYSICAL_ID_CHARS]
    # traceable on the console as well as on the panel: the short form is what
    # the figure carries, the canonical string is how it was derived
    print(f"fig2 panel-a physical_id {pid_short} ({physical_id}) "
          f"from {_canonical_field_string(field[REP_ORDER[0]])}")
    # panel a quotes one field; the response panels (b onward) must be the
    # response to that same rotation of it
    assert offset_index == PANEL_B_OFFSET, (
        f"panel a shows offset_index {offset_index} but panel b (unimodal "
        f"response) is conditioned on {PANEL_B_OFFSET}"
    )

    fig = new_figure(H_MM)

    # ---- a  same physical field, three serialized observations -------------
    # One wide panel reading left to right: the field itself, then the three
    # excerpt boxes that serialise it. The field as it was generated: the von Mises density rho(phi) proportional
    # to exp(kappa cos(phi - mu)), drawn as a smooth closed radial profile
    # r(phi) = 1 + A_AMP * rho/max(rho) around a thin unit circle. kappa and mu
    # come from the frozen record itself, and the drawing is only trusted after
    # the guard below has integrated this same density over the serialised bin
    # intervals and recovered the serialised 24-bin masses.
    rec_a = field["centers_24_standard"]
    kappa = float(rec_a["concentration"])
    mu = float(rec_a["offset_radians"])
    phi = np.linspace(-math.pi, math.pi, 721)
    rho = np.exp(kappa * np.cos(phi - mu))
    r_prof = 1.0 + A_AMP * rho / rho.max()

    _, mass = _encoded_field(rec_a)
    edges = _BIN_INTERVAL_RE.findall(
        field["intervals_24_decimal6"]["serialized_prompt"])
    quad = np.array([
        np.trapezoid(np.exp(kappa * np.cos(np.linspace(float(lo), float(hi), 65) - mu)),
                     np.linspace(float(lo), float(hi), 65))
        for lo, hi, _ in edges
    ])
    quad = quad / quad.sum()
    d_quad = float(np.abs(quad - mass).max())
    assert d_quad < A_DENSITY_TOL, (
        "panel a: the drawn von Mises density, integrated over the serialised "
        "bin intervals, does not reproduce the serialised 24-bin masses "
        f"(max |diff| = {d_quad:.3e} > {A_DENSITY_TOL:.0e}) - the drawing "
        "would not be a picture of the field the payloads encode"
    )

    axA = mm_axes(fig, COL_X[0], ROW_TOP[0], A_W, A_W)
    # optically centre the circle-plus-lobe extent, not the circle alone: the
    # lobe points towards mu, so it would otherwise sit off-balance in the box
    x_all = np.concatenate([np.cos(phi), r_prof * np.cos(phi)])
    y_all = np.concatenate([np.sin(phi), r_prof * np.sin(phi)])
    half = max(float(x_all.max() - x_all.min()),
               float(y_all.max() - y_all.min())) / 2.0 + A_PAD
    cx = float(x_all.max() + x_all.min()) / 2.0
    cy = float(y_all.max() + y_all.min()) / 2.0
    axA.set_xlim(cx - half, cx + half)
    axA.set_ylim(cy - half, cy + half)
    axA.set_aspect("equal")
    axA.axis("off")
    # band between the unit circle and the density profile: light neutral fill
    # under a slightly darker hairline, one smooth lobe instead of spokes
    band_x = np.concatenate([r_prof * np.cos(phi), np.cos(phi[::-1])])
    band_y = np.concatenate([r_prof * np.sin(phi), np.sin(phi[::-1])])
    axA.fill(band_x, band_y, facecolor="#DADADA", alpha=0.6,
             edgecolor="none", zorder=2)
    axA.plot(r_prof * np.cos(phi), r_prof * np.sin(phi), color="#8C8C8C",
             lw=LW_HAIR, zorder=4)
    # A stepped outline of the 24-bin masses was tried inside the lobe and
    # dropped: at kappa = 9 the density falls fast enough within one bin that
    # the constant steps cross the smooth profile at every bin edge, which
    # reads as ragged saw-teeth at print size. The density-to-payload tie is
    # carried by the quadrature assert above and stated in the caption.
    axA.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#333333",
                             lw=LW_LINE, zorder=5))
    axA.plot([1.0], [0.0], "o", color=INK, ms=MS_MEAN, zorder=6)   # focal phase
    # name the drawn curve inside the panel, clear of the lobe's upper flank
    axA.text(-0.88, 1.06, r"$\rho(\varphi)$", ha="center", va="center",
             fontsize=FS_TINY, color=MUTED)
    panel_label_at(fig, 4.0, LETTER_Y[0], "a",
                   "Same physical field, three serialized observations")
    # short physical_id on the title line, so the field the panel quotes can
    # be traced back to the stimulus record it was derived from
    fig_text_mm(fig, 68.0, LETTER_Y[0], f"id {pid_short}", ha="left",
                va="baseline", fontsize=FS_TINY, color=MUTED)
    # strip-level pointer, right-aligned on the same title line
    fig_text_mm(fig, B_RIGHT, LETTER_Y[0],
                "excerpts; complete serialisations in Supplementary Fig. S2",
                ha="right", va="baseline", fontsize=FS_SMALL, color=MUTED)
    # three single-line captions rather than one multi-line block: each line is
    # kept narrow enough to stay clear of the excerpt strip's left edge at
    # B_LEFT, and the whole block sits 1 mm left of the drawing's centreline so
    # line 1 does not read on into the per-box notes, which start level with
    # it. The
    # kappa quoted on line 1 is the record's own concentration, not a literal.
    cap_x = COL_X[0] + A_W / 2 - 1.0
    fig_text_mm(fig, cap_x, ROW_TOP[0] + A_W + 2.5,
                f"unimodal $\\kappa$ = {kappa:g} · focal at 0", ha="center",
                va="baseline", fontsize=FS_SMALL, color=MUTED)
    # lines 2-3: the honesty note. What is drawn is the generating density
    # (named rho(phi) inside the panel); every payload in the excerpt boxes -
    # the six moment features included - is a function of its 24-bin masses
    # (verified by the quadrature assert above and by
    # _assert_one_encoded_field). Both lines sit level with the per-box notes,
    # which start at B_LEFT, so each is kept short enough for its centred span
    # to end clear of them.
    fig_text_mm(fig, cap_x, ROW_TOP[0] + A_W + 5.1,
                "all three payloads derive", ha="center",
                va="baseline", fontsize=FS_TINY, color=MUTED)
    fig_text_mm(fig, cap_x, ROW_TOP[0] + A_W + 7.6,
                "from its 24-bin masses", ha="center",
                va="baseline", fontsize=FS_TINY, color=MUTED)

    # ---- a (continued)  the three excerpt boxes ----------------------------
    # The three serialisations differ in line length by more than a factor of
    # two, so equal-width boxes would force the monospace face down to ~5.2 pt.
    # Each box is instead sized to its own measured block, which lets all three
    # share one font size at the FS_SMALL used elsewhere in the set.
    strip_w = B_RIGHT - B_LEFT
    axB = mm_axes(fig, B_LEFT, B_TOP - 2.6, strip_w, B_H)
    axB.set_xlim(0.0, strip_w)
    axB.set_ylim(B_H, 0.0)          # mm, measured downwards from the strip top
    axB.set_xticks([])
    axB.set_yticks([])
    for sp in axB.spines.values():
        sp.set_visible(False)
    axB.patch.set_visible(False)
    heads: list = []
    body: list = []
    notes: list = []
    for rep in REP_ORDER:
        c = REP[rep]
        prompt = field[rep]["serialized_prompt"]
        heads.append(
            axB.text(0.0, 2.6 - B_HEAD_DY, REP_SHORT[rep], ha="left", va="baseline",
                     color=c, fontweight="bold", fontsize=FS_BODY, clip_on=False)
        )
        body.append(
            axB.text(0.0, 2.6 + B_INSET, _excerpt_lines(prompt),
                     ha="left", va="top", fontsize=FS_SMALL, family="monospace",
                     linespacing=1.5, color="#333333", zorder=3, clip_on=False)
        )
        n_chars = len(prompt)
        notes.append(
            axB.text(0.0, 0.0, f"full serialized observation:\n{n_chars} characters",
                     ha="left", va="top", fontsize=FS_TINY, color=MUTED,
                     linespacing=1.35, clip_on=False)
        )

    # Width: give every box its own measured block plus a uniform inset, and
    # share out whatever strip width is left over equally. Only if even that
    # does not fit is the (single, shared) font size reduced.
    fig.canvas.draw()
    widths = [abs(_mm_bbox(fig, t, axB).width) for t in body]
    fs = FS_SMALL
    ink_w = strip_w - 2 * B_GAP - 6 * B_INSET
    if sum(widths) > ink_w:
        fs = max(FS_TINY, math.floor(FS_SMALL * (ink_w / sum(widths)) * 20.0) / 20.0)
        for t in body:
            t.set_fontsize(fs)
        fig.canvas.draw()
        widths = [abs(_mm_bbox(fig, t, axB).width) for t in body]
    box_ws = [w + 2 * B_INSET for w in widths]
    slack = max(0.0, strip_w - 2 * B_GAP - sum(box_ws))
    box_ws = [w + slack / 3.0 for w in box_ws]

    # Height: one shared box height, taken from the tallest block.
    text_h = max(abs(_mm_bbox(fig, t, axB).height) for t in body)
    # the measured extent carries one line's worth of leading under the last
    # baseline; drop it so the ink sits optically centred in the box
    box_h = text_h + 2 * B_INSET - 0.5 * fs * 25.4 / 72.0

    x0 = 0.0
    for i, rep in enumerate(REP_ORDER):
        c = REP[rep]
        heads[i].set_x(x0)
        body[i].set_x(x0 + B_INSET)
        notes[i].set_x(x0)
        notes[i].set_y(2.6 + box_h + B_NOTE_DY)
        axB.add_patch(
            mpatches.FancyBboxPatch(
                (x0 + 0.35, 2.6 + 0.35), box_ws[i] - 0.7, box_h - 0.7,
                boxstyle="round,pad=0.35,rounding_size=0.7",
                facecolor=c, alpha=0.09, edgecolor=c, lw=LW_THIN, zorder=1,
            )
        )
        x0 += box_ws[i] + B_GAP

    # ---- b  unimodal trinomial --------------------------------------------
    axB = mm_axes(fig, COL_X[0], ROW_TOP[1], COL_W, ROW_H)
    panel_label(axB, "b", "Unimodal response")
    triB = _mean_trinomial(curves, PROFILE, PANEL_B_OFFSET)
    stacked_trinomial(axB, triB, show_legend=False, annotate=False)
    _tight_trinomial_labels(axB, triB)
    axB.set_ylim(0, 1.19)
    axB.set_ylabel("probability")
    despine(axB)
    trim_spines(axB, x=False)
    # direct segment labels instead of a legend
    pm = triB["moments_m1_m3"]
    pc = triB["centers_24_standard"]
    axB.text(0, pm[0] + pm[1] + pm[2] / 2, ACTION_LABELS[2], ha="center", va="center",
             color="white", fontsize=FS_SMALL)
    axB.text(1, pc[0] / 2, ACTION_LABELS[0], ha="center", va="center",
             color="white", fontsize=FS_SMALL)
    axB.text(1, pc[0] + pc[1] / 2, ACTION_LABELS[1], ha="center", va="center",
             color="#333333", fontsize=FS_SMALL)

    # ---- c  exact antipodal trinomial (shares b's y axis) -------------------
    axC = mm_axes(fig, COL_X[1], ROW_TOP[1], COL_W, ROW_H)
    panel_label(axC, "c", "Exact antipodal response")
    triC = _from_atlas(atlas, "antipodal_equal_k6")
    stacked_trinomial(axC, triC, show_legend=False, annotate=False)
    _tight_trinomial_labels(axC, triC)
    axC.set_ylim(0, 1.19)
    axC.set_yticklabels([])          # duplicate of b
    despine(axC)
    trim_spines(axC, x=False)

    # ---- d  near-zero imbalance activation ---------------------------------
    axD = mm_axes(fig, COL_X[2], ROW_TOP[1], COL_W, ROW_H)
    panel_label(axD, "d", "Near-zero activation")
    for rep in REP_ORDER:
        sub = eps[eps["representation"] == rep].sort_values("epsilon")
        near = sub[sub["epsilon"].isin([-0.02, 0.0, 0.02])]
        axD.plot(near["epsilon"], near["activity"], "-o",
                 color=REP[rep], ms=MS_SERIES, lw=LW_LINE)
    axD.axvline(0, color="#BBBBBB", lw=LW_HAIR, zorder=0)
    axD.set_xlabel(r"$\varepsilon$")
    axD.set_ylabel(r"$A$")
    axD.set_xlim(-0.0245, 0.048)
    axD.set_ylim(0, 1.10)
    axD.set_yticks([0, 0.5, 1.0])
    axD.set_xticks([-0.02, 0, 0.02])
    for rep, y in (("moments_m1_m3", 0.83),
                   ("intervals_24_decimal6", 0.93),
                   ("centers_24_standard", 1.03)):
        direct_label(axD, 0.0225, y, REP_SHORT[rep], REP[rep], fontsize=FS_SMALL)
    despine(axD)
    trim_spines(axD)

    # ---- e  signed sweep A(eps) / f  signed response a1(eps) ---------------
    # formerly f / f': the pair keeps its side-by-side alignment under the
    # new letters, which retire the prime mark
    axE = mm_axes(fig, COL_X[0], ROW_TOP[2], COL_W, ROW_H)
    axF = mm_axes(fig, COL_X[1], ROW_TOP[2], COL_W, ROW_H)
    panel_label(axE, "e", r"Signed sweep: activity $A(\varepsilon)$")
    panel_label(axF, "f", r"Signed response $a_1(\varepsilon)$")
    dip, tail = {}, {}
    for rep in REP_ORDER:
        sub = eps[eps["representation"] == rep].sort_values("epsilon")
        axE.plot(sub["epsilon"], sub["activity"], "-o", color=REP[rep],
                  ms=MS_SERIES, lw=LW_LINE)
        axF.plot(sub["epsilon"], sub["a1"], "-o", color=REP[rep],
                  ms=MS_SERIES, lw=LW_LINE)
        dip[rep] = float(sub[sub["epsilon"] == 0.0]["activity"].iloc[0])
        tail[rep] = float(sub["a1"].iloc[-1])
    for ax in (axE, axF):
        ax.axvline(0, color="#BBBBBB", lw=LW_HAIR, zorder=0)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_xlim(-0.125, 0.295)
        ax.set_xticks([-0.1, 0.0, 0.1, 0.2])
        despine(ax)
    axE.set_ylabel(r"$A$")
    axF.set_ylabel(r"$a_1$")
    axF.axhline(0, color="#BBBBBB", lw=LW_HAIR, zorder=0)
    axE.set_ylim(0, 1.05)
    axE.set_yticks([0, 0.5, 1.0])
    # e: label the collapse, which is where the three operators separate
    for rep in REP_ORDER:
        direct_label(axE, 0.028, dip[rep], REP_SHORT[rep], REP[rep], fontsize=FS_SMALL)
    # f: label the line ends, which are well separated there
    for rep in REP_ORDER:
        direct_label(axF, 0.212, tail[rep], REP_SHORT[rep], REP[rep], fontsize=FS_SMALL)
    trim_spines(axE)
    trim_spines(axF)

    # ---- g  operator dTV map ----------------------------------------------
    # panel g sits 8 mm right of its column so the spelled-out family names
    # clear the direct series labels at the right edge of panel f
    G_X = COL_X[2] + 8.0
    G_W = 28.0
    axG = mm_axes(fig, G_X, ROW_TOP[2], G_W, ROW_H)
    panel_label(axG, "g", r"Operator $d_{\mathrm{TV}}$ map")
    families = [
        "unimodal_k9",
        "antipodal_equal_k6",
        "asymmetric_w075_sep2_k6",
        "sparse_N8_unimodal_k6",
        "sparse_N16_antipodal_k6",
        "bimodal_equal_sep2_k6",
    ]
    short = ["unimodal", "antipodal", "asymmetric",
             "sparse unimodal", "sparse antipodal", "two-peaked"]
    mat = np.zeros((len(families), 3))
    pair_names = ["M–C", "M–I", "C–I"]
    for i, fam in enumerate(families):
        probs = _from_atlas(atlas, fam)
        pm, pc, pi = [np.array(probs[r]) for r in REP_ORDER]
        mat[i, 0] = 0.5 * np.abs(pm - pc).sum()
        mat[i, 1] = 0.5 * np.abs(pm - pi).sum()
        mat[i, 2] = 0.5 * np.abs(pc - pi).sum()
    vmax = max(0.4, float(mat.max()))
    im = axG.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto",
                    interpolation="nearest")
    axG.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    axG.set_yticks(np.arange(-0.5, len(families), 1), minor=True)
    axG.grid(which="minor", color="white", lw=LW_HAIR)
    axG.tick_params(which="minor", length=0)
    axG.tick_params(which="major", length=0)
    for sp in axG.spines.values():
        sp.set_visible(False)
    axG.set_yticks(range(len(families)))
    axG.set_yticklabels(short, fontsize=FS_TICK)
    axG.set_xticks([0, 1, 2])
    axG.set_xticklabels(pair_names, fontsize=FS_TICK)
    annotate_cells(axG, mat, fmt="{:.2f}", cmap=plt.get_cmap("viridis"),
                   norm=im.norm, fontsize=FS_TINY)
    cbar_mm(fig, im, G_X + G_W + 2.0, ROW_TOP[2] + 3.4, 2.2, ROW_H - 3.4,
            ticks=[0, round(vmax / 2, 2), round(vmax, 2)])
    fig_text_mm(fig, G_X + G_W + 3.1, ROW_TOP[2] + 1.6,
                r"$d_{\mathrm{TV}}$", ha="center", va="baseline", fontsize=FS_SMALL)

    # ---- symbol key ---------------------------------------------------------
    # Reviewer pass: every symbol the panels use, glossed in the reader's
    # language rather than the paper's. This is a figure-level key, so it is
    # drawn in figure millimetres rather than inside any one panel's axes.
    #
    # Placement: the board has no free block wide enough for a definition
    # column, so the key occupies the largest measured whitespace on the sheet
    # - the empty lower-right of panel f (a_1 < 0 at eps > 0 is unpopulated)
    # continuing into the gutter before panel g's row labels. The measured
    # free region runs from x 84.8 mm (the steep a_1 branch at eps = 0) to
    # x 118.9 mm, where panel g's longest row labels ("sparse unimodal",
    # "sparse antipodal") begin. Every gloss below is measured to end by
    # 117 mm, so no line depends on which panel-g row it happens to sit
    # beside, and the block never approaches the 178 mm ink limit.
    #
    # The glosses are shortened against that 27.4 mm column, not against the
    # page: "probability" is dropped from the A and a_0 lines because panel b
    # and panel c already label that axis, and "component" is shortened to
    # "part" on the a_1 line.
    KEY_X = 86.2          # symbol column
    KEY_DX = 4.0          # symbol column -> gloss column
    KEY_Y0 = 105.3        # baseline of the first line (clears the a_1 = 0 rule)
    KEY_DY = 2.6          # line pitch
    SYMBOL_KEY = (
        (r"$A$", "advance or retard, not stay"),
        (r"$a_0$", "advance minus retard"),
        (r"$a_1$", "one-cycle directional part"),
        (r"$\varepsilon$", "directional peer imbalance"),
        (r"$d_{\mathrm{TV}}$", "0 identical, 1 non-overlapping"),
    )
    for i, (sym, gloss) in enumerate(SYMBOL_KEY):
        y = KEY_Y0 + i * KEY_DY
        fig_text_mm(fig, KEY_X, y, sym, ha="left", va="baseline",
                    fontsize=FS_TINY, color=INK)
        fig_text_mm(fig, KEY_X + KEY_DX, y, gloss, ha="left", va="baseline",
                    fontsize=FS_TINY, color=MUTED)

    return save_fig(fig, "fig2_microscopic_operators")
