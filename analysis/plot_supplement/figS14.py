"""Supplementary Figure S14 — Frozen replay-field selection and audit.

Nine panels (a–i) showing that the 48 physical fields replayed under all three
target encodings were chosen by the mechanical rule frozen in
``docs/MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md``, that the six strata and three
source encodings were covered, that no fallback and no hand edit was used, and
that the selected set is hash-frozen before the paid replay was authorised.

Every printed count, ratio, hash and clock time is read from the frozen
artifacts at build time:

* ``analysis/matched_rep_collective/replay_fields_v0_1.json``
* ``analysis/matched_rep_collective/replay_selection_audit_v0_1.json``
* ``analysis/matched_rep_collective/replay_hash_freeze_v0_1.json``
* ``analysis/matched_rep_collective/replay_target_prompts_v0_1.json``
* ``analysis/matched_rep_collective/replay_primary/decision.json``
* ``docs/MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md`` (stratum predicates, quotas)

Nothing in this module hard-codes a statistic; the stratum predicates are
parsed out of the lock document so the figure cannot drift away from the rule
it is auditing.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

from analysis.plot_supplement.style import (
    CAPSIZE,
    FS_BODY,
    FS_LETTER,
    FS_SMALL,
    FS_TICK,
    FS_TINY,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    MS_POINT,
    MUTED,
    PASS_GREEN,
    REP,
    REP_ABBR,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    annotate_cells,
    apply_style,
    despine,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label_at,
    save_fig,
    swarm_x,
    text_on,
    trim_spines,
)

MRC = ROOT / "analysis" / "matched_rep_collective"
FIELDS_JSON = MRC / "replay_fields_v0_1.json"
AUDIT_JSON = MRC / "replay_selection_audit_v0_1.json"
FREEZE_JSON = MRC / "replay_hash_freeze_v0_1.json"
PROMPTS_JSON = MRC / "replay_target_prompts_v0_1.json"
AUTH_JSON = MRC / "replay_auth_go.json"
GATE_JSON = MRC / "paid_gate_status.json"
DECISION_JSON = MRC / "replay_primary" / "decision.json"
LOCK_DOC = ROOT / "docs" / "MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md"

STRATA = [
    "early_transient",
    "pre_onset",
    "post_onset",
    "ordered_state",
    "negative_K",
    "high_r2",
]
# Short tick labels; the full name and the frozen predicate are printed in a.
STRATUM_TICK = {
    "early_transient": "early",
    "pre_onset": "pre-onset",
    "post_onset": "post-onset",
    "ordered_state": "ordered",
    "negative_K": "negative $K$",
    "high_r2": "high $r_2$",
}
STRATUM_NAME = {
    "early_transient": "early transient",
    "pre_onset": "pre-onset",
    "post_onset": "post-onset",
    "ordered_state": "ordered state",
    "negative_K": "negative $K$",
    "high_r2": "high $r_2$",
}

HASH_SHOW = 12          # visible hex characters of any sha256
BARCODE_BYTES = 8       # sha256 bytes drawn per field in the barcode
ARROW_SCALE = 4.0       # flow-arrow head size (not a type/mark metric)
S_POINT = MS_POINT ** 2  # scatter area matching the shared MS_POINT diameter
CELL_EDGE = "white"

# --- canvas plan (mm from the top-left) ------------------------------------
H_MM = 187

C1, C2, C3 = 14.0, 75.0, 136.0          # data-axes left edges per column
LET1, LET2, LET3 = 5.4, 66.4, 127.4     # shared letter columns
W1, W2, W3 = 47.0, 47.0, 41.0

R1_TOP, R2_TOP, R3_TOP = 9.5, 76.0, 130.0
LETTER_DY = 1.6
LET_Y1 = R1_TOP - LETTER_DY
LET_Y2 = R2_TOP - LETTER_DY
LET_Y3 = R3_TOP - LETTER_DY


# ---------------------------------------------------------------------------
# artifact reading
# ---------------------------------------------------------------------------
def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mtime_utc(path: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)


def _stamp_utc(stamp: str) -> dt.datetime:
    """Parse a run-directory stamp such as ``20260724T065125Z``."""
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=dt.timezone.utc)


def _pretty_predicate(raw: str) -> str:
    """Lock-doc LaTeX -> matplotlib mathtext, first clause only."""
    s = raw.replace("\\(", "$").replace("\\)", "$")
    s = s.replace("\\lvert", "|").replace("\\rvert", "|")
    s = s.replace("\\ge", "\\geq").replace("\\le", "\\leq")
    head, sep, _tail = s.partition(";")
    return (head.strip() + " …") if sep else head.strip()


def _lock_strata() -> dict[str, tuple[int, str]]:
    """{stratum: (quota, predicate)} parsed from the frozen lock document."""
    out: dict[str, tuple[int, str]] = {}
    for line in LOCK_DOC.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in STRATA:
            out[cells[0]] = (int(cells[1]), _pretty_predicate(cells[2]))
    return out


def _lock_source_quota() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in LOCK_DOC.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] in REP_ORDER:
            out[cells[0]] = int(cells[1])
    return out


def _facts() -> dict:
    """Recompute every number this figure prints from the frozen artifacts."""
    doc = _load_json(FIELDS_JSON)
    fields = doc["fields"]
    audit = _load_json(AUDIT_JSON)
    freeze = _load_json(FREEZE_JSON)
    prompts = _load_json(PROMPTS_JSON)
    decision = _load_json(DECISION_JSON)
    auth = _load_json(AUTH_JSON)
    gate = _load_json(GATE_JSON)
    lock_strata = _lock_strata()
    lock_source = _lock_source_quota()

    # counts -----------------------------------------------------------------
    sx = np.zeros((len(REP_ORDER), len(STRATA)), dtype=int)
    for f in fields:
        sx[REP_ORDER.index(f["source_representation"]),
           STRATA.index(f["stratum"])] += 1
    source_counts = sx.sum(axis=1)
    stratum_counts = sx.sum(axis=0)

    ks = sorted({float(f["K"]) for f in fields})
    k_by_source = np.zeros((len(ks), len(REP_ORDER)), dtype=int)
    for f in fields:
        k_by_source[ks.index(float(f["K"])),
                    REP_ORDER.index(f["source_representation"])] += 1

    hashes = [f["physical_hash"] for f in fields]

    # sha256 re-verification of the frozen field list ------------------------
    recomputed = hashlib.sha256(FIELDS_JSON.read_bytes()).hexdigest()
    frozen = freeze["replay_fields_v0_1_sha256"]
    in_decision = decision["hashes"]["fields_sha256"]
    sha_ok = recomputed == frozen == in_decision

    # stratum predicates that the frozen record alone can verify -------------
    def _sel(st):
        return [f for f in fields if f["stratum"] == st]

    et = _sel("early_transient")
    orderd = _sel("ordered_state")
    negk = _sel("negative_K")
    hi2 = _sel("high_r2")
    predicates = [
        ("early-transient predicate",
         sum(5 <= f["time"] <= 15 and f["r1"] < 0.35 for f in et), len(et)),
        ("ordered-state predicate",
         sum(f["r1"] >= 0.8 for f in orderd), len(orderd)),
        ("negative-$K$ predicate",
         sum(abs(f["K"] + 0.15) < 1e-12 for f in negk), len(negk)),
        ("high-$r_2$ predicate",
         sum(f["r2"] >= 0.45 and f["r2"] > f["r1"] for f in hi2), len(hi2)),
    ]

    # target-encoding coverage ----------------------------------------------
    targets = sorted({p["target_representation"] for p in prompts["prompts"]})
    covered = {(p["field_id"], p["target_representation"]) for p in prompts["prompts"]}
    expected = {(f["field_id"], t) for f in fields for t in targets}
    missing_targets = len(expected - covered)

    acq = decision["acquisition"]

    # freeze timeline (UTC) --------------------------------------------------
    sessions = doc["sessions"]
    acq_start = min(_stamp_utc(Path(p).name) for p in sessions.values())
    elapsed = []
    n_runs, n_steps = set(), set()
    for p in sessions.values():
        summ = _load_json(ROOT / p / "session_summary.json")
        elapsed.append(float(summ["elapsed_seconds"]))
        n_runs.add(int(summ["n_runs"]))
        n_steps.add(int(summ["runs"][0]["n_steps"]))
    acq_hours = max(elapsed) / 3600.0

    replay_start = _stamp_utc(Path(acq["run_dir"]).name)
    authorized = dt.datetime.fromisoformat(auth["authorized_at"]).astimezone(
        dt.timezone.utc)
    timeline = [
        (dt.datetime.fromisoformat(gate["updated_utc"]).astimezone(dt.timezone.utc),
         "selection rule locked", "pre-outcome gate record"),
        (acq_start, "collective acquisition",
         f"{len(sessions)} encodings × {sorted(n_runs)[0]} runs, "
         f"{acq_hours:.1f} h"),
        (_mtime_utc(FIELDS_JSON), f"{len(fields)} fields locked",
         f"+ {prompts['n_target_prompts']} target prompts"),
        (_mtime_utc(FREEZE_JSON), "field hashes locked",
         "physical + prompt aggregates"),
        (authorized, "paid replay authorised", "human sign-off"),
        (replay_start, "3×3 replay acquisition",
         f"{acq['n_calls']:,} calls, {acq['elapsed_seconds'] / 60.0:.1f} min"),
        (_mtime_utc(DECISION_JSON), "primary inference locked",
         "decision.json"),
    ]

    return {
        "doc": doc,
        "fields": fields,
        "audit": audit,
        "freeze": freeze,
        "decision": decision,
        "acq": acq,
        "lock_strata": lock_strata,
        "lock_source": lock_source,
        "sx": sx,
        "source_counts": source_counts,
        "stratum_counts": stratum_counts,
        "ks": ks,
        "k_by_source": k_by_source,
        "hashes": hashes,
        "n_unique": len(set(hashes)),
        "recomputed_sha": recomputed,
        "sha_ok": sha_ok,
        "predicates": predicates,
        "n_prompts": prompts["n_target_prompts"],
        "n_targets": len(targets),
        "missing_targets": missing_targets,
        "n_runs": sorted(n_runs)[0],
        "n_steps": sorted(n_steps)[0],
        "n_bins": len(fields[0]["physical_histogram_24"]),
        "peer": int(fields[0]["peer_count"]),
        "timeline": timeline,
        "freeze_line_index": 4,   # nothing paid happens above this row
    }


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
A_LEFT, A_W, A_H = 5.5, 62.0, 60.0
BC_TOP = 15.0            # b and c sit below the row-1 letter band
B_H = 25.5
C1_H, C2_TOP, C2_H = 18.0, 39.0, 25.0


def _panel_a(fig, F) -> None:
    """Selection algorithm: trajectories -> predicates -> dedup -> quotas."""
    ax = mm_panel(fig, A_LEFT, R1_TOP, A_W, A_H)
    ax.set_xlim(0, A_W)
    ax.set_ylim(A_H, 0)

    quota_s = F["lock_strata"][STRATA[0]][0]
    quota_r = F["lock_source"][REP_ORDER[0]]
    steps = [
        f"{len(F['doc']['sessions'])} source encodings × {F['n_runs']} runs "
        f"× {F['n_steps']} steps",
        f"tag every frame by the {len(STRATA)} locked stratum predicates",
        f"deduplicate by {F['n_bins']}-bin physical_hash (peer = {F['peer']})",
        "greedy round-robin fill of source × stratum quotas",
        f"{len(F['fields'])} locked fields: {quota_s} per stratum, "
        f"{quota_r} per source",
    ]
    box_h, pitch, x0, w = 5.6, 7.4, 0.6, A_W - 1.2
    for k, txt in enumerate(steps):
        y = 1.2 + k * pitch
        last = k == len(steps) - 1
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y), w, box_h,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor="#EDF3EC" if last else "#F5F5F5",
            edgecolor=PASS_GREEN if last else "#B4B4B4",
            lw=LW_THIN if last else LW_HAIR))
        ax.text(x0 + w / 2, y + box_h / 2, txt, ha="center", va="center",
                fontsize=FS_TINY, color=PASS_GREEN if last else INK)
        if not last:
            ax.annotate("", xy=(A_W / 2, y + pitch - 0.25),
                        xytext=(A_W / 2, y + box_h + 0.25),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                        lw=LW_HAIR, shrinkA=0, shrinkB=0,
                                        mutation_scale=ARROW_SCALE))

    y_key = 1.2 + len(steps) * pitch + 1.6
    ax.text(x0, y_key, f"locked strata ({quota_s} fields each), "
                       "predicate as locked:",
            ha="left", va="center", fontsize=FS_TINY, color=INK,
            fontweight="bold")
    for k, st in enumerate(STRATA):
        yy = y_key + 3.4 + k * 3.0
        ax.text(x0 + 0.4, yy, STRATUM_NAME[st], ha="left", va="center",
                fontsize=FS_TINY, color=INK)
        ax.text(x0 + 18.4, yy, F["lock_strata"][st][1], ha="left", va="center",
                fontsize=FS_TINY, color=MUTED)
    panel_label_at(fig, LET1, LET_Y1, "a", "Locked selection algorithm")


def _panel_b(fig, F) -> None:
    """Source × stratum count matrix with both margins."""
    ax = mm_axes(fig, C2, BC_TOP, W2, B_H)
    m = F["sx"]
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=float(m.max()) * 1.9,
                   interpolation="nearest", aspect="auto")
    annotate_cells(ax, m, fmt="{:.0f}", cmap=im.cmap, norm=im.norm,
                   fontsize=FS_SMALL)
    ax.set_xticks(np.arange(-0.5, len(STRATA), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(REP_ORDER), 1), minor=True)
    ax.grid(which="minor", color=CELL_EDGE, lw=LW_THIN)
    ax.tick_params(which="both", length=0, pad=1.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(range(len(STRATA)))
    ax.set_yticks(range(len(REP_ORDER)))
    ax.set_xticklabels([STRATUM_TICK[s] for s in STRATA], fontsize=FS_TICK,
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticklabels([REP_ABBR[r] for r in REP_ORDER], fontsize=FS_TICK)
    for tick, r in zip(ax.get_yticklabels(), REP_ORDER):
        tick.set_color(REP[r])

    for j, tot in enumerate(F["stratum_counts"]):
        ax.text(j, -0.78, str(int(tot)), ha="center", va="center",
                fontsize=FS_SMALL, color=INK, fontweight="bold", clip_on=False)
    for i, tot in enumerate(F["source_counts"]):
        ax.text(len(STRATA) - 0.24, i, str(int(tot)), ha="left", va="center",
                fontsize=FS_SMALL, color=INK, fontweight="bold", clip_on=False)
    ax.text(len(STRATA) - 0.24, -0.78, "Σ", ha="left", va="center",
            fontsize=FS_SMALL, color=MUTED, clip_on=False)

    occupied = int((m >= 1).sum())
    quota_s = F["lock_strata"][STRATA[0]][0]
    quota_r = F["lock_source"][REP_ORDER[0]]
    notes = [
        f"{occupied}/{m.size} source × stratum cells occupied,",
        f"{int(m.min())}–{int(m.max())} fields per cell",
        f"column totals = stratum quota {quota_s}",
        f"row totals = source quota {quota_r}",
        "the lock requires ≥1 field per cell",
        "wherever the candidate pool allows",
    ]
    for k, txt in enumerate(notes):
        fig_text_mm(fig, C2 - 1.0, BC_TOP + B_H + 11.0 + k * 2.9, txt,
                    fontsize=FS_TINY, color=MUTED, ha="left", va="baseline")
    panel_label_at(fig, LET2, LET_Y1, "b", "Source × stratum balance")


def _panel_c(fig, F) -> None:
    """Coupling coverage: counts by K and the selection-time r1 at each K."""
    ks = F["ks"]
    xs = np.arange(len(ks))
    klab = [("$" + f"{k:+.2f}".replace("-", "−") + "$") for k in ks]

    ax1 = mm_axes(fig, C3, BC_TOP, W3, C1_H)
    bottom = np.zeros(len(ks))
    for i, r in enumerate(REP_ORDER):
        vals = F["k_by_source"][:, i].astype(float)
        ax1.bar(xs, vals, bottom=bottom, width=0.62, color=REP[r],
                edgecolor=CELL_EDGE, linewidth=LW_HAIR)
        bottom = bottom + vals
    for x, tot in zip(xs, bottom):
        ax1.text(x, tot + 0.5, str(int(tot)), ha="center", va="bottom",
                 fontsize=FS_SMALL, color=INK)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([])
    ax1.set_xlim(-0.6, len(ks) - 0.4)
    ax1.set_ylim(0, float(bottom.max()) * 1.28)
    ax1.set_yticks([0, 5, 10, 15])
    ax1.set_ylabel("fields")
    despine(ax1)
    trim_spines(ax1, x=False)

    ax2 = mm_axes(fig, C3, C2_TOP, W3, C2_H)
    for i, k in enumerate(ks):
        grp = [f for f in F["fields"] if float(f["K"]) == k]
        vals = np.array([f["r1"] for f in grp])
        cols = [REP[f["source_representation"]] for f in grp]
        ax2.scatter(swarm_x(vals, i, 0.24), vals, s=S_POINT, c=cols, alpha=0.85,
                    linewidths=0, zorder=3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(klab, fontsize=FS_TICK)
    ax2.set_xlim(-0.6, len(ks) - 0.4)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_xlabel("coupling $K$")
    ax2.set_ylabel("$r_1$ at selection")
    despine(ax2)
    trim_spines(ax2, x=False)
    panel_label_at(fig, LET3, LET_Y1, "c", "Coupling coverage")


def _thresholds() -> dict[str, float]:
    """Numeric thresholds parsed out of the frozen lock-document predicates."""
    raw = {}
    for line in LOCK_DOC.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in STRATA:
            raw[cells[0]] = cells[2]
    def _grab(key, pattern):
        m = re.search(pattern, raw[key])
        return float(m.group(1))
    return {
        "onset": _grab("pre_onset", r"r_1\\ge([0-9.]+)"),
        "ordered": _grab("ordered_state", r"r_1\\ge([0-9.]+)"),
        "high_r2": _grab("high_r2", r"r_2\\ge([0-9.]+)"),
    }


def _stratum_axis(ax, labels: bool) -> None:
    ax.set_xticks(range(len(STRATA)))
    ax.set_xlim(-0.6, len(STRATA) - 0.4)
    if labels:
        ax.set_xticklabels([STRATUM_TICK[s] for s in STRATA], fontsize=FS_TICK,
                           rotation=30, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels([])


def _stratum_scatter(ax, F, value, width: float = 0.24) -> None:
    for i, st in enumerate(STRATA):
        grp = [f for f in F["fields"] if f["stratum"] == st]
        vals = np.array([value(f) for f in grp], dtype=float)
        cols = [REP[f["source_representation"]] for f in grp]
        ax.scatter(swarm_x(vals, i, width), vals, s=S_POINT, c=cols, alpha=0.85,
                   linewidths=0, zorder=3)


D_H, E_H, E_GAP, F_H = 34.0, 15.0, 19.0, 16.0


def _panel_d(fig, F) -> None:
    thr = _thresholds()
    ax = mm_axes(fig, C1, R2_TOP, W1, D_H)
    for y in (thr["onset"], thr["ordered"]):
        ax.axhline(y, color=RULE, lw=LW_HAIR, ls=(0, (2.5, 1.8)), zorder=1)
    ax.text(len(STRATA) - 0.45, thr["onset"], f"$r_1$={thr['onset']:.1f}",
            ha="right", va="bottom", fontsize=FS_TINY, color=MUTED)
    ax.text(len(STRATA) - 0.45, thr["ordered"], f"$r_1$={thr['ordered']:.1f}",
            ha="right", va="bottom", fontsize=FS_TINY, color=MUTED)
    _stratum_scatter(ax, F, lambda f: f["r1"])
    for i, r in enumerate(REP_ORDER):
        ax.text(-0.45 + i * 2.05, 1.145, REP_SHORT[r], ha="left", va="center",
                fontsize=FS_TINY, color=REP[r], fontweight="bold")
    _stratum_axis(ax, labels=True)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("$r_1$ at selection")
    despine(ax)
    trim_spines(ax, x=False)
    panel_label_at(fig, LET1, LET_Y2, "d", "Selection-time $r_1$")


def _panel_e(fig, F) -> None:
    thr = _thresholds()
    ax1 = mm_axes(fig, C2, R2_TOP, W2, E_H)
    ax1.axhline(thr["high_r2"], color=RULE, lw=LW_HAIR, ls=(0, (2.5, 1.8)),
                zorder=1)
    ax1.text(-0.5, thr["high_r2"] + 0.04, f"$r_2$={thr['high_r2']:.2f}",
             ha="left", va="bottom", fontsize=FS_TINY, color=MUTED)
    _stratum_scatter(ax1, F, lambda f: f["r2"])
    _stratum_axis(ax1, labels=False)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_ylabel("$r_2$")
    despine(ax1)
    trim_spines(ax1, x=False)

    ax2 = mm_axes(fig, C2, R2_TOP + E_GAP, W2, E_H)
    ax2.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=1)
    _stratum_scatter(ax2, F, lambda f: f["r2"] - f["r1"])
    _stratum_axis(ax2, labels=True)
    q = np.array([f["r2"] - f["r1"] for f in F["fields"]])
    lim = float(np.max(np.abs(q))) * 1.12
    ax2.set_ylim(-lim, lim)
    ax2.set_yticks([-0.5, 0, 0.5])
    ax2.set_yticklabels(["$-0.5$", "$0$", "$0.5$"])
    ax2.set_ylabel("$Q_2=r_2-r_1$")
    despine(ax2)
    trim_spines(ax2, x=False)
    panel_label_at(fig, LET2, LET_Y2, "e", "Selection-time $r_2$ and $Q_2$")


def _check(ax, x: float, y: float, s: float = 1.0, color: str = PASS_GREEN):
    """A tick glyph drawn as two strokes (no font-coverage risk)."""
    ax.plot([x, x + 0.42 * s, x + 1.25 * s], [y + 0.42 * s, y + s, y - 0.55 * s],
            color=color, lw=LW_LINE, solid_capstyle="round",
            solid_joinstyle="round", clip_on=False, zorder=5)


def _panel_f(fig, F) -> None:
    ax = mm_axes(fig, C3, R2_TOP, W3, F_H)
    arr = np.array([[int(h[2 * i: 2 * i + 2], 16) for h in F["hashes"]]
                    for i in range(BARCODE_BYTES)], dtype=float)
    ax.imshow(arr, cmap="Greys", vmin=0, vmax=255, aspect="auto",
              interpolation="nearest")
    n = len(F["hashes"])
    ticks = [0, 11, 23, 35, n - 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t + 1) for t in ticks], fontsize=FS_TICK)
    ax.set_yticks([])
    ax.tick_params(length=1.4, pad=1.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("field, selection order")
    ax.set_ylabel(f"first {BARCODE_BYTES}\nhash bytes", fontsize=FS_TINY,
                  labelpad=1.2)

    box = mm_panel(fig, C3, R2_TOP + F_H + 8.5, W3, 24.0)
    box.set_xlim(0, W3)
    box.set_ylim(24.0, 0)
    dup = len(F["hashes"]) - F["n_unique"]
    box.text(0.0, 1.4, f"{F['n_unique']}/{n} unique physical hashes, "
                       f"{dup} duplicate", ha="left", va="center",
             fontsize=FS_TINY, color=PASS_GREEN, fontweight="bold")
    box.text(0.0, 5.2, "SHA-256 of the locked field list", ha="left",
             va="center", fontsize=FS_TINY, color=INK)
    rows = [
        ("recomputed", F["recomputed_sha"]),
        ("freeze file", F["freeze"]["replay_fields_v0_1_sha256"]),
        ("decision.json", F["decision"]["hashes"]["fields_sha256"]),
    ]
    for k, (lab, val) in enumerate(rows):
        y = 8.6 + k * 3.2
        box.text(0.8, y, lab, ha="left", va="center", fontsize=FS_TINY,
                 color=MUTED)
        box.text(16.4, y, val[:HASH_SHOW] + "…", ha="left", va="center",
                 fontsize=FS_TINY, color=INK)
    if F["sha_ok"]:
        box.plot([33.2, 33.2], [8.2, 15.0], color=PASS_GREEN, lw=LW_THIN,
                 clip_on=False)
        _check(box, 34.6, 11.6, s=1.15)
    box.text(0.0, 20.0, "physical-field aggregate", ha="left", va="center",
             fontsize=FS_TINY, color=MUTED)
    box.text(0.0, 23.0,
             F["freeze"]["physical_field_aggregate_sha256"][:HASH_SHOW] + "…",
             ha="left", va="center", fontsize=FS_TINY, color=INK)
    panel_label_at(fig, LET3, LET_Y2, "f", "Physical-hash freeze")


G_H, H_W, H_H, I_W, I_H = 32.0, 52.0, 52.0, 42.0, 50.0


def _panel_g(fig, F) -> None:
    """Every field's frozen rank inside its stratum × source cell."""
    quota = F["lock_strata"][STRATA[0]][0]
    ax = mm_axes(fig, C1, R3_TOP, W1, G_H)
    ax.set_xlim(-0.5, quota - 0.5)
    ax.set_ylim(len(STRATA) - 0.5, -0.5)
    for i, st in enumerate(STRATA):
        grp = sorted((f for f in F["fields"] if f["stratum"] == st),
                     key=lambda f: (f["selection_rank"], f["field_id"]))
        for j, f in enumerate(grp):
            col = REP[f["source_representation"]]
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1.0, 1.0, facecolor=col,
                edgecolor=CELL_EDGE, lw=LW_THIN))
            ax.text(j, i, str(f["selection_rank"]), ha="center", va="center",
                    fontsize=FS_SMALL, color=text_on(col))
    n_cover = sum(1 for f in F["fields"] if f["tie_break"]
                  == "source_x_stratum_coverage")
    split = n_cover / len(STRATA) - 0.5
    ax.plot([split, split], [-0.5, len(STRATA) - 0.5], color=INK, lw=LW_LINE,
            zorder=6)
    ax.set_xticks(range(quota))
    ax.set_xticklabels([str(j + 1) for j in range(quota)], fontsize=FS_TICK)
    ax.set_yticks(range(len(STRATA)))
    ax.set_yticklabels([STRATUM_TICK[s] for s in STRATA], fontsize=FS_SMALL)
    ax.tick_params(length=0, pad=1.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("fill position within stratum")

    lines = [
        "digit = locked selection_rank, colour = source encoding",
        f"left of the rule: rank 0, the {n_cover} source × stratum coverage picks",
        "right: quota fill, ranks 3–7",
        "score: $|r_1-$median$|$ in stratum; high $r_2$: max $Q_2$;",
        "negative $K$: max $|\\tau|$; ties broken by physical hash",
    ]
    for k, txt in enumerate(lines):
        fig_text_mm(fig, C1 - 3.0, R3_TOP + G_H + 7.0 + k * 3.0, txt,
                    fontsize=FS_TINY, color=MUTED, ha="left", va="baseline")
    panel_label_at(fig, LET1, LET_Y3, "g", "Selection rank and rule compliance")


def _panel_h(fig, F) -> None:
    """Compliance and audit-flag table; every value recomputed."""
    ax = mm_panel(fig, C2, R3_TOP, H_W, H_H)
    ax.set_xlim(0, H_W)
    ax.set_ylim(H_H, 0)

    n = len(F["fields"])
    quota_s = F["lock_strata"][STRATA[0]][0]
    quota_r = F["lock_source"][REP_ORDER[0]]
    sx = F["sx"]
    aud = F["audit"]
    acq = F["acq"]
    src = "/".join(str(int(v)) for v in F["source_counts"])
    blocks = [
        ("Rule compliance (recomputed)", [
            ("fields selected", f"{n} / {F['doc']['n_fields']}",
             n == int(F["doc"]["n_fields"])),
            ("stratum quota", f"{len(STRATA)} × {quota_s}",
             all(int(v) == quota_s for v in F["stratum_counts"])),
            ("source quota", src,
             all(int(v) == quota_r for v in F["source_counts"])),
            ("source × stratum cells", f"{int((sx >= 1).sum())} / {sx.size}",
             bool((sx >= 1).all())),
        ] + [(lab, f"{ok} / {tot}", ok == tot)
             for lab, ok, tot in F["predicates"]]),
        ("Freeze audit", [
            ("unique physical hashes", f"{F['n_unique']} / {n}",
             F["n_unique"] == n),
            ("duplicates in locked set", f"{n - F['n_unique']}",
             F["n_unique"] == n),
            ("stratum fallbacks", f"{int(aud['n_fallbacks'])}",
             int(aud["n_fallbacks"]) == 0),
            ("hand edits / substitutions", f"{len(F['doc']['hand_edits'])}",
             len(F["doc"]["hand_edits"]) == 0),
            ("surrogate use in selection",
             "yes" if F["doc"]["surrogate_use"] else "no",
             not F["doc"]["surrogate_use"]),
            ("outcome-A reselection",
             "yes" if F["doc"]["outcome_A_dependent_reselection"] else "no",
             not F["doc"]["outcome_A_dependent_reselection"]),
            ("missing target encodings",
             f"{F['missing_targets']} of {F['n_prompts']}",
             F["missing_targets"] == 0),
            ("replay calls valid",
             f"{acq['n_valid']:,} / {acq['n_calls']:,}",
             acq["n_valid"] == acq["n_calls"]),
            ("field-list SHA-256 match", "3 / 3 records", F["sha_ok"]),
        ]),
    ]

    x_lab, x_val, x_ok = 0.6, 41.0, 45.6
    y = 2.2
    for head, rows in blocks:
        ax.text(x_lab, y, head, ha="left", va="center", fontsize=FS_TINY,
                color=INK, fontweight="bold")
        ax.plot([0.0, H_W - 3.0], [y + 1.5, y + 1.5], color=RULE, lw=LW_HAIR,
                clip_on=False)
        y += 3.4
        for lab, val, ok in rows:
            ax.text(x_lab, y, lab, ha="left", va="center", fontsize=FS_TINY,
                    color=INK)
            ax.text(x_val, y, val, ha="right", va="center", fontsize=FS_TINY,
                    color=INK)
            if ok:
                _check(ax, x_ok, y, s=0.95)
            else:
                ax.text(x_ok, y, "!", ha="left", va="center",
                        fontsize=FS_TINY, color="#B0201F", fontweight="bold")
            y += 2.55
        y += 1.2
    panel_label_at(fig, LET2, LET_Y3, "h", "Fallback and manual-edit audit")


def _panel_i(fig, F) -> None:
    ax = mm_panel(fig, C3, R3_TOP, I_W, I_H)
    ax.set_xlim(0, I_W)
    ax.set_ylim(I_H, 0)

    events = F["timeline"]
    pitch, y0, gap = 6.3, 2.6, 4.4
    split = F["freeze_line_index"]
    ys = [y0 + k * pitch + (gap if k >= split else 0.0)
          for k in range(len(events))]
    ax.plot([1.3, 1.3], [ys[0], ys[-1]], color=RULE, lw=LW_THIN, zorder=1)
    y_rule = (ys[split - 1] + ys[split]) / 2.0 + 1.5
    ax.plot([0.0, I_W - 1.0], [y_rule, y_rule], color=PASS_GREEN, lw=LW_HAIR,
            ls=(0, (2.0, 1.6)), zorder=2)
    ax.text(I_W - 1.0, y_rule + 1.5, "target acquisition below this line",
            ha="right", va="center", fontsize=FS_TINY, color=PASS_GREEN)

    for (when, label, detail), y in zip(events, ys):
        ax.plot([1.3], [y], marker="o", ms=MS_POINT, color=INK, zorder=3)
        ax.text(3.2, y, when.strftime("%H:%M:%SZ"), ha="left", va="center",
                fontsize=FS_TINY, color=MUTED)
        ax.text(13.0, y, label, ha="left", va="center", fontsize=FS_TINY,
                color=INK)
        ax.text(13.0, y + 2.5, detail, ha="left", va="center",
                fontsize=FS_TINY, color=MUTED)
    day = events[0][0].strftime("%Y-%m-%d")
    ax.text(0.0, I_H - 0.5, f"all events UTC, {day}; self-recorded times",
            ha="left", va="center", fontsize=FS_TINY, color=MUTED)
    panel_label_at(fig, LET3, LET_Y3, "i", "Recorded chronology")


def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    F = _facts()

    fig = new_figure(H_MM)
    _panel_a(fig, F)
    _panel_b(fig, F)
    _panel_c(fig, F)
    _panel_d(fig, F)
    _panel_e(fig, F)
    _panel_f(fig, F)
    _panel_g(fig, F)
    _panel_h(fig, F)
    _panel_i(fig, F)

    return save_fig(fig, "figS14_replay_field_audit")
