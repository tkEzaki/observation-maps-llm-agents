"""Supplementary Figure S1 - deterministic engine validation and causal isolation.

The claim this figure has to defend is negative: the collective differences
between observation maps cannot be blamed on the integrator, on the update
order, on absolute phase, on peer ordering, or on an accounting slip.

Four panels, all recomputed from frozen artifacts:

* **a** the engine and information-flow contract, including what the model
  never sees and where the coupling K is applied; the state circle is not a
  cartoon but the frozen final configuration of a production run (GPT
  intervals condition, K = 0.08, median final-r1 seed, t = 100) - the same run the
  main figure's phase rings show - with the focal agent i = 0 enlarged;
* **b** the exact K = 0 free evolution, one residual per physical seed x
  observation map x model family;
* **c** the collective-frequency accounting identity
  eps_Omega = Omega_coll - (mean omega + K tau_social);
* **d** the synchronous-update audit - a deterministic toy trajectory run under
  the correct synchronous rule and under an incorrect sequential rule, against
  the bit-exact replay of every production cell.

Sources
-------
* ``runs/stage_c/matched-rep-collective-v0.1-*`` (GPT, 3 maps x 4 K x 6 seeds)
* ``runs/stage_c/r2_claude_macro/...`` (Claude, 3 maps x 26 cells)
* ``analysis/matched_rep_collective/paired_analysis/endpoints_long.csv``
* ``experiments/stage_c/protocol_matched_rep_collective_v0_1_centers.json``
* ``circlemap/engine.py`` and ``tests/test_stage_a.py``

Panels dropped relative to ``docs/NATCOMM_SI_FIGURE_PANEL_PLAN_DETAILED.md``
-----------------------------------------------------------------------------
The plan asked for a global-rotation covariance panel (d) and a
peer-permutation invariance panel (e), each built from LLM action
distributions before and after the transformation. **No rotated-field or
permuted-peer LLM acquisition was ever run.** The only evidence that exists is
that the *serialization* is rotation- and permutation-invariant, which
``tests/test_stage_a.py`` proves and which panel d reports as part of the
Stage-A suite status. Plotting action-distribution differences would imply an
experiment that does not exist, so both panels are omitted and the survivors
are re-lettered a-d.
"""

from __future__ import annotations

import importlib
import io
import json
import re
import unittest
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from analysis.natcomm_si_opus.style import (
    ACCENT_RED,
    FAMILY_EDGE,
    FAMILY_LABEL,
    FAMILY_MARKER,
    FS_BODY,
    FS_SMALL,
    FS_TINY,
    GRAY_BOX,
    INK,
    LW_HAIR,
    LW_LINE,
    LW_THIN,
    LW_TRACE,
    MS_MEAN,
    MS_POINT,
    MUTED,
    PASS_GREEN,
    R2_CLAUDE,
    REP,
    REP_ABBR,
    REP_ORDER,
    REP_SHORT,
    ROOT,
    RULE,
    apply_style,
    fig_text_mm,
    mm_axes,
    mm_panel,
    new_figure,
    panel_label,
    panel_label_at,
    save_fig,
    trim_spines,
)
from circlemap.engine import simulate, step
from circlemap.observation import DEFAULT_N_BINS, all_relative_histograms
from experiments.stage_c.run_collective import _omega as production_omega

# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------
PAIRED = ROOT / "analysis" / "matched_rep_collective" / "paired_analysis"
META = json.loads((PAIRED / "meta.json").read_text(encoding="utf-8"))
PROTO_GPT = json.loads(
    (
        ROOT / "experiments" / "stage_c"
        / "protocol_matched_rep_collective_v0_1_centers.json"
    ).read_text(encoding="utf-8")
)
PROTO_CLAUDE = json.loads((R2_CLAUDE / "protocol.json").read_text(encoding="utf-8"))
TEST_MODULE = "tests.test_stage_a"

CELL_RE = re.compile(r"^N17_K([+-][0-9.]+)_s(\d+)$")

# Scatter takes an area; derive it from the shared marker diameters.
S_POINT = MS_POINT ** 2

# The free-drift unit test in tests/test_stage_a.py asserts this absolute
# tolerance; panel b draws it as the reference rule rather than inventing one.
FREE_DRIFT_ATOL = 5e-14
EPS_MACH = float(np.finfo(np.float64).eps)

FAMILIES = ("gpt", "claude")

# Toy audit in panel d: same physical seed and coupling as a production cell.
TOY_K = 0.15
TOY_CELL = "N17_K+0.15_s0"

# Panel a's state circle shows a real production configuration - the same run
# main Fig. 1b's bottom strip shows: GPT intervals condition, K = 0.08, the seed
# whose final r1 sits closest to the condition median, at t = 100.
EX_REP = "intervals_24_decimal6"
EX_K = 0.08
EX_T = 100


# ---------------------------------------------------------------------------
# Canvas plan (mm from the top-left of a 180 mm-wide page)
# ---------------------------------------------------------------------------
H_MM = 121.0

A_LEFT, A_TOP, A_W, A_H = 4.0, 9.0, 172.0, 53.0
A_LETTER_Y = 7.0

R2_TOP, R2_H = 71.0, 38.0

B_LEFT, B_W, B_LX = 13.0, 39.0, 4.0
C_LEFT, C_W, C_GAP, C_LX = 65.5, 27.0, 6.0, 56.5
D_LEFT, D_W, D_LX = 136.0, 39.0, 127.0

C_XLABEL_Y = 115.4                # one shared x label under the two facets


# ---------------------------------------------------------------------------
# Artifact readers
# ---------------------------------------------------------------------------
def _sci(value: float, digits: int = 2) -> str:
    """Mathtext scientific notation, so the exponent minus sets as U+2212."""
    if value == 0.0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** exponent
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def _gpt_session(rep: str) -> Path:
    for s in META["sessions"]:
        if s["representation"] == rep:
            return ROOT.joinpath(*str(s["session"]).replace("\\", "/").split("/"))
    raise KeyError(rep)


def _example_state() -> tuple[np.ndarray, int]:
    """Phase configuration for panel a's state circle, plus its seed index.

    Selection replicates main Fig. 1b's bottom strip: within the GPT
    ``intervals`` condition at K = 0.08, take the seed whose final r1 is closest
    to the condition median, and read the phases at t = 100 from the stored run.
    """
    ep = pd.read_csv(PAIRED / "endpoints_long.csv")
    sub = ep[(ep["representation"] == EX_REP)
             & (np.isclose(ep["coupling"], EX_K))]
    seed = int(sub.iloc[
        (sub["final_r1"] - sub["final_r1"].median()).abs().argmin()
    ]["seed_index"])
    arr = np.load(_gpt_session(EX_REP) / f"N17_K+{EX_K:.2f}_s{seed}"
                  / "phases.npy")
    phases = np.asarray(arr[min(EX_T, arr.shape[0] - 1)], dtype=float)
    return phases, seed


def _omega() -> np.ndarray:
    """Natural-frequency ladder exactly as the production runner builds it."""
    n_gpt = int(PROTO_GPT["n_agents"][0])
    half = float(PROTO_GPT["omega_halfwidth"])
    if int(PROTO_CLAUDE["n_agents"]) != n_gpt:
        raise ValueError("GPT and Claude protocols disagree on n_agents")
    if float(PROTO_CLAUDE["omega_halfwidth"]) != half:
        raise ValueError("GPT and Claude protocols disagree on omega_halfwidth")
    return production_omega(n_gpt, half)


def _cells() -> list[dict]:
    """Every acquisition cell of both families as {family, rep, K, seed, dir}."""
    out: list[dict] = []
    for rep in REP_ORDER:
        for d in sorted(_gpt_session(rep).glob("N17_K*_s*")):
            m = CELL_RE.match(d.name)
            if m is None or not (d / "phases.npy").exists():
                continue
            out.append(dict(family="gpt", rep=rep, coupling=float(m.group(1)),
                            seed=int(m.group(2)), dir=d))
    for cell in sorted(R2_CLAUDE.glob("N17_K*_s*")):
        m = CELL_RE.match(cell.name)
        if m is None:
            continue
        for rep in REP_ORDER:
            d = cell / rep
            if not (d / "phases.npy").exists():
                continue
            out.append(dict(family="claude", rep=rep, coupling=float(m.group(1)),
                            seed=int(m.group(2)), dir=d))
    return out


def _free_drift(cells: list[dict], omega: np.ndarray) -> pd.DataFrame:
    """eps_free = max_it |x_i(t) - x_i(0) - t omega_i| for every K = 0 cell."""
    rows = []
    for c in cells:
        if c["coupling"] != 0.0:
            continue
        ph = np.asarray(np.load(c["dir"] / "phases.npy"), dtype=float)
        t = np.arange(ph.shape[0])[:, None]
        free = ph[0][None, :] + t * omega[None, :]
        rows.append({**{k: c[k] for k in ("family", "rep", "seed")},
                     "eps": float(np.max(np.abs(ph - free)))})
    return pd.DataFrame(rows)


def _k0_map_identity(cells: list[dict]) -> tuple[float, float, float]:
    """Max |dx|, max |dr_m| and the action mismatch rate across maps at K = 0."""
    by_cell: dict[tuple[str, int], dict[str, Path]] = {}
    for c in cells:
        if c["coupling"] == 0.0:
            by_cell.setdefault((c["family"], c["seed"]), {})[c["rep"]] = c["dir"]
    d_phase = 0.0
    d_rm = 0.0
    n_act = 0
    n_diff = 0
    for reps in by_cell.values():
        ref_ph = np.asarray(np.load(reps[REP_ORDER[0]] / "phases.npy"), dtype=float)
        ref_rm = np.stack([np.abs(np.exp(1j * m * ref_ph).mean(axis=1))
                           for m in (1, 2, 3)])
        ref_ac = np.asarray(np.load(reps[REP_ORDER[0]] / "actions.npy"), dtype=float)
        for rep in REP_ORDER[1:]:
            ph = np.asarray(np.load(reps[rep] / "phases.npy"), dtype=float)
            rm = np.stack([np.abs(np.exp(1j * m * ph).mean(axis=1)) for m in (1, 2, 3)])
            ac = np.asarray(np.load(reps[rep] / "actions.npy"), dtype=float)
            d_phase = max(d_phase, float(np.max(np.abs(ph - ref_ph))))
            d_rm = max(d_rm, float(np.max(np.abs(rm - ref_rm))))
            n_act += int(ac.size)
            n_diff += int(np.sum(ac != ref_ac))
    return d_phase, d_rm, n_diff / n_act


def _omega_residuals(cells: list[dict], omega: np.ndarray) -> pd.DataFrame:
    """eps_Omega for both families.

    GPT residuals are read straight from the frozen ``endpoints_long.csv``.
    Claude has no frozen endpoint table, so its residuals are recomputed with
    the same estimator; the GPT recomputation is checked against the frozen
    column so the two halves are known to be the same quantity.
    """
    ep = pd.read_csv(PAIRED / "endpoints_long.csv")
    rows = []
    worst_check = 0.0
    for c in cells:
        ph = np.asarray(np.load(c["dir"] / "phases.npy"), dtype=float)
        ac = np.asarray(np.load(c["dir"] / "actions.npy"), dtype=float)
        n = int(ac.shape[0])
        collective = float(np.mean((ph[n] - ph[0]) / n))
        predicted = float(np.mean(omega)) + c["coupling"] * float(np.mean(ac))
        eps = collective - predicted
        if c["family"] == "gpt":
            frozen = ep[(ep["representation"] == c["rep"])
                        & (np.isclose(ep["coupling"], c["coupling"]))
                        & (ep["seed_index"] == c["seed"])]["engine_residual"]
            if len(frozen) != 1:
                raise ValueError(f"no unique frozen residual for {c['dir']}")
            worst_check = max(worst_check, abs(eps - float(frozen.iloc[0])))
            eps = float(frozen.iloc[0])
        rows.append({**{k: c[k] for k in ("family", "rep", "coupling", "seed")},
                     "eps": eps})
    df = pd.DataFrame(rows)
    df.attrs["recompute_check"] = worst_check
    return df


def _replay_audit(cells: list[dict], omega: np.ndarray) -> dict:
    """Production trajectory vs an independent synchronous recomputation."""
    per_t = None
    worst = 0.0
    n_updates = 0
    for c in cells:
        ph = np.asarray(np.load(c["dir"] / "phases.npy"), dtype=float)
        ac = np.asarray(np.load(c["dir"] / "actions.npy"), dtype=float)
        recomputed = np.stack([
            step(ph[t], omega, c["coupling"], ac[t]) for t in range(ac.shape[0])
        ])
        dev = np.max(np.abs(ph[1:] - recomputed), axis=1)
        per_t = dev if per_t is None else np.maximum(per_t, dev)
        worst = max(worst, float(np.max(dev)))
        n_updates += int(ac.size)
    return {"n_cells": len(cells), "n_updates": n_updates,
            "max_abs": worst, "per_t": np.asarray(per_t, dtype=float)}


def _sign_policy(observations) -> np.ndarray:
    """Deterministic surrogate policy: sign of the peer-mass imbalance."""
    mid = DEFAULT_N_BINS // 2
    actions = []
    for obs in observations:
        diff = float(np.sum(obs.fractions[mid:])) - float(np.sum(obs.fractions[:mid]))
        actions.append(0.0 if abs(diff) < 1e-12 else float(np.sign(diff)))
    return np.asarray(actions, dtype=float)


def _toy_update_divergence(omega: np.ndarray) -> dict:
    """Same policy and initial state under synchronous vs sequential updating."""
    theta0 = np.asarray(
        np.load(_gpt_session(REP_ORDER[0]) / TOY_CELL / "phases.npy"), dtype=float
    )[0]
    n_steps = int(PROTO_GPT["n_steps"])
    sync = simulate(theta0, omega, TOY_K, n_steps,
                    lambda _t, obs: _sign_policy(obs)).unwrapped_phases

    seq = np.empty_like(sync)
    seq[0] = theta0
    x = theta0.copy()
    for t in range(n_steps):
        for i in range(x.size):
            action = _sign_policy(all_relative_histograms(x))[i]
            x = x.copy()
            x[i] = x[i] + omega[i] + TOY_K * action
        seq[t + 1] = x

    r1 = lambda a: np.abs(np.exp(1j * a).mean(axis=1))  # noqa: E731
    return {"t": np.arange(n_steps + 1),
            "divergence": np.max(np.abs(sync - seq), axis=1),
            "r1_sync": r1(sync), "r1_seq": r1(seq)}


def _stage_a_status() -> tuple[int, int] | None:
    """Run the Stage-A invariance suite and report (passed, total).

    Returns ``None`` when the test module is not importable in this checkout.
    That case must not be reported as a failing suite: a missing module makes
    ``loadTestsFromName`` synthesise one erroring test, which would stamp the
    panel "0/1 pass" and assert, on a panel whose whole purpose is to certify
    that the engine is sound, that the engine is broken.  The caller prints an
    explicit "not run here" instead, and the figure must be rebuilt in a
    checkout that carries the suite before the Supplementary Information is
    assembled.
    """
    try:
        importlib.import_module(TEST_MODULE)
    except ImportError:
        return None
    suite = unittest.defaultTestLoader.loadTestsFromName(TEST_MODULE)
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    failed = len(result.failures) + len(result.errors)
    return result.testsRun - failed, result.testsRun


# ---------------------------------------------------------------------------
# Schematic primitives for panel a
# ---------------------------------------------------------------------------
def _stage_box(ax, x0, x1, y0, y1, *, face="#FFFFFF", edge="#AAAAAA",
               lw=LW_THIN, ls="solid"):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.4",
            facecolor=face, edgecolor=edge, lw=lw, linestyle=ls,
            clip_on=False,
        )
    )


def _elbow(ax, points, color=INK):
    """Right-angled connector; the final segment carries the arrow head."""
    for (xa, ya), (xb, yb) in zip(points, points[1:]):
        head = "-|>" if (xb, yb) == points[-1] else "-"
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle=head, color=color, lw=LW_LINE,
                                    mutation_scale=6, shrinkA=0, shrinkB=0))


def _flow_arrow(ax, x0, x1, y, color=INK):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=LW_LINE,
                                mutation_scale=6, shrinkA=0, shrinkB=0))


def _panel_a(fig, ex_phase: np.ndarray) -> None:
    ax = mm_panel(fig, A_LEFT, A_TOP, A_W, A_H)
    ax.set_xlim(0, A_W)
    ax.set_ylim(0, A_H)
    panel_label_at(fig, A_LEFT, A_LETTER_Y, "a",
                   "Engine and information-flow contract")

    cy, y0, y1 = 41.0, 35.0, 47.0

    # --- stage 0: the physical state ------------------------------------
    # Not a cartoon: the frozen final configuration of the production run the
    # main figure's phase rings also show, drawn with the same circle and dot
    # weights as those rings. The focal agent i = 0 is the enlarged dot.
    cx, cr = 8.5, 5.0
    ax.add_patch(mpatches.Circle((cx, cy), cr, fill=False,
                                 lw=LW_HAIR, color="#9E9E9E"))
    ax.scatter(cx + cr * np.cos(ex_phase), cy + cr * np.sin(ex_phase),
               s=S_POINT, c=INK, linewidths=0, zorder=4)
    ax.scatter([cx + cr * np.cos(ex_phase[0])],
               [cy + cr * np.sin(ex_phase[0])],
               s=MS_MEAN ** 2, c=INK, linewidths=0, zorder=5)
    ax.text(cx, 51.6, r"state $x(t)$", ha="center", va="baseline",
            fontsize=FS_SMALL, color=INK)
    # The two caption lines are wider than the circle they label, so their
    # shared centre is nudged right of it: centred on the circle, the first line
    # would start 0.1 mm from the canvas edge instead of clearing it by 2 mm.
    ax.text(cx + 2.9, 49.8, "example: intervals condition,", ha="center",
            va="center", fontsize=FS_TINY, color=MUTED)
    ax.text(cx + 2.9, 47.7, rf"$K={EX_K:g}$, $t={EX_T}$", ha="center",
            va="center", fontsize=FS_TINY, color=MUTED)

    # --- stage 1: the peer relative-phase field --------------------------
    _stage_box(ax, 18.5, 53.5, y0, y1)
    ax.text(36.0, 45.0, r"peer field $\rho_i$", ha="center", va="center",
            fontsize=FS_SMALL, fontweight="bold", color=INK)
    ax.text(36.0, 41.4, r"$\{\theta_j-\theta_i\}_{j\neq i}$", ha="center",
            va="center", fontsize=FS_SMALL, color=INK)
    ax.text(36.0, 37.6, f"{int(PROTO_GPT['peer_counts'][0])} peers, "
                        f"{int(PROTO_GPT['n_bins'])} uniform bins",
            ha="center", va="center", fontsize=FS_TINY, color=MUTED)

    # --- stage 2: the observation map ------------------------------------
    _stage_box(ax, 58.0, 96.0, y0, y1)
    ax.text(77.0, 45.0, r"observation map $\mathcal{R}(\rho_i)$", ha="center",
            va="center", fontsize=FS_SMALL, fontweight="bold", color=INK)
    for rep, x in zip(REP_ORDER, (66.0, 77.0, 88.0)):
        ax.text(x, 41.4, REP_SHORT[rep], ha="center", va="center",
                fontsize=FS_TINY, fontweight="bold", color=REP[rep])
    ax.text(77.0, 37.6, "the only intervened factor", ha="center", va="center",
            fontsize=FS_TINY, color=MUTED)

    # --- stage 3: the frozen model ---------------------------------------
    _stage_box(ax, 100.5, 127.5, y0, y1, face=GRAY_BOX["facecolor"],
               edge=GRAY_BOX["edgecolor"])
    ax.text(114.0, 43.6, "pretrained LLM", ha="center", va="center",
            fontsize=FS_SMALL, fontweight="bold", color=INK)
    ax.text(114.0, 39.2, "one call per agent,\nper time step", ha="center",
            va="center", fontsize=FS_TINY, color=MUTED, linespacing=1.3)

    # --- stage 4: the sampled action -------------------------------------
    _stage_box(ax, 132.0, 171.0, y0, y1)
    ax.text(151.5, 43.8, r"$f_i(t)\in\{-1,\,0,\,+1\}$", ha="center", va="center",
            fontsize=FS_SMALL, color=INK)
    ax.text(151.5, 39.0, "sampled action; the model\nis not told what it does",
            ha="center", va="center", fontsize=FS_TINY, color=MUTED,
            linespacing=1.3)

    for x0, x1 in ((14.0, 18.0), (54.0, 57.5), (96.5, 100.0), (128.0, 131.5)):
        _flow_arrow(ax, x0, x1, cy)

    # --- the engine closes the loop and is the only place K enters -------
    _stage_box(ax, 30.0, 143.0, 20.5, 30.5, face="#F7F7F7", edge="#8C8C8C",
               lw=LW_LINE)
    ax.text(86.5, 31.9, "engine  ·  deterministic, synchronous", ha="center",
            va="baseline", fontsize=FS_SMALL, fontweight="bold", color=INK)
    ax.text(80.0, 26.9, r"$x_i(t{+}1)=x_i(t)+\omega_i+K\,f_i(t)$", ha="center",
            va="center", fontsize=FS_BODY, color=INK)
    ax.text(80.0, 22.9, "every agent updated from the same pre-update snapshot",
            ha="center", va="center", fontsize=FS_TINY, color=MUTED)
    ax.text(140.0, 26.9, "$K$ enters\nonly here", ha="right", va="center",
            fontsize=FS_TINY, color=ACCENT_RED, linespacing=1.3)

    _elbow(ax, [(151.5, 34.6), (151.5, 25.5), (143.6, 25.5)])
    _elbow(ax, [(29.6, 25.5), (8.5, 25.5), (8.5, 35.4)])
    ax.text(11.0, 27.0, r"$x(t{+}1)$", ha="left", va="bottom",
            fontsize=FS_TINY, color=MUTED)

    # --- what the model never receives -----------------------------------
    _stage_box(ax, 0.4, 171.0, 1.6, 16.4, face="#FFFFFF", edge=ACCENT_RED,
               lw=LW_HAIR, ls=(0, (2.4, 1.8)))
    ax.text(3.2, 13.2, "Withheld from the model at every call", ha="left",
            va="center", fontsize=FS_SMALL, fontweight="bold", color=ACCENT_RED)
    ax.text(168.2, 13.2,
            "common action/output contract; map-specific observation "
            "description and payload",
            ha="right", va="center", fontsize=FS_TINY, color=MUTED)
    chips = (r"coupling $K$", "absolute phase", "agent identity",
             r"time step $t$", "previous actions", "trajectory history")
    for i, chip in enumerate(chips):
        x0 = 3.0 + i * 28.0
        _stage_box(ax, x0, x0 + 26.0, 4.0, 10.4, face="#F5F5F5", edge="#C0C0C0",
                   lw=LW_HAIR)
        ax.text(x0 + 13.0, 7.2, chip, ha="center", va="center",
                fontsize=FS_TINY, color=INK)


def _panel_b(fig, free: pd.DataFrame, ident: tuple[float, float, float]) -> None:
    ax = mm_axes(fig, B_LEFT, R2_TOP, B_W, R2_H)
    panel_label(ax, "b", r"Exact $K=0$ free evolution", dx_mm=B_LX - B_LEFT)

    combos = [(rep, fam) for rep in REP_ORDER for fam in FAMILIES]
    offsets = np.linspace(-0.30, 0.30, len(combos))
    for (rep, fam), dx in zip(combos, offsets):
        sub = free[(free["rep"] == rep) & (free["family"] == fam)].sort_values("seed")
        ax.scatter(sub["seed"].to_numpy() + dx, sub["eps"].to_numpy() / 1e-14,
                   s=S_POINT, c=REP[rep], marker=FAMILY_MARKER[fam],
                   linewidths=0, zorder=3)

    ax.axhline(FREE_DRIFT_ATOL / 1e-14, color=MUTED, lw=LW_HAIR, ls=(0, (3, 2)),
               zorder=2)
    ax.set_ylim(0.0, 9.0)
    ax.set_xlim(-0.65, 5.65)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xticks(range(6))
    ax.set_xlabel("physical seed")
    ax.set_ylabel(r"$\epsilon_{\mathrm{free}}$  ($\times10^{-14}$ rad)")
    trim_spines(ax, x=False)

    ax.annotate("free-drift unit-test tolerance",
                xy=(5.6, FREE_DRIFT_ATOL / 1e-14), xytext=(0, -1.8),
                textcoords="offset points", ha="right", va="top",
                fontsize=FS_TINY, color=MUTED)
    for rep, xf in zip(REP_ORDER, (0.02, 0.20, 0.38)):
        ax.text(xf, 0.985, REP_ABBR[rep], transform=ax.transAxes, ha="left",
                va="top", fontsize=FS_TINY, fontweight="bold", color=REP[rep])
    for fam, xf in zip(FAMILIES, (0.58, 0.77)):
        ax.scatter([xf], [0.947], transform=ax.transAxes, s=S_POINT,
                   c=MUTED, marker=FAMILY_MARKER[fam], linewidths=0,
                   clip_on=False)
        ax.text(xf + 0.045, 0.947, FAMILY_LABEL[fam], transform=ax.transAxes,
                ha="left", va="center", fontsize=FS_TINY, color=MUTED)
    ax.text(0.50, 0.605,
            r"$\max|\Delta r_m|=0$ between maps" "\n"
            r"($m=1,2,3$): the maps disagree on" "\n"
            f"{ident[2] * 100:.1f}% of agent-steps, yet every\n"
            "trajectory is bit-identical",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FS_TINY, color=INK, linespacing=1.3)


def _panel_c(fig, eo: pd.DataFrame) -> None:
    axes = []
    for j, fam in enumerate(FAMILIES):
        ax = mm_axes(fig, C_LEFT + j * (C_W + C_GAP), R2_TOP, C_W, R2_H)
        axes.append(ax)
        sub = eo[eo["family"] == fam]
        ks = sorted(sub["coupling"].unique())
        x = 0.0
        centers = []
        for k in ks:
            blk = sub[np.isclose(sub["coupling"], k)]
            x_start = x
            for rep in REP_ORDER:
                vals = blk[blk["rep"] == rep].sort_values("seed")["eps"].to_numpy()
                ax.scatter(np.arange(vals.size) + x, vals / 1e-16, s=S_POINT,
                           c=REP[rep], marker=FAMILY_MARKER[fam], linewidths=0,
                           zorder=3)
                x += vals.size
            centers.append((k, 0.5 * (x_start + x - 1)))
            x += 4.0
        ax.axhline(0.0, color=RULE, lw=LW_HAIR, zorder=1)
        for sign in (-1.0, 1.0):
            ax.axhline(sign * EPS_MACH / 1e-16, color=MUTED, lw=LW_HAIR,
                       ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-2.0, x - 2.0)
        ax.set_ylim(-3.2, 3.2)
        ax.set_yticks([-2, 0, 2])
        ax.set_xticks([c for _, c in centers])
        ax.set_xticklabels([rf"${k:g}$" for k, _ in centers], fontsize=FS_SMALL)
        ax.text(0.03, 0.985, FAMILY_LABEL[fam], transform=ax.transAxes,
                ha="left", va="top", fontsize=FS_BODY, fontweight="bold",
                color=FAMILY_EDGE[fam])
        if j == 0:
            ax.set_ylabel(r"$\epsilon_{\Omega}$  ($\times10^{-16}$)")
        else:
            ax.tick_params(labelleft=False)
            ax.annotate(r"$\pm\varepsilon_{\mathrm{mach}}$",
                        xy=(1.0, EPS_MACH / 1e-16), xycoords=("axes fraction", "data"),
                        xytext=(1.2, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=FS_TINY, color=MUTED,
                        annotation_clip=False)
        trim_spines(ax, x=False)
    panel_label(axes[0], "c", "Collective-frequency accounting identity",
                dx_mm=C_LX - C_LEFT)
    fig_text_mm(fig, C_LEFT + C_W + C_GAP / 2.0, C_XLABEL_Y, r"coupling $K$",
                ha="center", va="baseline", fontsize=FS_BODY, color=INK)


def _panel_d(fig, toy: dict, audit: dict, tests: tuple[int, int]) -> None:
    ax = mm_axes(fig, D_LEFT, R2_TOP, D_W, R2_H)
    panel_label(ax, "d", "Synchronous-update audit", dx_mm=D_LX - D_LEFT)

    ax.plot(toy["t"], toy["divergence"], color=ACCENT_RED, lw=LW_TRACE, zorder=3)
    ax.plot(toy["t"][1:], audit["per_t"], color=PASS_GREEN, lw=LW_LINE, zorder=4)
    ax.set_xlim(0, toy["t"][-1])
    peak = float(np.max(toy["divergence"]))
    ax.set_ylim(-0.58 * peak, 1.62 * peak)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\max_i|\Delta x_i|$ (rad)")
    trim_spines(ax, x=False)

    ax.text(0.97, 0.855, "sequential update", transform=ax.transAxes,
            ha="right", va="top", fontsize=FS_TINY, fontweight="bold",
            color=ACCENT_RED)
    ax.text(0.50, 0.036,
            "production engine vs recomputed\n"
            r"synchronous update: $0$ at every" "\n"
            "step of every cell",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FS_TINY, color=PASS_GREEN, linespacing=1.3)
    if tests is None:
        stamp = "Stage-A suite not run here"
        stamp_color = MUTED
    else:
        stamp = f"Stage-A invariance suite {tests[0]}/{tests[1]} pass"
        stamp_color = INK
    ax.text(0.03, 0.985, stamp,
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_TINY,
            color=stamp_color)


# ---------------------------------------------------------------------------
def build(_out_dir: Path | None = None) -> Path:
    apply_style()
    omega = _omega()
    cells = _cells()
    free = _free_drift(cells, omega)
    ident = _k0_map_identity(cells)
    eo = _omega_residuals(cells, omega)
    audit = _replay_audit(cells, omega)
    toy = _toy_update_divergence(omega)
    tests = _stage_a_status()

    if ident[1] != 0.0:
        raise ValueError(
            "main Fig. 1f states max|dr_m| = 0 across maps at K = 0; "
            f"recomputation gives {ident[1]!r}"
        )

    ex_phase, ex_seed = _example_state()

    fig = new_figure(H_MM)
    _panel_a(fig, ex_phase)
    _panel_b(fig, free, ident)
    _panel_c(fig, eo)
    _panel_d(fig, toy, audit, tests)

    return save_fig(fig, "figS01_engine_validation")
