"""Sensitivity of the GPT collective phenotype classification to the locking rule.

The headline in the main text is operational: under the prespecified rule
``final r1 > 0.9`` the moments representation locks on 6/6 physical seeds at
every positive coupling while both histogram representations lock on 0/6,
giving an exact two-sided paired sign test p = 2/2^6 = 0.03125. This module
asks whether that statement survives the two choices buried in the rule --- the
threshold and the length of the trailing window over which r1 must stay above
it --- and writes the swept quantities to a JSON artifact.

Nothing here tunes anything. Before any sweep is run, the classification at the
prespecified setting is checked cell for cell against the frozen artifacts
(``endpoints_long.csv``, ``exact_sign_locking.csv``); a mismatch raises. The
classification primitives themselves are imported from
``analyze_paired_endpoints`` rather than re-implemented.

Usage
-----
    python -m analysis.matched_rep_collective.phenotype_rule_sensitivity

Writes ``analysis/matched_rep_collective/paired_analysis/
phenotype_rule_sensitivity.json``.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.matched_rep_collective.analyze_paired_endpoints import (
    _cluster_phenotype,
    _sustained_lock_time,
)

ROOT = Path(__file__).resolve().parents[2]
PAIRED = ROOT / "analysis" / "matched_rep_collective" / "paired_analysis"
OUT = PAIRED / "phenotype_rule_sensitivity.json"

REP_ORDER = (
    "moments_m1_m3",
    "centers_24_standard",
    "intervals_24_decimal6",
)

# The prespecified operational rule. These are the rule, not statistics: they
# are the values the sweeps run around.
THR_FROZEN = 0.9
W_FROZEN = 1          # the suffix rule is satisfied by a single final step
K_POS = (0.08, 0.15)
N_SEEDS = 6

THR_GRID = np.round(np.arange(0.70, 1.0001, 0.005), 4)
WINDOWS = np.arange(1, 101)


# ---------------------------------------------------------------- loading


def _load_runs() -> dict:
    """Trajectory harmonics for every representation, coupling and seed."""
    runs: dict = {}
    for i, rep in enumerate(REP_ORDER):
        ts = pd.read_csv(
            PAIRED / f"_tmp_session_{i}" / "trajectory_harmonics_timeseries.csv"
        )
        for (k, s), g in ts.groupby(["coupling", "seed_index"]):
            g = g.sort_values("t")
            runs[(rep, float(k), int(s))] = {
                "r1": g["r1"].to_numpy(dtype=float),
                "r2": g["r2"].to_numpy(dtype=float),
            }
    return runs


# ------------------------------------------------------------- rule forms


def _lock(runs, rep, k, s, thr: float, window: int = W_FROZEN) -> bool:
    """Generalised lock rule: r1 > thr over the trailing ``window`` steps."""
    r1 = runs[(rep, k, s)]["r1"]
    return bool(np.all(r1[-window:] > thr))


def _lock_fraction(runs, rep, thr: float, window: int = W_FROZEN,
                   ks=K_POS) -> float:
    vals = [_lock(runs, rep, k, s, thr, window)
            for k in ks for s in range(N_SEEDS)]
    return float(np.mean(vals))


def _exact_sign_p(ref_locks: np.ndarray, other_locks: np.ndarray) -> float:
    """Two-sided exact sign test on paired lock indicators (frozen form)."""
    n_pos = int(np.sum((ref_locks == 1) & (other_locks == 0)))
    n_neg = int(np.sum((ref_locks == 0) & (other_locks == 1)))
    n_nz = n_pos + n_neg
    if n_nz == 0:
        return 1.0
    k = max(n_pos, n_neg)
    tail = sum(comb(n_nz, i) for i in range(k, n_nz + 1)) / (2 ** n_nz)
    return min(1.0, 2.0 * tail)


# ----------------------------------------------------------- verification


def _verify_frozen(runs, ep: pd.DataFrame) -> dict:
    """Re-classify at the prespecified setting and check it against the freeze."""
    bad: list[str] = []
    for (rep, k, s), tr in runs.items():
        row = ep[
            (ep["representation"] == rep)
            & (np.isclose(ep["coupling"], k))
            & (ep["seed_index"] == s)
        ].iloc[0]
        pheno = _cluster_phenotype(
            {"final_r1": tr["r1"][-1], "final_r2": tr["r2"][-1]}
        )
        lock = int(tr["r1"][-1] > THR_FROZEN)
        sustained = int(np.isfinite(_sustained_lock_time(tr["r1"], THR_FROZEN)))
        if pheno != row["cluster_phenotype"]:
            bad.append(
                f"{rep} K={k} s{s}: phenotype {pheno} != {row['cluster_phenotype']}"
            )
        if lock != int(row["locking"]):
            bad.append(f"{rep} K={k} s{s}: locking {lock} != {int(row['locking'])}")
        if sustained != int(row["sustained_lock"]):
            bad.append(
                f"{rep} K={k} s{s}: sustained {sustained} != "
                f"{int(row['sustained_lock'])}"
            )
    if bad:
        raise RuntimeError(
            "Re-classification at the prespecified threshold does NOT reproduce "
            "the frozen phenotype table.\n" + "\n".join(bad[:12])
        )

    sign = pd.read_csv(PAIRED / "exact_sign_locking.csv")
    p_pos: list[float] = []
    for k in K_POS:
        ref = np.array(
            [int(_lock(runs, REP_ORDER[0], k, s, THR_FROZEN))
             for s in range(N_SEEDS)]
        )
        for other in REP_ORDER[1:]:
            oth = np.array(
                [int(_lock(runs, other, k, s, THR_FROZEN))
                 for s in range(N_SEEDS)]
            )
            p = _exact_sign_p(ref, oth)
            frozen_p = float(
                sign[
                    (np.isclose(sign["coupling"], k))
                    & (sign["contrast"] == f"{other}__minus__{REP_ORDER[0]}")
                ]["exact_sign_p_two_sided"].iloc[0]
            )
            if not np.isclose(p, frozen_p):
                raise RuntimeError(
                    f"Recomputed exact sign p={p} != frozen {frozen_p} "
                    f"(K={k}, {other})."
                )
            p_pos.append(p)

    n_lock = {
        rep: int(sum(_lock(runs, rep, k, s, THR_FROZEN)
                     for k in K_POS for s in range(N_SEEDS)))
        for rep in REP_ORDER
    }
    return {
        "p_exact": float(np.unique(np.round(p_pos, 12))[0]),
        "n_lock_pos": n_lock,
        "n_pos_runs": len(K_POS) * N_SEEDS,
    }


# ------------------------------------------------------------------ sweeps


def main() -> None:
    runs = _load_runs()
    ep = pd.read_csv(PAIRED / "endpoints_long.csv")
    frozen = _verify_frozen(runs, ep)

    # a. threshold sweep at the prespecified window
    thr_sweep = {
        rep: [_lock_fraction(runs, rep, float(t)) for t in THR_GRID]
        for rep in REP_ORDER
    }

    # the threshold band over which the 6/6-vs-0/6 statement holds unchanged
    finals_pos = {
        rep: np.array([runs[(rep, k, s)]["r1"][-1]
                       for k in K_POS for s in range(N_SEEDS)])
        for rep in REP_ORDER
    }
    band_lo = float(max(finals_pos[r].max() for r in REP_ORDER[1:]))
    band_hi = float(finals_pos[REP_ORDER[0]].min())

    # exact sign test recomputed at every swept threshold
    sign_sweep = []
    for t in THR_GRID:
        row = {"threshold": float(t)}
        for k in K_POS:
            ref = np.array(
                [int(_lock(runs, REP_ORDER[0], k, s, float(t)))
                 for s in range(N_SEEDS)]
            )
            for other in REP_ORDER[1:]:
                oth = np.array(
                    [int(_lock(runs, other, k, s, float(t)))
                     for s in range(N_SEEDS)]
                )
                row[f"p_K{k}_{other}"] = _exact_sign_p(ref, oth)
        sign_sweep.append(row)

    # b. sustained-window sweep at the prespecified threshold
    frac_w = {
        rep: np.array([_lock_fraction(runs, rep, THR_FROZEN, int(w))
                       for w in WINDOWS])
        for rep in REP_ORDER
    }
    w_all = int(WINDOWS[frac_w[REP_ORDER[0]] == 1.0].max())
    w_only = int(
        WINDOWS[
            (frac_w[REP_ORDER[0]] > 0)
            & (frac_w[REP_ORDER[1]] == 0)
            & (frac_w[REP_ORDER[2]] == 0)
        ].max()
    )

    out = {
        "prespecified_rule": {
            "final_r1_threshold": THR_FROZEN,
            "trailing_window": W_FROZEN,
            "positive_couplings": list(K_POS),
            "n_seeds": N_SEEDS,
        },
        "verified_against_freeze": frozen,
        "threshold_sweep": {
            "grid": [float(t) for t in THR_GRID],
            "lock_fraction": {r: thr_sweep[r] for r in REP_ORDER},
            "unchanged_band": [band_lo, band_hi],
            "exact_sign_p": sign_sweep,
        },
        "window_sweep": {
            "grid": [int(w) for w in WINDOWS],
            "lock_fraction": {r: frac_w[r].tolist() for r in REP_ORDER},
            "w_all_moments_lock": w_all,
            "w_moments_only_lock": w_only,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  unchanged threshold band: {band_lo:.3f}-{band_hi:.3f}")
    print(f"  all moments runs lock for W <= {w_all}; "
          f"moments-only locking for every W <= {w_only}")


if __name__ == "__main__":
    main()
