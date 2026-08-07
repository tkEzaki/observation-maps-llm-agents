"""LLM-vs-LLM paired endpoint board for matched representation collectives.

v0.2 revisions (post Outcome A review):
- phenotypes: polar_locked / partial_polar_order / high_r2_nonpolar / low_polar_active
- Q2 = r2-r1, Q2_given_r1 = r2-r1^2
- sustained first-passage locking time
- pairwise contrasts include intervals-centers
- t_r1_gt_* labeled as first-passage (not locking time)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage_c.offline_diagnostics_v0_1 import (  # noqa: E402
    analyze_trajectories,
)
from analysis.stage_c.offline_diagnostics_v0_1 import _rk_series  # noqa: E402

OUT_DEFAULT = ROOT / "analysis" / "matched_rep_collective"

REPS = [
    "moments_m1_m3",
    "centers_24_standard",
    "intervals_24_decimal6",
]
PAIRWISE = [
    ("centers_24_standard", "moments_m1_m3"),
    ("intervals_24_decimal6", "moments_m1_m3"),
    ("intervals_24_decimal6", "centers_24_standard"),
]


def _cluster_phenotype(row: dict) -> str:
    final_r1 = float(row["final_r1"])
    final_r2 = float(row["final_r2"])
    if final_r1 >= 0.9:
        return "polar_locked"
    if final_r2 >= 0.5 and final_r2 > final_r1:
        return "high_r2_nonpolar"
    if final_r1 >= 0.35:
        return "partial_polar_order"
    return "low_polar_active"


def _first_hit(series: np.ndarray, thresh: float) -> float:
    hits = np.where(series > thresh)[0]
    return float(hits[0]) if hits.size else float("nan")


def _sustained_lock_time(r1: np.ndarray, thresh: float = 0.9) -> float:
    """min {t: r1(s)>thresh for all s in [t, T]}."""
    n = len(r1)
    if n == 0:
        return float("nan")
    ok = r1 > thresh
    # suffix all-true
    for t in range(n):
        if bool(np.all(ok[t:])):
            return float(t)
    return float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _enrich_row(row: dict, session: Path) -> None:
    run_dir = session / row["run_id"]
    phases = np.load(run_dir / "phases.npy")
    r1 = _rk_series(phases, 1)
    r2 = _rk_series(phases, 2)
    row["final_Q2"] = float(row["final_r2"]) - float(row["final_r1"])
    row["mean_Q2"] = float(row["mean_r2"]) - float(row["mean_r1"])
    row["final_Q2_given_r1"] = float(row["final_r2"]) - float(row["final_r1"]) ** 2
    row["mean_Q2_given_r1"] = float(row["mean_r2"]) - float(row["mean_r1"]) ** 2
    # Rename semantics: keep columns but document as first-passage
    row["t_r1_gt_0.5_first_passage"] = float(row["t_r1_gt_0.5"])
    row["t_r1_gt_0.9_first_passage"] = float(row["t_r1_gt_0.9"])
    row["t_r1_gt_0.9_sustained"] = _sustained_lock_time(r1, 0.9)
    row["locking"] = int(float(row["final_r1"]) > 0.9)
    row["sustained_lock"] = int(np.isfinite(row["t_r1_gt_0.9_sustained"]))
    row["cluster_phenotype"] = _cluster_phenotype(row)


def _load_session_rows(session: Path, tmp_out: Path) -> tuple[dict, list[dict]]:
    tmp_out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((session / "protocol.json").read_text(encoding="utf-8"))
    rows = analyze_trajectories(session, tmp_out)
    for row in rows:
        row["representation"] = protocol["representation"]
        row["protocol_version"] = protocol["protocol_version"]
        row["match_group"] = protocol.get("match_group")
        _enrich_row(row, session)
    return protocol, rows


def _paired_contrasts(long_rows: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in long_rows:
        key = (int(row["seed_index"]), float(row["coupling"]), int(row["n_agents"]))
        by_key[key][row["representation"]] = row

    metrics = [
        "final_r1",
        "mean_r1",
        "final_r2",
        "mean_r2",
        "final_r3",
        "mean_r3",
        "final_Q2",
        "mean_Q2",
        "final_Q2_given_r1",
        "mean_Q2_given_r1",
        "mean_activity",
        "mean_tau_social",
        "t_r1_gt_0.5_first_passage",
        "t_r1_gt_0.9_first_passage",
        "t_r1_gt_0.9_sustained",
        "omega_coll",
        "locking",
        "sustained_lock",
    ]
    out = []
    for key, rep_map in sorted(by_key.items()):
        seed_i, coupling, n_agents = key
        for other, ref in PAIRWISE:
            if other not in rep_map or ref not in rep_map:
                continue
            base = rep_map[ref]
            alt = rep_map[other]
            row = {
                "seed_index": seed_i,
                "coupling": coupling,
                "n_agents": n_agents,
                "init_seed": base["init_seed"],
                "run_id": base["run_id"],
                "contrast": f"{other}__minus__{ref}",
                "contrast_role": (
                    "primary_vs_moments"
                    if ref == "moments_m1_m3"
                    else "secondary_intervals_vs_centers"
                ),
                "phenotype_ref": base["cluster_phenotype"],
                "phenotype_other": alt["cluster_phenotype"],
            }
            for m in metrics:
                a = float(alt[m])
                b = float(base[m])
                if np.isnan(a) or np.isnan(b):
                    row[f"delta_{m}"] = float("nan")
                else:
                    row[f"delta_{m}"] = a - b
            out.append(row)
    return out


def _seed_block_summary(contrasts: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in contrasts:
        groups[(float(row["coupling"]), row["contrast"])].append(row)

    metrics = (
        [k for k in contrasts[0].keys() if k.startswith("delta_")] if contrasts else []
    )
    out = []
    rng = np.random.default_rng(0)
    for (coupling, contrast), rows in sorted(groups.items()):
        n = len(rows)
        summary = {
            "coupling": coupling,
            "contrast": contrast,
            "contrast_role": rows[0].get("contrast_role", ""),
            "n_seeds": n,
            "n_finite_note": (
                "percentile bootstrap on finite paired deltas only; "
                "do not treat singleton uncensored first-passage CIs as primary"
            ),
        }
        for m in metrics:
            vals = np.asarray([float(r[m]) for r in rows], dtype=float)
            finite = vals[np.isfinite(vals)]
            summary[f"{m}_n_finite"] = int(finite.size)
            if finite.size == 0:
                summary[f"{m}_mean"] = float("nan")
                summary[f"{m}_ci_low"] = float("nan")
                summary[f"{m}_ci_high"] = float("nan")
                continue
            summary[f"{m}_mean"] = float(np.mean(finite))
            if finite.size < 2:
                summary[f"{m}_ci_low"] = float(finite[0])
                summary[f"{m}_ci_high"] = float(finite[0])
                continue
            boots = [
                float(np.mean(rng.choice(finite, size=finite.size, replace=True)))
                for _ in range(2000)
            ]
            summary[f"{m}_ci_low"] = float(np.quantile(boots, 0.025))
            summary[f"{m}_ci_high"] = float(np.quantile(boots, 0.975))
        out.append(summary)
    return out


def _exact_sign_locking(long_rows: list[dict]) -> list[dict]:
    """Two-sided exact sign test: moments locks vs other does not, per K."""
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in long_rows:
        key = (int(row["seed_index"]), float(row["coupling"]))
        by_key[key][row["representation"]] = row

    rows = []
    for other, ref in PAIRWISE:
        if ref != "moments_m1_m3":
            # secondary hierarchy: intervals vs centers locking (usually both 0)
            pass
        for K in sorted({k[1] for k in by_key}):
            signs = []
            details = []
            for seed_i in range(6):
                cell = by_key.get((seed_i, K), {})
                if other not in cell or ref not in cell:
                    continue
                lock_ref = int(cell[ref]["locking"])
                lock_other = int(cell[other]["locking"])
                # +1 if ref locks and other does not (moments advantage)
                if lock_ref == 1 and lock_other == 0:
                    signs.append(+1)
                elif lock_ref == 0 and lock_other == 1:
                    signs.append(-1)
                else:
                    signs.append(0)
                details.append(
                    {
                        "seed_index": seed_i,
                        "lock_ref": lock_ref,
                        "lock_other": lock_other,
                    }
                )
            n_pos = sum(1 for s in signs if s > 0)
            n_neg = sum(1 for s in signs if s < 0)
            n_nz = n_pos + n_neg
            # two-sided exact binomial under p=1/2 on non-ties
            if n_nz == 0:
                p = 1.0
            else:
                # P(|X| extreme): 2 * sum_{k=max(n_pos,n_neg)}^{n} Binom
                from math import comb

                k = max(n_pos, n_neg)
                tail = sum(comb(n_nz, i) for i in range(k, n_nz + 1)) / (2**n_nz)
                p = min(1.0, 2.0 * tail)
            rows.append(
                {
                    "coupling": K,
                    "contrast": f"{other}__minus__{ref}",
                    "n_seeds": len(signs),
                    "n_ref_lock_other_not": n_pos,
                    "n_other_lock_ref_not": n_neg,
                    "n_ties": sum(1 for s in signs if s == 0),
                    "exact_sign_p_two_sided": p,
                    "note": (
                        "For positive K moments 6/6 lock & other 0/6: "
                        "p=2/2^6=0.03125"
                    ),
                    "per_seed": details,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT / "paired_analysis")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict] = []
    protocols = []
    for i, session in enumerate(args.session):
        protocol, rows = _load_session_rows(session, out_dir / f"_tmp_session_{i}")
        protocols.append(
            {
                "session": str(session),
                "representation": protocol["representation"],
                "protocol_version": protocol["protocol_version"],
                "n_runs": len(rows),
            }
        )
        long_rows.extend(rows)

    _write_csv(out_dir / "endpoints_long.csv", long_rows)
    contrasts = _paired_contrasts(long_rows)
    _write_csv(out_dir / "paired_contrasts.csv", contrasts)
    summary = _seed_block_summary(contrasts)
    _write_csv(out_dir / "seed_block_summary.csv", summary)

    phenotype_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in long_rows:
        phenotype_counts[row["representation"]][row["cluster_phenotype"]] += 1
    pheno_rows = [
        {"representation": rep, "phenotype": ph, "count": n}
        for rep, d in sorted(phenotype_counts.items())
        for ph, n in sorted(d.items())
    ]
    _write_csv(out_dir / "phenotype_counts.csv", pheno_rows)

    sign_rows = _exact_sign_locking(long_rows)
    # flatten per_seed for JSON; keep CSV without nested
    sign_csv = [
        {k: v for k, v in r.items() if k != "per_seed"} for r in sign_rows
    ]
    _write_csv(out_dir / "exact_sign_locking.csv", sign_csv)
    (out_dir / "exact_sign_locking.json").write_text(
        json.dumps(sign_rows, indent=2), encoding="utf-8"
    )

    # Q2 means by rep x K
    q2_rows = []
    by = defaultdict(list)
    for r in long_rows:
        by[(r["representation"], float(r["coupling"]))].append(r)
    for K in sorted({float(r["coupling"]) for r in long_rows}):
        for rep in REPS:
            rows = by[(rep, K)]
            q2_rows.append(
                {
                    "representation": rep,
                    "coupling": K,
                    "mean_final_r1": float(np.mean([float(x["final_r1"]) for x in rows])),
                    "mean_final_r2": float(np.mean([float(x["final_r2"]) for x in rows])),
                    "mean_final_Q2": float(np.mean([float(x["final_Q2"]) for x in rows])),
                    "mean_final_Q2_given_r1": float(
                        np.mean([float(x["final_Q2_given_r1"]) for x in rows])
                    ),
                    "lock_frac": float(np.mean([float(x["locking"]) for x in rows])),
                    "sustained_lock_frac": float(
                        np.mean([float(x["sustained_lock"]) for x in rows])
                    ),
                    "n": len(rows),
                }
            )
    _write_csv(out_dir / "q2_by_rep_K.csv", q2_rows)

    meta = {
        "n_sessions": len(args.session),
        "n_long_rows": len(long_rows),
        "n_contrasts": len(contrasts),
        "sessions": protocols,
        "reference_representation": "moments_m1_m3",
        "surrogate_free": True,
        "analysis_revision": "v0.2_outcome_A_review",
        "phenotype_labels": [
            "polar_locked",
            "partial_polar_order",
            "high_r2_nonpolar",
            "low_polar_active",
        ],
        "primary_endpoints": [
            "locking",
            "sustained_lock",
            "final_r1",
            "mean_r1",
            "final_Q2",
            "cluster_phenotype",
        ],
        "note_t09": (
            "t_r1_gt_0.9_first_passage is first hitting time, not locking time; "
            "prefer t_r1_gt_0.9_sustained / sustained_lock for lock claims"
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
