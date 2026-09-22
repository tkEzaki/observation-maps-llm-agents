"""Summarize matched-rep collective paired analysis → decision JSON + print."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("analysis/matched_rep_collective/paired_analysis")
REPS = [
    "moments_m1_m3",
    "centers_24_standard",
    "intervals_24_decimal6",
]
METRICS = [
    "delta_final_r1",
    "delta_mean_r1",
    "delta_final_r2",
    "delta_mean_r2",
    "delta_mean_activity",
    "delta_mean_tau_social",
    "delta_t_r1_gt_0.5",
    "delta_t_r1_gt_0.9",
    "delta_locking",
    "delta_omega_coll",
]


def _nanmean(xs) -> float:
    a = np.asarray(list(xs), dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def main() -> int:
    long = list(csv.DictReader((OUT / "endpoints_long.csv").open(encoding="utf-8")))
    summary = list(
        csv.DictReader((OUT / "seed_block_summary.csv").open(encoding="utf-8"))
    )
    pheno = list(csv.DictReader((OUT / "phenotype_counts.csv").open(encoding="utf-8")))

    print("=== phenotype counts ===")
    for r in pheno:
        print(dict(r))

    by = defaultdict(list)
    for r in long:
        by[(r["representation"], float(r["coupling"]))].append(r)

    means = []
    print("\n=== means by rep x K ===")
    for K in sorted({float(r["coupling"]) for r in long}):
        print(f"--- K={K:+g} ---")
        for rep in REPS:
            rows = by[(rep, K)]
            row = {
                "representation": rep,
                "coupling": K,
                "final_r1": _nanmean(r["final_r1"] for r in rows),
                "mean_r1": _nanmean(r["mean_r1"] for r in rows),
                "final_r2": _nanmean(r["final_r2"] for r in rows),
                "mean_r2": _nanmean(r["mean_r2"] for r in rows),
                "mean_activity": _nanmean(r["mean_activity"] for r in rows),
                "mean_tau_social": _nanmean(r["mean_tau_social"] for r in rows),
                "t_r1_gt_0.5": _nanmean(r["t_r1_gt_0.5"] for r in rows),
                "t_r1_gt_0.9": _nanmean(r["t_r1_gt_0.9"] for r in rows),
                "locking": _nanmean(r["locking"] for r in rows),
                "omega_coll": _nanmean(r["omega_coll"] for r in rows),
                "n": len(rows),
            }
            means.append(row)
            short = rep.split("_")[0]
            print(
                f"  {short:10s} final_r1={row['final_r1']:.3f} "
                f"mean_r1={row['mean_r1']:.3f} mean_r2={row['mean_r2']:.3f} "
                f"A={row['mean_activity']:.3f} tau={row['mean_tau_social']:.3f} "
                f"t05={row['t_r1_gt_0.5']:.1f} t09={row['t_r1_gt_0.9']:.1f} "
                f"lock={row['locking']:.2f}"
            )

    print("\n=== seed-block CI signs ===")
    sig_events = []
    for row in summary:
        contrast = row["contrast"]
        K = float(row["coupling"])
        print(f"K={K:+g} {contrast} n={row['n_seeds']}")
        for k in METRICS:
            mean = float(row[f"{k}_mean"])
            lo = float(row[f"{k}_ci_low"])
            hi = float(row[f"{k}_ci_high"])
            if lo > 0:
                sign = "+"
            elif hi < 0:
                sign = "-"
            else:
                sign = "0"
            print(f"  {k:28s} {mean:+.3f} [{lo:+.3f},{hi:+.3f}] sig={sign}")
            if sign != "0":
                sig_events.append(
                    {
                        "coupling": K,
                        "contrast": contrast,
                        "metric": k,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "sign": sign,
                    }
                )

    # Outcome classification (preregistered)
    primary = {
        "delta_final_r1",
        "delta_mean_r1",
        "delta_mean_activity",
        "delta_t_r1_gt_0.5",
    }
    pathway = {
        "delta_mean_activity",
        "delta_mean_tau_social",
        "delta_t_r1_gt_0.5",
        "delta_t_r1_gt_0.9",
        "delta_mean_r2",
        "delta_final_r2",
        "delta_omega_coll",
    }
    primary_hits = [e for e in sig_events if e["metric"] in primary]
    pathway_hits = [e for e in sig_events if e["metric"] in pathway]
    locking_hits = [e for e in sig_events if e["metric"] == "delta_locking"]
    final_r1_hits = [e for e in sig_events if e["metric"] == "delta_final_r1"]

    # Phenotype story: different modal phenotypes across reps?
    pheno_by_rep = defaultdict(dict)
    for r in pheno:
        pheno_by_rep[r["representation"]][r["phenotype"]] = int(r["count"])

    outcome = "null"
    rationale = []
    if primary_hits and (
        any(e["metric"] == "delta_final_r1" for e in primary_hits)
        or locking_hits
        or any(
            pheno_by_rep[a] != pheno_by_rep[b]
            for a in REPS
            for b in REPS
            if a != b
        )
    ):
        # Check if phenotypes differ substantially
        modes = {
            rep: max(d.items(), key=lambda kv: kv[1])[0] if d else None
            for rep, d in pheno_by_rep.items()
        }
        if len(set(modes.values())) > 1 or final_r1_hits or locking_hits:
            outcome = "A"
            rationale.append(
                "Primary endpoint seed-block CIs exclude 0 and/or phenotypes differ."
            )
        elif pathway_hits:
            outcome = "B"
            rationale.append(
                "Terminal r1 similar pattern; pathway metrics show systematic paired deltas."
            )
    elif pathway_hits and not final_r1_hits:
        outcome = "B"
        rationale.append(
            "No clear final_r1 separation; pathway metrics have seed-block-significant deltas."
        )
    elif pathway_hits:
        outcome = "B"
        rationale.append("Pathway metrics significant under seed-block CIs.")
    else:
        rationale.append("No seed-block-significant paired contrasts on primary/pathway set.")

    # Refine: if outcome A claimed mainly from pathway, demote
    if outcome == "A" and not final_r1_hits and not locking_hits:
        # phenotype mode check
        modes = {
            rep: max(d.items(), key=lambda kv: kv[1])[0] if d else None
            for rep, d in pheno_by_rep.items()
        }
        if len(set(modes.values())) <= 1:
            outcome = "B"
            rationale.append("Demoted A→B: phenotypes share modal class; pathway differs.")

    decision = {
        "status": "matched_rep_collective_v0_1_analyzed",
        "sessions": {
            "moments": "runs/stage_c/matched-rep-collective-v0.1-moments_ed41bff41a9a/20260724T065125Z",
            "centers": "runs/stage_c/matched-rep-collective-v0.1-centers_65c4f06d18a1/20260724T065125Z",
            "intervals": "runs/stage_c/matched-rep-collective-v0.1-intervals_1d533f018b96/20260724T065128Z",
        },
        "n_runs_per_rep": 24,
        "valid_calls_per_rep": 40800,
        "outcome": outcome,
        "rationale": rationale,
        "phenotype_counts": pheno_by_rep,
        "n_significant_events": len(sig_events),
        "n_primary_significant": len(primary_hits),
        "n_pathway_significant": len(pathway_hits),
        "significant_events": sig_events,
        "means_by_rep_K": means,
        "next": {
            "if_A_or_B": [
                "run cross-encoding replay under MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md",
                "conditional second-model R1",
            ],
            "if_null": ["specialist venue / microscopic+C/T narrative"],
        },
    }
    path = OUT / "decision.json"
    path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print("\n=== OUTCOME ===", outcome)
    for line in rationale:
        print(" ", line)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
