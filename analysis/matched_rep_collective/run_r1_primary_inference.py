"""R1 cross-model primary inference + replication PASS gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.matched_rep_collective.run_replay_primary_inference import (  # noqa: E402
    N_PERM,
    TARGETS,
    aggregate,
    global_interaction_test,
    global_source_test,
    global_target_test,
    matrix_3x3,
)

FIELDS = ROOT / "analysis/matched_rep_collective/replay_fields_v0_1.json"
OUT = ROOT / "analysis/matched_rep_collective/r1_primary"
RUNS = {
    "gpt": ROOT
    / "runs/matched_rep_collective_replay"
    / "matched-rep-collective-3x3-replay-v0.1_1e976b79b71d"
    / "20260724T114515Z",
    "claude": ROOT
    / "runs/matched_rep_collective_r1"
    / "matched-rep-r1-cross-model-v0.1_claude_claude-haiku-4-5-20251001"
    / "20260724T142109Z",
    "gemini": ROOT
    / "runs/matched_rep_collective_r1"
    / "matched-rep-r1-cross-model-v0.1_gemini_gemini-3.5-flash"
    / "20260724T142109Z",
}


def _load_rows(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _pair_noise_flags(target_res: dict) -> dict:
    within = target_res["noise_floor"]["mean_within_target_block_TV"]
    boots = target_res["pairwise_bootstrap"]
    flags = {}
    n_pass = 0
    for k, b in boots.items():
        # pair exceeds noise floor if mean TV > within mean
        ok = b["mean"] > within
        flags[k] = {"mean_TV": b["mean"], "exceeds_within": ok, "ci95": b["ci95"]}
        if ok:
            n_pass += 1
    return {"pairs": flags, "n_pairs_exceed_noise": n_pass}


def _invalid_audit(rows: list[dict]) -> dict:
    n = len(rows)
    n_valid = sum(1 for r in rows if r.get("valid"))
    by_t = {t: {"n": 0, "invalid": 0} for t in TARGETS}
    for r in rows:
        t = r["target_representation"]
        by_t[t]["n"] += 1
        if not r.get("valid"):
            by_t[t]["invalid"] += 1
    rates = {t: by_t[t]["invalid"] / max(by_t[t]["n"], 1) for t in TARGETS}
    return {
        "n": n,
        "n_valid": n_valid,
        "valid_rate": n_valid / max(n, 1),
        "invalid_rate_by_target": rates,
        "max_target_invalid_rate": float(max(rates.values())),
        "skew_ok": float(max(rates.values())) < 0.10,
        "overall_ok": (n_valid / max(n, 1)) >= 0.95,
    }


def evaluate_family(name: str, run_dir: Path, fields: list[dict]) -> dict:
    rows = _load_rows(run_dir)
    session = json.loads((run_dir / "session_summary.json").read_text(encoding="utf-8"))
    panel = aggregate(fields, rows)["panel"]
    print(f"[{name}] target test...", flush=True)
    target = global_target_test(panel)
    print(f"[{name}] source/interaction (secondary)...", flush=True)
    source = global_source_test(panel)
    ix = global_interaction_test(panel)
    mat = matrix_3x3(panel)
    inv = _invalid_audit(rows)
    pair_flags = _pair_noise_flags(target)
    nf = target["noise_floor"]
    between_minus_within = (
        nf["mean_between_target_TV"] - nf["mean_within_target_block_TV"]
    )
    # bootstrap CI of between-within using paired field diffs already in noise test
    noise_paired = target["noise_floor_paired_test"]
    gates = {
        "global_target_effect_p_lt_0.05": target["p_one_sided"] < 0.05,
        "between_exceeds_within": nf["mean_between_target_TV"]
        > nf["mean_within_target_block_TV"],
        "between_minus_within_positive": between_minus_within > 0
        and noise_paired["p_one_sided"] < 0.05,
        "at_least_2_of_3_pairs_exceed_noise": pair_flags["n_pairs_exceed_noise"] >= 2,
        "invalid_rate_ok": inv["overall_ok"] and inv["skew_ok"],
    }
    return {
        "family": name,
        "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        "model": session.get("model"),
        "acquisition": {
            "n_trace": len(rows),
            "n_valid": inv["n_valid"],
            "valid_rate": inv["valid_rate"],
            "actual_cost_usd_session": session.get("actual_cost_usd_session"),
            "invalid_audit": inv,
        },
        "primary_target": {
            "p": target["p_one_sided"],
            "mean_pairwise_TV": target["observed_mean_pairwise_TV"],
            "pairwise_bootstrap": target["pairwise_bootstrap"],
            "noise_floor": nf,
            "noise_floor_paired_test": noise_paired,
            "deviance_p": target["secondary_multinomial_deviance"]["p_one_sided"],
        },
        "secondary_source_p": source["p_one_sided"],
        "secondary_interaction_p": ix["p_one_sided"],
        "matrix_3x3_A": mat["A"],
        "pair_noise_flags": pair_flags,
        "replication_gates": gates,
        "replication_pass": all(gates.values()),
    }


def fieldwise_cross_model(fields, results_panels: dict) -> dict:
    """Correlate fieldwise mean pairwise TV across models (secondary)."""
    # results_panels: family -> panel
    out = {}
    fams = list(results_panels.keys())
    tv_by = {f: [] for f in fams}
    for i, field in enumerate(fields):
        for fam, panel in results_panels.items():
            fr = panel[i]
            ps = [np.asarray(fr["targets"][t]["p"]) for t in TARGETS]
            tvs = []
            for a in range(3):
                for b in range(a + 1, 3):
                    tvs.append(0.5 * np.sum(np.abs(ps[a] - ps[b])))
            tv_by[fam].append(float(np.mean(tvs)))
    for i, a in enumerate(fams):
        for b in fams[i + 1 :]:
            x = np.asarray(tv_by[a])
            y = np.asarray(tv_by[b])
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
            out[f"{a}__vs__{b}"] = {"pearson_fieldwise_mean_TV": corr}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = json.loads(FIELDS.read_text(encoding="utf-8"))["fields"]
    family_results = {}
    panels = {}
    for name, run_dir in RUNS.items():
        if name == "gpt":
            # GPT used 32 responses; still evaluate gates for reference
            pass
        res = evaluate_family(name, run_dir, fields)
        family_results[name] = res
        rows = _load_rows(run_dir)
        panels[name] = aggregate(fields, rows)["panel"]
        print(
            f"[{name}] PASS={res['replication_pass']} p={res['primary_target']['p']:.4g} "
            f"valid={res['acquisition']['valid_rate']:.3f}",
            flush=True,
        )

    cross = fieldwise_cross_model(fields, panels)
    decision = {
        "status": "R1_CROSS_MODEL_PRIMARY_V0_1",
        "n_response_r1": 16,
        "families": family_results,
        "cross_model_secondary": cross,
        "overall": {
            "claude_pass": family_results["claude"]["replication_pass"],
            "gemini_pass": family_results["gemini"]["replication_pass"],
            "both_second_families_pass": (
                family_results["claude"]["replication_pass"]
                and family_results["gemini"]["replication_pass"]
            ),
            "gpt_reference_operator_supported": family_results["gpt"][
                "replication_pass"
            ],
        },
        "claim": (
            "Observation representations causally select distinct collective phases, "
            "while identical physical states elicit representation-dependent microscopic "
            "response operators — replicated across OpenAI, Claude, and Gemini families."
            if (
                family_results["claude"]["replication_pass"]
                and family_results["gemini"]["replication_pass"]
            )
            else (
                "Second-model replication incomplete or failed gates; "
                "see per-family replication_gates."
            )
        ),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision["overall"], indent=2))
    print(f"Wrote {OUT / 'decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
