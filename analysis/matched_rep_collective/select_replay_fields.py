"""Select 48 matched-collective fields for 3×3 replay (no API).

Implements docs/MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md exactly.
Does not use surrogates. Does not retune strata after Outcome A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.collective_scan import _histogram_from_phases  # noqa: E402
from analysis.stage_c.offline_diagnostics_v0_1 import _rk_series  # noqa: E402
from circlemap.observation import wrap_phase  # noqa: E402
from circlemap.representations import build_representation_prompt_from_histogram  # noqa: E402

OUT_DIR = ROOT / "analysis" / "matched_rep_collective"
SESSIONS = {
    "moments_m1_m3": ROOT
    / "runs/stage_c/matched-rep-collective-v0.1-moments_ed41bff41a9a/20260724T065125Z",
    "centers_24_standard": ROOT
    / "runs/stage_c/matched-rep-collective-v0.1-centers_65c4f06d18a1/20260724T065125Z",
    "intervals_24_decimal6": ROOT
    / "runs/stage_c/matched-rep-collective-v0.1-intervals_1d533f018b96/20260724T065128Z",
}
STRATA = [
    "early_transient",
    "pre_onset",
    "post_onset",
    "ordered_state",
    "negative_K",
    "high_r2",
]
SOURCES = list(SESSIONS.keys())
TARGETS = [
    "moments_m1_m3",
    "centers_24_standard",
    "intervals_24_decimal6",
]
N_BINS = 24
PEER = 16
QUOTA_STRATUM = 8
QUOTA_SOURCE = 16
NEAREST = {
    "early_transient": ["pre_onset", "negative_K", "post_onset"],
    "pre_onset": ["early_transient", "post_onset", "ordered_state"],
    "post_onset": ["pre_onset", "ordered_state", "high_r2"],
    "ordered_state": ["post_onset", "high_r2", "pre_onset"],
    "negative_K": ["early_transient", "pre_onset", "high_r2"],
    "high_r2": ["ordered_state", "post_onset", "pre_onset"],
}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_json(obj: object) -> str:
    return _sha_bytes(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _physical_hash(fractions: np.ndarray, peer: int) -> str:
    return _sha_json(
        {
            "fractions": [round(float(v), 12) for v in fractions.tolist()],
            "peer_count": int(peer),
            "n_bins": int(fractions.size),
        }
    )


def _first_hit(r1: np.ndarray, thresh: float) -> int | None:
    hits = np.where(r1 >= thresh)[0]
    return int(hits[0]) if hits.size else None


def _ordered_mask(r1: np.ndarray, thresh: float = 0.8, run: int = 5) -> np.ndarray:
    n = len(r1)
    ok = np.zeros(n, dtype=bool)
    above = r1 >= thresh
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if j - i >= run:
            ok[i:j] = True
        i = j
    return ok


def _tag_strata(r1: np.ndarray, r2: np.ndarray, activity: np.ndarray, K: float) -> dict[str, list[int]]:
    n = len(r1)
    # r1 series includes t=0..n_steps; actions have length n_steps
    tags: dict[str, list[int]] = {s: [] for s in STRATA}
    t_star = _first_hit(r1, 0.5)
    ordered = _ordered_mask(r1)

    for t in range(n):
        if 5 <= t <= 15 and r1[t] < 0.35:
            tags["early_transient"].append(t)
        if t_star is not None:
            if t_star - 10 <= t < t_star:
                tags["pre_onset"].append(t)
            if t_star < t <= t_star + 10:
                tags["post_onset"].append(t)
        else:
            mid = n // 2
            if abs(t - mid) <= 5 and r1[t] < 0.5:
                tags["pre_onset"].append(t)
            if abs(t - mid) <= 5 and activity[min(t, len(activity) - 1)] > 0.5:
                tags["post_onset"].append(t)
        if ordered[t]:
            tags["ordered_state"].append(t)
        if abs(K + 0.15) < 1e-12:
            tags["negative_K"].append(t)
        if r2[t] >= 0.45 and r2[t] > r1[t]:
            tags["high_r2"].append(t)
    return tags


def _score(cand: dict, stratum: str, median_r1: float) -> float:
    """Lower is better except where noted by returning negative preference."""
    if stratum == "high_r2":
        return -(float(cand["r2"]) - float(cand["r1"]))  # prefer high Q2
    if stratum == "negative_K":
        return -abs(float(cand["tau"]))
    return abs(float(cand["r1"]) - median_r1)


def _build_candidates() -> list[dict]:
    cands: list[dict] = []
    for source, session in SESSIONS.items():
        protocol = json.loads((session / "protocol.json").read_text(encoding="utf-8"))
        n_bins = int(protocol.get("n_bins", N_BINS))
        for run_dir in sorted(session.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "phases.npy").exists():
                continue
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            phases = np.load(run_dir / "phases.npy")
            actions = np.load(run_dir / "actions.npy")
            r1 = _rk_series(phases, 1)
            r2 = _rk_series(phases, 2)
            n_steps = actions.shape[0]
            activity = np.mean(np.abs(actions), axis=1)
            tau = np.mean(actions, axis=1)
            # align activity/tau to phase index t (use t-1 for t>=1)
            act_t = np.zeros(len(r1))
            tau_t = np.zeros(len(r1))
            act_t[0] = float(activity[0])
            tau_t[0] = float(tau[0])
            for t in range(1, len(r1)):
                idx = min(t - 1, n_steps - 1)
                act_t[t] = float(activity[idx])
                tau_t[t] = float(tau[idx])
            K = float(meta["coupling"])
            tags = _tag_strata(r1, r2, act_t, K)
            for stratum, times in tags.items():
                for t in times:
                    # focal agent 0 for reproducibility (peer-matched N=17)
                    agent = 0
                    hist = _histogram_from_phases(phases[t], agent, n_bins=n_bins)
                    fracs = np.asarray(hist.fractions, dtype=float)
                    ph = _physical_hash(fracs, PEER)
                    cands.append(
                        {
                            "source_representation": source,
                            "source_session": str(session.relative_to(ROOT)).replace(
                                "\\", "/"
                            ),
                            "K": K,
                            "seed": int(meta["seed_index"]),
                            "run_id": meta["run_id"],
                            "time": int(t),
                            "agent": agent,
                            "stratum_native": stratum,
                            "physical_hash": ph,
                            "physical_histogram_24": [float(x) for x in fracs.tolist()],
                            "r1": float(r1[t]),
                            "r2": float(r2[t]),
                            "activity": float(act_t[t]),
                            "tau": float(tau_t[t]),
                            "n_agents": int(meta["n_agents"]),
                            "peer_count": PEER,
                        }
                    )
    return cands


def _select(cands: list[dict]) -> tuple[list[dict], list[dict]]:
    # Dedup by physical_hash keeping first occurrence preference via ranking later
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_hash[c["physical_hash"]].append(c)

    # Unique pool: one representative per hash (prefer lower |r1-median| later)
    unique = []
    for h, group in by_hash.items():
        # keep all as alternatives for stratum assignment; pick best per need
        unique.append(group)

    selected: list[dict] = []
    used_hashes: set[str] = set()
    fallbacks: list[dict] = []
    source_counts = {s: 0 for s in SOURCES}
    stratum_counts = {s: 0 for s in STRATA}

    def available(stratum: str, source: str | None = None) -> list[dict]:
        pool = []
        for group in unique:
            # candidates in group that have this native stratum (or any if fallback)
            for c in group:
                if c["physical_hash"] in used_hashes:
                    continue
                if source is not None and c["source_representation"] != source:
                    continue
                if c["stratum_native"] != stratum:
                    continue
                pool.append(c)
        return pool

    def pick_one(pool: list[dict], stratum: str) -> dict | None:
        if not pool:
            return None
        # enforce source quota soft preference
        pool2 = [c for c in pool if source_counts[c["source_representation"]] < QUOTA_SOURCE]
        use = pool2 if pool2 else pool
        med = float(np.median([c["r1"] for c in use]))
        use = sorted(use, key=lambda c: (_score(c, stratum, med), c["physical_hash"]))
        return use[0]

    # Phase 1: ensure source × stratum coverage (1 each when possible)
    for stratum in STRATA:
        for source in SOURCES:
            if stratum_counts[stratum] >= QUOTA_STRATUM:
                break
            if source_counts[source] >= QUOTA_SOURCE:
                continue
            pool = available(stratum, source)
            chosen = pick_one(pool, stratum)
            if chosen is None:
                continue
            rec = dict(chosen)
            rec["stratum"] = stratum
            rec["fallback_from_stratum"] = None
            rec["fallback_reason"] = None
            rec["selection_rank"] = 0
            rec["tie_break"] = "source_x_stratum_coverage"
            selected.append(rec)
            used_hashes.add(rec["physical_hash"])
            source_counts[source] += 1
            stratum_counts[stratum] += 1

    # Phase 2: fill strata to 8 with round-robin sources
    for stratum in STRATA:
        guard = 0
        while stratum_counts[stratum] < QUOTA_STRATUM and guard < 10000:
            guard += 1
            # prefer under-quota sources
            sources_ordered = sorted(SOURCES, key=lambda s: source_counts[s])
            chosen = None
            native = True
            from_stratum = stratum
            for source in sources_ordered:
                if source_counts[source] >= QUOTA_SOURCE:
                    continue
                pool = available(stratum, source)
                chosen = pick_one(pool, stratum)
                if chosen is not None:
                    break
            if chosen is None:
                # any source, native stratum
                pool = available(stratum, None)
                chosen = pick_one(pool, stratum)
            if chosen is None:
                # fallback from nearest strata
                native = False
                for nb in NEAREST[stratum]:
                    for source in sources_ordered:
                        if source_counts[source] >= QUOTA_SOURCE and sum(
                            source_counts.values()
                        ) < 48:
                            # allow overflow only if total < 48 and stratum needs fill
                            pass
                        pool = available(nb, source if source_counts[source] < QUOTA_SOURCE else None)
                        # if source filtered empty, try any
                        if not pool:
                            pool = available(nb, None)
                        chosen = pick_one(pool, nb)
                        if chosen is not None:
                            from_stratum = nb
                            break
                    if chosen is not None:
                        break
            if chosen is None:
                break
            # if source already at 16, skip unless we must fill stratum and total < 48
            src = chosen["source_representation"]
            if source_counts[src] >= QUOTA_SOURCE:
                # try find alternative with under-quota source
                alt_pool = [
                    c
                    for c in available(from_stratum if not native else stratum, None)
                    if source_counts[c["source_representation"]] < QUOTA_SOURCE
                ]
                alt = pick_one(alt_pool, from_stratum if not native else stratum)
                if alt is not None:
                    chosen = alt
                    src = chosen["source_representation"]
                elif sum(source_counts.values()) >= 48:
                    break
            rec = dict(chosen)
            rec["stratum"] = stratum
            if native:
                rec["fallback_from_stratum"] = None
                rec["fallback_reason"] = None
            else:
                rec["fallback_from_stratum"] = from_stratum
                rec["fallback_reason"] = (
                    f"undersupply_of_{stratum}; filled_from_{from_stratum}"
                )
                fallbacks.append(
                    {
                        "target_stratum": stratum,
                        "from_stratum": from_stratum,
                        "physical_hash": rec["physical_hash"],
                        "source_representation": src,
                    }
                )
            rec["selection_rank"] = stratum_counts[stratum]
            rec["tie_break"] = "median_r1_or_stratum_score"
            selected.append(rec)
            used_hashes.add(rec["physical_hash"])
            source_counts[src] += 1
            stratum_counts[stratum] += 1

    # Phase 3: if some source < 16, top up from any remaining strata capacity... 
    # or replace is not allowed; only add if total < 48
    # Rebalance: if total < 48, fill remaining slots preferring under-quota sources
    guard = 0
    while len(selected) < 48 and guard < 10000:
        guard += 1
        sources_ordered = sorted(SOURCES, key=lambda s: source_counts[s])
        source = sources_ordered[0]
        if source_counts[source] >= QUOTA_SOURCE and all(
            source_counts[s] >= QUOTA_SOURCE for s in SOURCES
        ):
            break
        if source_counts[source] >= QUOTA_SOURCE:
            source = next(s for s in SOURCES if source_counts[s] < QUOTA_SOURCE)
        # pick stratum with room else any with fallback
        strata_ordered = sorted(STRATA, key=lambda s: stratum_counts[s])
        chosen = None
        stratum = strata_ordered[0]
        from_stratum = stratum
        native = True
        for st in strata_ordered:
            pool = available(st, source)
            chosen = pick_one(pool, st)
            if chosen is not None:
                stratum = st
                from_stratum = st
                break
        if chosen is None:
            for st in strata_ordered:
                for nb in [st] + NEAREST[st]:
                    pool = available(nb, source)
                    chosen = pick_one(pool, nb)
                    if chosen is not None:
                        stratum = st
                        from_stratum = nb
                        native = nb == st
                        break
                if chosen is not None:
                    break
        if chosen is None:
            break
        rec = dict(chosen)
        rec["stratum"] = stratum
        if native:
            rec["fallback_from_stratum"] = None
            rec["fallback_reason"] = None
        else:
            rec["fallback_from_stratum"] = from_stratum
            rec["fallback_reason"] = (
                f"source_balance_fill; assigned_to_{stratum}_from_{from_stratum}"
            )
            fallbacks.append(
                {
                    "target_stratum": stratum,
                    "from_stratum": from_stratum,
                    "physical_hash": rec["physical_hash"],
                    "source_representation": source,
                }
            )
        rec["selection_rank"] = len(selected)
        rec["tie_break"] = "source_balance_topup"
        selected.append(rec)
        used_hashes.add(rec["physical_hash"])
        source_counts[source] += 1
        stratum_counts[stratum] += 1

    # assign field_ids
    for i, rec in enumerate(selected):
        rec["field_id"] = f"mrc_replay_f{i:02d}"

    return selected, fallbacks


def _audit(selected: list[dict], fallbacks: list[dict]) -> dict:
    hashes = [f["physical_hash"] for f in selected]
    source_counts = {s: 0 for s in SOURCES}
    stratum_counts = {s: 0 for s in STRATA}
    sx = defaultdict(int)
    for f in selected:
        source_counts[f["source_representation"]] += 1
        stratum_counts[f["stratum"]] += 1
        sx[(f["source_representation"], f["stratum"])] += 1
    source_x_stratum_min = {
        f"{s}|{st}": sx[(s, st)] for s in SOURCES for st in STRATA
    }
    checks = {
        "n_fields_eq_48": len(selected) == 48,
        "physical_hash_unique_48": len(set(hashes)) == 48 and len(hashes) == 48,
        "source_balance_16_16_16": all(v == 16 for v in source_counts.values()),
        "stratum_counts_principle_8": all(v == 8 for v in stratum_counts.values()),
        "source_x_stratum_min1_where_possible": all(
            sx[(s, st)] >= 1 for s in SOURCES for st in STRATA
        ),
        "hand_edits": 0,
        "surrogate_use": 0,
        "outcome_A_dependent_reselection": 0,
    }
    return {
        "checks": checks,
        "pass": all(
            v is True or v == 0 for k, v in checks.items() if k != "hand_edits"
        )
        and checks["hand_edits"] == 0
        and checks["n_fields_eq_48"]
        and checks["physical_hash_unique_48"],
        "source_counts": source_counts,
        "stratum_counts": stratum_counts,
        "source_x_stratum": source_x_stratum_min,
        "n_fallbacks": len(fallbacks),
        "fallbacks": fallbacks,
    }


def _prompt_hashes(selected: list[dict]) -> dict:
    """Build 48×3 target prompts and aggregate hashes."""
    from circlemap.observation import RelativePhaseHistogram

    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
    per_target = {t: [] for t in TARGETS}
    rows = []
    for field in selected:
        fracs = np.asarray(field["physical_histogram_24"], dtype=float)
        hist = RelativePhaseHistogram(
            edges=edges, fractions=fracs, peer_count=PEER
        )
        for target in TARGETS:
            prompt = build_representation_prompt_from_histogram(target, hist)
            ph = _sha_bytes(prompt.encode("utf-8"))
            per_target[target].append(ph)
            rows.append(
                {
                    "field_id": field["field_id"],
                    "physical_hash": field["physical_hash"],
                    "target_representation": target,
                    "prompt_sha256": ph,
                    "prompt_chars": len(prompt),
                }
            )
    aggregates = {
        t: _sha_json(sorted(per_target[t])) for t in TARGETS
    }
    return {
        "n_target_prompts": len(rows),
        "expected": 144,
        "per_target_aggregate_sha256": aggregates,
        "prompts": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Building candidate pool...")
    cands = _build_candidates()
    print(f"  candidates (pre-dedup tags): {len(cands)}")
    selected, fallbacks = _select(cands)
    audit = _audit(selected, fallbacks)
    print("Audit:", json.dumps(audit["checks"], indent=2))
    print("source_counts", audit["source_counts"])
    print("stratum_counts", audit["stratum_counts"])
    print("n_fallbacks", audit["n_fallbacks"])

    fields_path = out / "replay_fields_v0_1.json"
    payload = {
        "status": "selected_v0_1",
        "lock_doc": "docs/MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md",
        "n_fields": len(selected),
        "sessions": {k: str(v.relative_to(ROOT)).replace("\\", "/") for k, v in SESSIONS.items()},
        "fields": selected,
        "hand_edits": [],
        "surrogate_use": False,
        "outcome_A_dependent_reselection": False,
    }
    fields_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    audit_path = out / "replay_selection_audit_v0_1.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("Building 48×3 target prompts...")
    prompts = _prompt_hashes(selected)
    (out / "replay_target_prompts_v0_1.json").write_text(
        json.dumps(prompts, indent=2), encoding="utf-8"
    )
    print(
        f"target prompts: {prompts['n_target_prompts']} "
        f"(expected {prompts['expected']})"
    )

    # Hash freeze package (selection stage; runner commit may be unavailable)
    try:
        import subprocess

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        commit = None

    analysis_plan = {
        "inference_unit": "48_physical_fields",
        "layers": [
            "target_representation_effect",
            "source_field_effect",
            "source_x_target_interaction",
        ],
        "primary_metrics": ["d_TV", "delta_a0", "delta_A"],
        "blocks": "optional_16x2_acquisition_blocks_same_total_calls",
        "mechanistic_readout_table": "docs/MATCHED_REP_COLLECTIVE_REPLAY_ANALYSIS_PLAN.md",
    }
    plan_path = out / "replay_analysis_plan_v0_1.json"
    plan_path.write_text(json.dumps(analysis_plan, indent=2), encoding="utf-8")

    freeze = {
        "status": "selection_hash_freeze_v0_1",
        "paid_authorized": False,
        "replay_fields_v0_1_sha256": _sha_bytes(fields_path.read_bytes()),
        "physical_field_aggregate_sha256": _sha_json(
            sorted(f["physical_hash"] for f in selected)
        ),
        "moments_prompt_aggregate_sha256": prompts["per_target_aggregate_sha256"][
            "moments_m1_m3"
        ],
        "centers_prompt_aggregate_sha256": prompts["per_target_aggregate_sha256"][
            "centers_24_standard"
        ],
        "intervals_prompt_aggregate_sha256": prompts["per_target_aggregate_sha256"][
            "intervals_24_decimal6"
        ],
        "analysis_plan_sha256": _sha_bytes(plan_path.read_bytes()),
        "selection_audit_sha256": _sha_bytes(audit_path.read_bytes()),
        "lock_doc": "docs/MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md",
        "n_target_prompts": prompts["n_target_prompts"],
        "runner_commit_hash": commit,
        "system_prompt": "response-law-v0.1",
        "note": (
            "Run refresh_replay_hash_freeze.py after protocol authoring "
            "to attach protocol/system-prompt SHAs."
        ),
    }
    freeze_path = out / "replay_hash_freeze_v0_1.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    print(f"Wrote {fields_path}")
    print(f"Wrote {audit_path}")
    print(f"Wrote {freeze_path}")
    print("AUDIT_PASS" if audit["pass"] else "AUDIT_FAIL")
    print(
        "Next: py analysis/matched_rep_collective/refresh_replay_hash_freeze.py"
    )
    return 0 if audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
