"""Analyze same-information observation-map control vs cross-rep / noise floor."""

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
    _tv,
    aggregate,
    cluster_bootstrap_mean,
    field_tvs,
    global_target_test,
    permute_target_labels_on_responses,
    target_perm_stat,
)

OMAP_RUN = (
    ROOT
    / "runs/matched_rep_collective_omap"
    / "matched-rep-omap-control-v0.1_gpt-5.4-mini"
    / "20260724T161234Z"
)
GPT_REPLAY = (
    ROOT
    / "runs/matched_rep_collective_replay"
    / "matched-rep-collective-3x3-replay-v0.1_1e976b79b71d"
    / "20260724T114515Z"
)
FIELDS = ROOT / "analysis/matched_rep_collective/replay_fields_v0_1.json"
OUT = ROOT / "analysis/matched_rep_collective/omap_primary"
VARIANTS = (
    "moments_original",
    "moments_reformatted",
    "moments_length_matched",
)
RNG = np.random.default_rng(20260725)


def _load(run_dir: Path) -> list[dict]:
    rows = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = json.loads(FIELDS.read_text(encoding="utf-8"))["fields"]
    omap_rows = _load(OMAP_RUN)
    session = json.loads((OMAP_RUN / "session_summary.json").read_text(encoding="utf-8"))
    assert len(omap_rows) == 2304
    n_valid = sum(1 for r in omap_rows if r.get("valid"))

    # Reuse aggregate by treating control variants as "targets"
    panel = aggregate(fields, omap_rows)["panel"]
    # Fix: aggregate keys off target_representation which is the variant name — OK
    # But TARGETS constant is moments/centers/intervals — aggregate uses TARGETS!
    # Need custom aggregate for variants.

    from collections import defaultdict
    from analysis.matched_rep_collective.run_replay_primary_inference import (
        _stats_from_counts,
    )

    by_ft = defaultdict(lambda: np.zeros(3, dtype=int))
    by_ftb = defaultdict(lambda: np.zeros(3, dtype=int))
    raw_by = defaultdict(list)
    for r in omap_rows:
        if not r.get("valid"):
            continue
        v = int(r["action_value"])
        idx = {-1: 0, 0: 1, 1: 2}[v]
        fid = r["field_id"]
        t = r["target_representation"]
        b = int(r["acquisition_block"])
        by_ft[(fid, t)][idx] += 1
        by_ftb[(fid, t, b)][idx] += 1
        raw_by[fid].append((t, idx, b))

    panel = []
    for f in fields:
        fid = f["field_id"]
        entry = {
            "field_id": fid,
            "source": f["source_representation"],
            "stratum": f["stratum"],
            "K": float(f["K"]),
            "raw": raw_by[fid],
            "targets": {},
            "blocks": {},
        }
        for t in VARIANTS:
            entry["targets"][t] = _stats_from_counts(by_ft[(fid, t)])
            for b in (0, 1):
                entry["blocks"][f"{t}|b{b}"] = _stats_from_counts(by_ftb[(fid, t, b)])
        panel.append(entry)

    # Monkeypatch-style: temporarily use VARIANTS in local stats
    def mean_pairwise(panel_local):
        tvs = []
        for fr in panel_local:
            ps = [np.asarray(fr["targets"][t]["p"]) for t in VARIANTS]
            for i in range(3):
                for j in range(i + 1, 3):
                    tvs.append(_tv(ps[i], ps[j]))
        return float(np.mean(tvs)), tvs

    def permute(panel_local):
        out = []
        for fr in panel_local:
            raw = list(fr["raw"])
            labels = [t for t, _, _ in raw]
            RNG.shuffle(labels)
            new_raw = [(labels[i], raw[i][1], raw[i][2]) for i in range(len(raw))]
            by_t = {t: np.zeros(3, dtype=int) for t in VARIANTS}
            by_tb = {(t, b): np.zeros(3, dtype=int) for t in VARIANTS for b in (0, 1)}
            for t, j, b in new_raw:
                by_t[t][j] += 1
                by_tb[(t, b)][j] += 1
            targets = {t: _stats_from_counts(by_t[t]) for t in VARIANTS}
            blocks = {
                f"{t}|b{b}": _stats_from_counts(by_tb[(t, b)])
                for t in VARIANTS
                for b in (0, 1)
            }
            out.append({**fr, "targets": targets, "blocks": blocks, "raw": new_raw})
        return out

    obs, field_pair_tvs = mean_pairwise(panel)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = mean_pairwise(permute(panel))[0]
    p_same = float((np.sum(null >= obs) + 1) / (N_PERM + 1))

    # Pairwise same-info
    pairs = {
        "original__vs__reformatted": [],
        "original__vs__length_matched": [],
        "reformatted__vs__length_matched": [],
    }
    within = []
    within_field_ix = []
    between_field_ix = []
    for field_ix, fr in enumerate(panel):
        o = np.asarray(fr["targets"]["moments_original"]["p"])
        r = np.asarray(fr["targets"]["moments_reformatted"]["p"])
        L = np.asarray(fr["targets"]["moments_length_matched"]["p"])
        pairs["original__vs__reformatted"].append(_tv(o, r))
        pairs["original__vs__length_matched"].append(_tv(o, L))
        pairs["reformatted__vs__length_matched"].append(_tv(r, L))
        between_field_ix.extend([field_ix] * 3)
        for t in VARIANTS:
            p0 = np.asarray(fr["blocks"][f"{t}|b0"]["p"])
            p1 = np.asarray(fr["blocks"][f"{t}|b1"]["p"])
            within.append(_tv(p0, p1))
            within_field_ix.append(field_ix)

    pair_boot = {k: cluster_bootstrap_mean(np.asarray(v)) for k, v in pairs.items()}
    within_boot = cluster_bootstrap_mean(
        np.asarray(within), clusters=np.asarray(within_field_ix)
    )
    between_boot = cluster_bootstrap_mean(
        np.asarray(field_pair_tvs), clusters=np.asarray(between_field_ix)
    )

    # Cross-rep reference from GPT replay (moments vs centers/intervals)
    gpt_rows = _load(GPT_REPLAY)
    gpt_panel = aggregate(fields, gpt_rows)["panel"]
    cross = {"M_C": [], "M_I": [], "C_I": []}
    for fr in gpt_panel:
        pm = np.asarray(fr["targets"]["moments_m1_m3"]["p"])
        pc = np.asarray(fr["targets"]["centers_24_standard"]["p"])
        pi = np.asarray(fr["targets"]["intervals_24_decimal6"]["p"])
        cross["M_C"].append(_tv(pm, pc))
        cross["M_I"].append(_tv(pm, pi))
        cross["C_I"].append(_tv(pc, pi))
    cross_boot = {k: cluster_bootstrap_mean(np.asarray(v)) for k, v in cross.items()}
    cross_mean = float(np.mean(cross["M_C"] + cross["M_I"] + cross["C_I"]))

    same_mean = obs
    within_mean = float(np.mean(within))
    # Primary readout
    much_smaller = same_mean < 0.5 * cross_mean
    near_noise = same_mean <= within_mean * 2.0  # within factor 2 of block noise

    if much_smaller and (same_mean < cross_mean) and near_noise:
        verdict = (
            "PASS_format_length_not_sufficient — same-info TVs near block noise "
            "and much smaller than cross-representation TVs."
        )
        claim = (
            "The effect cannot be explained by superficial reformatting or "
            "prompt length alone."
        )
    elif same_mean >= 0.5 * cross_mean:
        verdict = (
            "SERIALIZATION_SENSITIVE — same-info variants still separate substantially."
        )
        claim = (
            "The operator is sensitive not only to retained information but also "
            "to its serialization."
        )
    else:
        verdict = "MIXED — inspect effect sizes."
        claim = "Mixed; see decision.json effect sizes."

    decision = {
        "status": "OMAP_CONTROL_PRIMARY_V0_1",
        "run_dir": str(OMAP_RUN.relative_to(ROOT)).replace("\\", "/"),
        "acquisition": {
            "n_calls": 2304,
            "n_valid": n_valid,
            "valid_rate": n_valid / 2304,
            "actual_cost_usd_session": session.get("actual_cost_usd_session"),
        },
        "same_info_global_test": {
            "test": "field_blocked_variant_label_permutation_mean_pairwise_TV",
            "observed_mean_pairwise_TV": same_mean,
            "null_mean": float(np.mean(null)),
            "p_one_sided": p_same,
            "n_perm": N_PERM,
        },
        "same_info_pairwise_bootstrap": pair_boot,
        "same_info_noise_floor": {
            "mean_between_variant_TV": same_mean,
            "mean_within_variant_block_TV": within_mean,
            "between_boot": between_boot,
            "within_boot": within_boot,
            "ratio": same_mean / max(within_mean, 1e-12),
        },
        "cross_rep_reference_gpt_replay": {
            "pairwise_bootstrap": cross_boot,
            "mean_pairwise_TV": cross_mean,
        },
        # Rounded lead numbers, the ones the manuscript and figures quote.
        "lead_with_pairwise": {
            "original_vs_reformatted": round(
                pair_boot["original__vs__reformatted"]["mean"], 3
            ),
            "original_vs_padded": round(
                pair_boot["original__vs__length_matched"]["mean"], 3
            ),
            "reformatted_vs_padded": round(
                pair_boot["reformatted__vs__length_matched"]["mean"], 3
            ),
            "block_noise": round(within_mean, 3),
            "same_info_mean": round(same_mean, 3),
            "cross_rep_mean": round(cross_mean, 3),
        },
        "central_proposition_doc": "docs/publication_RESULTS_OUTLINE.md",  # historical label kept verbatim: decision.json is frozen with this string
        "padding_wording": (
            "The neutral-padding manipulation produced the largest separation; "
            "do not claim prompt length alone."
        ),
        "explicit_non_claims": [
            "collective_phases_replicated_across_three_families",
            "macro_effect_from_serialization_alone",
            "prompt_length_alone_caused_effect",
            "operator_fully_mediates_macro",
            "source_or_feedback_established",
        ],
        "primary_contrast": {
            "same_info_over_cross_rep": same_mean / max(cross_mean, 1e-12),
            "same_info_over_block_noise": same_mean / max(within_mean, 1e-12),
            "much_smaller_than_cross_rep": much_smaller,
            "near_block_noise": near_noise,
        },
        "verdict": verdict,
        "claim": claim,
        "note": (
            "Cross-rep reference uses GPT 3×3 replay (32 responses); "
            "omap control uses 16 responses — compare magnitudes, not identity."
        ),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({
        "same_mean_TV": same_mean,
        "within_mean_TV": within_mean,
        "cross_mean_TV": cross_mean,
        "p_same": p_same,
        "ratio_same_over_cross": same_mean / cross_mean,
        "verdict": verdict,
        "claim": claim,
        "cost": session.get("actual_cost_usd_session"),
        "valid": n_valid,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
