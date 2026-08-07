"""INT-2b: last bounded offline revision before INT-3 (or STOP).

1) Support-map decomposition by peer × family
2) Peer16-only re-evaluation
3) Interval-specific discrete-feature bake-off
4) Locked support-gate thresholds + GO/STOP
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.centers.distances import circular_emd_pairwise  # noqa: E402
from analysis.centers.models import HybridNativeKernel  # noqa: E402
from analysis.centers.peer_stratified_models import (  # noqa: E402
    CircularEmdKernel,
    PeerSpecificGlobal,
    peer_labels,
)
from analysis.centers.run_cent4_final_offline import cluster_bootstrap_delta  # noqa: E402
from analysis.centers.run_cent4_revision import (  # noqa: E402
    fit_named,
    per_row_nll,
    row_metrics,
)
from analysis.intervals.interval_features import DiscreteAugmentedKernel  # noqa: E402
from analysis.intervals.run_int2_offline import fit_intervals_named  # noqa: E402
from analysis.stage3.applicability_domain import ApplicabilityDomain, N_COMMON  # noqa: E402
from analysis.stage3.models import KernelHurdleMultinomial, counts_to_probs  # noqa: E402
from analysis.stage3.risk_calibrator import RiskCalibrator, monotone_ok  # noqa: E402
from analysis.stage3.validation import leave_one_group_out_predictions  # noqa: E402

REP = "intervals_24_decimal6"
BRANCH = ROOT / "analysis" / "intervals_branch"

# --- Locked support-gate constants (do not retune after INT-3) ---
SUPPORT_GATE = {
    "k_neighbors": 16,
    "coverage_gap_nn_threshold": 0.35,
    "high_stay_threshold": 0.55,
    "active_threshold": 0.55,
    "local_dispersion_tv_threshold": 0.35,
    "max_gap_risk_fraction": 0.35,
    "max_noncompress_fraction": 0.35,
    "max_gap_plus_noncompress_fraction": 0.45,
    "min_support_ok_fraction": 0.40,
    "calibration_note": (
        "Thresholds frozen from INT-2 exploratory map + Centers transfer-failure "
        "pattern (neighbors_active_feature_gap). Not to be changed after seeing INT-3."
    ),
}


def coarse_family(profile_family: str) -> str:
    f = profile_family.lower()
    if f.startswith("al_"):
        return "trajectory_al"
    if "sparse" in f and "unimodal" in f:
        return "sparse_unimodal"
    if "sparse" in f and "antipodal" in f:
        return "sparse_antipodal"
    if "antipodal" in f:
        return "antipodal"
    if "unimodal" in f or f.startswith("kappa_"):
        return "unimodal"
    if "bimodal" in f or "asymmetric" in f:
        return "multimodal"
    if "drift" in f or "k0" in f:
        return "drift"
    return "other"


def diagnose_local(
    nearest: float,
    frac_high_stay: float,
    frac_active: float,
    mean_tv: float,
    *,
    gate: dict,
) -> str:
    if nearest > gate["coverage_gap_nn_threshold"]:
        return "no_neighbor"
    if mean_tv >= gate["local_dispersion_tv_threshold"]:
        return "local_noncompressible"
    if frac_high_stay < 0.25 and frac_active >= 0.5:
        return "active_feature_gap"
    if frac_high_stay >= 0.25 and mean_tv < gate["local_dispersion_tv_threshold"]:
        return "supported"
    return "mixed"


def local_stats(
    query_x: np.ndarray,
    train_x: np.ndarray,
    train_counts: np.ndarray,
    *,
    peer: int,
    self_index: int | None,
    gate: dict,
) -> dict:
    peers = peer_labels(train_x)
    mask = peers == peer
    if int(np.sum(mask)) < 3:
        return {
            "d_NN": float("nan"),
            "n_local": 0,
            "V_local": float("nan"),
            "mean_p_stay_local": float("nan"),
            "mean_a0_local": float("nan"),
            "diagnosis": "no_neighbor",
        }
    xt = train_x[mask]
    ct = train_counts[mask]
    d = circular_emd_pairwise(query_x[None, N_COMMON:], xt[:, N_COMMON:])[0]
    if self_index is not None:
        # map global index to local if query is from train
        global_idx = np.where(mask)[0]
        hit = np.where(global_idx == self_index)[0]
        if hit.size:
            d[int(hit[0])] = np.inf
    k = min(gate["k_neighbors"], int(np.sum(np.isfinite(d))))
    order = np.argsort(d)[:k]
    empir = counts_to_probs(ct[order], alpha=0.0)
    stay = empir[:, 1]
    a0 = empir[:, 2] - empir[:, 0]
    activity = empir[:, 0] + empir[:, 2]
    tvs = []
    for i in range(k):
        for j in range(i + 1, k):
            tvs.append(0.5 * float(np.sum(np.abs(empir[i] - empir[j]))))
    mean_tv = float(np.mean(tvs)) if tvs else 0.0
    nearest = float(d[order[0]])
    frac_stay = float(np.mean(stay >= gate["high_stay_threshold"]))
    frac_act = float(np.mean(activity >= gate["active_threshold"]))
    diag = diagnose_local(nearest, frac_stay, frac_act, mean_tv, gate=gate)
    return {
        "d_NN": nearest,
        "n_local": k,
        "V_local": mean_tv,
        "mean_p_stay_local": float(np.mean(stay)),
        "mean_a0_local": float(np.mean(a0)),
        "frac_neighbors_high_stay": frac_stay,
        "frac_neighbors_active": frac_act,
        "diagnosis": diag,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def evaluate_scope(
    x: np.ndarray,
    counts: np.ndarray,
    rows: list,
    *,
    label: str,
    model_factories: dict,
) -> dict:
    blocks = [r.acquisition_block for r in rows]
    peers = peer_labels(x)
    ranking = []
    oofs = {}
    for name, factory in model_factories.items():
        oof, _ = leave_one_group_out_predictions(factory, x, counts, blocks)
        mask = np.isfinite(oof[:, 0])
        metrics = row_metrics(oof[mask], counts[mask])
        ranking.append({"model": name, **metrics})
        oofs[name] = (oof, mask)
    ranking = sorted(ranking, key=lambda r: r["log_loss"])

    # Cluster bootstrap vs peer-global / peer16-global
    base_name = (
        "peer16_global" if "peer16_global" in oofs else "peer_specific_global"
    )
    nll_g = per_row_nll(oofs[base_name][0], counts)
    mask_g = oofs[base_name][1]
    cluster = {}
    for name in model_factories:
        if name == base_name:
            continue
        oof_m, mask_m = oofs[name]
        mask = mask_g & mask_m
        delta = nll_g[mask] - per_row_nll(oof_m, counts)[mask]
        fc = [rows[i].profile for i, m in enumerate(mask) if m]
        fam = [rows[i].profile_family for i, m in enumerate(mask) if m]
        cluster[name] = {
            "row_mean": float(np.mean(delta)),
            "physical_field_cluster": cluster_bootstrap_delta(delta, fc, seed=31),
            "profile_family_cluster": cluster_bootstrap_delta(delta, fam, seed=32),
        }

    # Support diagnosis for all rows (leave-one-out local)
    support_rows = []
    # Fit models on full scope for disagreement / TV to local
    fitted = {name: factory(x, counts) for name, factory in model_factories.items()}
    for i, row in enumerate(rows):
        peer = int(peers[i])
        loc = local_stats(
            x[i], x, counts, peer=peer, self_index=i, gate=SUPPORT_GATE
        )
        preds = {}
        for name, model in fitted.items():
            p = model.predict_proba(x[i : i + 1])[0]
            preds[name] = p
        # model disagreement = mean pairwise TV among structured preds
        names = [n for n in preds if "global" not in n]
        diss = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                diss.append(0.5 * float(np.sum(np.abs(preds[names[a]] - preds[names[b]]))))
        # Reconstruct local mean trinomial from stay and a0:
        # p0=stay, a0=p+-p-, A=1-stay => p+=(A+a0)/2, p-=(A-a0)/2
        stay = loc["mean_p_stay_local"]
        a0 = loc["mean_a0_local"]
        if np.isfinite(stay) and np.isfinite(a0):
            A = 1.0 - stay
            p_plus = 0.5 * (A + a0)
            p_minus = 0.5 * (A - a0)
            local_p = np.asarray(
                [max(p_minus, 0.0), max(stay, 0.0), max(p_plus, 0.0)], dtype=np.float64
            )
            local_p = local_p / max(float(np.sum(local_p)), 1e-12)
        else:
            local_p = np.full(3, np.nan)
        tv_local = {}
        for name, p in preds.items():
            if np.all(np.isfinite(local_p)):
                tv_local[name] = float(0.5 * np.sum(np.abs(p - local_p)))
            else:
                tv_local[name] = float("nan")
        # Model-bias flag: local high stay but model predicts active
        bias = {}
        for name, p in preds.items():
            if loc["mean_p_stay_local"] >= 0.4 and p[1] < 0.3:
                bias[name] = "model_bias_vs_local_stay"
            elif loc["frac_neighbors_active"] >= 0.5 and p[1] >= 0.55:
                bias[name] = "model_more_abstaining_than_local"
            else:
                bias[name] = "aligned_or_mixed"
        support_rows.append(
            {
                "scope": label,
                "profile": row.profile,
                "profile_family": row.profile_family,
                "coarse_family": coarse_family(row.profile_family),
                "peer_count": peer,
                **loc,
                "model_disagreement": float(np.mean(diss)) if diss else float("nan"),
                **{f"tv_hat_local_{k}": v for k, v in tv_local.items()},
                **{f"bias_{k}": v for k, v in bias.items()},
            }
        )

    # Risk on best structured
    structured = [r for r in ranking if "global" not in r["model"]]
    candidate = structured[0]["model"] if structured else ranking[0]["model"]
    oof_c, mask_c = oofs[candidate]
    empir = counts_to_probs(counts, alpha=0.0)
    tv = 0.5 * np.sum(np.abs(oof_c - empir), axis=1)
    ad = ApplicabilityDomain.fit(x, representation=REP)
    sc = ad.score(
        x[mask_c], unresolved_near_zero=np.zeros(int(np.sum(mask_c)), dtype=bool)
    )
    risk_rows = []
    for j, li in enumerate(np.where(mask_c)[0]):
        risk_rows.append(
            {
                "e_tv": float(tv[li]),
                "d_shape": float(sc["d_shape"][j]),
                "d_native": float(sc["d_native"][j]),
                "local_density": float(sc["local_density"][j]),
                "u_ensemble": 0.0,
                "d_orientation": float(sc["d_orientation"][j]),
                "policy_a": float(sc["policy_a"][j]),
                "g_peer": int(bool(sc["g_peer"][j])),
                "peer_count": int(peers[li]),
                "bin_sparsity": float(np.mean(x[li, N_COMMON:] > 1e-12)),
            }
        )
    calibrator = RiskCalibrator.fit(
        [
            {
                "representation": REP,
                "scheme": "acquisition_block_holdout",
                **row,
            }
            for row in risk_rows
        ]
    )
    r_hat = calibrator.predict_rows(
        [
            {
                "representation": REP,
                "scheme": "acquisition_block_holdout",
                **row,
            }
            for row in risk_rows
        ]
    )
    y = np.asarray([r["e_tv"] for r in risk_rows])
    spars = np.asarray([r["bin_sparsity"] for r in risk_rows])
    order = np.argsort(r_hat)
    n = len(order)
    enrich = float(
        np.mean(y[order[-max(1, n // 5) :]]) / max(np.mean(y[order[: max(1, n // 5)]]), 1e-8)
    )

    # Aggregate support
    diag_counts = Counter(r["diagnosis"] for r in support_rows)
    n_sup = len(support_rows)
    by_peer_family = defaultdict(Counter)
    for r in support_rows:
        by_peer_family[(r["peer_count"], r["coarse_family"])][r["diagnosis"]] += 1

    return {
        "label": label,
        "n": int(x.shape[0]),
        "ranking": ranking,
        "cluster_bootstrap": cluster,
        "candidate": candidate,
        "risk": {
            "spearman": calibrator.train_spearman,
            "monotone_ok": monotone_ok(calibrator.bin_calibration),
            "enrichment": enrich,
            "spearman_risk_vs_bin_sparsity": _spearman(r_hat, spars),
            "spearman_error_vs_bin_sparsity": _spearman(y, spars),
        },
        "support_rows": support_rows,
        "diagnosis_counts": dict(diag_counts),
        "diagnosis_fractions": {
            k: v / max(n_sup, 1) for k, v in diag_counts.items()
        },
        "by_peer_family": {
            f"peer{peer}::{fam}": dict(cnt)
            for (peer, fam), cnt in sorted(by_peer_family.items())
        },
        "base_name": base_name,
    }


def support_gate_pass(fractions: dict) -> tuple[bool, dict]:
    gap = fractions.get("active_feature_gap", 0.0)
    non = fractions.get("local_noncompressible", 0.0)
    ok = fractions.get("supported", 0.0) + fractions.get("mixed", 0.0)
    checks = {
        "gap_risk_le_max": gap <= SUPPORT_GATE["max_gap_risk_fraction"],
        "noncompress_le_max": non <= SUPPORT_GATE["max_noncompress_fraction"],
        "gap_plus_noncompress_le_max": (gap + non)
        <= SUPPORT_GATE["max_gap_plus_noncompress_fraction"],
        "support_ok_ge_min": ok >= SUPPORT_GATE["min_support_ok_fraction"],
    }
    return all(checks.values()), checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=BRANCH)
    args = parser.parse_args()
    branch = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir
    scope = branch / "collective_scope_v0"
    arrays = np.load(scope / "arrays.npz", allow_pickle=True)
    x = arrays["features"]
    counts = arrays["counts"]
    peers = arrays["peer_count"]

    rows = []
    with (scope / "rows.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                SimpleNamespace(
                    profile=row["profile"],
                    profile_family=row["profile_family"],
                    acquisition_block=row["acquisition_block"],
                )
            )

    print("Evaluating full collective scope {8,16}...")
    factories_full = {
        "peer_specific_global": lambda xt, ct: PeerSpecificGlobal.fit(xt, ct),
        "peer_specific_activity_direction": lambda xt, ct: fit_named(
            "peer_specific_activity_direction", xt, ct
        ),
        "peer_specific_hellinger": lambda xt, ct: fit_intervals_named(
            "peer_specific_hellinger", xt, ct
        ),
        "peer_specific_canonical_emd": lambda xt, ct: fit_named(
            "peer_specific_canonical_emd", xt, ct
        ),
        "canonical_emd_plus_interval_discrete": lambda xt, ct: DiscreteAugmentedKernel.fit(
            xt, ct
        ),
        "direct_native_emd": lambda xt, ct: CircularEmdKernel.fit(
            xt, ct, use_canonical=False, k_neighbors=32
        ),
    }
    full = evaluate_scope(x, counts, rows, label="collective_8_16", model_factories=factories_full)

    print("Evaluating peer16-only...")
    mask16 = peers == 16
    x16, c16 = x[mask16], counts[mask16]
    rows16 = [rows[i] for i, m in enumerate(mask16) if m]
    factories16 = {
        "peer16_global": lambda xt, ct: PeerSpecificGlobal.fit(xt, ct),
        "peer16_activity_direction": lambda xt, ct: KernelHurdleMultinomial.fit(
            xt, ct, k_neighbors=32
        ),
        "peer16_hellinger": lambda xt, ct: HybridNativeKernel.fit(
            xt, ct, native_metric="hellinger", k_neighbors=32
        ),
        "peer16_canonical_emd": lambda xt, ct: CircularEmdKernel.fit(
            xt, ct, use_canonical=True, k_neighbors=32
        ),
        "peer16_canonical_emd_plus_discrete": lambda xt, ct: DiscreteAugmentedKernel.fit(
            xt, ct
        ),
        "peer16_direct_native_emd": lambda xt, ct: CircularEmdKernel.fit(
            xt, ct, use_canonical=False, k_neighbors=32
        ),
    }
    peer16 = evaluate_scope(
        x16, c16, rows16, label="peer16_only", model_factories=factories16
    )

    # Persist support tables
    for tag, payload in (("full", full), ("peer16", peer16)):
        path = branch / f"int2b_support_detail_{tag}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload["support_rows"][0].keys()))
            writer.writeheader()
            writer.writerows(payload["support_rows"])

    # Gate evaluation
    full_pass, full_checks = support_gate_pass(full["diagnosis_fractions"])
    p16_pass, p16_checks = support_gate_pass(peer16["diagnosis_fractions"])

    def model_gate(result: dict) -> dict:
        base = result["base_name"]
        ranking = result["ranking"]
        pg = next(r for r in ranking if r["model"] == base)
        structured = [r for r in ranking if r["model"] != base]
        best = structured[0]
        cb = result["cluster_bootstrap"].get(best["model"], {})
        field_ok = bool(
            cb.get("physical_field_cluster", {}).get("ci_entirely_positive")
        )
        return {
            "best_structured": best["model"],
            "beats_global_ll": bool(best["log_loss"] + 0.02 < pg["log_loss"]),
            "a0_improved": bool(best["a0_mae"] < pg["a0_mae"]),
            "sign_improved": bool(
                (best.get("a0_sign_acc") or 0) > (pg.get("a0_sign_acc") or 0)
            ),
            "cluster_field_ci_positive": field_ok,
            "pass": bool(
                best["log_loss"] + 0.02 < pg["log_loss"]
                and best["a0_mae"] < pg["a0_mae"]
                and field_ok
            ),
        }

    full_model = model_gate(full)
    p16_model = model_gate(peer16)

    risk_ok_full = bool(
        full["risk"]["spearman"] > 0
        and full["risk"]["enrichment"] > 1.0
        and abs(full["risk"]["spearman"])
        > 0.5 * abs(full["risk"]["spearman_risk_vs_bin_sparsity"] or 0)
    )
    risk_ok_p16 = bool(
        peer16["risk"]["spearman"] > 0 and peer16["risk"]["enrichment"] > 1.0
    )

    # Family collapse check: any coarse family with 100% gap/noncompress and n>=10
    def family_collapse(result: dict) -> list[str]:
        bad = []
        # rebuild counts by family
        by_fam = defaultdict(Counter)
        for r in result["support_rows"]:
            by_fam[r["coarse_family"]][r["diagnosis"]] += 1
        for fam, cnt in by_fam.items():
            n = sum(cnt.values())
            if n < 10:
                continue
            bad_n = cnt.get("active_feature_gap", 0) + cnt.get(
                "local_noncompressible", 0
            )
            if bad_n / n >= 0.9:
                bad.append(fam)
        return bad

    collapsed_full = family_collapse(full)
    collapsed_p16 = family_collapse(peer16)

    # Discrete feature helped support?
    # Compare diagnosis under predictions is already in support; check if discrete
    # model reduces tv_hat_local vs canonical on gap fields
    gap_rows = [r for r in peer16["support_rows"] if r["diagnosis"] == "active_feature_gap"]
    discrete_help = None
    if gap_rows:
        key_base = "tv_hat_local_peer16_canonical_emd"
        key_disc = "tv_hat_local_peer16_canonical_emd_plus_discrete"
        if key_base in gap_rows[0] and key_disc in gap_rows[0]:
            base = np.asarray([r[key_base] for r in gap_rows], dtype=np.float64)
            disc = np.asarray([r[key_disc] for r in gap_rows], dtype=np.float64)
            discrete_help = {
                "n_gap": len(gap_rows),
                "mean_tv_canonical": float(np.nanmean(base)),
                "mean_tv_discrete": float(np.nanmean(disc)),
                "improved_fraction": float(np.nanmean(disc < base - 1e-6)),
            }

    int3_go = bool(
        full_model["pass"]
        and p16_model["pass"]
        and full_pass
        and p16_pass
        and risk_ok_full
        and risk_ok_p16
        and not collapsed_full
        and not collapsed_p16
    )

    # STOP if peer16 support still fails and manifold is dominated by
    # gap + noncompress (Centers-like transport failure pattern).
    p16_gap = peer16["diagnosis_fractions"].get("active_feature_gap", 0.0)
    p16_non = peer16["diagnosis_fractions"].get("local_noncompressible", 0.0)
    p16_bad = p16_gap + p16_non
    discrete_clears_support = bool(p16_pass)  # discrete did not flip the gate
    stop = bool(
        (not p16_pass)
        and p16_bad >= 0.70
        and (not discrete_clears_support)
        and full_model["pass"]  # OOF still good ⇒ failure is transport/support, not IID fit
    )

    decision = {
        "status": "intervals_int2b_final_offline_revision",
        "support_gate_locked": SUPPORT_GATE,
        "full_collective": {
            "ranking": full["ranking"],
            "cluster_bootstrap": full["cluster_bootstrap"],
            "risk": full["risk"],
            "diagnosis_counts": full["diagnosis_counts"],
            "diagnosis_fractions": full["diagnosis_fractions"],
            "by_peer_family": full["by_peer_family"],
            "model_gate": full_model,
            "support_gate_checks": full_checks,
            "support_gate_pass": full_pass,
            "collapsed_families": collapsed_full,
        },
        "peer16_only": {
            "bundle": "intervals_collective_bundle_v1_N17",
            "n_train": peer16["n"],
            "ranking": peer16["ranking"],
            "cluster_bootstrap": peer16["cluster_bootstrap"],
            "risk": peer16["risk"],
            "diagnosis_counts": peer16["diagnosis_counts"],
            "diagnosis_fractions": peer16["diagnosis_fractions"],
            "by_peer_family": peer16["by_peer_family"],
            "model_gate": p16_model,
            "support_gate_checks": p16_checks,
            "support_gate_pass": p16_pass,
            "collapsed_families": collapsed_p16,
            "discrete_help_on_gap_fields": discrete_help,
        },
        "int3_prospective_pilot_go": int3_go,
        "intervals_collective_stop": stop,
        "paid_authorized": False,
        "freeze_ready": False,
        "locked_stop_wording": (
            "Interval-bin serialization yields an in-distribution-compressible "
            "microscopic response operator, but the available finite-peer training "
            "manifold does not support reliable prospective transport to "
            "collective-like fields."
        ),
        "next": (
            "draft INT-3 protocol + cost card (still needs explicit auth)"
            if int3_go
            else (
                "STOP intervals collective branch; archive as microscopic-only"
                if stop
                else "document residual offline options without paid spend"
            )
        ),
        "p_fail_plumbing_required_before_paid": True,
        "raw_response_retention_smoke_required_before_paid": True,
    }

    # Drop huge support_rows from JSON decision; keep paths
    decision["artifacts"] = {
        "support_full": "int2b_support_detail_full.csv",
        "support_peer16": "int2b_support_detail_peer16.csv",
    }
    (branch / "int2b_offline_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "full_diag": full["diagnosis_fractions"],
                "full_by_peer_family": full["by_peer_family"],
                "full_model_gate": full_model,
                "full_support_pass": full_pass,
                "peer16_diag": peer16["diagnosis_fractions"],
                "peer16_ranking_top": peer16["ranking"][:4],
                "peer16_model_gate": p16_model,
                "peer16_support_pass": p16_pass,
                "discrete_help": discrete_help,
                "int3_go": int3_go,
                "stop": stop,
                "next": decision["next"],
            },
            indent=2,
        )
    )
    if int3_go:
        return 0
    return 3 if stop else 2


if __name__ == "__main__":
    raise SystemExit(main())
