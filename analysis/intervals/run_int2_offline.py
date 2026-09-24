"""Intervals INT-2 offline: bake-off, cluster bootstrap, risk, support map, GO/NO-GO."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.centers.distances import (  # noqa: E402
    circular_emd_pairwise,
    hellinger_pairwise,
    js_pairwise,
)
from analysis.centers.peer_stratified_models import (  # noqa: E402
    COLLECTIVE_PEERS,
    CircularEmdKernel,
    DescriptorCollapseMixture,
    HierarchicalPeerKernel,
    PeerSpecificGlobal,
    PeerStratifiedModel,
    fit_collective_candidates,
    peer_labels,
)
from analysis.centers.run_cent4_final_offline import (  # noqa: E402
    cluster_bootstrap_delta,
)
from analysis.centers.run_cent4_revision import (  # noqa: E402
    fit_named,
    per_row_nll,
    row_metrics,
)
from analysis.stage3.applicability_domain import ApplicabilityDomain, N_COMMON  # noqa: E402
from analysis.stage3.models import (  # noqa: E402
    KernelHurdleMultinomial,
    SoftmaxStumpBoost,
    counts_to_probs,
)
from analysis.stage3.risk_calibrator import RiskCalibrator, monotone_ok  # noqa: E402
from analysis.stage3.validation import leave_one_group_out_predictions  # noqa: E402

REP = "intervals_24_decimal6"
BRANCH = ROOT / "analysis" / "intervals_branch"


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


def fit_intervals_named(name: str, x: np.ndarray, counts: np.ndarray):
    if name in {
        "pooled_global",
        "peer_specific_global",
        "peer_specific_circular_emd",
        "peer_specific_canonical_emd",
        "hierarchical_emd_kernel",
        "peer_specific_activity_direction",
        "descriptor_collapse_mixture",
        "pooled_kernel_nn",
    }:
        return fit_named(name, x, counts)
    if name == "softmax_stump_boost":
        return SoftmaxStumpBoost.fit(x, counts, n_estimators=30)
    if name == "peer_specific_hellinger":

        def hell(xt, ct):
            # reuse HybridNativeKernel from centers.models
            from analysis.centers.models import HybridNativeKernel

            return HybridNativeKernel.fit(xt, ct, native_metric="hellinger", k_neighbors=32)

        from analysis.centers.peer_stratified_models import _fit_peer_map

        experts, fb = _fit_peer_map(x, counts, hell)
        return PeerStratifiedModel(name=name, experts=experts, fallback=fb)
    raise KeyError(name)


def native_distance_screen(x: np.ndarray, counts: np.ndarray, blocks: list[str]) -> dict:
    """Quick AB holdout compare of native metrics inside peer-stratified EMD-style kernels."""
    from analysis.centers.models import HybridNativeKernel

    results = []
    for metric, factory in (
        (
            "circular_emd",
            lambda xt, ct: HybridNativeKernel.fit(
                xt, ct, native_metric="circular_emd", k_neighbors=24
            ),
        ),
        (
            "hellinger",
            lambda xt, ct: HybridNativeKernel.fit(
                xt, ct, native_metric="hellinger", k_neighbors=32
            ),
        ),
    ):
        oof, _ = leave_one_group_out_predictions(factory, x, counts, blocks)
        mask = np.isfinite(oof[:, 0])
        results.append({"native_metric": metric, **row_metrics(oof[mask], counts[mask])})
    # JS is expensive pairwise; skip full OOF — report sample pairwise mean scale only
    native = x[:64, N_COMMON:]
    js = js_pairwise(native, native)
    results.append(
        {
            "native_metric": "jensen_shannon_sample",
            "note": "scale check only (64x64)",
            "mean_offdiag_js": float(np.mean(js[np.triu_indices(js.shape[0], 1)])),
        }
    )
    return {"ab_holdout_native_kernels": results}


def build_support_map(x: np.ndarray, counts: np.ndarray, rows: list) -> list[dict]:
    """Local support diagnostics on a subsample of collective-like profiles."""
    peers = peer_labels(x)
    # Prefer sparse / antipodal / unimodal families
    idxs = [
        i
        for i, r in enumerate(rows)
        if any(
            key in r.profile_family.lower()
            for key in ("sparse", "antipodal", "unimodal", "kappa")
        )
    ]
    if len(idxs) > 80:
        rng = np.random.default_rng(0)
        idxs = sorted(rng.choice(idxs, size=80, replace=False).tolist())
    out = []
    for i in idxs:
        peer = int(peers[i])
        mask = peers == peer
        xt = x[mask]
        ct = counts[mask]
        local_pos = int(np.where(np.where(mask)[0] == i)[0][0])
        d = circular_emd_pairwise(x[i : i + 1, N_COMMON:], xt[:, N_COMMON:])[0]
        # exclude self
        d[local_pos] = np.inf
        order = np.argsort(d)[:16]
        empir = counts_to_probs(ct[order], alpha=0.0)
        stay = empir[:, 1]
        activity = empir[:, 0] + empir[:, 2]
        tvs = []
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                tvs.append(0.5 * float(np.sum(np.abs(empir[a] - empir[b]))))
        nearest = float(d[order[0]]) if order.size else float("nan")
        frac_stay = float(np.mean(stay >= 0.55))
        frac_act = float(np.mean(activity >= 0.55))
        mean_tv = float(np.mean(tvs)) if tvs else 0.0
        if nearest > 0.35:
            diag = "coverage_gap"
        elif frac_stay < 0.25 and frac_act >= 0.5:
            diag = "neighbors_active_feature_gap_risk"
        elif mean_tv >= 0.35:
            diag = "local_noncompressibility_risk"
        else:
            diag = "support_ok_or_mixed"
        out.append(
            {
                "profile": rows[i].profile,
                "profile_family": rows[i].profile_family,
                "peer_count": peer,
                "nearest_emd": nearest,
                "neighbor_mean_stay": float(np.mean(stay)),
                "neighbor_mean_activity": float(np.mean(activity)),
                "neighbor_response_mean_tv": mean_tv,
                "frac_neighbors_high_stay": frac_stay,
                "diagnosis": diag,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=BRANCH)
    args = parser.parse_args()
    branch = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir
    scope = branch / "collective_scope_v0"
    if not (scope / "arrays.npz").exists():
        raise SystemExit("run snapshot_and_atlas.py first")

    arrays = np.load(scope / "arrays.npz", allow_pickle=True)
    x = arrays["features"]
    counts = arrays["counts"]
    peers = arrays["peer_count"]
    bin_sp = arrays["bin_sparsity"]

    rows = []
    with (scope / "rows.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                SimpleNamespace(
                    profile=row["profile"],
                    profile_family=row["profile_family"],
                    acquisition_block=row["acquisition_block"],
                    source_family=row["source_family"],
                    sparse_realization=row["sparse_realization"],
                    offset_group=row["offset_group"],
                    physical_hash=row["physical_hash"],
                    prompt_hash=row["prompt_hash"],
                )
            )

    blocks = [r.acquisition_block for r in rows]
    print(f"INT-2 bake-off on intervals collective n={x.shape[0]} ...")

    model_names = [
        "pooled_global",
        "peer_specific_global",
        "peer_specific_circular_emd",
        "peer_specific_canonical_emd",
        "peer_specific_hellinger",
        "hierarchical_emd_kernel",
        "peer_specific_activity_direction",
        "softmax_stump_boost",
        "descriptor_collapse_mixture",
    ]

    ab_rank = []
    oof_store = {}
    for name in model_names:
        print(f"  AB OOF :: {name}")

        def factory(xt, ct, _n=name):
            return fit_intervals_named(_n, xt, ct)

        oof, folds = leave_one_group_out_predictions(factory, x, counts, blocks)
        mask = np.isfinite(oof[:, 0])
        metrics = row_metrics(oof[mask], counts[mask])
        by_peer = {}
        pl = peer_labels(x)
        for peer in COLLECTIVE_PEERS:
            pm = mask & (pl == peer)
            if int(np.sum(pm)) >= 5:
                by_peer[str(peer)] = row_metrics(oof[pm], counts[pm])
        ab_rank.append({"model": name, **metrics, "by_peer": by_peer})
        oof_store[name] = (oof, mask)
    ab_rank = sorted(ab_rank, key=lambda r: r["log_loss"])

    # Cluster bootstrap vs peer-specific global for top structured
    # fix duplicate nll_g
    nll_g = per_row_nll(oof_store["peer_specific_global"][0], counts)
    mask_g = oof_store["peer_specific_global"][1]
    cluster_boot = {}
    for name in (
        "peer_specific_activity_direction",
        "peer_specific_canonical_emd",
        "peer_specific_circular_emd",
        "peer_specific_hellinger",
        "softmax_stump_boost",
    ):
        if name not in oof_store:
            continue
        oof_m, mask_m = oof_store[name]
        mask = mask_g & mask_m
        delta = nll_g[mask] - per_row_nll(oof_m, counts)[mask]
        fc = [rows[i].profile for i, m in enumerate(mask) if m]
        fam = [rows[i].profile_family for i, m in enumerate(mask) if m]
        cluster_boot[name] = {
            "row_mean": float(np.mean(delta)),
            "physical_field_cluster": cluster_bootstrap_delta(delta, fc, seed=21),
            "profile_family_cluster": cluster_bootstrap_delta(delta, fam, seed=22),
        }

    native_screen = native_distance_screen(x, counts, blocks)

    # Risk on OOF winner (best structured)
    structured = [r for r in ab_rank if r["model"] != "peer_specific_global"]
    candidate = structured[0]["model"] if structured else ab_rank[0]["model"]
    oof_c, mask_c = oof_store[candidate]
    empir = counts_to_probs(counts, alpha=0.0)
    tv = 0.5 * np.sum(np.abs(oof_c - empir), axis=1)
    ad = ApplicabilityDomain.fit(x, representation=REP)
    sc = ad.score(
        x[mask_c],
        unresolved_near_zero=np.zeros(int(np.sum(mask_c)), dtype=bool),
    )
    risk_rows = []
    local = np.where(mask_c)[0]
    for j, li in enumerate(local):
        risk_rows.append(
            {
                "representation": REP,
                "scheme": "acquisition_block_holdout",
                "d_shape": float(sc["d_shape"][j]),
                "d_native": float(sc["d_native"][j]),
                "local_density": float(sc["local_density"][j]),
                "u_ensemble": 0.0,
                "d_orientation": float(sc["d_orientation"][j]),
                "policy_a": float(sc["policy_a"][j]),
                "g_peer": int(bool(sc["g_peer"][j])),
                "e_tv": float(tv[li]),
                "peer_count": int(peers[li]),
                "bin_sparsity": float(bin_sp[li]),
            }
        )
    calibrator = RiskCalibrator.fit(risk_rows)
    r_hat = calibrator.predict_rows(risk_rows)
    y = np.asarray([r["e_tv"] for r in risk_rows], dtype=np.float64)
    spars = np.asarray([r["bin_sparsity"] for r in risk_rows], dtype=np.float64)
    order = np.argsort(r_hat)
    n = len(order)
    lo, hi = order[: max(1, n // 5)], order[-max(1, n // 5) :]
    enrichment = float(np.mean(y[hi]) / max(np.mean(y[lo]), 1e-8))
    peer_sp = {}
    for peer in COLLECTIVE_PEERS:
        idx = [i for i, r in enumerate(risk_rows) if r["peer_count"] == peer]
        if len(idx) >= 8:
            peer_sp[str(peer)] = _spearman(r_hat[idx], y[idx])
    risk_report = {
        "candidate": candidate,
        "spearman": calibrator.train_spearman,
        "monotone_ok": monotone_ok(calibrator.bin_calibration),
        "enrichment_top20_vs_bottom20": enrichment,
        "peer_spearman": peer_sp,
        "spearman_risk_vs_bin_sparsity": _spearman(r_hat, spars),
        "spearman_error_vs_bin_sparsity": _spearman(y, spars),
    }
    calibrator.save(branch / "risk_calibrator_intervals_collective_v0.json")
    ad.save(branch / "applicability_domain_intervals_collective_v0")
    with (branch / "oof_residual_table_intervals_v0.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(risk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(risk_rows)

    support = build_support_map(x, counts, rows)
    with (branch / "support_map_intervals_v0.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(support[0].keys()))
        writer.writeheader()
        writer.writerows(support)
    from collections import Counter

    support_counts = Counter(r["diagnosis"] for r in support)

    # Gates
    pg = next(r for r in ab_rank if r["model"] == "peer_specific_global")
    best_s = structured[0]
    delta = cluster_boot.get(best_s["model"], {})
    field_ci = delta.get("physical_field_cluster", {})
    gate_model = bool(
        best_s["log_loss"] + 0.02 < pg["log_loss"]
        and field_ci.get("ci_entirely_positive")
        and best_s["a0_mae"] < pg["a0_mae"]
    )
    # peer collapse check
    peer_ok = True
    for peer in ("8", "16"):
        if peer in best_s.get("by_peer", {}) and peer in pg.get("by_peer", {}):
            if best_s["by_peer"][peer]["log_loss"] > pg["by_peer"][peer]["log_loss"] + 0.05:
                peer_ok = False
    gate_risk = bool(
        risk_report["spearman"] > 0
        and enrichment > 1.0
        and all(v >= 0 for v in peer_sp.values())
        if peer_sp
        else risk_report["spearman"] > 0
    )
    # Support gate: flag if feature-gap risk dominates
    n_gap = support_counts.get("neighbors_active_feature_gap_risk", 0)
    n_noncomp = support_counts.get("local_noncompressibility_risk", 0)
    gate_support = bool((n_gap + n_noncomp) / max(len(support), 1) < 0.5)

    int3_go = bool(gate_model and peer_ok and gate_risk and gate_support)

    decision = {
        "status": "intervals_int2_offline",
        "representation": REP,
        "n_train_collective": int(x.shape[0]),
        "peer_train_counts": {
            str(p): int(np.sum(peers == p)) for p in COLLECTIVE_PEERS
        },
        "acquisition_block_ranking": [
            {k: v for k, v in r.items() if k != "by_peer"} for r in ab_rank
        ],
        "acquisition_block_by_peer": {
            r["model"]: r.get("by_peer", {}) for r in ab_rank
        },
        "cluster_bootstrap": cluster_boot,
        "native_distance_screen": native_screen,
        "risk": risk_report,
        "support_diagnosis_counts": dict(support_counts),
        "gates": {
            "model_beats_peer_global_cluster": gate_model,
            "peer_no_collapse": peer_ok,
            "risk_positive": gate_risk,
            "support_map_not_dominated_by_gap": gate_support,
        },
        "candidate_model": candidate,
        "int3_prospective_pilot_go": int3_go,
        "paid_authorized": False,
        "freeze_ready": False,
        "centers_lessons_applied": [
            "peer_scope_split",
            "peer_specific_global_baseline",
            "cluster_bootstrap_required",
            "support_map_before_paid",
            "no_mechanical_centers_transplant",
        ],
    }
    (branch / "int2_offline_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "top5": decision["acquisition_block_ranking"][:5],
                "gates": decision["gates"],
                "int3_go": int3_go,
                "candidate": candidate,
                "risk": risk_report,
                "support_counts": dict(support_counts),
                "cluster_best": delta,
            },
            indent=2,
        )
    )
    return 0 if int3_go else 2


if __name__ == "__main__":
    raise SystemExit(main())
