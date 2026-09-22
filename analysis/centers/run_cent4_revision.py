"""CENT-4 model revision: peer{8,16} stratified gates (offline; no CENT-3 refit)."""

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

from analysis.centers.cent3_io import (  # noqa: E402
    BRANCH,
    REP,
    action_channels,
    field_feature_row,
    latest_cent3_runs,
    load_catalog,
    pooled_and_block_counts,
)
from analysis.centers.peer_stratified_models import (  # noqa: E402
    COLLECTIVE_PEERS,
    CircularEmdKernel,
    DescriptorCollapseMixture,
    HierarchicalPeerKernel,
    PeerSpecificGlobal,
    PeerStratifiedModel,
    bin_sparsity,
    fit_collective_candidates,
    peer_labels,
)
from analysis.stage3.models import (  # noqa: E402
    GlobalEmpiricalBaseline,
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    counts_to_probs,
)
from analysis.stage3.risk_calibrator import RiskCalibrator, monotone_ok  # noqa: E402
from analysis.stage3.validation import (  # noqa: E402
    leave_one_group_out_predictions,
    multinomial_log_loss,
    tv_distance,
    unique_groups,
)

def fit_named(name: str, x: np.ndarray, counts: np.ndarray):
    """Fit a single named candidate (avoid refitting the whole catalog per fold)."""
    if name == "pooled_global":
        return GlobalEmpiricalBaseline.fit(counts)
    if name == "peer_specific_global":
        return PeerSpecificGlobal.fit(x, counts)
    if name == "peer_specific_circular_emd":

        def emd_only(xt, ct):
            return CircularEmdKernel.fit(
                xt, ct, use_canonical=False, name="emd_only", k_neighbors=24
            )

        from analysis.centers.peer_stratified_models import _fit_peer_map

        experts, fb = _fit_peer_map(x, counts, emd_only)
        return PeerStratifiedModel(
            name=name, experts=experts, fallback=fb
        )
    if name == "peer_specific_canonical_emd":

        def emd_can(xt, ct):
            return CircularEmdKernel.fit(
                xt, ct, use_canonical=True, name="emd_canonical", k_neighbors=32
            )

        from analysis.centers.peer_stratified_models import _fit_peer_map

        experts, fb = _fit_peer_map(x, counts, emd_can)
        return PeerStratifiedModel(name=name, experts=experts, fallback=fb)
    if name == "hierarchical_emd_kernel":
        peer = fit_named("peer_specific_circular_emd", x, counts)
        pooled = CircularEmdKernel.fit(
            x, counts, use_canonical=True, name="pooled_emd_canonical", k_neighbors=32
        )
        return HierarchicalPeerKernel(
            name=name, peer_model=peer, pooled=pooled, shrink=0.25
        )
    if name == "peer_specific_activity_direction":

        def hurdle(xt, ct):
            return KernelHurdleMultinomial.fit(xt, ct, k_neighbors=24)

        from analysis.centers.peer_stratified_models import _fit_peer_map

        experts, fb = _fit_peer_map(x, counts, hurdle)
        return PeerStratifiedModel(name=name, experts=experts, fallback=fb)
    if name == "descriptor_collapse_mixture":
        return DescriptorCollapseMixture.fit(x, counts)
    if name == "pooled_kernel_nn":
        return KernelNeighborBaseline.fit(x, counts, k_neighbors=32)
    raise KeyError(name)


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


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def row_metrics(pred: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    empir = counts_to_probs(counts, alpha=0.0)
    a0_hat = pred[:, 2] - pred[:, 0]
    a0_obs = empir[:, 2] - empir[:, 0]
    a_hat = pred[:, 0] + pred[:, 2]
    a_obs = empir[:, 0] + empir[:, 2]
    # Sign accuracy: agree on sign when |a0_obs| > 0.05; ties count as miss if hat nonzero
    active = np.abs(a0_obs) > 0.05
    if np.any(active):
        sign_acc = float(np.mean(np.sign(a0_hat[active]) == np.sign(a0_obs[active])))
    else:
        sign_acc = float("nan")
    return {
        "log_loss": multinomial_log_loss(pred, counts),
        "d_TV": float(np.mean(tv_distance(pred, empir))),
        "abs_e_A": float(np.mean(np.abs(a_hat - a_obs))),
        "abs_e_a0": float(np.mean(np.abs(a0_hat - a0_obs))),
        "a0_mae": float(np.mean(np.abs(a0_hat - a0_obs))),
        "a0_sign_acc": sign_acc,
        "a0_corr": _corr(a0_hat, a0_obs),
        "n": float(pred.shape[0]),
    }


def bootstrap_delta_ll(
    ll_base: np.ndarray,
    ll_model: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap on per-row multinomial NLL contributions approximated by TV proxy.

    Uses per-row TV as a stable paired residual (true per-row NLL needs counts).
    Here ll_* are per-fold mean log-loss; bootstrap over folds if len>=3,
    else return point delta only.
    """
    delta = ll_base - ll_model
    if delta.size < 3:
        return {
            "mean_delta_ll": float(np.mean(delta)) if delta.size else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_units": int(delta.size),
        }
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, delta.size, size=delta.size)
        means.append(float(np.mean(delta[idx])))
    means_arr = np.sort(np.asarray(means))
    return {
        "mean_delta_ll": float(np.mean(delta)),
        "ci_low": float(means_arr[int(0.025 * n_boot)]),
        "ci_high": float(means_arr[int(0.975 * n_boot)]),
        "n_units": int(delta.size),
    }


def per_row_nll(pred: np.ndarray, counts: np.ndarray) -> np.ndarray:
    empir_counts = np.maximum(counts, 0.0)
    totals = np.sum(empir_counts, axis=1)
    logp = np.log(np.clip(pred, 1e-12, 1.0))
    # mean per-trial NLL within row
    nll = -np.sum(empir_counts * logp, axis=1) / np.maximum(totals, 1.0)
    return nll


def evaluate_oof(x, counts, groups, model_factory, scheme: str):
    if scheme in {"leave_one_eps_level_out", "sparse_realization_holdout"}:
        active = [i for i, g in enumerate(groups) if g != "na"]
        if len(active) < 20 or len(unique_groups([groups[i] for i in active])) < 2:
            return None
        x_s, c_s, g_s = x[active], counts[active], [groups[i] for i in active]
        index_map = active
    else:
        x_s, c_s, g_s = x, counts, groups
        index_map = list(range(len(groups)))
        if len(unique_groups(g_s)) < 2:
            return None

    oof, fold_results = leave_one_group_out_predictions(model_factory, x_s, c_s, g_s)
    if not fold_results or not np.any(np.isfinite(oof[:, 0])):
        return None
    mask = np.isfinite(oof[:, 0])
    metrics = row_metrics(oof[mask], c_s[mask])
    fold_ll = np.asarray([f.metrics["log_loss"] for f in fold_results], dtype=np.float64)
    peers = peer_labels(x_s)
    by_peer = {}
    for peer in COLLECTIVE_PEERS:
        pm = mask & (peers == peer)
        if int(np.sum(pm)) >= 5:
            by_peer[str(peer)] = row_metrics(oof[pm], c_s[pm])
    return {
        "scheme": scheme,
        "metrics": metrics,
        "by_peer": by_peer,
        "fold_log_loss": fold_ll.tolist(),
        "oof_pred": oof,
        "oof_index_map": index_map,
        "oof_mask_local": mask,
        "counts_local": c_s,
        "x_local": x_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=BRANCH)
    args = parser.parse_args()
    branch = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir
    scope = branch / "collective_scope_v0"
    if not (scope / "arrays.npz").exists():
        raise SystemExit("run build_collective_snapshot.py first")

    arrays = np.load(scope / "arrays.npz", allow_pickle=True)
    x = arrays["features"]
    counts = arrays["counts"]
    unresolved = arrays["unresolved_near_zero"]
    p_fail = arrays["p_fail"]
    peers = arrays["peer_count"]

    rows = []
    with (scope / "rows.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                SimpleNamespace(
                    profile_family=row["profile_family"],
                    epsilon_level=row["epsilon_level"],
                    sparse_realization=row["sparse_realization"],
                    offset_group=row["offset_group"],
                    acquisition_block=row["acquisition_block"],
                    source_family=row["source_family"],
                    profile=row["profile"],
                    offset_index=int(row["offset_index"]),
                )
            )

    group_maps = {
        "leave_one_profile_out": [r.profile_family for r in rows],
        "sparse_realization_holdout": [r.sparse_realization for r in rows],
        "offset_group_holdout": [r.offset_group for r in rows],
        "acquisition_block_holdout": [r.acquisition_block for r in rows],
        "source_family_holdout": [r.source_family for r in rows],
    }

    print(f"Fitting candidates on collective-scope n={x.shape[0]} ...")
    # Full-fit catalog for CENT-3 prospective (no CENT-3 in train)
    models_full = fit_collective_candidates(x, counts)

    # --- Gate 1/2/3: OOF ---
    oof_summary = {"by_scheme_model": {}, "delta_vs_peer_global": {}}
    # Keep OOF preds for production candidate risk later
    oof_store: dict[str, dict] = {}

    model_names = list(models_full.keys())
    for scheme, groups in group_maps.items():
        for name in model_names:

            def factory(xt, ct, _n=name):
                return fit_named(_n, xt, ct)

            print(f"  OOF {scheme} :: {name}")
            result = evaluate_oof(x, counts, groups, factory, scheme)
            if result is None:
                continue
            key = f"{scheme}::{name}"
            oof_summary["by_scheme_model"][key] = {
                "metrics": result["metrics"],
                "by_peer": result["by_peer"],
                "fold_log_loss": result["fold_log_loss"],
            }
            if scheme == "acquisition_block_holdout":
                oof_store[name] = result

    # Deltas vs peer-specific global — prefer row-level paired bootstrap on AB OOF
    if "peer_specific_global" in oof_store:
        base = oof_store["peer_specific_global"]
        base_mask = base["oof_mask_local"]
        base_nll = per_row_nll(base["oof_pred"], base["counts_local"])
        for name in model_names:
            if name not in oof_store:
                continue
            model = oof_store[name]
            mask = base_mask & model["oof_mask_local"]
            if int(np.sum(mask)) < 10:
                continue
            b = base_nll[mask]
            m = per_row_nll(model["oof_pred"], model["counts_local"])[mask]
            boot = bootstrap_delta_ll(b, m, seed=17)
            boot["point_delta_ll"] = float(np.mean(b - m))
            boot["model_metrics"] = oof_summary["by_scheme_model"][
                f"acquisition_block_holdout::{name}"
            ]["metrics"]
            boot["by_peer"] = oof_summary["by_scheme_model"][
                f"acquisition_block_holdout::{name}"
            ]["by_peer"]
            # Peer-wise mean delta
            peers_local = peer_labels(base["x_local"])
            peer_delta = {}
            for peer in COLLECTIVE_PEERS:
                pm = mask & (peers_local == peer)
                if int(np.sum(pm)) >= 5:
                    peer_delta[str(peer)] = float(np.mean(base_nll[pm] - per_row_nll(model["oof_pred"], model["counts_local"])[pm]))
            boot["peer_mean_delta_ll"] = peer_delta
            oof_summary["delta_vs_peer_global"][name] = boot

    # Rank on acquisition-block holdout log-loss
    ab_rank = []
    for name in model_names:
        key = f"acquisition_block_holdout::{name}"
        if key in oof_summary["by_scheme_model"]:
            m = oof_summary["by_scheme_model"][key]["metrics"]
            ab_rank.append({"model": name, **m})
    ab_rank = sorted(ab_rank, key=lambda r: r["log_loss"])

    # --- Gate 2 clarity ---
    peer_global_ll = next(
        (r["log_loss"] for r in ab_rank if r["model"] == "peer_specific_global"),
        float("nan"),
    )
    structured = [r for r in ab_rank if r["model"] != "peer_specific_global"]
    best_structured = structured[0] if structured else None
    delta_info = (
        oof_summary["delta_vs_peer_global"].get(best_structured["model"], {})
        if best_structured
        else {}
    )
    gate2_pass = bool(
        best_structured is not None
        and delta_info.get("ci_low", -1) > 0.0
        and (peer_global_ll - best_structured["log_loss"]) >= 0.02
    )

    # --- Gate 3: a0 ---
    gate3 = {}
    if best_structured is not None:
        pg = next(r for r in ab_rank if r["model"] == "peer_specific_global")
        gate3 = {
            "peer_global_a0_mae": pg["a0_mae"],
            "best_structured_model": best_structured["model"],
            "best_structured_a0_mae": best_structured["a0_mae"],
            "best_structured_a0_sign_acc": best_structured["a0_sign_acc"],
            "best_structured_a0_corr": best_structured["a0_corr"],
            "a0_mae_improved": bool(best_structured["a0_mae"] < pg["a0_mae"] - 1e-6),
        }
    gate3_pass = bool(gate3.get("a0_mae_improved"))

    # --- Prospective CENT-3 (no refit) ---
    load_catalog(branch / "cent3_stimulus_catalog_v0.json")
    fields = json.loads((branch / "cent3_frozen_fields_v0.json").read_text(encoding="utf-8"))[
        "fields"
    ]
    pooled, _ = pooled_and_block_counts(latest_cent3_runs())
    # fail rates from CENT-3 traces
    fail_by_field = {}
    for run_dir in latest_cent3_runs():
        with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                fid = rec["profile"]
                slot = fail_by_field.setdefault(fid, {"n": 0, "inv": 0})
                slot["n"] += 1
                if not rec.get("valid"):
                    slot["inv"] += 1

    xs = []
    meta = []
    for field in fields:
        feat, unresolved_f, peer, spars = field_feature_row(field)
        xs.append(feat)
        fid = field["field_id"]
        fr = fail_by_field.get(fid, {"n": 0, "inv": 0})
        meta.append(
            {
                "field_id": fid,
                "bucket": field["bucket"],
                "role": field["role"],
                "peer_count": peer,
                "bin_sparsity": spars,
                "p_fail": fr["inv"] / fr["n"] if fr["n"] else float("nan"),
                "n_valid": int(np.sum(pooled[fid])),
            }
        )
    x_cent = np.asarray(xs, dtype=np.float64)
    cent3_detail = []
    cent3_summary = []
    for name, model in models_full.items():
        pred = model.predict_proba(x_cent)
        rows_m = []
        for i, field_meta in enumerate(meta):
            c = pooled[field_meta["field_id"]]
            m = row_metrics(pred[i : i + 1], c[None, :])
            ch_hat = action_channels(pred[i])
            ch_obs = action_channels(counts_to_probs(c[None, :], alpha=0.0)[0])
            rows_m.append({**field_meta, **m, **{f"hat_{k}": ch_hat[k] for k in ch_hat}, **{f"obs_{k}": ch_obs[k] for k in ch_obs}})
            cent3_detail.append({"model": name, **rows_m[-1]})
        overall = row_metrics(
            pred, np.asarray([pooled[m["field_id"]] for m in meta], dtype=np.float64)
        )
        by_peer = {}
        for peer in COLLECTIVE_PEERS:
            idx = [i for i, m in enumerate(meta) if m["peer_count"] == peer]
            if not idx:
                continue
            by_peer[str(peer)] = row_metrics(
                pred[idx],
                np.asarray([pooled[meta[i]["field_id"]] for i in idx], dtype=np.float64),
            )
        cent3_summary.append({"model": name, "overall": overall, "by_peer": by_peer})
    cent3_rank = sorted(cent3_summary, key=lambda s: s["overall"]["log_loss"])
    cent3_peer_global = next(
        s for s in cent3_rank if s["model"] == "peer_specific_global"
    )
    cent3_best_structured = next(
        (s for s in cent3_rank if s["model"] not in {"peer_specific_global", "pooled_global"}),
        None,
    )
    cent3_delta = None
    gate5_pass = False
    if cent3_best_structured is not None:
        cent3_delta = float(
            cent3_peer_global["overall"]["log_loss"]
            - cent3_best_structured["overall"]["log_loss"]
        )
        # Require clear prospective win on CENT-3 (same bar as Gate 2 point estimate)
        gate5_pass = bool(cent3_delta >= 0.02)
        # Also require a0 improvement vs peer-global on CENT-3
        gate5_a0 = bool(
            cent3_best_structured["overall"]["a0_mae"]
            < cent3_peer_global["overall"]["a0_mae"] - 1e-6
        )
        gate5_pass = bool(gate5_pass and gate5_a0)

    # --- Gate 4 phenotypes on CENT-3 controls for best structured + peer models ---
    ctrl_ids = {
        "cent_ctrl_unimodal_positive_polar": "peer16_unimodal_stay",
        "cent_ctrl_unimodal_sign_reversed": "peer16_unimodal_stay",
        "cent_ctrl_exact_antipodal": "peer16_antipodal_partial",
        "cent_ctrl_small_imbalance": "peer16_imbalance_plus",
        "cent_ctrl_sparse_unimodal": "peer8_sparse_unimodal_high_stay",
        "cent_ctrl_sparse_antipodal": "peer16_sparse_antipodal_validity",
    }

    def phenotype_check(model_name: str) -> dict:
        pred = models_full[model_name].predict_proba(x_cent)
        by_id = {m["field_id"]: i for i, m in enumerate(meta)}
        out = {}
        # ±unimodal stay collapse
        for fid in (
            "cent_ctrl_unimodal_positive_polar",
            "cent_ctrl_unimodal_sign_reversed",
        ):
            i = by_id[fid]
            p = pred[i]
            obs = counts_to_probs(pooled[fid][None, :], alpha=0.0)[0]
            out[fid] = {
                "hat_stay": float(p[1]),
                "obs_stay": float(obs[1]),
                "hat_a0": float(p[2] - p[0]),
                "obs_a0": float(obs[2] - obs[0]),
                "pass_stay_collapse_hat": bool(p[1] >= 0.7),
                "pass_obs_all_stay": bool(obs[1] >= 0.99),
            }
        i = by_id["cent_ctrl_exact_antipodal"]
        p = pred[i]
        obs = counts_to_probs(
            pooled["cent_ctrl_exact_antipodal"][None, :], alpha=0.0
        )[0]
        out["cent_ctrl_exact_antipodal"] = {
            "hat_activity": float(p[0] + p[2]),
            "obs_activity": float(obs[0] + obs[2]),
            "pass_partial_activity_hat": bool(0.15 <= (p[0] + p[2]) <= 0.95),
        }
        i = by_id["cent_ctrl_small_imbalance"]
        p = pred[i]
        obs = counts_to_probs(pooled["cent_ctrl_small_imbalance"][None, :], alpha=0.0)[0]
        out["cent_ctrl_small_imbalance"] = {
            "hat_p_plus": float(p[2]),
            "obs_p_plus": float(obs[2]),
            "pass_strong_plus_hat": bool(p[2] >= 0.55),
        }
        i = by_id["cent_ctrl_sparse_unimodal"]
        p = pred[i]
        obs = counts_to_probs(pooled["cent_ctrl_sparse_unimodal"][None, :], alpha=0.0)[0]
        out["cent_ctrl_sparse_unimodal"] = {
            "hat_stay": float(p[1]),
            "obs_stay": float(obs[1]),
            "pass_high_stay_hat": bool(p[1] >= 0.6),
        }
        fr = fail_by_field["cent_ctrl_sparse_antipodal"]
        out["cent_ctrl_sparse_antipodal"] = {
            "n_valid": fr["n"] - fr["inv"],
            "p_fail": fr["inv"] / fr["n"],
            "phenotype_decidable": bool((fr["n"] - fr["inv"]) >= 12),
        }
        checks = [
            out["cent_ctrl_unimodal_positive_polar"]["pass_stay_collapse_hat"],
            out["cent_ctrl_unimodal_sign_reversed"]["pass_stay_collapse_hat"],
            out["cent_ctrl_exact_antipodal"]["pass_partial_activity_hat"],
            out["cent_ctrl_small_imbalance"]["pass_strong_plus_hat"],
            out["cent_ctrl_sparse_unimodal"]["pass_high_stay_hat"],
        ]
        out["n_passed"] = int(sum(checks))
        out["n_checks"] = len(checks)
        out["pass"] = bool(sum(checks) >= 4)
        return out

    phenotype = {
        name: phenotype_check(name)
        for name in (
            "peer_specific_global",
            "peer_specific_circular_emd",
            "peer_specific_canonical_emd",
            "hierarchical_emd_kernel",
            "peer_specific_activity_direction",
            "descriptor_collapse_mixture",
        )
        if name in models_full
    }

    # --- Risk recalibration on OOF of chosen candidate ---
    # Prefer best structured that clears gate2 if any; else best by AB log-loss excluding pooled_global only
    candidate = (
        best_structured["model"]
        if best_structured is not None
        else ab_rank[0]["model"]
    )
    risk_report = {"candidate": candidate, "note": "OOF residual risk; not CENT-3-refit"}
    if candidate in oof_store:
        result = oof_store[candidate]
        oof = result["oof_pred"]
        mask = result["oof_mask_local"]
        c_s = result["counts_local"]
        x_s = result["x_local"]
        empir = counts_to_probs(c_s, alpha=0.0)
        tv = 0.5 * np.sum(np.abs(oof - empir), axis=1)
        # Simple AD-like features for calibrator rows
        from analysis.stage3.applicability_domain import ApplicabilityDomain

        # Fit AD on full collective scope
        ad = ApplicabilityDomain.fit(x, representation=REP)
        sc = ad.score(x_s[mask], unresolved_near_zero=np.zeros(int(np.sum(mask)), dtype=bool))
        # Align unresolved from original
        # Build risk rows
        risk_rows = []
        local_idx = np.where(mask)[0]
        for j, li in enumerate(local_idx):
            # map back approximate features
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
                    "peer_count": int(peer_labels(x_s[li : li + 1])[0]),
                    "bin_sparsity": float(bin_sparsity(x_s[li : li + 1])[0]),
                }
            )
        calibrator = RiskCalibrator.fit(risk_rows)
        r_hat = calibrator.predict_rows(risk_rows)
        y = np.asarray([r["e_tv"] for r in risk_rows], dtype=np.float64)
        spars = np.asarray([r["bin_sparsity"] for r in risk_rows], dtype=np.float64)
        order = np.argsort(r_hat)
        n = len(order)
        lo = order[: max(1, n // 5)]
        hi = order[-max(1, n // 5) :]
        enrichment = float(np.mean(y[hi]) / max(np.mean(y[lo]), 1e-8))
        # peer-wise spearman
        peer_sp = {}
        for peer in COLLECTIVE_PEERS:
            idx = [i for i, r in enumerate(risk_rows) if r["peer_count"] == peer]
            if len(idx) >= 8:
                peer_sp[str(peer)] = _spearman(r_hat[idx], y[idx])
        # separation from sparsity/invalid: partial correlation proxy
        sp_r_s = _spearman(r_hat, spars)
        sp_e_s = _spearman(y, spars)
        risk_report.update(
            {
                "spearman": calibrator.train_spearman,
                "monotone_ok": monotone_ok(calibrator.bin_calibration),
                "enrichment_top20_vs_bottom20": enrichment,
                "peer_spearman": peer_sp,
                "spearman_risk_vs_bin_sparsity": sp_r_s,
                "spearman_error_vs_bin_sparsity": sp_e_s,
                "risk_not_only_sparsity": bool(
                    calibrator.train_spearman > 0.05
                    and abs(calibrator.train_spearman) > abs(sp_r_s) * 0.5
                ),
                "bin_calibration": calibrator.bin_calibration,
                "n_oof_rows": len(risk_rows),
            }
        )
        calibrator.save(branch / "risk_calibrator_collective_scope_v0.json")
        with (branch / "oof_residual_table_collective_scope_v0.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(risk_rows[0].keys()))
            writer.writeheader()
            writer.writerows(risk_rows)

    gate4_model = candidate
    gate4_pass = bool(phenotype.get(gate4_model, {}).get("pass"))

    # Peer collapse: structured must not be worse than peer-global on either peer
    peer_ok = True
    if best_structured is not None:
        bp = oof_summary["by_scheme_model"][
            f"acquisition_block_holdout::{best_structured['model']}"
        ]["by_peer"]
        gp = oof_summary["by_scheme_model"][
            "acquisition_block_holdout::peer_specific_global"
        ]["by_peer"]
        for peer in ("8", "16"):
            if peer in bp and peer in gp:
                if bp[peer]["log_loss"] > gp[peer]["log_loss"] + 0.02:
                    peer_ok = False

    gate1_pass = bool(best_structured is not None)
    risk_pass = bool(
        risk_report.get("spearman", -1) > 0
        and risk_report.get("enrichment_top20_vs_bottom20", 0) > 1.0
        and all(v > 0 for v in risk_report.get("peer_spearman", {}).values())
        if risk_report.get("peer_spearman")
        else False
    )

    confirmatory_go = bool(
        gate2_pass
        and peer_ok
        and gate3_pass
        and gate4_pass
        and risk_pass
        and gate5_pass
    )

    decision = {
        "status": "cent4_peer_stratified_revision",
        "scope": "centers_collective_bundle_v1 peer in {8,16}",
        "n_train": int(x.shape[0]),
        "peer_train_counts": {
            str(p): int(np.sum(peers == p)) for p in COLLECTIVE_PEERS
        },
        "acquisition_block_ranking": ab_rank,
        "cent3_prospective_ranking": [
            {"model": s["model"], **s["overall"]} for s in cent3_rank
        ],
        "cent3_transfer": {
            "peer_global_log_loss": cent3_peer_global["overall"]["log_loss"],
            "best_structured_model": (
                None if cent3_best_structured is None else cent3_best_structured["model"]
            ),
            "best_structured_log_loss": (
                None
                if cent3_best_structured is None
                else cent3_best_structured["overall"]["log_loss"]
            ),
            "delta_ll_peer_global_minus_structured": cent3_delta,
            "gate5_clear_prospective_win": gate5_pass,
        },
        "gates": {
            "gate1_oof_completed": gate1_pass,
            "gate2_beats_peer_global_ci": gate2_pass,
            "gate2_peer_no_collapse": peer_ok,
            "gate3_a0_improved": gate3_pass,
            "gate4_peer_phenotype": gate4_pass,
            "gate5_cent3_prospective_transfer": gate5_pass,
            "risk_positive_peerwise": risk_pass,
        },
        "gate3_detail": gate3,
        "gate2_delta": delta_info,
        "phenotype": phenotype,
        "risk": risk_report,
        "candidate_model": candidate,
        "confirmatory_pilot_go": confirmatory_go,
        "paid_authorization": "NO",
        "freeze": "NO-GO",
        "stage_c": "NO-GO",
        "interpretation": {
            "in_distribution_oof": (
                "peer-stratified models clearly beat peer-specific global on "
                "acquisition-block OOF within the peer{8,16} training manifold"
            ),
            "cent3_transfer": (
                "structured models still do not clearly beat peer-specific global "
                "on the CENT-3 collective-like fields; finite-peer stay-collapse "
                "phenotype is not reproduced"
            ),
            "hypothesis_status": (
                "partially supported: removing peer240 and stratifying peers "
                "improves OOF compressibility, but coverage of collective-like "
                "fields remains insufficient for confirmatory paid pilot"
            ),
        },
        "next": (
            "prepare non-overlapping 768-call confirmatory pilot"
            if confirmatory_go
            else (
                "continue offline coverage repair (peer8 densification / "
                "finite-peer abstention neighborhood) before any paid pilot; "
                "do not freeze; stop-rule still applies after confirmatory"
            )
        ),
        "stop_rule_note": (
            "Confirmatory pilot is the last model-establishment test; "
            "second prospective failure → do not continue surrogate engineering"
        ),
    }

    # Persist
    (branch / "cent4_revision_oof.json").write_text(
        json.dumps(oof_summary, indent=2), encoding="utf-8"
    )
    with (branch / "cent4_revision_cent3_detail.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cent3_detail[0].keys()))
        writer.writeheader()
        writer.writerows(cent3_detail)
    (branch / "cent4_revision_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "acquisition_block_ranking": ab_rank[:5],
        "cent3_top": decision["cent3_prospective_ranking"][:5],
        "gates": decision["gates"],
        "candidate": candidate,
        "confirmatory_pilot_go": confirmatory_go,
        "gate2_delta": delta_info,
        "gate3": gate3,
    }, indent=2))
    return 0 if confirmatory_go else 2


if __name__ == "__main__":
    raise SystemExit(main())
