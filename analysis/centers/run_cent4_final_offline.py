"""CENT-4 final bounded offline attempt.

1) Cluster bootstrap of ΔLL (field / profile-family / acquisition)
2) Local-support diagnosis on CENT-3 fields
3) Discrete bin features
4) Peer16-only (N=17) bundle viability
5) Final GO / STOP decision (no paid calls)
"""

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
    histogram_from_field,
    latest_cent3_runs,
    load_catalog,
    pooled_and_block_counts,
)
from analysis.centers.discrete_bin_features import (  # noqa: E402
    DISCRETE_FEATURE_NAMES,
    discrete_bin_features_from_native,
)
from analysis.centers.distances import circular_emd_pairwise  # noqa: E402
from analysis.centers.peer_stratified_models import (  # noqa: E402
    CircularEmdKernel,
    PeerSpecificGlobal,
    peer_labels,
)
from analysis.centers.run_cent4_revision import (  # noqa: E402
    fit_named,
    per_row_nll,
    row_metrics,
)
from analysis.stage3.applicability_domain import N_COMMON  # noqa: E402
from analysis.stage3.models import (  # noqa: E402
    KernelHurdleMultinomial,
    counts_to_probs,
)
from analysis.stage3.validation import leave_one_group_out_predictions  # noqa: E402


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


def cluster_bootstrap_delta(
    delta_row: np.ndarray,
    cluster_ids: list[str],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Resample clusters with replacement; average row-Δ within drawn clusters."""
    clusters = sorted(set(cluster_ids))
    if len(clusters) < 3:
        return {
            "mean_delta_ll": float(np.mean(delta_row)),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_clusters": len(clusters),
        }
    by = {c: np.where(np.asarray(cluster_ids) == c)[0] for c in clusters}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        vals = []
        for c in drawn:
            vals.append(float(np.mean(delta_row[by[c]])))
        means.append(float(np.mean(vals)))
    arr = np.sort(np.asarray(means, dtype=np.float64))
    return {
        "mean_delta_ll": float(np.mean(delta_row)),
        "ci_low": float(arr[int(0.025 * n_boot)]),
        "ci_high": float(arr[int(0.975 * n_boot)]),
        "n_clusters": len(clusters),
        "ci_entirely_positive": bool(arr[int(0.025 * n_boot)] > 0.0),
    }


def ab_oof_pair(
    x: np.ndarray,
    counts: np.ndarray,
    blocks: list[str],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (oof_base_nll, oof_model_nll, mask) for peer-global vs named model."""

    def factory_global(xt, ct):
        return PeerSpecificGlobal.fit(xt, ct)

    def factory_model(xt, ct, _n=model_name):
        return fit_named(_n, xt, ct)

    oof_g, _ = leave_one_group_out_predictions(factory_global, x, counts, blocks)
    oof_m, _ = leave_one_group_out_predictions(factory_model, x, counts, blocks)
    mask = np.isfinite(oof_g[:, 0]) & np.isfinite(oof_m[:, 0])
    return per_row_nll(oof_g, counts), per_row_nll(oof_m, counts), mask


def local_support_row(
    query_x: np.ndarray,
    train_x: np.ndarray,
    train_counts: np.ndarray,
    *,
    peer: int,
    k: int = 16,
) -> dict:
    peers = peer_labels(train_x)
    mask = peers == peer
    if int(np.sum(mask)) < 3:
        return {
            "n_peer_neighbors_available": int(np.sum(mask)),
            "diagnosis": "no_peer_support",
        }
    xt = train_x[mask]
    ct = train_counts[mask]
    native_q = query_x[None, N_COMMON : N_COMMON + 24]
    native_t = xt[:, N_COMMON : N_COMMON + 24]
    d_emd = circular_emd_pairwise(native_q, native_t)[0]
    # canonical Euclidean on standardized common
    common_q = query_x[:N_COMMON]
    common_t = xt[:, :N_COMMON]
    mean = np.mean(common_t, axis=0)
    scale = np.std(common_t, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    d_can = np.sqrt(
        np.sum(((common_q - mean) / scale - (common_t - mean) / scale) ** 2, axis=1)
    )
    order = np.argsort(d_emd)
    kk = min(k, order.size)
    idx = order[:kk]
    empir = counts_to_probs(ct[idx], alpha=0.0)
    stay = empir[:, 1]
    a0 = empir[:, 2] - empir[:, 0]
    activity = empir[:, 0] + empir[:, 2]
    # Response dispersion: mean pairwise TV among neighbor empiricinals
    tvs = []
    for i in range(kk):
        for j in range(i + 1, kk):
            tvs.append(0.5 * float(np.sum(np.abs(empir[i] - empir[j]))))
    mean_tv = float(np.mean(tvs)) if tvs else 0.0
    nearest = float(d_emd[idx[0]])
    frac_high_stay = float(np.mean(stay >= 0.55))
    frac_active = float(np.mean(activity >= 0.55))

    if nearest > 0.35 and kk < 5:
        diagnosis = "coverage_gap"
    elif frac_high_stay < 0.25 and frac_active >= 0.5:
        diagnosis = "neighbors_active_feature_gap"
    elif frac_high_stay >= 0.25 and mean_tv < 0.25:
        diagnosis = "neighbors_have_stay_model_bias_if_pred_active"
    elif mean_tv >= 0.35:
        diagnosis = "local_noncompressibility"
    else:
        diagnosis = "mixed_or_partial_support"

    return {
        "n_peer_neighbors_available": int(np.sum(mask)),
        "k_used": kk,
        "nearest_emd": nearest,
        "mean_emd_k": float(np.mean(d_emd[idx])),
        "nearest_canonical": float(d_can[idx[0]]),
        "mean_canonical_k": float(np.mean(d_can[idx])),
        "neighbor_mean_stay": float(np.mean(stay)),
        "neighbor_mean_a0": float(np.mean(a0)),
        "neighbor_mean_activity": float(np.mean(activity)),
        "neighbor_stay_std": float(np.std(stay)),
        "neighbor_a0_std": float(np.std(a0)),
        "neighbor_response_mean_tv": mean_tv,
        "frac_neighbors_high_stay": frac_high_stay,
        "frac_neighbors_active": frac_active,
        "diagnosis": diagnosis,
    }


class _DiscreteAugmentedEmd:
    """Canonical + circular-EMD kernel with discrete bin features in the canonical side."""

    name: str

    @classmethod
    def fit(cls, x: np.ndarray, counts: np.ndarray) -> "_DiscreteAugmentedEmd":
        native = x[:, N_COMMON:]
        common = x[:, :N_COMMON]
        disc = discrete_bin_features_from_native(native)
        from analysis.centers.peer_stratified_models import _standardize_fit

        xs, mean, scale = _standardize_fit(np.concatenate([common, disc], axis=1))
        obj = cls.__new__(cls)
        obj.name = "peer16_canonical_emd_plus_discrete"
        obj.common_train = xs
        obj.common_mean = mean
        obj.common_scale = scale
        obj.native_train = native.astype(np.float64)
        obj.p_train = counts_to_probs(counts, alpha=0.5)
        obj.length_scale_native = 0.75
        obj.length_scale_common = 1.5
        obj.k_neighbors = 32
        obj.n_disc = disc.shape[1]
        return obj

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        from analysis.centers.peer_stratified_models import _standardize_apply

        native = x[:, N_COMMON:]
        common = x[:, :N_COMMON]
        disc = discrete_bin_features_from_native(native)
        xs = _standardize_apply(
            np.concatenate([common, disc], axis=1), self.common_mean, self.common_scale
        )
        n = x.shape[0]
        out = np.empty((n, 3), dtype=np.float64)
        k = min(self.k_neighbors, self.native_train.shape[0])
        inv_n = 1.0 / (self.length_scale_native**2)
        inv_c = 1.0 / (self.length_scale_common**2)
        for start in range(0, n, 64):
            stop = min(start + 64, n)
            d_n = circular_emd_pairwise(native[start:stop], self.native_train)
            diff = xs[start:stop, None, :] - self.common_train[None, :, :]
            dist2 = (d_n**2) * inv_n + np.sum(diff * diff, axis=-1) * inv_c
            if k < self.native_train.shape[0]:
                idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
                rows = np.arange(stop - start)[:, None]
                dist2_k = dist2[rows, idx]
                p_k = self.p_train[idx]
            else:
                dist2_k = dist2
                p_k = np.repeat(self.p_train[None, :, :], stop - start, axis=0)
            w = np.exp(-0.5 * dist2_k)
            w = w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-12)
            out[start:stop] = np.sum(w[:, :, None] * p_k, axis=1)
        return out


def stay_collapse_family_holdout(
    x: np.ndarray,
    counts: np.ndarray,
    families: list[str],
    peers: np.ndarray,
) -> dict:
    """Hold out unimodal-like families; check predicted stay on held-out rows."""
    # Families that look unimodal / sparse-unimodal
    targets = {
        i
        for i, f in enumerate(families)
        if ("unimodal" in f.lower() or f.startswith("kappa_")) and int(peers[i]) == 16
    }
    # Peer16 training may have few unimodal — also use high abs_z1 & low antipodal
    from circlemap.field_features import COMMON_FEATURE_NAMES

    abs_z1 = x[:, COMMON_FEATURE_NAMES.index("abs_z1")]
    bal = x[:, COMMON_FEATURE_NAMES.index("antipodal_balance")]
    auto = {
        i
        for i in range(len(families))
        if int(peers[i]) == 16 and abs_z1[i] > 0.85 and bal[i] < 0.35
    }
    hold_idx = sorted(targets | auto)
    if len(hold_idx) < 8:
        return {"status": "insufficient_unimodal_peer16_rows", "n": len(hold_idx)}
    hold = np.zeros(len(families), dtype=bool)
    hold[hold_idx] = True
    train = ~hold
    model = CircularEmdKernel.fit(
        x[train], counts[train], use_canonical=True, k_neighbors=32
    )
    hurdle = KernelHurdleMultinomial.fit(x[train], counts[train], k_neighbors=32)
    disc = _DiscreteAugmentedEmd.fit(x[train], counts[train])
    pred_m = model.predict_proba(x[hold])
    pred_h = hurdle.predict_proba(x[hold])
    pred_d = disc.predict_proba(x[hold])
    empir = counts_to_probs(counts[hold], alpha=0.0)
    return {
        "status": "ok",
        "n_holdout": int(np.sum(hold)),
        "obs_mean_stay": float(np.mean(empir[:, 1])),
        "canonical_emd_mean_stay_hat": float(np.mean(pred_m[:, 1])),
        "hurdle_mean_stay_hat": float(np.mean(pred_h[:, 1])),
        "discrete_emd_mean_stay_hat": float(np.mean(pred_d[:, 1])),
        "frac_obs_stay_ge_0_7": float(np.mean(empir[:, 1] >= 0.7)),
        "frac_emd_hat_stay_ge_0_7": float(np.mean(pred_m[:, 1] >= 0.7)),
        "frac_hurdle_hat_stay_ge_0_7": float(np.mean(pred_h[:, 1] >= 0.7)),
        "frac_disc_hat_stay_ge_0_7": float(np.mean(pred_d[:, 1] >= 0.7)),
        "pass_emd": bool(
            np.mean(pred_m[:, 1]) >= 0.55 or np.mean(pred_m[:, 1] >= 0.7) >= 0.4
        ),
        "pass_hurdle": bool(
            np.mean(pred_h[:, 1]) >= 0.55 or np.mean(pred_h[:, 1] >= 0.7) >= 0.4
        ),
        "pass_discrete": bool(
            np.mean(pred_d[:, 1]) >= 0.55 or np.mean(pred_d[:, 1] >= 0.7) >= 0.4
        ),
    }


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
                    offset_index=int(row["offset_index"]),
                    source_family=row["source_family"],
                    peer_count=int(float(row["peer_count"])),
                )
            )

    blocks = [r.acquisition_block for r in rows]
    # Physical field cluster ≈ profile (stimulus id), independent of offset
    field_clusters = [r.profile for r in rows]
    family_clusters = [r.profile_family for r in rows]
    acq_clusters = [r.acquisition_block for r in rows]

    models_to_boot = [
        "peer_specific_activity_direction",
        "peer_specific_canonical_emd",
        "peer_specific_circular_emd",
    ]
    cluster_boot = {}
    print("Cluster bootstrap ΔLL vs peer-specific global...")
    for name in models_to_boot:
        print(f"  AB OOF for {name}")
        nll_g, nll_m, mask = ab_oof_pair(x, counts, blocks, name)
        delta = nll_g[mask] - nll_m[mask]
        fc = [field_clusters[i] for i, m in enumerate(mask) if m]
        fam = [family_clusters[i] for i, m in enumerate(mask) if m]
        acq = [acq_clusters[i] for i, m in enumerate(mask) if m]
        cluster_boot[name] = {
            "row_bootstrap_mean": float(np.mean(delta)),
            "physical_field_cluster": cluster_bootstrap_delta(delta, fc, seed=1),
            "profile_family_cluster": cluster_bootstrap_delta(delta, fam, seed=2),
            "acquisition_group_cluster": cluster_bootstrap_delta(delta, acq, seed=3),
        }

    # --- Peer16-only ---
    print("Peer16-only evaluation...")
    mask16 = peers == 16
    x16 = x[mask16]
    c16 = counts[mask16]
    rows16 = [rows[i] for i, m in enumerate(mask16) if m]
    blocks16 = [r.acquisition_block for r in rows16]
    peer16_rank = []
    peer16_models = {
        "peer16_global": PeerSpecificGlobal.fit(x16, c16),
        "peer16_circular_emd": CircularEmdKernel.fit(
            x16, c16, use_canonical=False, k_neighbors=32
        ),
        "peer16_canonical_emd": CircularEmdKernel.fit(
            x16, c16, use_canonical=True, k_neighbors=32
        ),
        "peer16_activity_direction": KernelHurdleMultinomial.fit(x16, c16, k_neighbors=32),
        "peer16_canonical_emd_plus_discrete": _DiscreteAugmentedEmd.fit(x16, c16),
    }

    def _oof_named(name, factory):
        oof, folds = leave_one_group_out_predictions(factory, x16, c16, blocks16)
        mask = np.isfinite(oof[:, 0])
        return row_metrics(oof[mask], c16[mask]), oof, mask

    oof16 = {}
    for name, model in peer16_models.items():
        # Refit per fold via factory
        if name == "peer16_global":

            def factory(xt, ct):
                return PeerSpecificGlobal.fit(xt, ct)

        elif name == "peer16_circular_emd":

            def factory(xt, ct):
                return CircularEmdKernel.fit(
                    xt, ct, use_canonical=False, k_neighbors=32
                )

        elif name == "peer16_canonical_emd":

            def factory(xt, ct):
                return CircularEmdKernel.fit(
                    xt, ct, use_canonical=True, k_neighbors=32
                )

        elif name == "peer16_activity_direction":

            def factory(xt, ct):
                return KernelHurdleMultinomial.fit(xt, ct, k_neighbors=32)

        else:

            def factory(xt, ct):
                return _DiscreteAugmentedEmd.fit(xt, ct)

        metrics, oof, mask = _oof_named(name, factory)
        peer16_rank.append({"model": name, **metrics})
        oof16[name] = (oof, mask)
    peer16_rank = sorted(peer16_rank, key=lambda r: r["log_loss"])

    # ΔLL peer16 global vs best structured
    nll_g16, _, mask_g = ab_oof_pair(
        x16,
        c16,
        blocks16,
        "peer_specific_global",  # works: PeerSpecificGlobal on peer16-only is fine
    )
    # Recompute properly with peer16 factories
    def fac_g(xt, ct):
        return PeerSpecificGlobal.fit(xt, ct)

    def fac_best(xt, ct):
        return KernelHurdleMultinomial.fit(xt, ct, k_neighbors=32)

    oof_g16, _ = leave_one_group_out_predictions(fac_g, x16, c16, blocks16)
    oof_h16, _ = leave_one_group_out_predictions(fac_best, x16, c16, blocks16)
    m16 = np.isfinite(oof_g16[:, 0]) & np.isfinite(oof_h16[:, 0])
    delta16 = per_row_nll(oof_g16, c16)[m16] - per_row_nll(oof_h16, c16)[m16]
    fc16 = [rows16[i].profile for i, f in enumerate(m16) if f]
    fam16 = [rows16[i].profile_family for i, f in enumerate(m16) if f]
    peer16_cluster = {
        "hurdle_vs_global_row_mean": float(np.mean(delta16)),
        "physical_field_cluster": cluster_bootstrap_delta(delta16, fc16, seed=11),
        "profile_family_cluster": cluster_bootstrap_delta(delta16, fam16, seed=12),
    }
    # Also EMD vs global
    def fac_emd(xt, ct):
        return CircularEmdKernel.fit(xt, ct, use_canonical=True, k_neighbors=32)

    oof_e16, _ = leave_one_group_out_predictions(fac_emd, x16, c16, blocks16)
    m_e = np.isfinite(oof_g16[:, 0]) & np.isfinite(oof_e16[:, 0])
    delta_e = per_row_nll(oof_g16, c16)[m_e] - per_row_nll(oof_e16, c16)[m_e]
    fc_e = [rows16[i].profile for i, f in enumerate(m_e) if f]
    peer16_cluster["canonical_emd_vs_global"] = {
        "row_mean": float(np.mean(delta_e)),
        "physical_field_cluster": cluster_bootstrap_delta(delta_e, fc_e, seed=13),
    }

    # Stay-collapse family holdout on peer16
    stay_hold = stay_collapse_family_holdout(
        x16,
        c16,
        [r.profile_family for r in rows16],
        peers[mask16],
    )

    # --- Local support on CENT-3 ---
    print("CENT-3 local-support diagnosis...")
    load_catalog(branch / "cent3_stimulus_catalog_v0.json")
    fields = json.loads((branch / "cent3_frozen_fields_v0.json").read_text(encoding="utf-8"))[
        "fields"
    ]
    pooled, _ = pooled_and_block_counts(latest_cent3_runs())

    # Fit full peer{8,16} models for predictions
    models_full = {
        "peer_specific_global": PeerSpecificGlobal.fit(x, counts),
        "peer_specific_canonical_emd": fit_named("peer_specific_canonical_emd", x, counts),
        "peer_specific_activity_direction": fit_named(
            "peer_specific_activity_direction", x, counts
        ),
        "peer16_global": PeerSpecificGlobal.fit(x16, c16),
        "peer16_canonical_emd": CircularEmdKernel.fit(
            x16, c16, use_canonical=True, k_neighbors=32
        ),
        "peer16_activity_direction": KernelHurdleMultinomial.fit(
            x16, c16, k_neighbors=32
        ),
        "peer16_discrete_emd": _DiscreteAugmentedEmd.fit(x16, c16),
    }

    support_rows = []
    for field in fields:
        feat, _u, peer, spars = field_feature_row(field)
        fid = field["field_id"]
        hist = histogram_from_field(field)
        disc = discrete_bin_features_from_native(hist.fractions)
        support = local_support_row(feat, x, counts, peer=peer, k=16)
        obs = counts_to_probs(pooled[fid][None, :], alpha=0.0)[0]
        ch = action_channels(obs)
        preds = {}
        for name, model in models_full.items():
            if name.startswith("peer16") and peer != 16:
                continue
            p = model.predict_proba(feat[None, :])[0]
            preds[name] = {
                "p_stay": float(p[1]),
                "a0": float(p[2] - p[0]),
                "A": float(p[0] + p[2]),
            }
        # Refine diagnosis using model prediction vs neighbor stay
        diag = support["diagnosis"]
        if diag == "neighbors_have_stay_model_bias_if_pred_active":
            hat = preds.get("peer_specific_canonical_emd", {}).get("p_stay", 0.0)
            if hat < 0.4 and support["neighbor_mean_stay"] >= 0.4:
                diag = "model_bias"
            elif hat >= 0.55:
                diag = "neighbors_support_stay_and_model_agrees"
        support_rows.append(
            {
                "field_id": fid,
                "bucket": field["bucket"],
                "role": field["role"],
                "peer_count": peer,
                "bin_sparsity": spars,
                "obs_stay": ch["p_stay"],
                "obs_a0": ch["a0"],
                "obs_A": ch["A"],
                **{f"disc_{n}": float(disc[i]) for i, n in enumerate(DISCRETE_FEATURE_NAMES)},
                **support,
                "diagnosis_refined": diag,
                "predictions": preds,
            }
        )

    # Control-focused summary
    controls = [
        r
        for r in support_rows
        if r["field_id"].startswith("cent_ctrl_")
    ]

    # CENT-3 development ranking for peer16-only models on peer16 fields
    peer16_fields = [f for f in fields if int(f["stimulus"]["peer_count"]) == 16]
    x_c16 = []
    c_c16 = []
    for f in peer16_fields:
        feat, _, _, _ = field_feature_row(f)
        x_c16.append(feat)
        c_c16.append(pooled[f["field_id"]])
    x_c16 = np.asarray(x_c16)
    c_c16 = np.asarray(c_c16, dtype=np.float64)
    cent3_peer16_rank = []
    for name, model in models_full.items():
        if not name.startswith("peer16") and name not in {
            "peer_specific_global",
            "peer_specific_canonical_emd",
            "peer_specific_activity_direction",
        }:
            continue
        # For peer_specific_* on peer16 fields ok; peer16_* ok
        if name.startswith("peer16") or True:
            pred = model.predict_proba(x_c16)
            cent3_peer16_rank.append({"model": name, **row_metrics(pred, c_c16)})
    cent3_peer16_rank = sorted(cent3_peer16_rank, key=lambda r: r["log_loss"])

    # Phenotype on controls with peer16 models
    def control_phenotype(model_name: str) -> dict:
        model = models_full[model_name]
        out = {}
        for fid in (
            "cent_ctrl_unimodal_positive_polar",
            "cent_ctrl_unimodal_sign_reversed",
            "cent_ctrl_exact_antipodal",
            "cent_ctrl_small_imbalance",
        ):
            field = next(f for f in fields if f["field_id"] == fid)
            feat, _, _, _ = field_feature_row(field)
            p = model.predict_proba(feat[None, :])[0]
            obs = counts_to_probs(pooled[fid][None, :], alpha=0.0)[0]
            out[fid] = {
                "hat_stay": float(p[1]),
                "obs_stay": float(obs[1]),
                "hat_a0": float(p[2] - p[0]),
                "obs_a0": float(obs[2] - obs[0]),
                "hat_p_plus": float(p[2]),
                "obs_p_plus": float(obs[2]),
                "hat_A": float(p[0] + p[2]),
                "obs_A": float(obs[0] + obs[2]),
            }
        checks = [
            out["cent_ctrl_unimodal_positive_polar"]["hat_stay"] >= 0.7,
            out["cent_ctrl_unimodal_sign_reversed"]["hat_stay"] >= 0.7,
            0.15 <= out["cent_ctrl_exact_antipodal"]["hat_A"] <= 0.95,
            out["cent_ctrl_small_imbalance"]["hat_p_plus"] >= 0.55,
        ]
        out["n_passed"] = int(sum(checks))
        out["pass"] = bool(sum(checks) >= 3)
        return out

    phenotype16 = {
        name: control_phenotype(name)
        for name in (
            "peer16_global",
            "peer16_canonical_emd",
            "peer16_activity_direction",
            "peer16_discrete_emd",
            "peer_specific_canonical_emd",
            "peer_specific_activity_direction",
        )
        if name in models_full
    }

    # --- Final gates ---
    # Cluster bootstrap: require all three clusterings CI>0 for OOF winner
    winner = "peer_specific_activity_direction"
    cb = cluster_boot[winner]
    gate_cluster = all(
        cb[k]["ci_entirely_positive"]
        for k in (
            "physical_field_cluster",
            "profile_family_cluster",
            "acquisition_group_cluster",
        )
        if "ci_entirely_positive" in cb[k]
    )
    # acquisition has only 2 clusters → CI nan; treat point mean>0 + other two CI>0
    gate_cluster = bool(
        cb["physical_field_cluster"].get("ci_entirely_positive")
        and cb["profile_family_cluster"].get("ci_entirely_positive")
        and cb["acquisition_group_cluster"]["mean_delta_ll"] > 0
    )

    gate_peer16_oof = bool(
        peer16_rank[0]["model"] != "peer16_global"
        and peer16_rank[0]["log_loss"] + 0.02
        < next(r["log_loss"] for r in peer16_rank if r["model"] == "peer16_global")
    )
    # Prefer cluster evidence for peer16 hurdle
    gate_peer16_cluster = bool(
        peer16_cluster["physical_field_cluster"].get("ci_entirely_positive")
    )

    best_c3 = cent3_peer16_rank[0]
    c3_global = next(
        r
        for r in cent3_peer16_rank
        if r["model"] in {"peer16_global", "peer_specific_global"}
    )
    # Compare best structured among peer16_* and peer_specific_* excluding globals
    structured_c3 = [
        r
        for r in cent3_peer16_rank
        if "global" not in r["model"]
    ]
    best_struct_c3 = structured_c3[0] if structured_c3 else best_c3
    gate_c3_dev = bool(
        best_struct_c3["log_loss"] + 0.02 < c3_global["log_loss"]
        and best_struct_c3["a0_mae"] < c3_global["a0_mae"]
    )

    gate_stay = bool(
        stay_hold.get("pass_emd")
        or stay_hold.get("pass_hurdle")
        or stay_hold.get("pass_discrete")
        or any(phenotype16[n].get("pass") for n in phenotype16)
    )

    # Diagnosis histogram for unimodal controls
    uni_diag = {
        r["field_id"]: r["diagnosis_refined"]
        for r in support_rows
        if "unimodal" in r["field_id"]
    }

    confirmatory_go = bool(
        gate_cluster
        and (gate_peer16_oof and gate_peer16_cluster)
        and gate_c3_dev
        and gate_stay
    )

    stop_centers = bool(
        (not gate_c3_dev)
        and (not gate_stay)
        and (
            best_struct_c3["a0_mae"] >= c3_global["a0_mae"] - 1e-6
            or best_struct_c3["a0_sign_acc"] <= c3_global.get("a0_sign_acc", 0) + 0.05
        )
    )

    decision = {
        "status": "cent4_final_bounded_offline",
        "cent3_role": "development-transfer benchmark (not final prospective evidence)",
        "cluster_bootstrap": cluster_boot,
        "peer16_only": {
            "n_train": int(np.sum(mask16)),
            "acquisition_block_ranking": peer16_rank,
            "cluster_delta": peer16_cluster,
            "bundle_name": "centers_collective_bundle_v1_N17",
            "intended_N": 17,
        },
        "stay_collapse_family_holdout": stay_hold,
        "cent3_peer16_development_ranking": cent3_peer16_rank,
        "phenotype_peer16_models": phenotype16,
        "unimodal_control_diagnosis": uni_diag,
        "gates": {
            "cluster_bootstrap_oof_positive": gate_cluster,
            "peer16_oof_beats_global": gate_peer16_oof,
            "peer16_cluster_bootstrap_positive": gate_peer16_cluster,
            "cent3_dev_beats_global": gate_c3_dev,
            "stay_collapse_reproduced": gate_stay,
        },
        "confirmatory_pilot_go": confirmatory_go,
        "peer16_only_pilot_go": bool(
            gate_cluster
            and gate_peer16_oof
            and gate_peer16_cluster
            and gate_c3_dev
            and gate_stay
        ),
        "stop_centers_collective_path": stop_centers,
        "paid_authorization": "NO",
        "freeze": "NO-GO",
        "stage_c": "NO-GO",
        "locked_conclusion_if_stop": (
            "Center-bin serialization induces reproducible microscopic "
            "transmutation and peer-regime dependence, but its finite-peer "
            "response surface was not sufficiently stable or compressible for "
            "prospective collective prediction under the tested descriptors and coverage."
        ),
        "next": (
            "draft peer16-only confirmatory protocol"
            if confirmatory_go
            else (
                "STOP centers collective surrogate path; move to intervals "
                "or archive centers as microscopic-only result"
                if stop_centers
                else "document residual offline options without further paid spend"
            )
        ),
    }

    # Persist support table (flatten predictions)
    flat_support = []
    for row in support_rows:
        flat = {k: v for k, v in row.items() if k != "predictions"}
        for mname, pred in row["predictions"].items():
            for pk, pv in pred.items():
                flat[f"pred_{mname}_{pk}"] = pv
        flat_support.append(flat)

    with (branch / "cent4_final_local_support.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_support[0].keys()))
        writer.writeheader()
        writer.writerows(flat_support)

    (branch / "cent4_final_offline_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    # Compact controls JSON
    (branch / "cent4_final_controls_support.json").write_text(
        json.dumps(controls, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "gates": decision["gates"],
                "confirmatory_pilot_go": confirmatory_go,
                "stop_centers_collective_path": stop_centers,
                "cluster_winner": cb,
                "peer16_top": peer16_rank[:4],
                "cent3_peer16_top": cent3_peer16_rank[:5],
                "stay_hold": stay_hold,
                "unimodal_diag": uni_diag,
                "next": decision["next"],
            },
            indent=2,
        )
    )
    return 0 if confirmatory_go else (3 if stop_centers else 2)


if __name__ == "__main__":
    raise SystemExit(main())
