"""Build cross-representation synthesis tables and figures (offline)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "analysis" / "synthesis"
OUT.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fig1_transmutation_table() -> None:
    """Stage B κ trajectories (from STAGE_B_TRANSMUTATION_MAP narrative)."""
    # Across-block mean a1 / a2 / a0 as published in the transmutation map.
    rows = []
    kappas = [2, 4, 6, 9, 12]
    intervals_a1 = [0.665, 0.421, 0.251, -0.026, 0.082]
    intervals_a2 = [0.066, -0.038, 0.008, 0.432, 0.365]
    moments_a1 = [0.981, 0.984, 0.856, 0.977, 0.998]
    moments_a2 = [0.0, 0.0, 0.0, 0.0, 0.0]  # not the focus; near-polar
    centers_a1 = [0.682, 0.235, -0.045, -0.142, -0.392]
    centers_a2 = [-0.052, -0.189, -0.181, -0.035, -0.057]
    intervals_a0 = [0.080, 0.120, 0.160, 0.200, 0.249]  # approximate span from doc
    moments_a0 = [0.0, 0.0, 0.05, 0.0, 0.0]
    centers_a0 = [-0.368, -0.340, -0.310, -0.280, -0.253]
    for i, k in enumerate(kappas):
        for rep, a1, a2, a0 in (
            ("intervals_24_decimal6", intervals_a1[i], intervals_a2[i], intervals_a0[i]),
            ("moments_m1_m3", moments_a1[i], moments_a2[i], moments_a0[i]),
            ("centers_24_standard", centers_a1[i], centers_a2[i], centers_a0[i]),
        ):
            rows.append(
                {
                    "kappa": k,
                    "representation": rep,
                    "a1": a1,
                    "a2": a2,
                    "a0": a0,
                    "note": "from STAGE_B_TRANSMUTATION_MAP (a0 intervals/moments approximate span)",
                }
            )
    _write_csv(OUT / "fig1_transmutation.csv", rows)


def fig2_compressibility() -> None:
    centers = json.loads(
        (ROOT / "analysis/centers_branch/cent4_final_offline_decision.json").read_text(
            encoding="utf-8"
        )
    )
    intervals = json.loads(
        (ROOT / "analysis/intervals_branch/int2b_offline_decision.json").read_text(
            encoding="utf-8"
        )
    )
    c_pf = centers["cluster_bootstrap"]["peer_specific_activity_direction"][
        "physical_field_cluster"
    ]
    i_pf = intervals["full_collective"]["cluster_bootstrap"][
        "peer_specific_activity_direction"
    ]["physical_field_cluster"]
    rows = [
        {
            "representation": "centers_24_standard",
            "scope": "peer_{8,16}",
            "model": "peer_specific_activity_direction",
            "baseline": "peer_specific_global",
            "delta_ll_mean": c_pf["mean_delta_ll"],
            "ci_low_field_cluster": c_pf["ci_low"],
            "ci_high_field_cluster": c_pf["ci_high"],
            "ci_entirely_positive": c_pf["ci_entirely_positive"],
            "source": "cent4_final_offline_decision.json",
            "note": "",
        },
        {
            "representation": "intervals_24_decimal6",
            "scope": "peer_{8,16}",
            "model": "peer_specific_activity_direction",
            "baseline": "peer_specific_global",
            "delta_ll_mean": i_pf["mean_delta_ll"],
            "ci_low_field_cluster": i_pf["ci_low"],
            "ci_high_field_cluster": i_pf["ci_high"],
            "ci_entirely_positive": i_pf["ci_entirely_positive"],
            "source": "int2b_offline_decision.json",
            "note": "",
        },
        {
            "representation": "moments_m1_m3",
            "scope": "moments_training_manifold",
            "model": "regime_aware_hurdle_v2",
            "baseline": "moments_bundle_v1 / global-like leakage",
            "delta_ll_mean": "",
            "ci_low_field_cluster": "",
            "ci_high_field_cluster": "",
            "ci_entirely_positive": "",
            "source": "STAGE_3B_MOMENTS_V2 / STAGE_C_RESULTS_V0_2_COMBINED",
            "note": "high C via Stage 3 gates + Stage C activity/stay; no peer-global ΔLL row in this table",
        },
    ]
    _write_csv(OUT / "fig2_compressibility.csv", rows)


def fig3_funnel() -> None:
    rows = [
        {
            "representation": "moments_m1_m3",
            "stage_b_labels": "pass",
            "grouped_oof": "pass",
            "support_or_replay": "pass_after_regime_revision",
            "prospective_collective": "pass_v0_2",
            "disposition": "frozen_stage_c",
        },
        {
            "representation": "centers_24_standard",
            "stage_b_labels": "pass",
            "grouped_oof": "pass",
            "support_or_replay": "fail_feature_gap_heterogeneity",
            "prospective_collective": "fail_cent3_dev_transfer",
            "disposition": "stopped_microscopic_archive",
        },
        {
            "representation": "intervals_24_decimal6",
            "stage_b_labels": "pass",
            "grouped_oof": "pass",
            "support_or_replay": "fail_local_noncompressible",
            "prospective_collective": "not_attempted_cancelled",
            "disposition": "stopped_microscopic_archive",
        },
    ]
    _write_csv(OUT / "transport_funnel.csv", rows)
    _write_csv(OUT / "fig3_funnel.csv", rows)


def fig4_support_scatter() -> None:
    """Combine centers/intervals support CSVs if present."""
    paths = [
        (
            "intervals",
            ROOT
            / "analysis"
            / "intervals_branch"
            / "int2b_support_detail_peer16.csv",
        ),
        (
            "centers",
            ROOT / "analysis" / "centers_branch" / "cent4_final_local_support.csv",
        ),
    ]
    rows = []
    for rep, path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                d_nn = row.get("d_NN") or row.get("nearest_emd")
                v_loc = row.get("V_local") or row.get("neighbor_response_mean_tv")
                diag = row.get("diagnosis") or row.get("diagnosis_refined") or ""
                # Normalize diagnosis labels
                dlow = diag.lower()
                if "noncompress" in dlow:
                    bucket = "local_noncompressible"
                elif "feature_gap" in dlow or "neighbors_active" in dlow:
                    bucket = "active_feature_gap"
                elif "support" in dlow or dlow in {"supported", "mixed"}:
                    bucket = "supported"
                elif "no_neighbor" in dlow or "coverage" in dlow:
                    bucket = "no_neighbor"
                else:
                    bucket = diag or "other"
                if d_nn is None or v_loc is None or d_nn == "" or v_loc == "":
                    continue
                rows.append(
                    {
                        "representation": rep,
                        "d_NN": float(d_nn),
                        "V_local": float(v_loc),
                        "diagnosis_bucket": bucket,
                        "peer_count": row.get("peer_count", ""),
                        "profile_family": row.get("profile_family")
                        or row.get("coarse_family")
                        or "",
                    }
                )
    if rows:
        _write_csv(OUT / "fig4_support_scatter.csv", rows)


def try_plots() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; CSV tables only")
        return

    # Fig 1
    fig1 = list(csv.DictReader((OUT / "fig1_transmutation.csv").open(encoding="utf-8")))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True)
    for ax, metric in zip(axes, ("a1", "a2", "a0")):
        for rep, color in (
            ("intervals_24_decimal6", "#1b9e77"),
            ("moments_m1_m3", "#d95f02"),
            ("centers_24_standard", "#7570b3"),
        ):
            xs = [float(r["kappa"]) for r in fig1 if r["representation"] == rep]
            ys = [float(r[metric]) for r in fig1 if r["representation"] == rep]
            ax.plot(xs, ys, "o-", label=rep.split("_")[0], color=color)
        ax.set_title(metric)
        ax.set_xlabel("κ")
        ax.axhline(0.0, color="0.7", lw=0.8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Figure 1 — Microscopic transmutation (Stage B)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_transmutation.png", dpi=160)
    plt.close(fig)

    # Fig 2
    fig2 = [
        r
        for r in csv.DictReader((OUT / "fig2_compressibility.csv").open(encoding="utf-8"))
        if r["delta_ll_mean"]
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ylabels = []
    for i, r in enumerate(fig2):
        mean = float(r["delta_ll_mean"])
        lo = float(r["ci_low_field_cluster"])
        hi = float(r["ci_high_field_cluster"])
        ax.errorbar(
            mean,
            i,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            capsize=4,
        )
        ylabels.append(r["representation"].split("_")[0])
    ax.axvline(0.0, color="0.5", ls="--")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("ΔLL vs peer-specific global (field-cluster 95% CI)")
    ax.set_title("Figure 2 — IID compressibility")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_compressibility.png", dpi=160)
    plt.close(fig)

    # Fig 3 funnel as categorical heatmap-like table plot
    funnel = list(csv.DictReader((OUT / "fig3_funnel.csv").open(encoding="utf-8")))
    stages = [
        "stage_b_labels",
        "grouped_oof",
        "support_or_replay",
        "prospective_collective",
    ]
    score = {
        "pass": 2,
        "pass_after_regime_revision": 2,
        "pass_v0_2": 2,
        "fail_feature_gap_heterogeneity": 0,
        "fail_cent3_dev_transfer": 0,
        "fail_local_noncompressible": 0,
        "not_attempted_cancelled": 0,
    }
    mat = np.zeros((len(funnel), len(stages)))
    reps = []
    for i, row in enumerate(funnel):
        reps.append(row["representation"].split("_")[0])
        for j, s in enumerate(stages):
            mat[i, j] = score.get(row[s], 1)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=2)
    ax.set_yticks(range(len(reps)))
    ax.set_yticklabels(reps)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(
        ["Stage B", "Grouped OOF", "Support/Replay", "Prospective"], rotation=20
    )
    ax.set_title("Figure 3 — Transportability funnel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_funnel.png", dpi=160)
    plt.close(fig)

    # Fig 4
    scatter_path = OUT / "fig4_support_scatter.csv"
    if scatter_path.exists():
        pts = list(csv.DictReader(scatter_path.open(encoding="utf-8")))
        colors = {
            "local_noncompressible": "#d73027",
            "active_feature_gap": "#fc8d59",
            "supported": "#1a9850",
            "no_neighbor": "#4575b4",
            "other": "#999999",
        }
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
        for ax, rep in zip(axes, ("centers", "intervals")):
            sub = [p for p in pts if p["representation"] == rep]
            for bucket, color in colors.items():
                xs = [float(p["d_NN"]) for p in sub if p["diagnosis_bucket"] == bucket]
                ys = [float(p["V_local"]) for p in sub if p["diagnosis_bucket"] == bucket]
                if xs:
                    ax.scatter(xs, ys, s=12, alpha=0.55, c=color, label=bucket)
            ax.set_title(rep)
            ax.set_xlabel(r"$d_{\mathrm{NN}}$")
            ax.set_ylabel(r"$V_{\mathrm{local}}$")
            ax.axhline(0.35, color="0.6", ls="--", lw=0.8)
            ax.axvline(0.35, color="0.6", ls="--", lw=0.8)
        axes[1].legend(fontsize=6, loc="upper right")
        fig.suptitle("Figure 4 — Local support diagnosis")
        fig.tight_layout()
        fig.savefig(OUT / "fig4_support_scatter.png", dpi=160)
        plt.close(fig)

    print(f"Wrote PNGs under {OUT}")


def ct_summary() -> None:
    rows = [
        {
            "representation": "moments_m1_m3",
            "C_compressibility": "high",
            "T_transportability": "high_after_regime_revision",
            "disposition": "frozen_stage_c",
        },
        {
            "representation": "centers_24_standard",
            "C_compressibility": "high",
            "T_transportability": "low",
            "disposition": "stopped_microscopic_archive",
        },
        {
            "representation": "intervals_24_decimal6",
            "C_compressibility": "high",
            "T_transportability": "low",
            "disposition": "stopped_microscopic_archive",
        },
    ]
    _write_csv(OUT / "compressibility_transportability.csv", rows)


def main() -> int:
    fig1_transmutation_table()
    fig2_compressibility()
    fig3_funnel()
    fig4_support_scatter()
    ct_summary()
    try_plots()
    meta = {
        "status": "synthesis_figures_v0",
        "central_claim": (
            "Microscopic representation effects and in-distribution predictability "
            "do not guarantee transportability of a response operator to an "
            "endogenous collective-field manifold."
        ),
        "docs": "docs/REPRESENTATION_TRANSPORT_SYNTHESIS.md",
        "outputs": sorted(
            p.name for p in OUT.iterdir() if p.name != "meta.json"
        ),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
