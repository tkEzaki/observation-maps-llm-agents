"""Select ~36 Stage C collective fields for a fixed-field replay panel (no API)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.applicability_domain import ApplicabilityDomain  # noqa: E402
from analysis.stage3.collective_scan import _histogram_from_phases  # noqa: E402
from analysis.stage3.models import KernelHurdleMultinomial  # noqa: E402
from analysis.stage3.risk_calibrator import RiskCalibrator  # noqa: E402
from circlemap.field_features import extract_field_descriptors  # noqa: E402
from circlemap.representations import build_representation_prompt_from_histogram  # noqa: E402
from circlemap.observation import wrap_phase  # noqa: E402


def _rk(phases: np.ndarray, order: int = 1) -> float:
    return float(np.abs(np.mean(np.exp(1j * order * wrap_phase(phases)))))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pick_times(n_steps: int, kind: str) -> list[int]:
    if kind == "neg":
        return [2, n_steps // 2, n_steps - 1]
    if kind == "pos":
        # pre-lock / transition / locked proxies
        return [2, max(3, n_steps // 4), n_steps - 1]
    return [0, n_steps // 2, n_steps - 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        default=ROOT
        / "runs"
        / "stage_c"
        / "stage-c-v0.1_0291166d2e39"
        / "20260723T235620Z",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage_c_artifacts" / "replay_panel_v1",
    )
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    args = parser.parse_args()

    protocol = json.loads((args.session / "protocol.json").read_text(encoding="utf-8"))
    bundle = ROOT / protocol["bundle_root"]
    model = KernelHurdleMultinomial.load(bundle / "kernel_hurdle")
    ad = ApplicabilityDomain.load(bundle / "applicability_domain")
    calibrator = RiskCalibrator.load(bundle / "risk_calibrator.json")
    ood = json.loads((bundle / "ood_policy.json").read_text(encoding="utf-8"))
    risk_threshold = float(ood["risk"]["risk_threshold"])
    representation = protocol["representation"]

    # Target allocation
    plan = [
        ("neg", -0.15, 12),
        ("pos", 0.15, 12),
        ("zero", 0.0, 6),
    ]
    selected = []
    for kind, k_target, need in plan:
        run_dirs = [
            p
            for p in sorted(args.session.iterdir())
            if p.is_dir() and (p / "phases.npy").exists()
        ]
        cand_runs = []
        for run_dir in run_dirs:
            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            if abs(float(meta["coupling"]) - k_target) > 1e-12:
                continue
            cand_runs.append((run_dir, meta))
        # Round-robin over runs and times / agents
        times = _pick_times(int(protocol["n_steps"]), kind)
        pool = []
        for run_dir, meta in cand_runs:
            phases = np.load(run_dir / "phases.npy")
            for t in times:
                r1 = _rk(phases[t], 1)
                for agent in range(int(meta["n_agents"])):
                    pool.append((run_dir, meta, t, agent, r1, phases))
        # Diversify by (N, seed, t, r1 bin)
        rng = np.random.default_rng(2026072450 + int(round(abs(k_target) * 1000)))
        rng.shuffle(pool)
        seen = set()
        for run_dir, meta, t, agent, r1, phases in pool:
            key = (
                int(meta["n_agents"]),
                int(meta["seed_index"]),
                int(t),
                int(np.clip(r1, 0, 1) * 10),
            )
            if key in seen:
                continue
            hist = _histogram_from_phases(phases[t], agent)
            desc = extract_field_descriptors(representation, hist)
            x = np.concatenate([desc.common, desc.native])[None, :]
            scores = ad.score(x, unresolved_near_zero=np.asarray([desc.unresolved_near_zero]))
            risk = float(
                calibrator.predict_arrays(
                    d_shape=scores["d_shape"],
                    d_native=scores["d_native"],
                    local_density=scores["local_density"],
                    u_ensemble=np.zeros(1),
                    d_orientation=scores["d_orientation"],
                    policy_a=scores["policy_a"],
                )[0]
            )
            if bool(scores["g_peer"][0]):
                continue
            probs = model.predict_proba(x)[0]
            prompt = build_representation_prompt_from_histogram(representation, hist)
            field_id = (
                f"replay_{kind}_N{meta['n_agents']}_K{meta['coupling']:+g}"
                f"_s{meta['seed_index']}_t{t:02d}_a{agent:02d}"
            )
            selected.append(
                {
                    "field_id": field_id,
                    "source_bucket": kind,
                    "source_run_id": meta["run_id"],
                    "n_agents": meta["n_agents"],
                    "coupling": meta["coupling"],
                    "seed_index": meta["seed_index"],
                    "t": t,
                    "agent": agent,
                    "r1": r1,
                    "r2": _rk(phases[t], 2),
                    "r3": _rk(phases[t], 3),
                    "peer_count": int(hist.peer_count),
                    "fixed_fractions": [float(v) for v in hist.fractions],
                    "pred_p_retard": float(probs[0]),
                    "pred_p_stay": float(probs[1]),
                    "pred_p_advance": float(probs[2]),
                    "R": risk,
                    "high_risk": bool(risk > risk_threshold),
                    "prompt_sha256": _sha(prompt),
                    "physical_hash": _sha(
                        json.dumps(
                            {
                                "fractions": [float(v) for v in hist.fractions],
                                "peer_count": int(hist.peer_count),
                            },
                            sort_keys=True,
                        )
                    ),
                }
            )
            seen.add(key)
            if sum(1 for s in selected if s["source_bucket"] == kind) >= need:
                break

    # Stage B anchors (6) from pilot catalog controls / AL — fixed ids
    anchors = [
        "control_exact_antipodal_8_8_r00",
        "control_small_imbalance_9_7_r00",
        "control_lowrisk_sparse_N8_unimodal_k6",
        "control_sparse_N16_antipodal_equal",
        "al_f00_n17_p16",
        "al_f10_n9_p8",
    ]
    catalog = json.loads(
        (ROOT / "analysis" / "stage3a_artifacts" / "pilot_stimulus_catalog_v1.json").read_text(
            encoding="utf-8"
        )
    )
    for aid in anchors:
        stim = catalog[aid]
        fr = [float(v) for v in stim["fixed_fractions"]]
        peer = int(stim["peer_count"])
        edges = np.linspace(-np.pi, np.pi, 25)
        from circlemap.observation import RelativePhaseHistogram

        hist = RelativePhaseHistogram(
            edges=edges, fractions=np.asarray(fr), peer_count=peer
        )
        desc = extract_field_descriptors(representation, hist)
        x = np.concatenate([desc.common, desc.native])[None, :]
        scores = ad.score(x, unresolved_near_zero=np.asarray([desc.unresolved_near_zero]))
        risk = float(
            calibrator.predict_arrays(
                d_shape=scores["d_shape"],
                d_native=scores["d_native"],
                local_density=scores["local_density"],
                u_ensemble=np.zeros(1),
                d_orientation=scores["d_orientation"],
                policy_a=scores["policy_a"],
            )[0]
        )
        probs = model.predict_proba(x)[0]
        prompt = build_representation_prompt_from_histogram(representation, hist)
        selected.append(
            {
                "field_id": f"replay_anchor_{aid}",
                "source_bucket": "stage_b_anchor",
                "source_run_id": "",
                "n_agents": peer + 1,
                "coupling": float("nan"),
                "seed_index": -1,
                "t": -1,
                "agent": -1,
                "r1": float("nan"),
                "r2": float("nan"),
                "r3": float("nan"),
                "peer_count": peer,
                "fixed_fractions": fr,
                "pred_p_retard": float(probs[0]),
                "pred_p_stay": float(probs[1]),
                "pred_p_advance": float(probs[2]),
                "R": risk,
                "high_risk": bool(risk > risk_threshold),
                "prompt_sha256": _sha(prompt),
                "physical_hash": _sha(
                    json.dumps({"fractions": fr, "peer_count": peer}, sort_keys=True)
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Dedup by physical_hash
    uniq = []
    seen_h = set()
    for row in selected:
        if row["physical_hash"] in seen_h:
            continue
        seen_h.add(row["physical_hash"])
        uniq.append(row)

    fields_path = args.out_dir / "replay_fields_v1.json"
    fields_path.write_text(
        json.dumps({"n_fields": len(uniq), "fields": uniq}, indent=2),
        encoding="utf-8",
    )
    # CSV without fractions
    csv_rows = []
    for row in uniq:
        csv_rows.append({k: v for k, v in row.items() if k != "fixed_fractions"})
    with (args.out_dir / "replay_fields_v1.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    n_fields = len(uniq)
    expected_calls = n_fields * args.reps * args.blocks
    # rough cost from Stage C moments ~200 tokens in / 19 out
    est = expected_calls * (200 * 0.75 + 20 * 4.5) / 1_000_000
    protocol = {
        "protocol_name": "stage_c_collective_field_replay",
        "protocol_version": "replay-v0.1",
        "purpose": "measure p_stay on Stage-C-realized fields; keep Stage C v0.1 prospective",
        "representation": "moments_m1_m3",
        "fields_file": str(fields_path.relative_to(ROOT).as_posix()),
        "n_fields": n_fields,
        "repetitions_per_condition": args.reps,
        "n_blocks": args.blocks,
        "expected_calls": expected_calls,
        "prompt_version": "response-law-v0.1",
        "reuse_stage_c_actions_as_labels": False,
        "independent_seeds": True,
        "bucket_counts": {
            b: sum(1 for r in uniq if r["source_bucket"] == b)
            for b in sorted({r["source_bucket"] for r in uniq})
        },
    }
    proto_path = (
        ROOT / "experiments" / "stage_c" / "protocol_collective_replay_v0_1.json"
    )
    proto_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "n_fields": n_fields,
                "expected_calls": expected_calls,
                "est_cost_usd_rough": round(est, 4),
                "bucket_counts": protocol["bucket_counts"],
                "fields": str(fields_path.relative_to(ROOT).as_posix()),
                "protocol": str(proto_path.relative_to(ROOT).as_posix()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
