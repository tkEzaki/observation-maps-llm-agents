"""Offline Stage C predictions using the hash-locked moments bundle."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.applicability_domain import ApplicabilityDomain  # noqa: E402
from analysis.stage3.models import KernelHurdleMultinomial  # noqa: E402
from analysis.stage3.risk_calibrator import RiskCalibrator  # noqa: E402
from circlemap.field_features import extract_field_descriptors  # noqa: E402
from circlemap.observation import wrap_phase  # noqa: E402
from circlemap.engine import step  # noqa: E402

ACTIONS = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)


def _histogram(phases: np.ndarray, focal: int, n_bins: int = 24):
    from analysis.stage3.collective_scan import _histogram_from_phases

    return _histogram_from_phases(phases, focal, n_bins=n_bins)


def _omega(n: int, halfwidth: float) -> np.ndarray:
    if n == 1:
        return np.zeros(1)
    return np.linspace(-halfwidth, halfwidth, n, dtype=np.float64)


def _order_parameter(phases: np.ndarray) -> float:
    return float(np.abs(np.mean(np.exp(1j * wrap_phase(phases)))))


def _first_hit(series: list[float], threshold: float) -> float:
    for t, value in enumerate(series):
        if value >= threshold:
            return float(t)
    return float("nan")


def run_one(
    *,
    model,
    ad: ApplicabilityDomain,
    calibrator: RiskCalibrator,
    risk_threshold: float,
    n_agents: int,
    n_steps: int,
    coupling: float,
    omega_halfwidth: float,
    seed: int,
    representation: str,
    track_rich: bool = False,
) -> dict:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, size=n_agents)
    omega = _omega(n_agents, omega_halfwidth)
    n_high_risk = 0
    n_peer_reject = 0
    total = n_agents * n_steps
    r_hist: list[float] = []
    act_hist: list[float] = []
    tau_hist: list[float] = []
    for t in range(n_steps):
        xs = []
        unresolved = []
        for agent in range(n_agents):
            hist = _histogram(phases, agent)
            desc = extract_field_descriptors(representation, hist)
            xs.append(np.concatenate([desc.common, desc.native]))
            unresolved.append(bool(desc.unresolved_near_zero))
        x = np.asarray(xs, dtype=np.float64)
        scores = ad.score(x, unresolved_near_zero=np.asarray(unresolved))
        # Ensemble disagreement proxy: zero when only one production model.
        u_ens = np.zeros(n_agents, dtype=np.float64)
        risk = calibrator.predict_arrays(
            d_shape=scores["d_shape"],
            d_native=scores["d_native"],
            local_density=scores["local_density"],
            u_ensemble=u_ens,
            d_orientation=scores["d_orientation"],
            policy_a=scores["policy_a"],
        )
        g_peer = scores["g_peer"]
        n_peer_reject += int(np.sum(g_peer))
        n_high_risk += int(np.sum((~g_peer) & (risk > risk_threshold)))
        probs = model.predict_proba(x)
        actions = np.zeros(n_agents, dtype=np.float64)
        for i in range(n_agents):
            if g_peer[i]:
                actions[i] = 0.0  # hard-reject: no social torque
            else:
                actions[i] = float(rng.choice(ACTIONS, p=probs[i]))
        phases = step(phases, omega, coupling, actions)
        r_hist.append(_order_parameter(phases))
        if track_rich:
            act_hist.append(float(np.mean(np.abs(actions))))
            tau_hist.append(float(np.mean(actions)))
    out = {
        "n_agents": n_agents,
        "peer_count": n_agents - 1,
        "n_steps": n_steps,
        "coupling": coupling,
        "omega_halfwidth": omega_halfwidth,
        "seed": seed,
        "final_r": r_hist[-1] if r_hist else float("nan"),
        "mean_r": float(np.mean(r_hist)) if r_hist else float("nan"),
        "q_high_risk": n_high_risk / max(total, 1),
        "q_peer_reject": n_peer_reject / max(total, 1),
        "mean_R_proxy_note": "u_ensemble=0 single-model",
    }
    if track_rich:
        out.update(
            {
                "t_r1_gt_0.5": _first_hit(r_hist, 0.5),
                "t_r1_gt_0.9": _first_hit(r_hist, 0.9),
                "mean_activity": float(np.mean(act_hist)) if act_hist else float("nan"),
                "mean_tau_social": float(np.mean(tau_hist)) if tau_hist else float("nan"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "experiments" / "stage_c" / "protocol_stage_c_v0_1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage_c_artifacts",
    )
    parser.add_argument(
        "--n-steps-override",
        type=int,
        default=None,
        help="optional longer offline horizon (e.g. 100)",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    bundle = ROOT / protocol["bundle_root"]
    model = KernelHurdleMultinomial.load(bundle / "kernel_hurdle")
    ad = ApplicabilityDomain.load(bundle / "applicability_domain")
    calibrator = RiskCalibrator.load(bundle / "risk_calibrator.json")
    ood = json.loads((bundle / "ood_policy.json").read_text(encoding="utf-8"))
    risk_threshold = float(ood["risk"]["risk_threshold"])
    representation = protocol["representation"]
    n_steps = int(args.n_steps_override or protocol["n_steps"])

    rows = []
    for n_agents in protocol["n_agents"]:
        for coupling in protocol["couplings"]:
            for seed_i in range(int(protocol["n_seeds"])):
                seed = int(protocol["base_seed"]) + 1000 * int(n_agents) + 10 * seed_i
                row = run_one(
                    model=model,
                    ad=ad,
                    calibrator=calibrator,
                    risk_threshold=risk_threshold,
                    n_agents=int(n_agents),
                    n_steps=n_steps,
                    coupling=float(coupling),
                    omega_halfwidth=float(protocol["omega_halfwidth"]),
                    seed=seed,
                    representation=representation,
                )
                rows.append(row)
                print(
                    f"N={n_agents} K={coupling:+.2f} seed={seed_i}: "
                    f"mean_r={row['mean_r']:.3f} q_risk={row['q_high_risk']:.3f}"
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"surrogate_stage_c_T{n_steps}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol_version": protocol["protocol_version"],
        "n_steps": n_steps,
        "n_rows": len(rows),
        "mean_final_r_by_K": {},
        "mean_q_high_risk": float(np.mean([r["q_high_risk"] for r in rows])),
        "mean_q_peer_reject": float(np.mean([r["q_peer_reject"] for r in rows])),
        "artifact": str(out_csv.relative_to(ROOT).as_posix()),
    }
    for coupling in protocol["couplings"]:
        subset = [r for r in rows if abs(r["coupling"] - float(coupling)) < 1e-12]
        summary["mean_final_r_by_K"][str(coupling)] = float(
            np.mean([r["final_r"] for r in subset])
        )
    (args.out_dir / f"surrogate_stage_c_T{n_steps}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
