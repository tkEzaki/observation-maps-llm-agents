"""Stage C v0.1 offline diagnostics (§5) before any further paid runs.

Produces trajectory harmonics, per-(N,seed) tables, LLM-field AD/risk,
teacher-forced one-step scores, surrogate predictive replicates, and
activity / torque / engine-identity checks.
"""

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
from analysis.stage3.collective_scan import _histogram_from_phases  # noqa: E402
from analysis.stage3.models import KernelHurdleMultinomial  # noqa: E402
from analysis.stage3.risk_calibrator import RiskCalibrator  # noqa: E402
from analysis.stage3.validation import multinomial_log_loss  # noqa: E402
from circlemap.engine import step  # noqa: E402
from circlemap.field_features import extract_field_descriptors  # noqa: E402
from circlemap.observation import wrap_phase  # noqa: E402

ACTIONS = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
ACTION_INDEX = {"retard": 0, "stay": 1, "advance": 2}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rk(phases: np.ndarray, order: int) -> float:
    return float(np.abs(np.mean(np.exp(1j * order * wrap_phase(phases)))))


def _rk_series(phases_hist: np.ndarray, order: int) -> np.ndarray:
    return np.asarray([_rk(phases_hist[t], order) for t in range(phases_hist.shape[0])])


def _first_hit(series: np.ndarray, threshold: float) -> float:
    hits = np.where(series >= threshold)[0]
    return float(hits[0]) if hits.size else float("nan")


def _early_growth(series: np.ndarray, *, window: int = 10) -> float:
    w = min(window, series.size - 1)
    if w <= 1:
        return float("nan")
    y = np.log(np.maximum(series[: w + 1], 1e-8))
    t = np.arange(w + 1, dtype=np.float64)
    slope = np.polyfit(t, y, 1)[0]
    return float(slope)


def _omega(n: int, halfwidth: float) -> np.ndarray:
    return np.linspace(-halfwidth, halfwidth, n, dtype=np.float64)


def _load_session(session: Path) -> tuple[dict, list[dict]]:
    protocol = json.loads((session / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((session / "session_summary.json").read_text(encoding="utf-8"))
    return protocol, summary["runs"]


def analyze_trajectories(session: Path, out_dir: Path) -> list[dict]:
    protocol, runs = _load_session(session)
    rows = []
    series_rows = []
    for run in runs:
        run_dir = session / run["run_id"]
        phases = np.load(run_dir / "phases.npy")
        actions = np.load(run_dir / "actions.npy")
        r1 = _rk_series(phases, 1)
        r2 = _rk_series(phases, 2)
        r3 = _rk_series(phases, 3)
        for t in range(phases.shape[0]):
            series_rows.append(
                {
                    "run_id": run["run_id"],
                    "n_agents": run["n_agents"],
                    "coupling": run["coupling"],
                    "seed_index": run["seed_index"],
                    "t": t,
                    "r1": float(r1[t]),
                    "r2": float(r2[t]),
                    "r3": float(r3[t]),
                    "activity": (
                        float(np.mean(np.abs(actions[t]))) if t < actions.shape[0] else float("nan")
                    ),
                    "tau_social": (
                        float(np.mean(actions[t])) if t < actions.shape[0] else float("nan")
                    ),
                }
            )
        # Engine identity over full window
        n_steps = actions.shape[0]
        omega = _omega(int(run["n_agents"]), float(protocol["omega_halfwidth"]))
        eff = (phases[n_steps] - phases[0]) / max(n_steps, 1)
        omega_coll = float(np.mean(eff))
        mean_omega = float(np.mean(omega))
        tau = float(np.mean(actions))
        predicted = mean_omega + float(run["coupling"]) * tau
        rows.append(
            {
                "run_id": run["run_id"],
                "n_agents": run["n_agents"],
                "coupling": run["coupling"],
                "seed_index": run["seed_index"],
                "init_seed": run["init_seed"],
                "final_r1": float(r1[-1]),
                "final_r2": float(r2[-1]),
                "final_r3": float(r3[-1]),
                "mean_r1": float(np.mean(r1[1:])),
                "mean_r2": float(np.mean(r2[1:])),
                "mean_r3": float(np.mean(r3[1:])),
                "t_r1_gt_0.5": _first_hit(r1, 0.5),
                "t_r1_gt_0.9": _first_hit(r1, 0.9),
                "early_growth_r1": _early_growth(r1),
                "mean_activity": float(np.mean(np.abs(actions))),
                "mean_p_stay_emp": float(np.mean(actions == 0.0)),
                "mean_tau_social": tau,
                "omega_coll": omega_coll,
                "mean_omega": mean_omega,
                "omega_coll_predicted": predicted,
                "engine_residual": omega_coll - predicted,
            }
        )
    _write_csv(out_dir / "trajectory_summary_per_run.csv", rows)
    _write_csv(out_dir / "trajectory_harmonics_timeseries.csv", series_rows)
    return rows


def analyze_llm_risk_and_teacher(
    session: Path,
    *,
    model: KernelHurdleMultinomial,
    ad: ApplicabilityDomain,
    calibrator: RiskCalibrator,
    risk_threshold: float,
    representation: str,
    out_dir: Path,
) -> tuple[list[dict], list[dict]]:
    protocol, runs = _load_session(session)
    risk_rows = []
    teacher_rows = []
    for run in runs:
        run_dir = session / run["run_id"]
        phases = np.load(run_dir / "phases.npy")
        actions = np.load(run_dir / "actions.npy")
        n_agents = int(run["n_agents"])
        n_steps = int(actions.shape[0])
        risks_all = []
        high = 0
        peer_rej = 0
        total = 0
        # Aggregate teacher counts
        pred_list = []
        count_list = []
        stay_hat = []
        stay_obs = []
        mean_act_hat = []
        mean_act_obs = []
        for t in range(n_steps):
            xs = []
            unresolved = []
            for agent in range(n_agents):
                hist = _histogram_from_phases(phases[t], agent)
                desc = extract_field_descriptors(representation, hist)
                xs.append(np.concatenate([desc.common, desc.native]))
                unresolved.append(bool(desc.unresolved_near_zero))
            x = np.asarray(xs, dtype=np.float64)
            scores = ad.score(x, unresolved_near_zero=np.asarray(unresolved))
            risk = calibrator.predict_arrays(
                d_shape=scores["d_shape"],
                d_native=scores["d_native"],
                local_density=scores["local_density"],
                u_ensemble=np.zeros(n_agents),
                d_orientation=scores["d_orientation"],
                policy_a=scores["policy_a"],
            )
            g_peer = scores["g_peer"]
            probs = model.predict_proba(x)
            for agent in range(n_agents):
                total += 1
                risks_all.append(float(risk[agent]))
                if g_peer[agent]:
                    peer_rej += 1
                elif risk[agent] > risk_threshold:
                    high += 1
                a_obs = float(actions[t, agent])
                counts = np.zeros(3, dtype=np.float64)
                counts[ACTION_INDEX[{-1.0: "retard", 0.0: "stay", 1.0: "advance"}[a_obs]]] = 1.0
                pred_list.append(probs[agent])
                count_list.append(counts)
                stay_hat.append(float(probs[agent, 1]))
                stay_obs.append(1.0 if a_obs == 0.0 else 0.0)
                mean_act_hat.append(
                    float(-probs[agent, 0] + probs[agent, 2])
                )
                mean_act_obs.append(a_obs)

        pred = np.asarray(pred_list)
        counts = np.asarray(count_list)
        empir = counts  # one-hot
        brier = float(np.mean(np.sum((pred - empir) ** 2, axis=-1)))
        log_loss = multinomial_log_loss(pred, counts)
        risks_arr = np.asarray(risks_all)
        risk_rows.append(
            {
                "run_id": run["run_id"],
                "n_agents": run["n_agents"],
                "coupling": run["coupling"],
                "seed_index": run["seed_index"],
                "n_agent_steps": total,
                "mean_R": float(np.mean(risks_arr)),
                "R_p50": float(np.quantile(risks_arr, 0.5)),
                "R_p90": float(np.quantile(risks_arr, 0.9)),
                "R_p99": float(np.quantile(risks_arr, 0.99)),
                "Q_risk": high / max(total, 1),
                "q_peer_reject": peer_rej / max(total, 1),
                "teacher_log_loss": log_loss,
                "teacher_brier": brier,
                "pred_stay_frac": float(np.mean(stay_hat)),
                "obs_stay_frac": float(np.mean(stay_obs)),
                "pred_mean_action": float(np.mean(mean_act_hat)),
                "obs_mean_action": float(np.mean(mean_act_obs)),
                "mean_r1_run": float(_rk(phases[-1], 1)),
            }
        )
        # Per-(K,N) teacher already in risk_rows; also store fine bins by t decade
        for t_bin, t0, t1 in (
            ("t0_9", 0, 10),
            ("t10_19", 10, 20),
            ("t20_29", 20, 30),
            ("t30_39", 30, 40),
        ):
            if t1 > n_steps:
                continue
            # rebuild slice metrics cheaply from stored actions/phases
            pred_b, count_b = [], []
            for t in range(t0, min(t1, n_steps)):
                xs = []
                for agent in range(n_agents):
                    hist = _histogram_from_phases(phases[t], agent)
                    desc = extract_field_descriptors(representation, hist)
                    xs.append(np.concatenate([desc.common, desc.native]))
                probs = model.predict_proba(np.asarray(xs))
                for agent in range(n_agents):
                    a_obs = float(actions[t, agent])
                    counts_row = np.zeros(3)
                    counts_row[
                        ACTION_INDEX[{-1.0: "retard", 0.0: "stay", 1.0: "advance"}[a_obs]]
                    ] = 1.0
                    pred_b.append(probs[agent])
                    count_b.append(counts_row)
            pred_b = np.asarray(pred_b)
            count_b = np.asarray(count_b)
            teacher_rows.append(
                {
                    "run_id": run["run_id"],
                    "n_agents": run["n_agents"],
                    "coupling": run["coupling"],
                    "seed_index": run["seed_index"],
                    "t_bin": t_bin,
                    "log_loss": multinomial_log_loss(pred_b, count_b),
                    "brier": float(np.mean(np.sum((pred_b - count_b) ** 2, axis=-1))),
                    "pred_stay": float(np.mean(pred_b[:, 1])),
                    "obs_stay": float(np.mean(count_b[:, 1])),
                    "mean_r1_bin": float(
                        np.mean([_rk(phases[t], 1) for t in range(t0, min(t1, n_steps))])
                    ),
                }
            )
    _write_csv(out_dir / "llm_field_risk_teacher.csv", risk_rows)
    _write_csv(out_dir / "teacher_forced_by_tbin.csv", teacher_rows)
    return risk_rows, teacher_rows


def surrogate_replicates(
    *,
    model: KernelHurdleMultinomial,
    ad: ApplicabilityDomain,
    calibrator: RiskCalibrator,
    risk_threshold: float,
    protocol: dict,
    runs: list[dict],
    n_reps: int,
    out_dir: Path,
) -> list[dict]:
    representation = protocol["representation"]
    halfwidth = float(protocol["omega_halfwidth"])
    n_steps = int(protocol["n_steps"])
    rows = []
    for run in runs:
        n_agents = int(run["n_agents"])
        coupling = float(run["coupling"])
        init_seed = int(run["init_seed"])
        # Fixed init from LLM seed; vary only action-sampling stream.
        rng0 = np.random.default_rng(init_seed)
        phases0 = rng0.uniform(-np.pi, np.pi, size=n_agents)
        omega = _omega(n_agents, halfwidth)
        finals = []
        means = []
        r2f = []
        r3f = []
        taus = []
        acts = []
        for rep in range(n_reps):
            rng = np.random.default_rng(init_seed + 10_000_003 * (rep + 1))
            phases = phases0.copy()
            action_hist = []
            r1_hist = []
            for _t in range(n_steps):
                xs = []
                unresolved = []
                for agent in range(n_agents):
                    hist = _histogram_from_phases(phases, agent)
                    desc = extract_field_descriptors(representation, hist)
                    xs.append(np.concatenate([desc.common, desc.native]))
                    unresolved.append(bool(desc.unresolved_near_zero))
                x = np.asarray(xs)
                scores = ad.score(x, unresolved_near_zero=np.asarray(unresolved))
                probs = model.predict_proba(x)
                actions = np.zeros(n_agents)
                for i in range(n_agents):
                    if scores["g_peer"][i]:
                        actions[i] = 0.0
                    else:
                        actions[i] = float(rng.choice(ACTIONS, p=probs[i]))
                phases = step(phases, omega, coupling, actions)
                action_hist.append(actions)
                r1_hist.append(_rk(phases, 1))
            finals.append(_rk(phases, 1))
            means.append(float(np.mean(r1_hist)))
            r2f.append(_rk(phases, 2))
            r3f.append(_rk(phases, 3))
            acts.append(float(np.mean(np.abs(np.asarray(action_hist)))))
            taus.append(float(np.mean(np.asarray(action_hist))))
        finals_a = np.asarray(finals)
        llm_final = float(run["final_r"])
        # LLM mean_r from session summary
        llm_mean = float(run["mean_r"])
        rows.append(
            {
                "run_id": run["run_id"],
                "n_agents": n_agents,
                "coupling": coupling,
                "seed_index": run["seed_index"],
                "n_reps": n_reps,
                "surr_final_r1_mean": float(np.mean(finals_a)),
                "surr_final_r1_p05": float(np.quantile(finals_a, 0.05)),
                "surr_final_r1_p50": float(np.quantile(finals_a, 0.50)),
                "surr_final_r1_p95": float(np.quantile(finals_a, 0.95)),
                "surr_final_r2_mean": float(np.mean(r2f)),
                "surr_final_r3_mean": float(np.mean(r3f)),
                "surr_mean_activity": float(np.mean(acts)),
                "surr_mean_tau": float(np.mean(taus)),
                "llm_final_r1": llm_final,
                "llm_mean_r1": llm_mean,
                "llm_in_p05_p95": bool(
                    np.quantile(finals_a, 0.05)
                    <= llm_final
                    <= np.quantile(finals_a, 0.95)
                ),
                "llm_gt_p95": bool(llm_final > np.quantile(finals_a, 0.95)),
                "llm_lt_p05": bool(llm_final < np.quantile(finals_a, 0.05)),
            }
        )
        print(
            f"  replicates {run['run_id']}: "
            f"surr[{np.quantile(finals_a,0.05):.3f},"
            f"{np.quantile(finals_a,0.95):.3f}] "
            f"llm={llm_final:.3f} in={rows[-1]['llm_in_p05_p95']}"
        )
    _write_csv(out_dir / "surrogate_replicates.csv", rows)
    return rows


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
        default=ROOT / "analysis" / "stage_c_artifacts" / "offline_v0_1",
    )
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--skip-replicates", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol, runs = _load_session(args.session)
    bundle = ROOT / protocol["bundle_root"]
    model = KernelHurdleMultinomial.load(bundle / "kernel_hurdle")
    ad = ApplicabilityDomain.load(bundle / "applicability_domain")
    calibrator = RiskCalibrator.load(bundle / "risk_calibrator.json")
    ood = json.loads((bundle / "ood_policy.json").read_text(encoding="utf-8"))
    risk_threshold = float(ood["risk"]["risk_threshold"])
    representation = protocol["representation"]

    print("=== 5.1 / 5.2 / 5.6 trajectories + torque ===")
    traj = analyze_trajectories(args.session, out_dir)

    print("=== 5.3 / 5.4 LLM risk + teacher-forced ===")
    risk_rows, teacher_rows = analyze_llm_risk_and_teacher(
        args.session,
        model=model,
        ad=ad,
        calibrator=calibrator,
        risk_threshold=risk_threshold,
        representation=representation,
        out_dir=out_dir,
    )

    rep_rows = []
    if not args.skip_replicates:
        print(f"=== 5.5 surrogate replicates (n={args.n_reps}) ===")
        rep_rows = surrogate_replicates(
            model=model,
            ad=ad,
            calibrator=calibrator,
            risk_threshold=risk_threshold,
            protocol=protocol,
            runs=runs,
            n_reps=args.n_reps,
            out_dir=out_dir,
        )

    # Rollup
    by_k = {}
    for k in (-0.15, 0.0, 0.15):
        sub_t = [r for r in traj if abs(float(r["coupling"]) - k) < 1e-12]
        sub_r = [r for r in risk_rows if abs(float(r["coupling"]) - k) < 1e-12]
        by_k[str(k)] = {
            "final_r1": {f"N{r['n_agents']}_s{r['seed_index']}": r["final_r1"] for r in sub_t},
            "final_r2": {f"N{r['n_agents']}_s{r['seed_index']}": r["final_r2"] for r in sub_t},
            "final_r3": {f"N{r['n_agents']}_s{r['seed_index']}": r["final_r3"] for r in sub_t},
            "t_r1_gt_0.9": {
                f"N{r['n_agents']}_s{r['seed_index']}": r["t_r1_gt_0.9"] for r in sub_t
            },
            "mean_R": float(np.mean([r["mean_R"] for r in sub_r])),
            "Q_risk": float(np.mean([r["Q_risk"] for r in sub_r])),
            "teacher_log_loss": float(np.mean([r["teacher_log_loss"] for r in sub_r])),
            "obs_stay": float(np.mean([r["obs_stay_frac"] for r in sub_r])),
            "pred_stay": float(np.mean([r["pred_stay_frac"] for r in sub_r])),
            "mean_engine_residual": float(np.mean([r["engine_residual"] for r in sub_t])),
        }

    branch = "undetermined"
    mean_q = float(np.mean([r["Q_risk"] for r in risk_rows]))
    mean_ll = float(np.mean([r["teacher_log_loss"] for r in risk_rows]))
    # Heuristic branch for §7 (not a freeze gate)
    if mean_q < 0.05 and mean_ll < 0.9:
        branch = "A_low_risk_good_onestep_closed_loop_accumulation"
    elif mean_q >= 0.05:
        branch = "B_high_risk_fields_need_coverage_v2"
    else:
        branch = "C_low_risk_bad_onestep_surrogate_structure"

    n_in = sum(1 for r in rep_rows if r.get("llm_in_p05_p95")) if rep_rows else None
    summary = {
        "session": str(args.session),
        "branch_heuristic": branch,
        "mean_Q_risk_llm_fields": mean_q,
        "mean_teacher_log_loss": mean_ll,
        "mean_engine_residual_abs": float(
            np.mean([abs(r["engine_residual"]) for r in traj])
        ),
        "replicates_llm_in_interval": n_in,
        "replicates_n_runs": len(rep_rows) if rep_rows else 0,
        "by_K": by_k,
        "artifacts": {
            "trajectory_summary": "trajectory_summary_per_run.csv",
            "timeseries": "trajectory_harmonics_timeseries.csv",
            "risk_teacher": "llm_field_risk_teacher.csv",
            "teacher_tbin": "teacher_forced_by_tbin.csv",
            "replicates": "surrogate_replicates.csv",
        },
    }
    def _json_default(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        raise TypeError(type(obj))

    (out_dir / "offline_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
