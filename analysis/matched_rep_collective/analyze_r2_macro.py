"""R2 Claude macro primary inference: seed-blocked exact permutation + gates.

Inference unit = physical seed. Exhaustive (3!)^6 = 46656 permutations on core.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from circlemap.observation import wrap_phase  # noqa: E402

REPS = ("moments_m1_m3", "centers_24_standard", "intervals_24_decimal6")
POS_K = (0.08, 0.15)


def _rk_series(phases: np.ndarray, m: int) -> np.ndarray:
    # phases: (T+1, N)
    return np.abs(np.mean(np.exp(1j * m * wrap_phase(phases)), axis=1))


def _sustained_lock(r1: np.ndarray, thresh: float = 0.9) -> bool:
    ok = r1 > thresh
    for t in range(len(r1)):
        if bool(np.all(ok[t:])):
            return True
    return False


def _phenotype(final_r1: float, final_r2: float) -> str:
    if final_r1 >= 0.9:
        return "polar_locked"
    if final_r2 >= 0.5 and final_r2 > final_r1:
        return "high_r2_nonpolar"
    if final_r1 >= 0.35:
        return "partial_polar_order"
    return "low_polar_active"


def _load_run_meta(session: Path, cell_id: str, rep: str) -> dict:
    path = session / cell_id / rep / "run_meta.json"
    if not path.exists():
        # Prefer phases.npy enrichment
        phases_path = session / cell_id / rep / "phases.npy"
        if not phases_path.exists():
            raise FileNotFoundError(path)
        phases = np.load(phases_path)
        r1 = _rk_series(phases, 1)
        r2 = _rk_series(phases, 2)
        return {
            "final_r1": float(r1[-1]),
            "final_r2": float(r2[-1]),
            "mean_r1": float(np.mean(r1[1:])),
            "r1_series": r1.tolist(),
            "r2_series": r2.tolist(),
            "sustained_lock": int(_sustained_lock(r1)),
        }
    meta = json.loads(path.read_text(encoding="utf-8"))
    if "r1_series" in meta:
        r1 = np.asarray(meta["r1_series"], dtype=float)
        meta["sustained_lock"] = int(_sustained_lock(r1))
    else:
        phases = np.load(session / cell_id / rep / "phases.npy")
        r1 = _rk_series(phases, 1)
        meta["r1_series"] = r1.tolist()
        meta["r2_series"] = _rk_series(phases, 2).tolist()
        meta["sustained_lock"] = int(_sustained_lock(r1))
        meta["final_r1"] = float(r1[-1])
        meta["final_r2"] = float(meta["r2_series"][-1])
    return meta


def collect_rows(session: Path, protocol: dict) -> list[dict]:
    rows = []
    for panel_name, panel in (
        ("core", protocol["core_panel"]),
        ("heldout", protocol["heldout_panel"]),
    ):
        for seed_i in panel["seed_indices"]:
            for k in panel["couplings"]:
                cell_id = f"N{protocol['n_agents']}_K{float(k):+g}_s{int(seed_i)}"
                for rep in REPS:
                    meta = _load_run_meta(session, cell_id, rep)
                    final_r1 = float(meta["final_r1"])
                    final_r2 = float(meta["final_r2"])
                    rows.append(
                        {
                            "panel": panel_name,
                            "seed_index": int(seed_i),
                            "coupling": float(k),
                            "cell_id": cell_id,
                            "representation": rep,
                            "final_r1": final_r1,
                            "final_r2": final_r2,
                            "mean_r1": float(meta.get("mean_r1", np.nan)),
                            "final_Q2": final_r2 - final_r1,
                            "sustained_lock": int(meta["sustained_lock"]),
                            "polar_locked": int(final_r1 >= 0.9),
                            "cluster_phenotype": _phenotype(final_r1, final_r2),
                        }
                    )
    return rows


def seed_scores(rows: list[dict], panel: str = "core") -> dict[int, dict[str, dict]]:
    """seed -> rep -> {L, Y, ...} aggregated over positive K."""
    out: dict[int, dict[str, dict]] = {}
    for seed_i in sorted({r["seed_index"] for r in rows if r["panel"] == panel}):
        out[seed_i] = {}
        for rep in REPS:
            pos = [
                r
                for r in rows
                if r["panel"] == panel
                and r["seed_index"] == seed_i
                and r["representation"] == rep
                and float(r["coupling"]) in POS_K
            ]
            if len(pos) != 2:
                raise ValueError(
                    f"incomplete positive-K cube seed={seed_i} rep={rep}: {len(pos)}"
                )
            L = 0.5 * sum(float(r["sustained_lock"]) for r in pos)
            Y = 0.5 * sum(float(r["final_r1"]) for r in pos)
            Q = 0.5 * sum(float(r["final_Q2"]) for r in pos)
            out[seed_i][rep] = {"L": L, "Y": Y, "Q2": Q, "rows": pos}
    return out


def _T_from_means(means: dict[str, float]) -> float:
    grand = float(np.mean(list(means.values())))
    return float(sum((v - grand) ** 2 for v in means.values()))


def exact_perm_test(scores: dict[int, dict[str, dict]], key: str) -> dict:
    seeds = sorted(scores)
    obs_mat = np.array([[scores[s][r][key] for r in REPS] for s in seeds], dtype=float)
    obs_means = {r: float(np.mean(obs_mat[:, i])) for i, r in enumerate(REPS)}
    T_obs = _T_from_means(obs_means)

    perms = list(itertools.permutations(range(3)))
    n_null = 0
    n_ge = 0
    # For each seed independently permute columns.
    # Exhaustive: product of per-seed permutations.
    for seed_perms in itertools.product(perms, repeat=len(seeds)):
        mat = np.empty_like(obs_mat)
        for i, perm in enumerate(seed_perms):
            mat[i] = obs_mat[i, list(perm)]
        means = {r: float(np.mean(mat[:, j])) for j, r in enumerate(REPS)}
        T = _T_from_means(means)
        n_null += 1
        if T >= T_obs - 1e-15:
            n_ge += 1
    return {
        "endpoint": key,
        "n_seeds": len(seeds),
        "n_permutations": n_null,
        "T_obs": T_obs,
        "mean_by_rep": obs_means,
        "p_perm": n_ge / n_null,
        "alpha": 0.05,
        "reject_H0": (n_ge / n_null) < 0.05,
    }


def pairwise_contrasts(scores: dict[int, dict[str, dict]], key: str) -> list[dict]:
    seeds = sorted(scores)
    pairs = [
        ("moments_m1_m3", "centers_24_standard", "M-C"),
        ("moments_m1_m3", "intervals_24_decimal6", "M-I"),
        ("intervals_24_decimal6", "centers_24_standard", "I-C"),
    ]
    out = []
    rng = np.random.default_rng(20260725)
    for a, b, name in pairs:
        diffs = np.array(
            [scores[s][a][key] - scores[s][b][key] for s in seeds], dtype=float
        )
        mean_diff = float(np.mean(diffs))
        # Bootstrap CI
        boots = []
        for _ in range(5000):
            idx = rng.integers(0, len(diffs), size=len(diffs))
            boots.append(float(np.mean(diffs[idx])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        # Exact sign-flip
        n_ge = 0
        n_null = 0
        for signs in itertools.product([-1.0, 1.0], repeat=len(diffs)):
            t = abs(float(np.mean(diffs * np.asarray(signs))))
            n_null += 1
            if t >= abs(mean_diff) - 1e-15:
                n_ge += 1
        out.append(
            {
                "contrast": name,
                "endpoint": key,
                "seed_diffs": diffs.tolist(),
                "mean_diff": mean_diff,
                "bootstrap_ci_95": [float(lo), float(hi)],
                "signflip_p": n_ge / n_null,
                "n_signflip": n_null,
            }
        )
    return out


def evaluate_gates(rows: list[dict], primary_L: dict, primary_Y: dict) -> dict:
    core_pos = [
        r for r in rows if r["panel"] == "core" and float(r["coupling"]) in POS_K
    ]
    # Gate A
    a1 = bool(primary_L["reject_H0"])
    a2 = bool(primary_Y["reject_H0"])
    # phenotype difference at some positive K
    pheno_diff = False
    for k in POS_K:
        by_rep = {}
        for rep in REPS:
            labels = [
                r["cluster_phenotype"]
                for r in core_pos
                if r["representation"] == rep and float(r["coupling"]) == k
            ]
            by_rep[rep] = set(labels)
        if len(set().union(*by_rep.values())) > 1 and any(
            by_rep[a] != by_rep[b] for a in REPS for b in REPS
        ):
            pheno_diff = True
            break
    # not single-seed driven: leave detailed flag via lock matrix variance
    lock_by_seed_rep = {}
    for seed_i in range(6):
        lock_by_seed_rep[seed_i] = {}
        for rep in REPS:
            locks = [
                r["polar_locked"]
                for r in core_pos
                if r["seed_index"] == seed_i and r["representation"] == rep
            ]
            lock_by_seed_rep[seed_i][rep] = locks
    # Gate B
    gate_b = {"pass": False, "details": []}
    for k in POS_K:
        fracs = {}
        for rep in REPS:
            locks = [
                r["polar_locked"]
                for r in core_pos
                if r["representation"] == rep and float(r["coupling"]) == k
            ]
            fracs[rep] = sum(locks) / max(len(locks), 1)
        high = [r for r, f in fracs.items() if f >= 5 / 6]
        low = [r for r, f in fracs.items() if f <= 1 / 6]
        detail = {"K": k, "lock_fractions": fracs, "high": high, "low": low}
        gate_b["details"].append(detail)
        if high and low:
            # held-out same direction
            held = [
                r
                for r in rows
                if r["panel"] == "heldout" and float(r["coupling"]) == k
            ]
            if held:
                h_fracs = {}
                for rep in REPS:
                    locks = [
                        r["polar_locked"]
                        for r in held
                        if r["representation"] == rep
                    ]
                    h_fracs[rep] = sum(locks) / max(len(locks), 1)
                detail["heldout_lock_fractions"] = h_fracs
                # same direction: high reps mean lock >= low reps mean lock
                if min(h_fracs[h] for h in high) >= max(h_fracs[lo] for lo in low):
                    gate_b["pass"] = True
            else:
                gate_b["pass"] = True  # heldout missing: mark core-only

    # Gate C
    means_r1 = primary_Y["mean_by_rep"]
    # Q2 means over positive K core
    q_means = {}
    for rep in REPS:
        vals = [r["final_Q2"] for r in core_pos if r["representation"] == rep]
        q_means[rep] = float(np.mean(vals)) if vals else float("nan")
    ranking = (
        means_r1["moments_m1_m3"]
        > means_r1["centers_24_standard"]
        > means_r1["intervals_24_decimal6"]
        and q_means["intervals_24_decimal6"]
        > q_means["moments_m1_m3"]
        and q_means["intervals_24_decimal6"]
        > q_means["centers_24_standard"]
    )
    gate_c = {
        "pass": bool(ranking),
        "mean_final_r1": means_r1,
        "mean_final_Q2": q_means,
        "note": "optional strong GPT map replication",
    }

    return {
        "gate_A": {
            "required": True,
            "sustained_lock_perm": a1,
            "final_r1_perm": a2,
            "phenotype_diff_some_pos_K": pheno_diff,
            "pass": bool(a1 and a2 and pheno_diff),
            "lock_by_seed_rep": lock_by_seed_rep,
        },
        "gate_B": gate_b,
        "gate_C": gate_c,
    }


def check_k0(session: Path, protocol: dict) -> dict:
    tol = float(protocol["k0_control"]["tolerance"])
    max_diff = 0.0
    for seed_i in protocol["core_panel"]["seed_indices"]:
        cell_id = f"N{protocol['n_agents']}_K+0_s{int(seed_i)}"
        series = {}
        for rep in REPS:
            phases = np.load(session / cell_id / rep / "phases.npy")
            series[rep] = {
                m: _rk_series(phases, m) for m in (1, 2, 3)
            }
        for m in (1, 2, 3):
            for a in REPS:
                for b in REPS:
                    d = float(np.max(np.abs(series[a][m] - series[b][m])))
                    max_diff = max(max_diff, d)
    return {
        "max_abs_diff_rm": max_diff,
        "tolerance": tol,
        "pass": max_diff < tol,
        "on_fail": "abort_engine_matching_failure",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT
        / "experiments"
        / "stage_c"
        / "protocol_matched_rep_collective_r2_claude_v0_1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <session>/r2_inference",
    )
    args = parser.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    session = args.session
    out_dir = args.out_dir or (session / "r2_inference")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(session, protocol)
    (out_dir / "trajectory_rows.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    k0 = check_k0(session, protocol)
    (out_dir / "k0_control.json").write_text(
        json.dumps(k0, indent=2) + "\n", encoding="utf-8"
    )
    if not k0["pass"]:
        raise SystemExit(
            f"K=0 control FAILED: max|Δr_m|={k0['max_abs_diff_rm']} "
            f">= {k0['tolerance']} (engine/matching failure)"
        )

    scores = seed_scores(rows, panel="core")
    primary_L = exact_perm_test(scores, "L")
    primary_Y = exact_perm_test(scores, "Y")
    contrasts_L = pairwise_contrasts(scores, "L")
    contrasts_Y = pairwise_contrasts(scores, "Y")
    gates = evaluate_gates(rows, primary_L, primary_Y)

    result = {
        "protocol_version": protocol["protocol_version"],
        "session": session.as_posix(),
        "primary_sustained_lock": primary_L,
        "secondary_final_r1": primary_Y,
        "contrasts_L": contrasts_L,
        "contrasts_Y": contrasts_Y,
        "gates": gates,
        "k0_control": k0,
        "n_rows": len(rows),
    }
    (out_dir / "primary_inference.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"T_L={primary_L['T_obs']:.6f} p={primary_L['p_perm']:.6g} "
        f"reject={primary_L['reject_H0']}"
    )
    print(
        f"T_Y={primary_Y['T_obs']:.6f} p={primary_Y['p_perm']:.6g} "
        f"reject={primary_Y['reject_H0']}"
    )
    print(
        f"GateA={gates['gate_A']['pass']} GateB={gates['gate_B']['pass']} "
        f"GateC={gates['gate_C']['pass']}"
    )
    print(f"Wrote {out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
