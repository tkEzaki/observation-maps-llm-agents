"""Build Stage C v0.2 combined tables + bootstrap error reductions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "stage_c_artifacts" / "stage_c_v0_2_combined"
RNG = np.random.default_rng(20260724)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict, key: str) -> float:
    return float(row[key])


def _bootstrap_mean_ci(
    values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    means = []
    n = values.size
    for _ in range(n_boot):
        sample = values[RNG.integers(0, n, size=n)]
        means.append(float(np.mean(sample)))
    means_arr = np.sort(np.asarray(means))
    lo = float(np.quantile(means_arr, alpha / 2))
    hi = float(np.quantile(means_arr, 1 - alpha / 2))
    return {"mean": float(np.mean(values)), "lo": lo, "hi": hi, "n": int(n)}


def _abs_err(llm: float, surr: float) -> float:
    return abs(llm - surr)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = {
        "v0.2a": ROOT / "analysis" / "stage_c_artifacts" / "stage_c_v0_2a",
        "v0.2b": ROOT / "analysis" / "stage_c_artifacts" / "stage_c_v0_2b",
    }

    endpoint_rows = []
    delta_pool: dict[str, list[float]] = {
        "delta_e_A": [],
        "delta_e_r1": [],
        "delta_e_t09": [],
        "delta_e_tau": [],
        "delta_e_stay_teacher": [],
    }
    panel_summaries = {}

    for panel, root in panels.items():
        per = _read(root / "llm_vs_surrogate_per_run.csv")
        # per-run has one row per (run, bundle); pivot by run_id
        by_run: dict[str, dict] = {}
        for row in per:
            run_id = row["run_id"]
            slot = by_run.setdefault(
                run_id,
                {
                    "panel": panel,
                    "run_id": run_id,
                    "n_agents": (
                        int(float(row["n_agents"]))
                        if row.get("n_agents") not in (None, "")
                        else (17 if panel == "v0.2a" else 9)
                    ),
                    "coupling": float(row["coupling"]),
                    "seed_index": int(float(row["seed_index"])),
                    "llm_activity": float(row["llm_activity"]),
                    "llm_final_r1": float(row["llm_final_r1"]),
                    "llm_mean_r1": float(row["llm_mean_r1"]),
                    "llm_t05": float(row["llm_t05"]),
                    "llm_t09": float(row["llm_t09"]),
                    "llm_tau": float(row["llm_tau"]),
                    "llm_p_stay": float(row["llm_p_stay"]),
                },
            )
            b = row["bundle"]
            slot[f"{b}_activity"] = float(row["surr_activity"])
            slot[f"{b}_final_r1"] = float(row["surr_final_r1"])
            slot[f"{b}_mean_r1"] = float(row["surr_mean_r1"])
            slot[f"{b}_t05"] = float(row["surr_t05"])
            slot[f"{b}_t09"] = float(row["surr_t09"])
            slot[f"{b}_tau"] = float(row["surr_tau"])

        # teacher stay by run
        t1 = {
            r["run_id"]: r
            for r in _read(root / "teacher_v1" / "llm_field_risk_teacher.csv")
        }
        t2 = {
            r["run_id"]: r
            for r in _read(root / "teacher_v2" / "llm_field_risk_teacher.csv")
        }

        for run_id, slot in by_run.items():
            if "v1_activity" not in slot or "v2_activity" not in slot:
                continue
            eA_v1 = _abs_err(slot["llm_activity"], slot["v1_activity"])
            eA_v2 = _abs_err(slot["llm_activity"], slot["v2_activity"])
            er_v1 = _abs_err(slot["llm_final_r1"], slot["v1_final_r1"])
            er_v2 = _abs_err(slot["llm_final_r1"], slot["v2_final_r1"])
            et_v1 = _abs_err(slot["llm_t09"], slot["v1_t09"])
            et_v2 = _abs_err(slot["llm_t09"], slot["v2_t09"])
            etau_v1 = _abs_err(slot["llm_tau"], slot["v1_tau"])
            etau_v2 = _abs_err(slot["llm_tau"], slot["v2_tau"])

            stay_obs = float(t1[run_id]["obs_stay_frac"])
            stay_v1 = float(t1[run_id]["pred_stay_frac"])
            stay_v2 = float(t2[run_id]["pred_stay_frac"])
            es_v1 = abs(stay_v1 - stay_obs)
            es_v2 = abs(stay_v2 - stay_obs)

            row_out = {
                **slot,
                "v1_pred_stay": stay_v1,
                "v2_pred_stay": stay_v2,
                "obs_stay_teacher": stay_obs,
                "eA_v1": eA_v1,
                "eA_v2": eA_v2,
                "delta_e_A": eA_v1 - eA_v2,
                "er1_v1": er_v1,
                "er1_v2": er_v2,
                "delta_e_r1": er_v1 - er_v2,
                "et09_v1": et_v1,
                "et09_v2": et_v2,
                "delta_e_t09": et_v1 - et_v2,
                "etau_v1": etau_v1,
                "etau_v2": etau_v2,
                "delta_e_tau": etau_v1 - etau_v2,
                "estay_v1": es_v1,
                "estay_v2": es_v2,
                "delta_e_stay_teacher": es_v1 - es_v2,
                "locked_llm": int(slot["llm_final_r1"] > 0.9),
            }
            endpoint_rows.append(row_out)
            delta_pool["delta_e_A"].append(row_out["delta_e_A"])
            delta_pool["delta_e_r1"].append(row_out["delta_e_r1"])
            if np.isfinite(row_out["delta_e_t09"]):
                delta_pool["delta_e_t09"].append(row_out["delta_e_t09"])
            delta_pool["delta_e_tau"].append(row_out["delta_e_tau"])
            delta_pool["delta_e_stay_teacher"].append(row_out["delta_e_stay_teacher"])

        cost = json.loads((root / "actual_cost.json").read_text(encoding="utf-8"))
        panel_summaries[panel] = {
            "actual_cost_usd": cost["actual_cost_usd"],
            "total_calls": cost["total_calls"],
            "valid_rate": cost["valid_rate"],
            "n_runs": len(by_run),
        }

    # write per-run endpoints
    ep_path = OUT / "combined_endpoints_per_run.csv"
    with ep_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0].keys()))
        writer.writeheader()
        writer.writerows(endpoint_rows)

    # bootstrap overall and by panel
    boot = {"overall": {}, "by_panel": {}}
    for key, vals in delta_pool.items():
        boot["overall"][key] = _bootstrap_mean_ci(np.asarray(vals))
    for panel in panels:
        sub = [r for r in endpoint_rows if r["panel"] == panel]
        boot["by_panel"][panel] = {
            "delta_e_A": _bootstrap_mean_ci(np.asarray([r["delta_e_A"] for r in sub])),
            "delta_e_r1": _bootstrap_mean_ci(np.asarray([r["delta_e_r1"] for r in sub])),
            "delta_e_t09": _bootstrap_mean_ci(
                np.asarray([r["delta_e_t09"] for r in sub])
            ),
            "delta_e_tau": _bootstrap_mean_ci(
                np.asarray([r["delta_e_tau"] for r in sub])
            ),
            "delta_e_stay_teacher": _bootstrap_mean_ci(
                np.asarray([r["delta_e_stay_teacher"] for r in sub])
            ),
        }

    # cell-level means (N,K,panel)
    cells = {}
    for r in endpoint_rows:
        key = (r["panel"], int(r["n_agents"]), float(r["coupling"]))
        cells.setdefault(key, []).append(r)
    cell_rows = []
    for (panel, n_agents, coupling), rows in sorted(cells.items()):
        cell_rows.append(
            {
                "panel": panel,
                "n_agents": n_agents,
                "coupling": coupling,
                "n_seeds": len(rows),
                "llm_mean_activity": float(np.mean([r["llm_activity"] for r in rows])),
                "v1_mean_activity": float(np.mean([r["v1_activity"] for r in rows])),
                "v2_mean_activity": float(np.mean([r["v2_activity"] for r in rows])),
                "llm_mean_final_r1": float(np.mean([r["llm_final_r1"] for r in rows])),
                "v1_mean_final_r1": float(np.mean([r["v1_final_r1"] for r in rows])),
                "v2_mean_final_r1": float(np.mean([r["v2_final_r1"] for r in rows])),
                "llm_mean_t09": float(np.nanmean([r["llm_t09"] for r in rows])),
                "v1_mean_t09": float(np.nanmean([r["v1_t09"] for r in rows])),
                "v2_mean_t09": float(np.nanmean([r["v2_t09"] for r in rows])),
                "llm_mean_tau": float(np.mean([r["llm_tau"] for r in rows])),
                "v1_mean_tau": float(np.mean([r["v1_tau"] for r in rows])),
                "v2_mean_tau": float(np.mean([r["v2_tau"] for r in rows])),
                "llm_mean_p_stay": float(np.mean([r["llm_p_stay"] for r in rows])),
                "obs_stay_teacher": float(
                    np.mean([r["obs_stay_teacher"] for r in rows])
                ),
                "v1_pred_stay": float(np.mean([r["v1_pred_stay"] for r in rows])),
                "v2_pred_stay": float(np.mean([r["v2_pred_stay"] for r in rows])),
                "lock_frac_llm": float(np.mean([r["locked_llm"] for r in rows])),
                "mean_delta_e_A": float(np.mean([r["delta_e_A"] for r in rows])),
                "mean_delta_e_stay": float(
                    np.mean([r["delta_e_stay_teacher"] for r in rows])
                ),
            }
        )
    cell_path = OUT / "combined_endpoints_by_cell.csv"
    with cell_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cell_rows)

    # transition bounds from LLM lock fractions
    def lock_bound(panel: str, n_agents: int) -> dict:
        sub = [
            c
            for c in cell_rows
            if c["panel"] == panel
            and int(c["n_agents"]) == n_agents
            and float(c["coupling"]) >= 0
        ]
        sub = sorted(sub, key=lambda c: c["coupling"])
        last_zero = None
        first_full = None
        for c in sub:
            if c["lock_frac_llm"] <= 0:
                last_zero = c["coupling"]
            if c["lock_frac_llm"] >= 1.0 and first_full is None:
                first_full = c["coupling"]
        return {
            "last_K_with_zero_lock": last_zero,
            "first_K_with_all_lock": first_full,
            "statement": (
                f"0 < K_cross^LLM(N={n_agents}) <= {first_full}"
                if first_full is not None and (last_zero is not None or first_full > 0)
                else "undetermined"
            ),
        }

    transitions = {
        "N9_v0_2b": lock_bound("v0.2b", 9),
        "N17_v0_2a": lock_bound("v0.2a", 17),
        "surrogate_offline_cross": {"N9": 0.08, "N17": 0.10},
        "note": (
            "LLM onset earlier than surrogate crossing; grid is coarse — "
            "not a precision K_c estimate."
        ),
    }

    summary = {
        "n_runs_total": len(endpoint_rows),
        "panels": panel_summaries,
        "bootstrap_delta_e_v1_minus_v2": boot,
        "transitions": transitions,
        "artifacts": {
            "per_run": str(ep_path.relative_to(ROOT).as_posix()),
            "by_cell": str(cell_path.relative_to(ROOT).as_posix()),
        },
        "supported_claim": (
            "replay-informed regime-aware v2 prospectively improved activity/stay "
            "across unseen K, T=100, new seeds, N, and sign(K)"
        ),
        "not_supported": [
            "full sync-speed prediction",
            "true K_c prediction",
            "negative K = disordered",
            "arbitrary-N generalization",
            "representation-dependent macroscopic phase selection",
        ],
    }
    (OUT / "combined_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "n_runs": len(endpoint_rows),
        "bootstrap_A": boot["overall"]["delta_e_A"],
        "bootstrap_stay": boot["overall"]["delta_e_stay_teacher"],
        "bootstrap_t09": boot["overall"]["delta_e_t09"],
        "transitions": transitions,
    }, indent=2))


if __name__ == "__main__":
    main()
