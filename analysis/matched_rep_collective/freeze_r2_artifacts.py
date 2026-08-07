"""Freeze R2 Claude macro physical seeds, task order, and source hashes.

Paid acquisition must not run until estimate-only + auth_go are reviewed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from analysis.stage3.collective_scan import _histogram_from_phases  # noqa: E402
from circlemap.representations import (  # noqa: E402
    build_representation_prompt_from_histogram,
)
from experiments.stage_c.run_collective import _omega  # noqa: E402

OUT = ROOT / "analysis" / "matched_rep_collective" / "r2_macro"
PROTOCOL = (
    ROOT
    / "experiments"
    / "stage_c"
    / "protocol_matched_rep_collective_r2_claude_v0_1.json"
)
GPT_MATCH = ROOT / "analysis" / "matched_rep_collective" / "match_manifest.json"
REPS = ("moments_m1_m3", "centers_24_standard", "intervals_24_decimal6")


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_arr(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sha_json(obj: object) -> str:
    return _sha_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def _theta0(init_seed: int, n_agents: int) -> np.ndarray:
    rng = np.random.default_rng(init_seed)
    return rng.uniform(-np.pi, np.pi, size=n_agents)


def _init_seed(base_seed: int, n_agents: int, seed_i: int, coupling: float) -> int:
    return (
        int(base_seed)
        + 1000 * int(n_agents)
        + 10 * int(seed_i)
        + int(round(abs(float(coupling)) * 1000))
    )


def _run_id(n_agents: int, coupling: float, seed_i: int) -> str:
    return f"N{int(n_agents)}_K{float(coupling):+g}_s{int(seed_i)}"


def _sample_seed(
    *,
    family: str,
    model_id: str,
    representation: str,
    init_seed: int,
    coupling: float,
    agent: int,
    t: int,
) -> int:
    payload = (
        f"{family}|{model_id}|{representation}|{init_seed}|"
        f"{float(coupling):.12g}|{agent}|{t}"
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % 2_147_483_647


def build_cells(protocol: dict) -> list[dict]:
    n = int(protocol["n_agents"])
    base = int(protocol["base_seed"])
    half = float(protocol["omega_halfwidth"])
    omega = _omega(n, half)
    omega_sha = _sha_arr(omega)
    cells: list[dict] = []

    panels = [
        ("core", protocol["core_panel"]),
        ("heldout", protocol["heldout_panel"]),
    ]
    for panel_name, panel in panels:
        for seed_i in panel["seed_indices"]:
            for coupling in panel["couplings"]:
                init = _init_seed(base, n, seed_i, coupling)
                theta = _theta0(init, n)
                cells.append(
                    {
                        "panel": panel_name,
                        "run_id": _run_id(n, coupling, seed_i),
                        "n_agents": n,
                        "coupling": float(coupling),
                        "seed_index": int(seed_i),
                        "init_seed": int(init),
                        "theta0_sha256": _sha_arr(theta),
                        "omega_sha256": omega_sha,
                        "legacy_gpt_matched": panel_name == "core",
                    }
                )
    return cells


def verify_legacy_against_gpt(cells: list[dict], gpt_match: dict) -> dict:
    gpt_by_id = {c["run_id"]: c for c in gpt_match["physical_cells"]}
    checks = []
    ok = True
    for cell in cells:
        if cell["panel"] != "core":
            continue
        gpt = gpt_by_id.get(cell["run_id"])
        if gpt is None:
            checks.append({"run_id": cell["run_id"], "ok": False, "reason": "missing_in_gpt"})
            ok = False
            continue
        match = (
            gpt["init_seed"] == cell["init_seed"]
            and gpt["theta0_sha256"] == cell["theta0_sha256"]
            and gpt["omega_sha256"] == cell["omega_sha256"]
        )
        checks.append(
            {
                "run_id": cell["run_id"],
                "ok": match,
                "init_seed": cell["init_seed"],
                "theta0_sha256": cell["theta0_sha256"],
            }
        )
        ok = ok and match
    return {"pass": ok, "n_core_cells": len(checks), "checks": checks}


def build_task_manifest(protocol: dict, cells: list[dict]) -> dict:
    """Freeze acquisition order: core then heldout; within panel by seed then K."""
    order_key = {
        "core": 0,
        "heldout": 1,
    }
    k_order = {0.0: 0, 0.08: 1, 0.15: 2}
    sorted_cells = sorted(
        cells,
        key=lambda c: (
            order_key[c["panel"]],
            c["seed_index"],
            k_order[float(c["coupling"])],
        ),
    )
    tasks = []
    for i, cell in enumerate(sorted_cells):
        tasks.append(
            {
                "order_index": i,
                "cell_id": cell["run_id"],
                "panel": cell["panel"],
                "seed_index": cell["seed_index"],
                "coupling": cell["coupling"],
                "init_seed": cell["init_seed"],
                "representation_order": list(
                    protocol["acquisition"]["representation_order"]
                ),
                "schedule": "round_robin_reps_per_timestep",
                "n_steps": int(protocol["n_steps"]),
                "n_agents": int(protocol["n_agents"]),
                "expected_calls": (
                    int(protocol["n_steps"])
                    * int(protocol["n_agents"])
                    * len(REPS)
                ),
            }
        )
    return {
        "protocol_version": protocol["protocol_version"],
        "n_cells": len(tasks),
        "n_representations": len(REPS),
        "expected_calls_total": sum(t["expected_calls"] for t in tasks),
        "tasks": tasks,
    }


def build_prompt_hashes(protocol: dict) -> dict:
    n = int(protocol["n_agents"])
    n_bins = int(protocol["n_bins"])
    # Canonical flat histogram for encoder/prompt identity freeze.
    phases = np.linspace(-np.pi, np.pi, n, endpoint=False)
    hashes = {}
    for rep in REPS:
        hist = _histogram_from_phases(phases, 0, n_bins=n_bins)
        prompt = build_representation_prompt_from_histogram(rep, hist)
        hashes[rep] = {
            "canonical_prompt_sha256": _sha_text(prompt),
            "canonical_prompt_chars": len(prompt),
        }
    return {
        "prompt_version": protocol["prompt_version"],
        "canonical_state": "linspace_phases_endpoint_false_agent0",
        "representations": hashes,
        "aggregate_sha256": _sha_json(hashes),
    }


def build_source_hashes(protocol: dict) -> dict:
    paths = {
        "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
        "freeze_script": (
            Path(__file__).resolve().relative_to(ROOT).as_posix()
        ),
        "runner": "experiments/stage_c/run_r2_claude_macro.py",
        "analyzer": "analysis/matched_rep_collective/analyze_r2_macro.py",
        "gpt_match_manifest": GPT_MATCH.relative_to(ROOT).as_posix(),
        "hypothesis": protocol["docs"]["hypothesis"],
        "protocol_doc": protocol["docs"]["protocol"],
        "analysis_plan": protocol["docs"]["analysis_plan"],
    }
    file_hashes = {}
    for key, rel in paths.items():
        path = ROOT / rel
        if path.exists():
            file_hashes[key] = {
                "path": rel,
                "sha256": _sha_file(path),
            }
        else:
            file_hashes[key] = {"path": rel, "sha256": None, "missing": True}

    # Encoder source identity via representations module source.
    import circlemap.representations as reps_mod

    file_hashes["representations_module_source_sha256"] = _sha_text(
        inspect.getsource(reps_mod)
    )
    return {
        "protocol_version": protocol["protocol_version"],
        "files": file_hashes,
    }


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    gpt_match = json.loads(GPT_MATCH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    cells = build_cells(protocol)
    legacy = verify_legacy_against_gpt(cells, gpt_match)
    if not legacy["pass"]:
        raise SystemExit(f"legacy GPT physical IC mismatch: {legacy}")

    physical = {
        "protocol_version": protocol["protocol_version"],
        "base_seed": protocol["base_seed"],
        "n_agents": protocol["n_agents"],
        "omega_halfwidth": protocol["omega_halfwidth"],
        "model": protocol["model"],
        "family": protocol["family"],
        "n_cells": len(cells),
        "expected_calls": protocol["expected_calls"],
        "legacy_gpt_verification": legacy,
        "cells": cells,
        "sample_seed_scheme": (
            "sha256(family|model_id|representation|init_seed|K|agent|t) % 2^31-1"
        ),
        "sample_seed_example": {
            "payload_fields": [
                protocol["family"],
                protocol["model"],
                REPS[0],
                cells[0]["init_seed"],
                cells[0]["coupling"],
                0,
                0,
            ],
            "sample_seed": _sample_seed(
                family=protocol["family"],
                model_id=protocol["model"],
                representation=REPS[0],
                init_seed=cells[0]["init_seed"],
                coupling=cells[0]["coupling"],
                agent=0,
                t=0,
            ),
        },
    }
    task_manifest = build_task_manifest(protocol, cells)
    if task_manifest["expected_calls_total"] != int(protocol["expected_calls"]):
        raise SystemExit(
            "call budget mismatch: "
            f"{task_manifest['expected_calls_total']} vs {protocol['expected_calls']}"
        )

    prompt_hashes = build_prompt_hashes(protocol)
    source_hashes = build_source_hashes(protocol)

    auth_go = {
        "experiment_family": protocol["experiment_family"],
        "protocol_version": protocol["protocol_version"],
        "paid_authorized": False,
        "mock_smoke_authorized": True,
        "estimate_only_reviewed": False,
        "human_review_complete": False,
        "cost_ceiling_usd": protocol["cost_ceiling_usd"],
        "notes": "Flip paid_authorized only after estimate-only + human review GO.",
    }

    (OUT / "physical_seed_manifest.json").write_text(
        json.dumps(physical, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "task_manifest.json").write_text(
        json.dumps(task_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "prompt_hashes.json").write_text(
        json.dumps(prompt_hashes, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "source_hashes.json").write_text(
        json.dumps(source_hashes, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "auth_go.json").write_text(
        json.dumps(auth_go, indent=2) + "\n", encoding="utf-8"
    )

    # Placeholder until --estimate-only fills it.
    estimate_stub = {
        "status": "pending_estimate_only",
        "protocol_version": protocol["protocol_version"],
        "expected_calls": protocol["expected_calls"],
        "command": (
            "py -m experiments.stage_c.run_r2_claude_macro --estimate-only"
        ),
    }
    est_path = OUT / "estimate_only.json"
    if not est_path.exists() or json.loads(est_path.read_text()).get(
        "status"
    ) == "pending_estimate_only":
        est_path.write_text(
            json.dumps(estimate_stub, indent=2) + "\n", encoding="utf-8"
        )

    print(f"Wrote R2 freeze artifacts under {OUT.as_posix()}")
    print(f"  cells={len(cells)} calls={task_manifest['expected_calls_total']}")
    print(f"  legacy_gpt_match={legacy['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
