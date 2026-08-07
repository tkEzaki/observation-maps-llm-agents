"""Write match_manifest.json verifying physical IC identity across three protocols."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage_c.run_collective import _omega, _run_specs  # noqa: E402

PROTOCOLS = [
    ROOT
    / "experiments"
    / "stage_c"
    / "protocol_matched_rep_collective_v0_1_moments.json",
    ROOT
    / "experiments"
    / "stage_c"
    / "protocol_matched_rep_collective_v0_1_centers.json",
    ROOT
    / "experiments"
    / "stage_c"
    / "protocol_matched_rep_collective_v0_1_intervals.json",
]
OUT = ROOT / "analysis" / "matched_rep_collective" / "match_manifest.json"


def _theta0(init_seed: int, n_agents: int) -> np.ndarray:
    rng = np.random.default_rng(init_seed)
    return rng.uniform(-np.pi, np.pi, size=n_agents)


def _sha_arr(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> int:
    loaded = []
    for path in PROTOCOLS:
        proto = json.loads(path.read_text(encoding="utf-8"))
        loaded.append((path, proto))

    # Shared locks
    keys = [
        "base_seed",
        "n_agents",
        "couplings",
        "n_steps",
        "n_seeds",
        "omega_halfwidth",
        "n_bins",
        "match_group",
        "experiment_family",
        "surrogate_free",
        "prompt_version",
    ]
    ref = loaded[0][1]
    mismatches = []
    for path, proto in loaded[1:]:
        for key in keys:
            if proto.get(key) != ref.get(key):
                mismatches.append(
                    {
                        "key": key,
                        "ref": ref.get(key),
                        "other": proto.get(key),
                        "protocol": path.name,
                    }
                )
    if mismatches:
        raise SystemExit(f"protocol shared-field mismatch: {mismatches}")

    reps = [p["representation"] for _, p in loaded]
    if len(set(reps)) != 3:
        raise SystemExit(f"expected 3 distinct representations, got {reps}")

    versions = [p["protocol_version"] for _, p in loaded]
    if len(set(versions)) != 3:
        raise SystemExit(f"protocol_version must differ per rep: {versions}")

    specs = _run_specs(ref)
    cells = []
    for spec in specs:
        theta = _theta0(spec["init_seed"], spec["n_agents"])
        omega = _omega(spec["n_agents"], float(ref["omega_halfwidth"]))
        cells.append(
            {
                "run_id": spec["run_id"],
                "n_agents": spec["n_agents"],
                "coupling": spec["coupling"],
                "seed_index": spec["seed_index"],
                "init_seed": spec["init_seed"],
                "theta0_sha256": _sha_arr(theta),
                "omega_sha256": _sha_arr(omega),
            }
        )

    # Verify identical init_seed / theta / omega would be produced under each
    # protocol's _run_specs (representation-free formula).
    for path, proto in loaded[1:]:
        other = _run_specs(proto)
        if len(other) != len(specs):
            raise SystemExit("spec count mismatch across protocols")
        for a, b in zip(specs, other):
            if a != b:
                raise SystemExit(f"run spec mismatch {a} vs {b} ({path.name})")

    manifest = {
        "match_group": ref["match_group"],
        "base_seed": ref["base_seed"],
        "n_cells": len(cells),
        "expected_calls_per_rep": ref["expected_calls"],
        "expected_calls_total": int(ref["expected_calls"]) * 3,
        "shared_locks": {k: ref[k] for k in keys},
        "representations": [
            {
                "representation": proto["representation"],
                "protocol_version": proto["protocol_version"],
                "protocol_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path, proto in loaded
        ],
        "physical_cells": cells,
        "api_sample_seed_policy": (
            "hash(protocol_version|representation|base|run_id|t|agent); "
            "independent across encodings; physical IC matched"
        ),
        "status": "locked",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} matched cells × 3 reps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
