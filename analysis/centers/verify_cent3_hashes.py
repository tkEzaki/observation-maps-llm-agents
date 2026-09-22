"""Verify CENT-3 freeze manifest artifact hashes before paid run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _h(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest_path = ROOT / "analysis" / "centers_branch" / "cent3_freeze_manifest_v0.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "dataset_manifest": (
            ROOT / "analysis/centers_branch/dataset_manifest_v0.json",
            m["dataset_manifest_sha256"],
        ),
        "risk": (
            ROOT / "analysis/centers_branch/risk_calibrator_centers_v0.json",
            m["risk_calibrator_sha256"],
        ),
        "ad": (
            ROOT / "analysis/centers_branch/applicability_domain_centers_v0/meta.json",
            m["ad_sha256"],
        ),
        "cent12": (
            ROOT / "analysis/centers_branch/cent12_decision.json",
            m["cent12_decision_sha256"],
        ),
        "bank": (
            ROOT / "analysis/stage3a_artifacts/candidate_field_bank.npz",
            m["field_bank_sha256"],
        ),
        "fields": (
            ROOT / "analysis/centers_branch/cent3_frozen_fields_v0.json",
            m["frozen_fields_sha256"],
        ),
        "catalog": (
            ROOT / "analysis/centers_branch/cent3_stimulus_catalog_v0.json",
            m["stimulus_catalog_sha256"],
        ),
        "prospective": (
            ROOT / "analysis/centers_branch/cent3_prospective_lock_v0.json",
            m["prospective_lock_sha256"],
        ),
        "proto1": (
            ROOT / m["protocols"]["block_1"]["path"],
            m["protocols"]["block_1"]["sha256"],
        ),
        "proto2": (
            ROOT / m["protocols"]["block_2"]["path"],
            m["protocols"]["block_2"]["sha256"],
        ),
    }
    ok = True
    for name, (path, expected) in checks.items():
        got = _h(path)
        match = got == expected
        ok = ok and match
        status = "OK" if match else "MISMATCH"
        print(f"{name}: {status}")
        if not match:
            print(f"  path={path}")
            print(f"  expected={expected}")
            print(f"  got={got}")
    print("ALL_OK" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
