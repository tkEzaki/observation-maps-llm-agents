"""Write hash-locked Centers final close manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRANCH = ROOT / "analysis" / "centers_branch"


def _sha_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    decision_path = BRANCH / "cent4_final_offline_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    docs = {
        "CENTERS_FINAL_DECISION.md": ROOT / "docs" / "CENTERS_FINAL_DECISION.md",
        "CENTERS_FINAL_OFFLINE.md": ROOT / "docs" / "CENTERS_FINAL_OFFLINE.md",
        "CENTERS_CENT4_REVISION.md": ROOT / "docs" / "CENTERS_CENT4_REVISION.md",
        "CENTERS_CENT4_OFFLINE.md": ROOT / "docs" / "CENTERS_CENT4_OFFLINE.md",
        "CENTERS_BRANCH.md": ROOT / "docs" / "CENTERS_BRANCH.md",
    }
    artifacts = {
        "cent4_final_offline_decision.json": decision_path,
        "collective_scope_split_v0.json": BRANCH / "collective_scope_split_v0.json",
        "cent4_revision_decision.json": BRANCH / "cent4_revision_decision.json",
        "cent3_prospective_summary.json": BRANCH / "cent3_prospective_summary.json",
        "training_arrays_centers_v0.npz": BRANCH / "training_arrays_centers_v0.npz",
    }
    manifest = {
        "status": "stopped",
        "paid_authorized": False,
        "freeze_ready": False,
        "stage_c_authorized": False,
        "further_surrogate_engineering": False,
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "representation": "centers_24_standard",
        "layers": {
            "microscopic_transmutation": "supported",
            "peer_stratified_in_distribution_compressibility": "supported",
            "prospective_collective_transfer": "not_supported",
        },
        "cent3_role": (
            "initial prospective evaluation; subsequently used as a "
            "development-transfer benchmark"
        ),
        "locked_wording": (
            "Center-bin serialization induces reproducible microscopic "
            "transmutation and peer-regime dependence, but its finite-peer "
            "response surface was not sufficiently stable or compressible for "
            "prospective collective prediction under the tested descriptors "
            "and coverage."
        ),
        "stop_rule_fired": bool(decision.get("stop_centers_collective_path")),
        "confirmatory_pilot_go": False,
        "doc_sha256": {name: _sha_file(path) for name, path in docs.items()},
        "artifact_sha256": {name: _sha_file(path) for name, path in artifacts.items()},
        "forbidden_actions": [
            "confirmatory_768_call_pilot",
            "centers_collective_bundle_v1_freeze",
            "centers_collective_bundle_v1_N17_freeze",
            "centers_matched_stage_c",
            "cent3_informed_feature_chasing",
            "paid_stay_collapse_coverage_acquisition",
        ],
    }
    body = json.dumps(manifest, indent=2, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    out = BRANCH / "final_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "stop_rule_fired": manifest["stop_rule_fired"],
        "manifest_sha256": manifest["manifest_sha256"],
        "path": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
