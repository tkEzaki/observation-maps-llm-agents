"""Hash-lock Intervals collective STOP manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRANCH = ROOT / "analysis" / "intervals_branch"


def _sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    decision = json.loads(
        (BRANCH / "int2b_offline_decision.json").read_text(encoding="utf-8")
    )
    manifest = {
        "status": "stopped",
        "paid_authorized": False,
        "freeze_ready": False,
        "stage_c_authorized": False,
        "further_surrogate_engineering": False,
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "representation": "intervals_24_decimal6",
        "layers": {
            "microscopic_transmutation": "supported",
            "peer_stratified_in_distribution_compressibility": "supported",
            "prospective_collective_transfer": "not_supported",
        },
        "int3_prospective_pilot_go": False,
        "intervals_collective_stop": True,
        "locked_wording": decision.get("locked_stop_wording"),
        "support_gate_locked": decision.get("support_gate_locked"),
        "doc_sha256": {
            "INTERVALS_FINAL_DECISION.md": _sha(
                ROOT / "docs" / "INTERVALS_FINAL_DECISION.md"
            ),
            "INTERVALS_INT0_2.md": _sha(ROOT / "docs" / "INTERVALS_INT0_2.md"),
            "INTERVALS_BRANCH.md": _sha(ROOT / "docs" / "INTERVALS_BRANCH.md"),
        },
        "artifact_sha256": {
            "int2b_offline_decision.json": _sha(BRANCH / "int2b_offline_decision.json"),
            "int2_offline_decision.json": _sha(BRANCH / "int2_offline_decision.json"),
            "dataset_manifest_intervals_v0.json": _sha(
                BRANCH / "dataset_manifest_intervals_v0.json"
            ),
        },
        "forbidden_actions": [
            "int3_coverage_pilot",
            "int4_confirmatory_pilot",
            "intervals_bundle_v1_freeze",
            "intervals_collective_bundle_v1_N17_freeze",
            "moments_intervals_matched_stage_c",
            "further_feature_mixture_chasing_on_current_manifold",
        ],
        "next_priority": (
            "archive histogram collective paths; moments Stage C remains the "
            "only frozen collective representation path"
        ),
    }
    body = json.dumps(manifest, indent=2, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    out = BRANCH / "final_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "stop": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "path": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
