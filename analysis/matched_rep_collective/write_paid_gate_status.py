"""Paid-gate checklist status for matched representation collective.

Does not authorize spend. Records what remains before OpenAI --yes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTH = ROOT / "analysis" / "matched_rep_collective" / "auth_go.json"
OUT = ROOT / "analysis" / "matched_rep_collective" / "paid_gate_status.json"


def main() -> int:
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    estimate_log = ROOT / "analysis" / "matched_rep_collective" / "estimate_only_log.json"
    manifest = ROOT / "analysis" / "matched_rep_collective" / "match_manifest.json"
    smoke = ROOT / "analysis" / "matched_rep_collective" / "smoke" / "smoke_report.json"
    status = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "paid_authorized": bool(auth.get("paid_authorized")),
        "mock_smoke_authorized": bool(auth.get("mock_smoke_authorized")),
        "match_manifest_exists": manifest.exists(),
        "estimate_only_log_exists": estimate_log.exists(),
        "mock_smoke_ok": smoke.exists()
        and json.loads(smoke.read_text(encoding="utf-8")).get("status") == "smoke_ok",
        "gates_already_done": [
            "three per-rep character-heuristic --estimate-only logged on cost card",
            "match manifest locked (24 cells x 3 reps)",
            "mock smoke IC-match passed",
            "replay field rules locked pre-outcome",
            "protocols + surrogate-free auth path implemented",
        ],
        "gates_remaining_before_paid": [
            "human review checklist signed (docs/MATCHED_REP_COLLECTIVE_HUMAN_REVIEW.md)",
            "flip auth_go.json paid_authorized to true",
            "explicit user --yes",
            "then run N=17 x 4K x 6seeds x 3reps",
            "then 3x3 replay under MATCHED_REP_COLLECTIVE_REPLAY_LOCK.md",
            "then conditional R1/R2 second-model only if Outcome A or B",
        ],
        "explicit_non_actions": [
            "no INT/CENT surrogate reopen",
            "no OpenAI spend while paid_authorized is false",
            "no N=9 before N=17 effect",
        ],
        }
    OUT.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
