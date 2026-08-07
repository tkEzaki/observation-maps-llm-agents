"""Refresh hash-freeze manifest with protocol / system-prompt / docs SHAs.

Does not reselect fields. Safe to run after protocol authoring.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circlemap.representations import _instruction  # noqa: E402
from circlemap.response import PROMPT_VERSION  # noqa: E402

OUT = ROOT / "analysis" / "matched_rep_collective"
PROTOCOL = (
    ROOT
    / "experiments"
    / "stage_c"
    / "protocol_matched_rep_collective_replay_v0_1.json"
)
ANALYSIS_MD = ROOT / "docs" / "MATCHED_REP_COLLECTIVE_REPLAY_ANALYSIS_PLAN.md"
PLAN_JSON = OUT / "replay_analysis_plan_v0_1.json"
FREEZE = OUT / "replay_hash_freeze_v0_1.json"
FIELDS = OUT / "replay_fields_v0_1.json"
AUDIT = OUT / "replay_selection_audit_v0_1.json"
PROMPTS = OUT / "replay_target_prompts_v0_1.json"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None

    # Canonical instruction template (observation_description placeholder fixed)
    system_text = (
        f"PROMPT_VERSION={PROMPT_VERSION}\n"
        + _instruction("{observation_description}")
    )

    freeze.update(
        {
            "status": "selection_hash_freeze_v0_1_complete",
            "paid_authorized": False,
            "replay_fields_v0_1_sha256": _sha_bytes(FIELDS.read_bytes()),
            "selection_audit_sha256": _sha_bytes(AUDIT.read_bytes()),
            "target_prompts_v0_1_sha256": _sha_bytes(PROMPTS.read_bytes()),
            "protocol_sha256": _sha_bytes(PROTOCOL.read_bytes()),
            "protocol_path": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
            "system_prompt_version": PROMPT_VERSION,
            "system_prompt_sha256": _sha_bytes(system_text.encode("utf-8")),
            "analysis_plan_json_sha256": _sha_bytes(PLAN_JSON.read_bytes()),
            "analysis_plan_md_sha256": _sha_bytes(ANALYSIS_MD.read_bytes()),
            "analysis_plan_sha256": _sha_bytes(PLAN_JSON.read_bytes()),
            "runner_commit_hash": commit,
            "n_target_prompts": json.loads(PROMPTS.read_text(encoding="utf-8"))[
                "n_target_prompts"
            ],
            "note": (
                "Field hashes frozen. Paid still false until human sign-off "
                "+ estimate-only under ceiling + explicit authorization."
            ),
        }
    )
    FREEZE.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
