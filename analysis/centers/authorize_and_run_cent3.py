"""Authorize CENT-3 after GO, verify hashes/cost, run both paid blocks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.centers.verify_cent3_hashes import main as verify_hashes  # noqa: E402

BRANCH = ROOT / "analysis" / "centers_branch"
COST_CEILING_USD = 1.773
ENV_FILE = Path(r"<HOME>\Dropbox\research_current\gen_ai_logi\rq1\.env")


def _authorize() -> None:
    manifest_path = BRANCH / "cent3_freeze_manifest_v0.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    m["status"] = "human_review_accepted_paid_authorized"
    m["paid_run_authorized"] = True
    m["human_review"] = {
        "reviewer": "author",
        "date": "2026-07-24",
        "decision": "Human review accepted",
        "paid_authorization": "YES — CENT-3 pilot spend GO",
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "representation": "centers_24_standard",
            "n_fields": 36,
            "responses_per_condition": 16,
            "blocks": 2,
            "total_calls": 1152,
            "cost_ceiling_usd": COST_CEILING_USD,
            "prospective_lock_immutable": True,
            "no_post_hoc_field_drop": True,
        },
    }
    manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")


def _estimate_block(protocol: Path) -> float:
    cmd = [
        sys.executable,
        "-m",
        "experiments.stage_b_response_law.run_representation",
        "--backend",
        "openai",
        "--model",
        "gpt-5.4-mini",
        "--env-file",
        str(ENV_FILE),
        "--protocol",
        str(protocol),
        "--input-price-per-million",
        "0.75",
        "--output-price-per-million",
        "4.5",
        "--max-output-tokens",
        "16",
        "--concurrency",
        "20",
        "--estimate-only",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout or "estimate failed")
    ceiling = None
    for line in proc.stdout.splitlines():
        if "retry ceiling" in line.lower():
            # e.g. "retry ceiling (3 attempts): $0.8865"
            ceiling = float(line.split("$")[-1].strip())
    if ceiling is None:
        raise SystemExit(f"could not parse ceiling from:\n{proc.stdout}")
    print(proc.stdout)
    return ceiling


def _run_block(protocol: Path, run_log: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "experiments.stage_b_response_law.run_representation",
        "--backend",
        "openai",
        "--model",
        "gpt-5.4-mini",
        "--env-file",
        str(ENV_FILE),
        "--protocol",
        str(protocol),
        "--input-price-per-million",
        "0.75",
        "--output-price-per-million",
        "4.5",
        "--max-output-tokens",
        "16",
        "--concurrency",
        "20",
        "--yes",
        "--out-root",
        str(ROOT / "runs" / "centers_cent3"),
    ]
    with run_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **{k: v for k, v in __import__("os").environ.items()},
                "PYTHONPATH": str(ROOT),
            },
        )
        (BRANCH / "cent3_active_pid").write_text(str(proc.pid), encoding="utf-8")
        print(f"started pid={proc.pid} protocol={protocol.name} log={run_log}")
        return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-only", action="store_true")
    parser.add_argument("--skip-authorize", action="store_true")
    args = parser.parse_args()

    print("Verifying freeze hashes...")
    if verify_hashes() != 0:
        print("STOP: manifest artifact hash mismatch")
        return 2

    if not args.skip_authorize:
        _authorize()
        print("Authorization recorded in cent3_freeze_manifest_v0.json")
    if args.authorize_only:
        return 0

    p1 = ROOT / "experiments/stage_b_response_law/protocol_centers_cent3_v0_block_1.json"
    p2 = ROOT / "experiments/stage_b_response_law/protocol_centers_cent3_v0_block_2.json"
    print("Estimating costs...")
    c1 = _estimate_block(p1)
    c2 = _estimate_block(p2)
    total_ceiling = c1 + c2
    print(f"total_retry_ceiling_usd={total_ceiling:.4f} limit={COST_CEILING_USD}")
    if total_ceiling > COST_CEILING_USD + 1e-9:
        print("STOP: pre-run estimate exceeds authorized ceiling")
        return 3

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = BRANCH / "cent3_run_logs" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "preflight.json").write_text(
        json.dumps(
            {
                "total_retry_ceiling_usd": total_ceiling,
                "limit_usd": COST_CEILING_USD,
                "max_output_tokens": 16,
                "blocks": [str(p1), str(p2)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    t0 = time.time()
    rc1 = _run_block(p1, log_dir / "block_1.log")
    # rc=3 means some invalid parses but all planned calls completed.
    if rc1 not in {0, 3}:
        print(f"block_1 failed rc={rc1}")
        return rc1
    rc2 = _run_block(p2, log_dir / "block_2.log")
    if rc2 not in {0, 3}:
        print(f"block_2 failed rc={rc2}")
        return rc2
    (BRANCH / "cent3_active_pid").unlink(missing_ok=True)
    print(f"both blocks complete in {(time.time()-t0)/60:.1f} min")
    (log_dir / "status.json").write_text(
        json.dumps({"ok": True, "elapsed_s": time.time() - t0}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
