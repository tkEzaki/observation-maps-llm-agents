"""Serialization-length control runner (48 fields x 4 arms x 16 responses).

Uses the frozen physical fields of the identical-field replay. Arms differ only
in how the retained numbers are laid out as text: within each content class the
compact and standard forms carry the same values at the same precision, and no
filler text is added anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circlemap.backends import OpenAIResponseBackend, QuotaError  # noqa: E402
from circlemap.costing import estimate_cost, format_cost_estimate  # noqa: E402
from circlemap.envload import load_project_env  # noqa: E402
from circlemap.observation import RelativePhaseHistogram  # noqa: E402
from circlemap.serialization_length_controls import (  # noqa: E402
    SLC_VARIANTS,
    build_slc_prompt,
)
from experiments.stage_c.run_matched_rep_collective_replay import (  # noqa: E402
    FREEZE_PATH,
    REQUIRED_RETENTION,
    _load_trace,
    _sha_file,
    _sha_text,
    run_one_task,
    verify_freeze,
)

DEFAULT_PROTOCOL = Path(__file__).with_name(
    "protocol_matched_rep_collective_slc_v0_1.json"
)
AUTH_PATH = ROOT / "analysis/matched_rep_collective/slc_auth_go.json"
AUDIT_PATH = ROOT / "analysis/matched_rep_collective/slc_information_audit.json"


@dataclass(frozen=True)
class SlcTask:
    task_id: str
    field_id: str
    physical_hash: str
    source_representation: str
    stratum: str
    target_representation: str  # control variant name
    acquisition_block: int
    sample_index: int
    sample_seed: int
    prompt: str
    prompt_sha256: str


def build_tasks(protocol: dict, fields: list[dict]) -> list[SlcTask]:
    n_blocks = int(protocol["n_blocks"])
    per_block = int(protocol["responses_per_block"])
    variants = list(protocol["control_variants"])
    assert tuple(variants) == SLC_VARIANTS
    seed_prefix = protocol.get("seed_prefix", "mrc-slc-v0.1")
    edges = np.linspace(-np.pi, np.pi, 25)
    tasks: list[SlcTask] = []
    for field in fields:
        hist = RelativePhaseHistogram(
            edges=edges,
            fractions=np.asarray(field["physical_histogram_24"], dtype=float),
            peer_count=int(field.get("peer_count", 16)),
        )
        for variant in variants:
            prompt = build_slc_prompt(variant, hist)
            ph = _sha_text(prompt)
            for block in range(n_blocks):
                for sample in range(per_block):
                    payload = (
                        f"{seed_prefix}|{field['field_id']}|{variant}|"
                        f"{block}|{sample}"
                    )
                    sample_seed = (
                        int.from_bytes(
                            hashlib.sha256(payload.encode()).digest()[:8], "big"
                        )
                        % 2_147_483_647
                    )
                    task_id = (
                        f"{field['field_id']}__{variant}__"
                        f"block{block}__s{sample:02d}"
                    )
                    tasks.append(
                        SlcTask(
                            task_id=task_id,
                            field_id=field["field_id"],
                            physical_hash=field["physical_hash"],
                            source_representation=field["source_representation"],
                            stratum=field["stratum"],
                            target_representation=variant,
                            acquisition_block=block,
                            sample_index=sample,
                            sample_seed=sample_seed,
                            prompt=prompt,
                            prompt_sha256=ph,
                        )
                    )
    return tasks


def audit_tasks(tasks: list[SlcTask], protocol: dict) -> dict:
    expected = int(protocol["expected_calls"])
    per_block = int(protocol["responses_per_block"])
    n_blocks = int(protocol["n_blocks"])
    n_variants = len(protocol["control_variants"])
    per_cond: dict[tuple, int] = {}
    for t in tasks:
        key = (t.field_id, t.target_representation, t.acquisition_block)
        per_cond[key] = per_cond.get(key, 0) + 1
    checks = {
        "n_tasks_eq_expected": len(tasks) == expected,
        "n_fields_48": len({t.field_id for t in tasks}) == 48,
        "n_variants_match": len({t.target_representation for t in tasks}) == n_variants,
        "n_blocks_match": len({t.acquisition_block for t in tasks}) == n_blocks,
        "unique_task_ids": len({t.task_id for t in tasks}) == len(tasks),
        "per_condition_match": all(v == per_block for v in per_cond.values())
        and len(per_cond) == 48 * n_variants * n_blocks,
        "one_prompt_per_field_and_variant": len(
            {(t.field_id, t.target_representation, t.prompt_sha256) for t in tasks}
        )
        == 48 * n_variants,
    }
    return {"pass": all(checks.values()), "checks": checks, "n_tasks": len(tasks)}


def require_offline_audit() -> dict:
    """Refuse to spend money unless the information-equivalence audit passed."""
    if not AUDIT_PATH.exists():
        raise SystemExit(
            "SLC_AUDIT_MISSING: run "
            "analysis/matched_rep_collective/verify_slc_information_equivalence.py"
        )
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if not audit.get("pass"):
        raise SystemExit("SLC_AUDIT_FAIL")
    eq = audit["information_equivalence"]
    if eq["max_abs_moment_component_gap"] != 0.0 or eq["max_abs_bin_mass_gap"] != 0.0:
        raise SystemExit("SLC_AUDIT_INFORMATION_NOT_IDENTICAL")
    if not all(audit["standard_arms_byte_identical_to_primary"].values()):
        raise SystemExit("SLC_AUDIT_STANDARD_ARMS_DRIFTED")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--backend", choices=("openai",), default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--input-price-per-million", type=float, default=0.75)
    parser.add_argument("--output-price-per-million", type=float, default=4.5)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "runs" / "matched_rep_collective_slc",
    )
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--preflight-tasks", type=int, default=8)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    load_project_env()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    ceiling = float(protocol["cost_ceiling_usd"])
    fields_path = ROOT / protocol["fields_file"]
    fields = json.loads(fields_path.read_text(encoding="utf-8"))["fields"]
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

    freeze_report = verify_freeze(protocol, fields_path, fields, freeze)
    field_ok = all(
        freeze_report["checks"][k]
        for k in (
            "fields_file_sha",
            "physical_aggregate",
            "n_fields_48",
            "unique_physical_48",
        )
    )
    print(json.dumps(freeze_report, indent=2), flush=True)
    if not field_ok:
        raise SystemExit("FIELD_FREEZE_VERIFY_FAIL")

    audit = require_offline_audit()
    print("Offline information audit: PASS", flush=True)

    tasks = build_tasks(protocol, fields)
    construction = audit_tasks(tasks, protocol)
    print("Task construction:", json.dumps(construction, indent=2), flush=True)
    if not construction["pass"]:
        raise SystemExit("TASK_CONSTRUCTION_AUDIT_FAIL")

    estimate = estimate_cost(
        [t.prompt for t in tasks],
        max_output_tokens_per_call=args.max_output_tokens,
        max_attempts_per_call=args.max_attempts,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    print(
        format_cost_estimate(estimate).replace(
            "Stage B", "Serialization-length control"
        ),
        flush=True,
    )
    if estimate.estimated_base_cost_usd > ceiling:
        raise SystemExit(
            f"COST_CEILING_EXCEEDED: {estimate.estimated_base_cost_usd:.4f} > {ceiling}"
        )
    if args.estimate_only:
        return 0

    if not AUTH_PATH.exists():
        raise SystemExit("SLC_PAID_AUTH_MISSING")
    auth = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    if not auth.get("paid_authorized"):
        raise SystemExit("SLC_PAID_AUTH_FALSE")
    if not args.yes:
        raise SystemExit("Refusing paid serialization-length run without --yes")

    backend = OpenAIResponseBackend(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.resume_dir is not None:
        run_dir = args.resume_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = args.out_root / f"{protocol['protocol_version']}_{args.model}" / stamp
        run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "protocol_sha256": _sha_file(args.protocol),
        "fields_sha256": _sha_file(fields_path),
        "freeze_sha256": _sha_file(FREEZE_PATH),
        "audit_sha256": _sha_file(AUDIT_PATH),
        "backend": args.backend,
        "model": args.model,
        "max_attempts": args.max_attempts,
        "control_variants": list(protocol["control_variants"]),
        "estimate_base_usd": estimate.estimated_base_cost_usd,
        "cost_ceiling_usd": ceiling,
        "expected_calls": int(protocol["expected_calls"]),
        "prompt_chars": {
            v: audit["prompt_chars"][v]["median"] for v in protocol["control_variants"]
        },
    }
    cfg_path = run_dir / "resolved_config.json"
    if cfg_path.exists():
        existing = json.loads(cfg_path.read_text(encoding="utf-8"))
        for k in ("protocol_sha256", "fields_sha256", "backend", "model"):
            if existing.get(k) != config.get(k):
                raise ValueError(f"resume mismatch: {k}")
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (run_dir / "information_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    trace_path = run_dir / "trace.jsonl"
    completed = {r["task_id"] for r in _load_trace(trace_path)}
    pending = [t for t in tasks if t.task_id not in completed]
    print(f"pending calls: {len(pending)}/{len(tasks)}", flush=True)

    if args.preflight_tasks > 0 and pending:
        n_pre = min(args.preflight_tasks, len(pending))
        print(f"Preflight: {n_pre} in-budget tasks...", flush=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            for task in pending[:n_pre]:
                rec = run_one_task(task, backend, max_attempts=args.max_attempts)
                handle.write(json.dumps(rec, ensure_ascii=True) + "\n")
                handle.flush()
                missing = [k for k in REQUIRED_RETENTION if k not in rec]
                if missing:
                    raise SystemExit(f"PREFLIGHT_RETENTION_FAIL: {missing}")
        completed = {r["task_id"] for r in _load_trace(trace_path)}
        pending = [t for t in tasks if t.task_id not in completed]
        print(f"Preflight done; remaining {len(pending)}", flush=True)

    n_done = n_valid = 0
    tin = tout = 0
    t0 = time.perf_counter()
    with trace_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    run_one_task, task, backend, max_attempts=args.max_attempts
                ): task
                for task in pending
            }
            for fut in as_completed(futures):
                try:
                    record = fut.result()
                except QuotaError:
                    raise
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                n_done += 1
                tin += int(record.get("input_tokens") or 0)
                tout += int(record.get("output_tokens") or 0)
                if record.get("valid"):
                    n_valid += 1
                if n_done % 100 == 0 or n_done == len(pending):
                    print(
                        f"completed {n_done}/{len(pending)} "
                        f"valid_rate={n_valid / max(n_done, 1):.3f}",
                        flush=True,
                    )

    all_rows = _load_trace(trace_path)
    cost = (
        tin * args.input_price_per_million + tout * args.output_price_per_million
    ) / 1e6
    summary = {
        "run_dir": str(run_dir),
        "n_planned": len(tasks),
        "n_trace_rows": len(all_rows),
        "n_executed_this_session": n_done,
        "n_valid_this_session": n_valid,
        "valid_rate_session": n_valid / max(n_done, 1),
        "actual_input_tokens_session": tin,
        "actual_output_tokens_session": tout,
        "actual_cost_usd_session": cost,
        "elapsed_seconds": time.perf_counter() - t0,
        "complete": len(all_rows) >= len(tasks),
    }
    (run_dir / "session_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
