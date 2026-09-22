"""R1 cross-model operator replication on frozen 48-field panel.

Same fields/encodings as matched 3×3 replay; only the LLM family changes.
Default: 48×3×16 = 2304 calls per model (8×2 blocks).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circlemap.backends import (  # noqa: E402
    AnthropicResponseBackend,
    GoogleResponseBackend,
    MockResponseBackend,
    OpenAIResponseBackend,
)
from circlemap.costing import estimate_cost, format_cost_estimate  # noqa: E402
from circlemap.envload import load_project_env  # noqa: E402
from experiments.stage_c.run_matched_rep_collective_replay import (  # noqa: E402
    FREEZE_PATH,
    REQUIRED_RETENTION,
    TARGETS,
    audit_task_construction,
    build_tasks,
    run_one_task,
    verify_freeze,
    _load_trace,
    _sha_file,
    _validate_resume,
)

DEFAULT_PROTOCOL = Path(__file__).with_name(
    "protocol_matched_rep_collective_r1_v0_1.json"
)
AUTH_PATH = ROOT / "analysis" / "matched_rep_collective" / "r1_auth_go.json"


def _make_backend(args: argparse.Namespace, model_cfg: dict):
    backend = args.backend or model_cfg["backend"]
    model = args.model or model_cfg["model"]
    if backend == "mock":
        return MockResponseBackend(), "mock"
    if backend == "openai":
        return (
            OpenAIResponseBackend(
                model=model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout,
                base_url=args.base_url,
            ),
            model,
        )
    if backend == "anthropic":
        return (
            AnthropicResponseBackend(
                model=model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout,
            ),
            model,
        )
    if backend == "google":
        return (
            GoogleResponseBackend(
                model=model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout_seconds=args.timeout,
            ),
            model,
        )
    raise SystemExit(f"unsupported backend: {backend}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--family",
        choices=("claude", "gemini"),
        required=True,
        help="Which frozen second-model family arm to run.",
    )
    parser.add_argument(
        "--backend",
        choices=("mock", "openai", "anthropic", "google"),
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "runs" / "matched_rep_collective_r1",
    )
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--smoke-format", type=int, default=0)
    parser.add_argument("--preflight-tasks", type=int, default=8)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--skip-primary-report", action="store_true")
    args = parser.parse_args(argv)

    load_project_env()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    model_cfg = protocol["models"][args.family]
    in_price = (
        args.input_price_per_million
        if args.input_price_per_million is not None
        else float(model_cfg["input_price_per_million"])
    )
    out_price = (
        args.output_price_per_million
        if args.output_price_per_million is not None
        else float(model_cfg["output_price_per_million"])
    )
    ceiling = float(model_cfg["cost_ceiling_usd"])

    fields_path = ROOT / protocol["fields_file"]
    fields = json.loads(fields_path.read_text(encoding="utf-8"))["fields"]
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

    print("Verifying parent field freeze...", flush=True)
    freeze_report = verify_freeze(protocol, fields_path, fields, freeze)
    # protocol_sha in parent freeze refers to GPT replay protocol; for R1 we
    # only require field/prompt aggregates.
    field_ok = all(
        freeze_report["checks"][k]
        for k in (
            "fields_file_sha",
            "physical_aggregate",
            "n_fields_48",
            "unique_physical_48",
            "moments_prompt_agg",
            "centers_prompt_agg",
            "intervals_prompt_agg",
            "n_target_prompts_144",
        )
    )
    print(json.dumps(freeze_report, indent=2), flush=True)
    if not field_ok:
        raise SystemExit("FIELD_FREEZE_VERIFY_FAIL")

    tasks = build_tasks(protocol, fields)
    construction = audit_task_construction(tasks, protocol)
    print("Task construction:", json.dumps(construction, indent=2), flush=True)
    if not construction["pass"]:
        raise SystemExit("TASK_CONSTRUCTION_AUDIT_FAIL")

    estimate = estimate_cost(
        [t.prompt for t in tasks],
        max_output_tokens_per_call=max(args.max_output_tokens, 64),
        max_attempts_per_call=args.max_attempts,
        input_price_per_million=in_price,
        output_price_per_million=out_price,
    )
    print(
        format_cost_estimate(estimate).replace(
            "Stage B", f"R1 {args.family}"
        ),
        flush=True,
    )
    if estimate.estimated_base_cost_usd > ceiling:
        raise SystemExit(
            f"COST_CEILING_EXCEEDED: {estimate.estimated_base_cost_usd:.4f} > {ceiling}"
        )
    if args.estimate_only:
        return 0

    expected = int(protocol["expected_calls"])
    backend_name = args.backend or model_cfg["backend"]
    paid = backend_name in ("openai", "anthropic", "google")
    auth = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    if paid:
        if not auth.get("paid_authorized"):
            raise SystemExit("R1_PAID_AUTH_FALSE")
        families = auth.get("authorized_families") or ["claude", "gemini"]
        if args.family not in families:
            raise SystemExit(f"family {args.family} not in authorized_families")
        if not args.yes:
            raise SystemExit("Refusing paid R1 run without --yes")

    backend, model = _make_backend(args, model_cfg)

    if args.smoke_format > 0:
        print(f"Format smoke: {args.smoke_format} tasks", flush=True)
        ok = 0
        for task in tasks[: args.smoke_format]:
            rec = run_one_task(task, backend, max_attempts=args.max_attempts)
            print(
                {
                    "task_id": task.task_id,
                    "valid": rec["valid"],
                    "raw": (rec.get("last_raw_response") or "")[:120],
                    "action": rec.get("action_label"),
                },
                flush=True,
            )
            if rec.get("valid"):
                ok += 1
        print(f"SMOKE_FORMAT {ok}/{args.smoke_format}", flush=True)
        return 0 if ok == args.smoke_format else 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.resume_dir is not None:
        run_dir = args.resume_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = (
            args.out_root
            / f"{protocol['protocol_version']}_{args.family}_{model.replace('/', '_')}"
            / stamp
        )
        run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "family": args.family,
        "protocol_sha256": _sha_file(args.protocol),
        "fields_sha256": _sha_file(fields_path),
        "freeze_sha256": _sha_file(FREEZE_PATH),
        "backend": backend_name,
        "model": model,
        "max_attempts": args.max_attempts,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "estimate_base_usd": estimate.estimated_base_cost_usd,
        "cost_ceiling_usd": ceiling,
        "expected_calls": expected,
        "replication_pass_gates": protocol["replication_pass_gates"],
    }
    _validate_resume(run_dir / "resolved_config.json", {
        **config,
        "dependency_snapshot_sha256": freeze.get("dependency_snapshot_sha256"),
    } if False else {
        "protocol_sha256": config["protocol_sha256"],
        "fields_sha256": config["fields_sha256"],
        "freeze_sha256": config["freeze_sha256"],
        "backend": config["backend"],
        "model": config["model"],
        "max_attempts": config["max_attempts"],
        "dependency_snapshot_sha256": freeze.get("dependency_snapshot_sha256"),
    })
    # Soften resume: only enforce if file exists with matching core keys
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

    trace_path = run_dir / "trace.jsonl"
    completed = {r["task_id"] for r in _load_trace(trace_path)}
    pending = [t for t in tasks if t.task_id not in completed]
    print(f"pending calls: {len(pending)}/{len(tasks)}", flush=True)

    if paid and args.preflight_tasks > 0 and pending:
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
                if not rec.get("valid"):
                    print("PREFLIGHT_WARN invalid:", task.task_id, flush=True)
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
                record = fut.result()
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
    cost = (tin * in_price + tout * out_price) / 1e6
    summary = {
        "family": args.family,
        "model": model,
        "backend": backend_name,
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

    if not args.skip_primary_report and len(all_rows) >= len(tasks):
        from analysis.matched_rep_collective.analyze_replay_primary_3x3 import (
            run_primary_report,
        )

        report_path = run_primary_report(run_dir, fields)
        print(f"PRIMARY_3x3_REPORT: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
