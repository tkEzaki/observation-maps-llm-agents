"""Matched 3×3 collective replay runner (48 fields × 3 targets × 32).

Safety gates (paid openai):
- freeze manifest re-verification
- exact 4608 = 48×3×16×2 tasks
- estimate under $12 ceiling
- replay_auth_go.json paid_authorized must be true
- never swap fields on failure
- retain strict parse + all attempt raw responses + tokens
- after acquisition: primary 3×3 report before any surrogate

Mock backend is always allowed (no paid auth).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circlemap.backends import (  # noqa: E402
    MockResponseBackend,
    QuotaError,
    ResponseBackend,
)
from circlemap.costing import estimate_cost, format_cost_estimate  # noqa: E402
from circlemap.observation import RelativePhaseHistogram  # noqa: E402
from circlemap.representations import (  # noqa: E402
    build_representation_prompt_from_histogram,
)
from circlemap.response import ACTION_VALUES, parse_social_action  # noqa: E402
from experiments.stage_b_response_law.run import _make_backend  # noqa: E402

DEFAULT_PROTOCOL = Path(__file__).with_name(
    "protocol_matched_rep_collective_replay_v0_1.json"
)
AUTH_PATH = ROOT / "analysis" / "matched_rep_collective" / "replay_auth_go.json"
FREEZE_PATH = (
    ROOT / "analysis" / "matched_rep_collective" / "replay_hash_freeze_v0_1.json"
)
TARGETS = (
    "moments_m1_m3",
    "centers_24_standard",
    "intervals_24_decimal6",
)
CEILING_USD = 12.0
REQUIRED_RETENTION = (
    "field_id",
    "physical_hash",
    "target_representation",
    "acquisition_block",
    "sample_index",
    "strict_parsed_action",
    "last_raw_response",
    "attempt_raw_responses",
    "rescued_parse",
    "unrecoverable",
    "input_tokens",
    "output_tokens",
    "valid",
)


@dataclass(frozen=True)
class ReplayTask:
    task_id: str
    field_id: str
    physical_hash: str
    source_representation: str
    stratum: str
    target_representation: str
    acquisition_block: int
    sample_index: int
    sample_seed: int
    prompt: str
    prompt_sha256: str


@dataclass
class FlakyMockBackend:
    """Mock that injects invalid JSON on a deterministic fraction of seeds."""

    fail_mod: int = 17
    inner: MockResponseBackend | None = None
    name: str = "mock_flaky"
    model: str = "attractive-v1-flaky"

    def __post_init__(self) -> None:
        if self.inner is None:
            self.inner = MockResponseBackend()

    def call(self, prompt: str, *, seed: int):
        assert self.inner is not None
        result = self.inner.call(prompt, seed=seed)
        if seed % self.fail_mod == 0:
            return type(result)(
                raw_text='{"social_action": "stay"',  # truncated JSON
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        return result


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(obj: object) -> str:
    return _sha_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def _load_auth() -> dict:
    return json.loads(AUTH_PATH.read_text(encoding="utf-8"))


def _deterministic_rescue(raw_text: str | None):
    """Identical rescue rule for all 144 conditions (no hand edits)."""
    if not raw_text:
        return None
    text = raw_text.strip()
    # Strip common markdown fences before strict retry (uniform rule).
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return parse_social_action(text)
    except Exception:
        pass
    match = re.search(
        r'\{\s*"social_action"\s*:\s*"(advance|stay|retard)"\s*\}',
        raw_text or "",
    )
    if not match:
        # Truncation rescue: open object with complete social_action value only.
        match = re.search(
            r'"social_action"\s*:\s*"(advance|stay|retard)"',
            raw_text or "",
        )
        if not match:
            return None
        label = match.group(1)
        return type("Parsed", (), {"label": label, "value": ACTION_VALUES[label]})()
    label = match.group(1)
    return type("Parsed", (), {"label": label, "value": ACTION_VALUES[label]})()


def verify_freeze(
    protocol: dict,
    fields_path: Path,
    fields: list[dict],
    freeze: dict,
) -> dict:
    checks: dict[str, bool] = {}
    checks["fields_file_sha"] = (
        _sha_file(fields_path) == freeze["replay_fields_v0_1_sha256"]
    )
    phys = _sha_json(sorted(f["physical_hash"] for f in fields))
    checks["physical_aggregate"] = phys == freeze["physical_field_aggregate_sha256"]
    checks["n_fields_48"] = len(fields) == 48
    checks["unique_physical_48"] = len({f["physical_hash"] for f in fields}) == 48

    # Rebuild prompt aggregates
    edges = np.linspace(-np.pi, np.pi, 25)
    per_target = {t: [] for t in TARGETS}
    for field in fields:
        hist = RelativePhaseHistogram(
            edges=edges,
            fractions=np.asarray(field["physical_histogram_24"], dtype=float),
            peer_count=int(field.get("peer_count", 16)),
        )
        for target in TARGETS:
            prompt = build_representation_prompt_from_histogram(target, hist)
            per_target[target].append(_sha_text(prompt))
    checks["moments_prompt_agg"] = (
        _sha_json(sorted(per_target["moments_m1_m3"]))
        == freeze["moments_prompt_aggregate_sha256"]
    )
    checks["centers_prompt_agg"] = (
        _sha_json(sorted(per_target["centers_24_standard"]))
        == freeze["centers_prompt_aggregate_sha256"]
    )
    checks["intervals_prompt_agg"] = (
        _sha_json(sorted(per_target["intervals_24_decimal6"]))
        == freeze["intervals_prompt_aggregate_sha256"]
    )
    protocol_path = ROOT / freeze.get(
        "protocol_path",
        "experiments/stage_c/protocol_matched_rep_collective_replay_v0_1.json",
    )
    checks["protocol_sha"] = _sha_file(protocol_path) == freeze["protocol_sha256"]
    checks["n_target_prompts_144"] = freeze.get("n_target_prompts") == 144
    return {
        "pass": all(checks.values()),
        "checks": checks,
    }


def build_tasks(protocol: dict, fields: list[dict]) -> list[ReplayTask]:
    n_blocks = int(protocol["n_blocks"])
    per_block = int(protocol["responses_per_block"])
    assert n_blocks >= 1 and per_block >= 1
    seed_prefix = protocol.get("seed_prefix", "mrc-replay-v0.1")
    edges = np.linspace(-np.pi, np.pi, 25)
    tasks: list[ReplayTask] = []
    for field in fields:
        hist = RelativePhaseHistogram(
            edges=edges,
            fractions=np.asarray(field["physical_histogram_24"], dtype=float),
            peer_count=int(field.get("peer_count", 16)),
        )
        for target in TARGETS:
            prompt = build_representation_prompt_from_histogram(target, hist)
            ph = _sha_text(prompt)
            for block in range(n_blocks):
                for sample in range(per_block):
                    payload = (
                        f"{seed_prefix}|{field['field_id']}|{target}|"
                        f"{block}|{sample}"
                    )
                    sample_seed = (
                        int.from_bytes(
                            hashlib.sha256(payload.encode()).digest()[:8], "big"
                        )
                        % 2_147_483_647
                    )
                    task_id = (
                        f"{field['field_id']}__{target}__"
                        f"block{block}__s{sample:02d}"
                    )
                    tasks.append(
                        ReplayTask(
                            task_id=task_id,
                            field_id=field["field_id"],
                            physical_hash=field["physical_hash"],
                            source_representation=field["source_representation"],
                            stratum=field["stratum"],
                            target_representation=target,
                            acquisition_block=block,
                            sample_index=sample,
                            sample_seed=sample_seed,
                            prompt=prompt,
                            prompt_sha256=ph,
                        )
                    )
    return tasks


def audit_task_construction(tasks: list[ReplayTask], protocol: dict) -> dict:
    expected = int(protocol["expected_calls"])
    n_blocks_proto = int(protocol["n_blocks"])
    per_block = int(protocol["responses_per_block"])
    n_fields = len({t.field_id for t in tasks})
    n_targets = len({t.target_representation for t in tasks})
    n_blocks = len({t.acquisition_block for t in tasks})
    per_cond = {}
    for t in tasks:
        key = (t.field_id, t.target_representation, t.acquisition_block)
        per_cond[key] = per_cond.get(key, 0) + 1
    checks = {
        "n_tasks_eq_expected": len(tasks) == expected,
        "n_fields_48": n_fields == 48,
        "n_targets_3": n_targets == 3,
        "n_blocks_match": n_blocks == n_blocks_proto,
        "unique_task_ids": len({t.task_id for t in tasks}) == len(tasks),
        "per_condition_match": all(v == per_block for v in per_cond.values())
        and len(per_cond) == 48 * 3 * n_blocks_proto,
    }
    return {"pass": all(checks.values()), "checks": checks, "n_tasks": len(tasks)}


def run_one_task(
    task: ReplayTask,
    backend: ResponseBackend,
    *,
    max_attempts: int,
) -> dict:
    started = time.perf_counter()
    errors: list[str] = []
    attempt_raw: list[str | None] = []
    tin = tout = 0
    last_raw = None
    strict_action = None
    rescued = False
    action_label = None
    action_value = None
    valid = False
    unrecoverable = True

    for attempt in range(1, max_attempts + 1):
        try:
            # Same seed each attempt (Stage-B contract); retries only help transient backend errors.
            response = backend.call(task.prompt, seed=task.sample_seed)
            tin += int(response.input_tokens)
            tout += int(response.output_tokens)
            last_raw = response.raw_text
            attempt_raw.append(response.raw_text)
            try:
                parsed = parse_social_action(response.raw_text)
                strict_action = parsed.label
                action_label = parsed.label
                action_value = parsed.value
                valid = True
                rescued = False
                unrecoverable = False
                break
            except Exception as exc:
                errors.append(f"strict:{type(exc).__name__}:{exc}")
                rescued_parsed = _deterministic_rescue(response.raw_text)
                if rescued_parsed is not None:
                    action_label = rescued_parsed.label
                    action_value = rescued_parsed.value
                    strict_action = None
                    valid = True
                    rescued = True
                    unrecoverable = False
                    break
                errors.append("rescue:failed")
        except QuotaError:
            raise
        except Exception as exc:
            attempt_raw.append(None)
            errors.append(f"backend:{type(exc).__name__}:{exc}")

    record = {
        **{k: getattr(task, k) for k in (
            "task_id",
            "field_id",
            "physical_hash",
            "source_representation",
            "stratum",
            "target_representation",
            "acquisition_block",
            "sample_index",
            "sample_seed",
            "prompt_sha256",
        )},
        "prompt": None,
        "valid": valid,
        "strict_parsed_action": strict_action,
        "action_label": action_label,
        "action_value": action_value,
        "last_raw_response": last_raw,
        "attempt_raw_responses": attempt_raw,
        "rescued_parse": rescued,
        "unrecoverable": unrecoverable,
        "attempts": len(attempt_raw),
        "errors": errors,
        "input_tokens": tin,
        "output_tokens": tout,
        "backend": getattr(backend, "name", "unknown"),
        "model": getattr(backend, "model", "unknown"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    missing = [k for k in REQUIRED_RETENTION if k not in record]
    if missing:
        raise RuntimeError(f"retention schema missing keys: {missing}")
    return record


def _load_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _validate_resume(path: Path, expected: dict) -> None:
    if not path.exists():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "protocol_sha256",
        "fields_sha256",
        "freeze_sha256",
        "backend",
        "model",
        "max_attempts",
        "dependency_snapshot_sha256",
    )
    bad = [k for k in keys if existing.get(k) != expected.get(k)]
    if bad:
        raise ValueError("resume config mismatch: " + ", ".join(bad))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--backend", choices=("mock", "openai", "mock_flaky"), default="mock")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--input-price-per-million", type=float, default=0.75)
    parser.add_argument("--output-price-per-million", type=float, default=4.5)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "runs" / "matched_rep_collective_replay",
    )
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--verify-freeze-only", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--preflight-tasks",
        type=int,
        default=0,
        help="After paid auth: run N tasks, verify retention, then continue "
        "(counted inside the 4608 budget).",
    )
    parser.add_argument(
        "--skip-primary-report",
        action="store_true",
        help="Debug only; production must generate primary 3x3 first.",
    )
    args = parser.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    fields_path = ROOT / protocol["fields_file"]
    fields_payload = json.loads(fields_path.read_text(encoding="utf-8"))
    fields = fields_payload["fields"]
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

    print("Verifying freeze manifest...", flush=True)
    freeze_report = verify_freeze(protocol, fields_path, fields, freeze)
    print(json.dumps(freeze_report, indent=2), flush=True)
    if not freeze_report["pass"]:
        raise SystemExit("FREEZE_VERIFY_FAIL")
    if args.verify_freeze_only:
        print("FREEZE_VERIFY_ALL_OK", flush=True)
        return 0

    tasks = build_tasks(protocol, fields)
    construction = audit_task_construction(tasks, protocol)
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
        format_cost_estimate(estimate).replace("Stage B", "Matched 3x3 replay"),
        flush=True,
    )
    if estimate.estimated_base_cost_usd > CEILING_USD:
        raise SystemExit(
            f"COST_CEILING_EXCEEDED: base={estimate.estimated_base_cost_usd:.4f} > {CEILING_USD}"
        )
    if args.estimate_only:
        return 0

    paid = args.backend == "openai"
    auth = _load_auth()
    if paid:
        if not auth.get("paid_authorized"):
            raise SystemExit(
                "PAID_AUTH_FALSE: replay_auth_go.json paid_authorized is false; "
                "refusing OpenAI calls"
            )
        if not args.yes:
            raise SystemExit("Refusing paid run without --yes")
        if not args.model:
            parser.error("--model required for openai")
    else:
        print(
            f"backend={args.backend}: paid auth not required "
            f"(paid_authorized={auth.get('paid_authorized')})",
            flush=True,
        )

    # Backend
    if args.backend == "mock":
        backend: ResponseBackend = MockResponseBackend()
        model = "mock"
    elif args.backend == "mock_flaky":
        backend = FlakyMockBackend()
        model = "mock_flaky"
    else:
        backend = _make_backend(args)
        model = args.model

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    proto_hash = _sha_file(args.protocol)[:12]
    if args.resume_dir is not None:
        run_dir = args.resume_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = (
            args.out_root
            / f"{protocol['protocol_version']}_{proto_hash}"
            / stamp
        )
        run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "protocol_sha256": _sha_file(args.protocol),
        "fields_sha256": _sha_file(fields_path),
        "freeze_sha256": _sha_file(FREEZE_PATH),
        "dependency_snapshot_sha256": freeze.get("dependency_snapshot_sha256"),
        "runner_source_sha256": freeze.get("runner_source_sha256"),
        "encoder_sources_sha256": freeze.get("encoder_sources_sha256"),
        "selector_source_sha256": freeze.get("selector_source_sha256"),
        "backend": args.backend,
        "model": model,
        "max_attempts": args.max_attempts,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "freeze_verify": freeze_report,
        "task_construction": construction,
        "estimate_base_usd": estimate.estimated_base_cost_usd,
        "cost_ceiling_usd": CEILING_USD,
        "paid_authorized": bool(auth.get("paid_authorized")),
        "no_field_swap_on_failure": True,
        "primary_before_surrogate": True,
    }
    _validate_resume(run_dir / "resolved_config.json", config)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (run_dir / "freeze_verify.json").write_text(
        json.dumps(freeze_report, indent=2), encoding="utf-8"
    )

    trace_path = run_dir / "trace.jsonl"
    completed = {r["task_id"] for r in _load_trace(trace_path)}
    pending = [t for t in tasks if t.task_id not in completed]
    print(f"pending calls: {len(pending)}/{len(tasks)}", flush=True)

    # Optional in-budget preflight (paid path)
    if paid and args.preflight_tasks > 0 and pending:
        n_pre = min(args.preflight_tasks, len(pending))
        preflight = pending[:n_pre]
        print(f"Preflight checkpoint: {n_pre} tasks (in-budget)...", flush=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            for task in preflight:
                rec = run_one_task(task, backend, max_attempts=args.max_attempts)
                handle.write(json.dumps(rec, ensure_ascii=True) + "\n")
                handle.flush()
                missing = [k for k in REQUIRED_RETENTION if k not in rec]
                if missing:
                    raise SystemExit(f"PREFLIGHT_RETENTION_FAIL: {missing}")
        completed = {r["task_id"] for r in _load_trace(trace_path)}
        pending = [t for t in tasks if t.task_id not in completed]
        print(f"Preflight OK; remaining {len(pending)}", flush=True)

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
                if n_done % 200 == 0 or n_done == len(pending):
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
        "freeze_pass": True,
        "task_construction_pass": True,
        "complete": len(all_rows) >= len(tasks),
    }
    (run_dir / "session_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)

    if not args.skip_primary_report and len(all_rows) >= len(tasks):
        from analysis.matched_rep_collective.analyze_replay_primary_3x3 import (  # noqa: E402
            run_primary_report,
        )

        report_path = run_primary_report(run_dir, fields)
        print(f"PRIMARY_3x3_REPORT: {report_path}", flush=True)
        print("NOTE: surrogate refit is deferred until after primary report.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
