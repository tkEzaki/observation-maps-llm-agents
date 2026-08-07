"""Run Stage B fixed-field response measurements with a mandatory cost gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from circlemap.backends import (
    MockResponseBackend,
    OpenAIResponseBackend,
    QuotaError,
    ResponseBackend,
)
from circlemap.costing import estimate_cost, format_cost_estimate
from circlemap.response import (
    PROMPT_VERSION,
    build_response_prompt,
    fixed_field_histogram,
    parse_social_action,
)
from circlemap.response_analysis import analyze_response_records

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name("protocol_v0_1.json")


@dataclass(frozen=True)
class ProbeTask:
    task_id: str
    profile: str
    concentration: float
    offset_index: int
    offset_radians: float
    repetition: int
    sample_seed: int
    prompt: str
    prompt_sha256: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_protocol(path: Path, repetitions: int | None = None) -> dict:
    with path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("protocol prompt_version does not match implementation")
    if repetitions is not None:
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        protocol["repetitions_per_condition"] = repetitions
    return protocol


def build_tasks(protocol: dict) -> list[ProbeTask]:
    tasks = []
    n_offsets = int(protocol["n_offsets"])
    n_bins = int(protocol["n_bins"])
    repetitions = int(protocol["repetitions_per_condition"])
    base_seed = int(protocol["base_seed"])
    offsets = -np.pi + np.arange(n_offsets) * 2.0 * np.pi / n_offsets

    task_number = 0
    for profile, concentration_value in protocol[
        "concentration_profiles"
    ].items():
        concentration = float(concentration_value)
        for offset_index, offset in enumerate(offsets):
            histogram = fixed_field_histogram(
                float(offset),
                concentration,
                n_bins=n_bins,
                synthetic_peer_count=int(protocol["synthetic_peer_count"]),
            )
            prompt = build_response_prompt(histogram)
            prompt_hash = _sha256_text(prompt)
            for repetition in range(repetitions):
                task_id = (
                    f"{profile}__offset_{offset_index:02d}"
                    f"__rep_{repetition:04d}"
                )
                tasks.append(
                    ProbeTask(
                        task_id=task_id,
                        profile=profile,
                        concentration=concentration,
                        offset_index=offset_index,
                        offset_radians=float(offset),
                        repetition=repetition,
                        sample_seed=base_seed + task_number,
                        prompt=prompt,
                        prompt_sha256=prompt_hash,
                    )
                )
                task_number += 1
    return tasks


def run_task(
    task: ProbeTask,
    backend: ResponseBackend,
    *,
    max_attempts: int,
) -> dict:
    task_started = time.perf_counter()
    errors = []
    total_input_tokens = 0
    total_output_tokens = 0
    last_raw_response = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = backend.call(task.prompt, seed=task.sample_seed)
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            last_raw_response = response.raw_text
            action = parse_social_action(response.raw_text)
            return {
                **asdict(task),
                "prompt": None,
                "valid": True,
                "action_label": action.label,
                "action_value": action.value,
                "raw_response": response.raw_text,
                "attempts": attempt,
                "errors": errors,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "backend": backend.name,
                "model": backend.model,
                "elapsed_seconds": time.perf_counter() - task_started,
            }
        except QuotaError:
            raise
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    return {
        **asdict(task),
        "prompt": None,
        "valid": False,
        "action_label": None,
        "action_value": None,
        # Retain last backend text for offline missingness audit / deterministic reparse.
        "raw_response": last_raw_response,
        "attempts": max_attempts,
        "errors": errors,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "backend": backend.name,
        "model": backend.model,
        "elapsed_seconds": time.perf_counter() - task_started,
    }


def _load_existing_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _validate_resume_config(path: Path, expected: dict) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        existing = json.load(handle)
    keys = (
        "protocol_sha256",
        "backend",
        "model",
        "temperature",
        "max_output_tokens",
        "max_attempts",
        "base_url",
        "env_file",
    )
    mismatches = [
        key
        for key in keys
        if existing.get(key) != expected.get(key)
    ]
    if mismatches:
        raise ValueError(
            "run directory configuration mismatch for: "
            + ", ".join(mismatches)
        )
    return existing


def _make_backend(args: argparse.Namespace) -> ResponseBackend:
    if args.backend == "mock":
        return MockResponseBackend()
    if args.env_file is not None:
        if not args.env_file.is_file():
            raise FileNotFoundError(f"env file not found: {args.env_file}")
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "--env-file requires: py -m pip install -e .[openai]"
            ) from exc
        load_dotenv(args.env_file, encoding="utf-8-sig", override=False)
    return OpenAIResponseBackend(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout,
        base_url=args.base_url,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--backend", choices=("mock", "openai"), default="mock")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "runs" / "response_law",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    if args.backend == "openai":
        if not args.model:
            parser.error("--model is required for the openai backend")
        if args.input_price_per_million is None:
            parser.error(
                "--input-price-per-million is required for paid backends"
            )
        if args.output_price_per_million is None:
            parser.error(
                "--output-price-per-million is required for paid backends"
            )
    else:
        args.input_price_per_million = 0.0
        args.output_price_per_million = 0.0
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    protocol = load_protocol(args.protocol, repetitions=args.repetitions)
    protocol_text = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
    )
    protocol_hash = _sha256_text(protocol_text)
    tasks = build_tasks(protocol)

    if args.run_dir:
        run_dir = args.run_dir.resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (
            args.out_root.resolve()
            / f"{protocol['protocol_version']}_{protocol_hash[:12]}"
            / timestamp
        )
    resolved_model = (
        args.model if args.backend != "mock" else MockResponseBackend().model
    )
    prospective_config = {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "backend": args.backend,
        "model": resolved_model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "max_attempts": args.max_attempts,
        "concurrency": args.concurrency,
        "base_url": args.base_url,
        "env_file": str(args.env_file.resolve()) if args.env_file else None,
    }
    existing_config = _validate_resume_config(
        run_dir / "resolved_config.json",
        prospective_config,
    )
    trace_path = run_dir / "trace.jsonl"
    existing = _load_existing_trace(trace_path)
    task_ids = {task.task_id for task in tasks}
    unknown_ids = {
        record["task_id"] for record in existing
    } - task_ids
    if unknown_ids:
        raise ValueError("trace contains tasks outside the resolved protocol")
    completed_ids = {record["task_id"] for record in existing}
    pending = [task for task in tasks if task.task_id not in completed_ids]

    estimate = estimate_cost(
        [task.prompt for task in pending] or ["no pending calls"],
        max_output_tokens_per_call=args.max_output_tokens,
        max_attempts_per_call=args.max_attempts,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    if not pending:
        estimate = estimate.__class__(
            planned_calls=0,
            max_attempts_per_call=args.max_attempts,
            input_token_estimation_method=(
                "ceil(characters / 2.5) + 16"
            ),
            estimated_input_tokens=0,
            maximum_output_tokens=0,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            estimated_base_cost_usd=0.0,
            maximum_retry_cost_usd=0.0,
        )
    print(format_cost_estimate(estimate), flush=True)
    print(f"  completed calls found: {len(existing):,}", flush=True)
    print(f"  pending calls: {len(pending):,}", flush=True)

    if args.estimate_only:
        return 0
    if args.backend != "mock" and not args.yes:
        print(
            "External execution stopped before any API call. "
            "Review the estimate and rerun with --yes.",
            file=sys.stderr,
        )
        return 2

    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **prospective_config,
        "cost_estimate": estimate.as_dict(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if existing_config is None:
        with (run_dir / "resolved_config.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(resolved_config, handle, indent=2)
    with (run_dir / "cost_estimate.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(estimate.as_dict(), handle, indent=2)
    unique_stimuli = {}
    for task in tasks:
        unique_stimuli.setdefault(
            task.prompt_sha256,
            {
                "profile": task.profile,
                "concentration": task.concentration,
                "offset_index": task.offset_index,
                "offset_radians": task.offset_radians,
                "prompt_sha256": task.prompt_sha256,
                "serialized_prompt": task.prompt,
            },
        )
    with (run_dir / "stimuli.jsonl").open("w", encoding="utf-8") as handle:
        for stimulus in unique_stimuli.values():
            handle.write(json.dumps(stimulus, ensure_ascii=False) + "\n")

    execution_started = time.perf_counter()
    backend = _make_backend(args)
    new_records = []
    if pending:
        with trace_path.open("a", encoding="utf-8", buffering=1) as trace:
            bulk_tasks = pending
            completed_before_bulk = 0
            if args.backend != "mock":
                print(
                    "running one serial preflight call "
                    "(counted as the first scientific sample)",
                    flush=True,
                )
                preflight_record = run_task(
                    pending[0],
                    backend,
                    max_attempts=args.max_attempts,
                )
                trace.write(
                    json.dumps(preflight_record, ensure_ascii=False) + "\n"
                )
                new_records.append(preflight_record)
                if not preflight_record["valid"]:
                    with (run_dir / "run_status.json").open(
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(
                            {
                                "status": "aborted_preflight_invalid",
                                "record": preflight_record,
                            },
                            handle,
                            indent=2,
                        )
                    print(
                        "Preflight response was invalid after the frozen "
                        "retry policy; bulk execution was not started.",
                        file=sys.stderr,
                    )
                    return 4
                print("preflight passed", flush=True)
                bulk_tasks = pending[1:]
                completed_before_bulk = 1

            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(
                        run_task,
                        task,
                        backend,
                        max_attempts=args.max_attempts,
                    ): task
                    for task in bulk_tasks
                }
                try:
                    for completed_index, future in enumerate(
                        as_completed(futures),
                        start=completed_before_bulk + 1,
                    ):
                        record = future.result()
                        trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                        new_records.append(record)
                        if completed_index % 100 == 0:
                            print(
                                f"completed {completed_index:,}/"
                                f"{len(pending):,}",
                                flush=True,
                            )
                except QuotaError:
                    for future in futures:
                        future.cancel()
                    raise

    records = existing + new_records
    execution_elapsed = time.perf_counter() - execution_started
    criteria = protocol["criteria"]
    report = analyze_response_records(
        records,
        run_dir,
        max_harmonic=int(protocol["max_harmonic"]),
        n_permutations=int(protocol["phase_dependence_permutations"]),
        minimum_valid_rate=float(criteria["minimum_valid_rate"]),
        minimum_valid_samples_per_condition=int(
            criteria["minimum_valid_samples_per_condition"]
        ),
        phase_dependence_family_alpha=float(
            criteria["phase_dependence_family_alpha"]
        ),
        minimum_response_range=float(criteria["minimum_response_range"]),
        minimum_mirrored_valid_fraction=float(
            criteria["minimum_mirrored_valid_fraction"]
        ),
    )
    actual_input = sum(int(r.get("input_tokens", 0)) for r in records)
    actual_output = sum(int(r.get("output_tokens", 0)) for r in records)
    actual_cost = (
        actual_input * args.input_price_per_million
        + actual_output * args.output_price_per_million
    ) / 1_000_000
    summed_task_elapsed = sum(
        float(record.get("elapsed_seconds", 0.0))
        for record in new_records
    )
    executed_call_count = len(new_records)
    run_summary = {
        "run_dir": str(run_dir),
        "n_records": len(records),
        "n_valid": report["n_valid"],
        "valid_rate": report["valid_rate"],
        "actual_input_tokens": actual_input,
        "actual_output_tokens": actual_output,
        "actual_cost_usd": actual_cost,
        "configured_concurrency": args.concurrency,
        "api_execution_elapsed_seconds": execution_elapsed,
        "executed_calls_this_process": executed_call_count,
        "summed_task_elapsed_seconds": summed_task_elapsed,
        "effective_parallelism": (
            summed_task_elapsed / execution_elapsed
            if execution_elapsed > 0
            else 0.0
        ),
        "calls_per_second": (
            executed_call_count / execution_elapsed
            if execution_elapsed > 0
            else 0.0
        ),
        "criteria": report["criteria"],
    }
    with (run_dir / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(run_summary, handle, indent=2)
    print(json.dumps(run_summary, indent=2), flush=True)
    return 0 if report["criteria"]["all_stage_b_data_criteria_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
