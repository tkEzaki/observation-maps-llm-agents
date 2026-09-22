"""Run the Stage B representation-invariance screen with a cost gate."""

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

from circlemap.backends import QuotaError
from circlemap.costing import estimate_cost, format_cost_estimate
from circlemap.representations import (
    build_representation_prompt,
    coerce_representation_spec,
)
from circlemap.response import PROMPT_VERSION
from circlemap.stimuli import (
    STIMULUS_CATALOG,
    StimulusSpec,
    stimulus_concentration_proxy,
)
from experiments.stage_b_response_law.run import (
    _load_existing_trace,
    _make_backend,
    _validate_resume_config,
    run_task,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = Path(__file__).with_name(
    "protocol_representation_v0_1.json"
)


@dataclass(frozen=True)
class RepresentationTask:
    task_id: str
    representation: str
    profile: str
    concentration: float
    offset_index: int
    offset_radians: float
    repetition: int
    sample_seed: int
    prompt: str
    prompt_sha256: str
    seed_block_id: str | None = None


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_protocol(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _derive_sample_seed(
    base_seed: int,
    representation: str,
    profile: str,
    offset_index: int,
    repetition: int,
) -> int:
    payload = (
        f"hash-v1|{base_seed}|{representation}|{profile}|"
        f"{offset_index}|{repetition}"
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8],
        "big",
    ) % 2_147_483_647


def _resolve_offsets(protocol: dict) -> list[tuple[int, float]]:
    if "offset_radians" in protocol:
        offsets = [float(value) for value in protocol["offset_radians"]]
        if not offsets or not all(np.isfinite(offsets)):
            raise ValueError("offset_radians must be nonempty and finite")
        return list(enumerate(offsets))
    grid_size = int(
        protocol["offset_grid_size"]
        if "offset_grid_size" in protocol
        else protocol["n_offsets"]
    )
    offset_indices = (
        protocol["offset_indices"]
        if "offset_indices" in protocol
        else list(range(int(protocol["n_offsets"])))
    )
    return [
        (
            int(index),
            float(-np.pi + int(index) * 2.0 * np.pi / grid_size),
        )
        for index in offset_indices
    ]


def _load_stimulus_catalog(protocol: dict) -> None:
    """Register any protocol-bundled fixed stimuli into STIMULUS_CATALOG."""
    catalog_file = protocol.get("stimulus_catalog_file")
    if not catalog_file:
        return
    path = ROOT / str(catalog_file)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, spec_payload in payload.items():
        data = dict(spec_payload)
        data.pop("id", None)
        data.pop("derived_from_catalog", None)
        data.pop("shift_bins", None)
        if "fixed_fractions" in data:
            data["fixed_fractions"] = tuple(float(v) for v in data["fixed_fractions"])
        STIMULUS_CATALOG[str(name)] = StimulusSpec(**data)  # type: ignore[arg-type]


def _profile_items(protocol: dict) -> list[tuple[str, float, str | None]]:
    """Return (profile_id, concentration_proxy, stimulus_id_or_none)."""
    if "stimulus_profiles" in protocol and "concentration_profiles" in protocol:
        raise ValueError(
            "protocol cannot define both stimulus_profiles and "
            "concentration_profiles"
        )
    if "stimulus_profiles" in protocol:
        _load_stimulus_catalog(protocol)
        items = []
        for profile in protocol["stimulus_profiles"]:
            if profile not in STIMULUS_CATALOG:
                raise ValueError(f"unknown stimulus profile: {profile}")
            items.append(
                (
                    str(profile),
                    stimulus_concentration_proxy(profile),
                    str(profile),
                )
            )
        return items
    return [
        (str(profile), float(value), None)
        for profile, value in protocol["concentration_profiles"].items()
    ]


def build_tasks(protocol: dict) -> list[RepresentationTask]:
    if protocol.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(
            f"unsupported prompt_version: {protocol.get('prompt_version')}"
        )
    offsets = _resolve_offsets(protocol)
    repetitions = int(protocol["repetitions_per_condition"])
    base_seed = int(protocol["base_seed"])
    seed_strategy = protocol.get("seed_strategy", "sequential")
    seed_block_id = protocol.get("seed_block_id")
    tasks = []
    task_number = 0
    for representation_spec in protocol["representations"]:
        representation, _ = coerce_representation_spec(representation_spec)
        for profile, concentration, stimulus in _profile_items(protocol):
            for offset_index, offset in offsets:
                if stimulus is None:
                    prompt = build_representation_prompt(
                        representation_spec,
                        float(offset),
                        concentration,
                    )
                else:
                    prompt = build_representation_prompt(
                        representation_spec,
                        float(offset),
                        concentration,
                        stimulus=stimulus,
                    )
                prompt_hash = _hash(prompt)
                for repetition in range(repetitions):
                    task_id = (
                        f"{representation}__{profile}"
                        f"__offset_{offset_index:02d}"
                        f"__rep_{repetition:04d}"
                    )
                    if seed_strategy == "hash_v1":
                        sample_seed = _derive_sample_seed(
                            base_seed,
                            representation,
                            profile,
                            offset_index,
                            repetition,
                        )
                    elif seed_strategy == "sequential":
                        sample_seed = base_seed + task_number
                    else:
                        raise ValueError(
                            f"unknown seed_strategy: {seed_strategy}"
                        )
                    tasks.append(
                        RepresentationTask(
                            task_id=task_id,
                            representation=representation,
                            profile=profile,
                            concentration=concentration,
                            offset_index=offset_index,
                            offset_radians=float(offset),
                            repetition=repetition,
                            sample_seed=sample_seed,
                            prompt=prompt,
                            prompt_sha256=prompt_hash,
                            seed_block_id=seed_block_id,
                        )
                    )
                    task_number += 1
    if len({task.sample_seed for task in tasks}) != len(tasks):
        raise ValueError("sample-seed collision detected")
    if "schedule_seed" in protocol:
        rng = np.random.default_rng(int(protocol["schedule_seed"]))
        order = rng.permutation(len(tasks))
        tasks = [tasks[int(index)] for index in order]
    return tasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "runs" / "response_law_representation",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.backend == "openai":
        if not args.model:
            parser.error("--model is required for the openai backend")
        if args.input_price_per_million is None:
            parser.error("--input-price-per-million is required")
        if args.output_price_per_million is None:
            parser.error("--output-price-per-million is required")
    else:
        args.input_price_per_million = 0.0
        args.output_price_per_million = 0.0
    if args.concurrency <= 0 or args.max_attempts <= 0:
        parser.error("concurrency and max-attempts must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol)
    protocol_text = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol_hash = _hash(protocol_text)
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

    model = args.model if args.backend == "openai" else "attractive-v1"
    prospective_config = {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "backend": args.backend,
        "model": model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "max_attempts": args.max_attempts,
        "base_url": args.base_url,
        "env_file": str(args.env_file.resolve()) if args.env_file else None,
    }
    existing_config = _validate_resume_config(
        run_dir / "resolved_config.json",
        prospective_config,
    )
    trace_path = run_dir / "trace.jsonl"
    existing = _load_existing_trace(trace_path)
    known_ids = {task.task_id for task in tasks}
    unknown_ids = {
        record["task_id"]
        for record in existing
        if record["task_id"] not in known_ids
    }
    if unknown_ids:
        raise ValueError(
            f"trace contains {len(unknown_ids)} unknown task IDs"
        )
    completed_ids = {record["task_id"] for record in existing}
    pending = [task for task in tasks if task.task_id not in completed_ids]
    if pending:
        estimate = estimate_cost(
            [task.prompt for task in pending],
            max_output_tokens_per_call=args.max_output_tokens,
            max_attempts_per_call=args.max_attempts,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
        )
        print(format_cost_estimate(estimate), flush=True)
    else:
        estimate = None
        print("No pending calls.", flush=True)
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
    if not pending:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **prospective_config,
        "concurrency": args.concurrency,
        "cost_estimate": estimate.as_dict() if estimate else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if existing_config is None:
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
    (run_dir / "cost_estimate.json").write_text(
        json.dumps(estimate.as_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    unique_stimuli = {}
    for task in tasks:
        unique_stimuli.setdefault(
            task.prompt_sha256,
            {
                "representation": task.representation,
                "seed_block_id": task.seed_block_id,
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
    with trace_path.open("a", encoding="utf-8", buffering=1) as trace:
        bulk_tasks = pending
        completed_before_bulk = 0
        if args.backend != "mock":
            print("running one serial preflight call", flush=True)
            record = run_task(
                pending[0],
                backend,
                max_attempts=args.max_attempts,
            )
            trace.write(json.dumps(record, ensure_ascii=False) + "\n")
            new_records.append(record)
            if not record["valid"]:
                print("Preflight invalid; bulk execution stopped.", file=sys.stderr)
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
                    if completed_index % 200 == 0:
                        print(
                            f"completed {completed_index:,}/{len(pending):,}",
                            flush=True,
                        )
            except QuotaError:
                for future in futures:
                    future.cancel()
                raise

    execution_elapsed = time.perf_counter() - execution_started
    records = existing + new_records
    valid = sum(record.get("valid", False) for record in records)
    actual_input = sum(int(record.get("input_tokens", 0)) for record in records)
    actual_output = sum(int(record.get("output_tokens", 0)) for record in records)
    actual_cost = (
        actual_input * args.input_price_per_million
        + actual_output * args.output_price_per_million
    ) / 1_000_000
    summed_task_elapsed = sum(
        float(record.get("elapsed_seconds", 0.0))
        for record in new_records
    )
    summary = {
        "run_dir": str(run_dir),
        "n_records": len(records),
        "n_valid": valid,
        "valid_rate": valid / len(records),
        "actual_input_tokens": actual_input,
        "actual_output_tokens": actual_output,
        "actual_cost_usd": actual_cost,
        "configured_concurrency": args.concurrency,
        "api_execution_elapsed_seconds": execution_elapsed,
        "summed_task_elapsed_seconds": summed_task_elapsed,
        "effective_parallelism": summed_task_elapsed / execution_elapsed,
        "calls_per_second": len(new_records) / execution_elapsed,
        "seed_block_id": protocol.get("seed_block_id"),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if valid == len(records) else 3


if __name__ == "__main__":
    raise SystemExit(main())
