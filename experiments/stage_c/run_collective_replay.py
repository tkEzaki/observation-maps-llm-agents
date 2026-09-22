"""Collective-field replay panel: fixed Stage-C fields → Stage-B response law.

Smoke-tests prompt hashes against original Stage C observations before paid run.
"""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.collective_scan import _histogram_from_phases  # noqa: E402
from circlemap.backends import QuotaError  # noqa: E402
from circlemap.costing import estimate_cost, format_cost_estimate  # noqa: E402
from circlemap.observation import RelativePhaseHistogram  # noqa: E402
from circlemap.representations import build_representation_prompt_from_histogram  # noqa: E402
from circlemap.response import parse_social_action  # noqa: E402
from experiments.stage_b_response_law.run import (  # noqa: E402
    _load_existing_trace,
    _make_backend,
    run_task,
)

DEFAULT_PROTOCOL = Path(__file__).with_name("protocol_collective_replay_v0_1.json")
STAGE_C_SESSION = (
    ROOT
    / "runs"
    / "stage_c"
    / "stage-c-v0.1_0291166d2e39"
    / "20260723T235620Z"
)


@dataclass(frozen=True)
class ReplayTask:
    task_id: str
    representation: str
    profile: str
    offset_index: int
    offset_radians: float
    repetition: int
    sample_seed: int
    prompt: str
    prompt_sha256: str
    seed_block_id: str
    # aliases for run_task / Stage B tooling
    concentration: float = 0.0


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _histogram_from_field(field: dict) -> RelativePhaseHistogram:
    fr = np.asarray(field["fixed_fractions"], dtype=np.float64)
    n_bins = int(fr.size)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1, dtype=np.float64)
    return RelativePhaseHistogram(
        edges=edges,
        fractions=fr,
        peer_count=int(field["peer_count"]),
    )


def smoke_test_prompt_hashes(
    fields: list[dict],
    *,
    representation: str,
    stage_c_session: Path,
) -> dict:
    """Require hash(replay prompt) == hash(original Stage C observation prompt)."""
    checked = 0
    mismatches = []
    for field in fields:
        hist = _histogram_from_field(field)
        replay_prompt = build_representation_prompt_from_histogram(
            representation, hist
        )
        replay_hash = _sha(replay_prompt)
        stored = field.get("prompt_sha256")
        if stored and stored != replay_hash:
            mismatches.append(
                {
                    "field_id": field["field_id"],
                    "kind": "stored_vs_replay",
                    "stored": stored,
                    "replay": replay_hash,
                }
            )
        # Stage-C-sourced fields: rebuild from phases.npy
        run_id = field.get("source_run_id") or ""
        t = int(field.get("t", -1))
        agent = int(field.get("agent", -1))
        if run_id and t >= 0 and agent >= 0:
            phases = np.load(stage_c_session / run_id / "phases.npy")
            hist_c = _histogram_from_phases(phases[t], agent)
            # Exact mass match
            if not np.allclose(hist_c.fractions, hist.fractions, atol=1e-12):
                mismatches.append(
                    {
                        "field_id": field["field_id"],
                        "kind": "fraction_mismatch_vs_phases",
                        "max_abs": float(
                            np.max(np.abs(hist_c.fractions - hist.fractions))
                        ),
                    }
                )
            orig_prompt = build_representation_prompt_from_histogram(
                representation, hist_c
            )
            orig_hash = _sha(orig_prompt)
            checked += 1
            if orig_hash != replay_hash:
                mismatches.append(
                    {
                        "field_id": field["field_id"],
                        "kind": "stage_c_vs_replay",
                        "stage_c": orig_hash,
                        "replay": replay_hash,
                    }
                )
            # Also verify against Stage C trace record if present
            trace_path = stage_c_session / run_id / "trace.jsonl"
            if trace_path.exists():
                with trace_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        rec = json.loads(line)
                        if int(rec["t"]) == t and int(rec["agent"]) == agent:
                            if rec["prompt_sha256"] != replay_hash:
                                mismatches.append(
                                    {
                                        "field_id": field["field_id"],
                                        "kind": "trace_vs_replay",
                                        "trace": rec["prompt_sha256"],
                                        "replay": replay_hash,
                                    }
                                )
                            break
    return {
        "stage_c_sourced_checked": checked,
        "n_fields": len(fields),
        "n_mismatches": len(mismatches),
        "pass": len(mismatches) == 0,
        "mismatches": mismatches[:10],
    }


def build_tasks(protocol: dict, fields: list[dict]) -> list[ReplayTask]:
    representation = protocol["representation"]
    reps = int(protocol["repetitions_per_condition"])
    n_blocks = int(protocol["n_blocks"])
    tasks: list[ReplayTask] = []
    for block_i in range(1, n_blocks + 1):
        block_id = f"block_{block_i}"
        base = 2026072460 + block_i
        for field in fields:
            hist = _histogram_from_field(field)
            prompt = build_representation_prompt_from_histogram(
                representation, hist
            )
            prompt_hash = _sha(prompt)
            for rep in range(reps):
                payload = (
                    f"replay-v1|{base}|{field['field_id']}|{rep}|{block_id}"
                )
                sample_seed = (
                    int.from_bytes(
                        hashlib.sha256(payload.encode()).digest()[:8], "big"
                    )
                    % 2_147_483_647
                )
                task_id = f"{field['field_id']}__{block_id}__r{rep:02d}"
                tasks.append(
                    ReplayTask(
                        task_id=task_id,
                        representation=representation,
                        profile=field["field_id"],
                        offset_index=0,
                        offset_radians=0.0,
                        repetition=rep,
                        sample_seed=sample_seed,
                        prompt=prompt,
                        prompt_sha256=prompt_hash,
                        seed_block_id=block_id,
                    )
                )
    return tasks


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--input-price-per-million", type=float, default=0.75)
    parser.add_argument("--output-price-per-million", type=float, default=4.5)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "runs" / "collective_replay",
    )
    parser.add_argument("--stage-c-session", type=Path, default=STAGE_C_SESSION)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    fields_path = ROOT / protocol["fields_file"]
    payload = json.loads(fields_path.read_text(encoding="utf-8"))
    fields = payload["fields"]

    print("Running prompt-hash smoke test...", flush=True)
    smoke = smoke_test_prompt_hashes(
        fields,
        representation=protocol["representation"],
        stage_c_session=args.stage_c_session,
    )
    smoke_path = (
        ROOT
        / "analysis"
        / "stage_c_artifacts"
        / "replay_panel_v1"
        / "prompt_hash_smoke.json"
    )
    smoke_path.write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    print(json.dumps(smoke, indent=2), flush=True)
    if not smoke["pass"]:
        raise SystemExit("SMOKE FAIL: refusing to run replay")

    if args.smoke_only:
        print("Smoke PASS - exiting (--smoke-only)", flush=True)
        return 0

    if args.backend == "openai" and not args.model:
        parser.error("--model required for openai")

    tasks = build_tasks(protocol, fields)
    assert len(tasks) == int(protocol["expected_calls"]), (
        len(tasks),
        protocol["expected_calls"],
    )

    if args.estimate_only or (args.backend == "openai" and not args.yes):
        estimate = estimate_cost(
            [t.prompt for t in tasks],
            max_output_tokens_per_call=args.max_output_tokens,
            max_attempts_per_call=args.max_attempts,
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
        )
        print(format_cost_estimate(estimate).replace("Stage B", "Replay"), flush=True)
        if args.estimate_only:
            return 0
        if args.backend == "openai" and not args.yes:
            raise SystemExit("Refusing paid run without --yes")

    # Persist prospective v1 predictions (pre-acquisition lock)
    pred_lock = {
        "locked_before_acquisition": True,
        "fields": [
            {
                "field_id": f["field_id"],
                "pred_p_retard": f["pred_p_retard"],
                "pred_p_stay": f["pred_p_stay"],
                "pred_p_advance": f["pred_p_advance"],
                "R": f["R"],
                "prompt_sha256": f["prompt_sha256"],
                "physical_hash": f["physical_hash"],
            }
            for f in fields
        ],
    }
    pred_path = (
        ROOT
        / "analysis"
        / "stage_c_artifacts"
        / "replay_panel_v1"
        / "v1_predictions_locked.json"
    )
    pred_path.write_text(json.dumps(pred_lock, indent=2), encoding="utf-8")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    proto_hash = _sha_file(args.protocol)[:12]
    run_dir = args.out_root / f"{protocol['protocol_version']}_{proto_hash}" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    model = args.model if args.backend == "openai" else "mock"
    config = {
        "protocol": protocol,
        "protocol_sha256": _sha_file(args.protocol),
        "fields_sha256": _sha_file(fields_path),
        "v1_predictions_sha256": _sha_file(pred_path),
        "smoke": smoke,
        "backend": args.backend,
        "model": model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "max_attempts": args.max_attempts,
        "concurrency": args.concurrency,
        "prompt_version": protocol["prompt_version"],
        "representation": protocol["representation"],
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    backend = _make_backend(args)
    trace_path = run_dir / "trace.jsonl"
    completed = {r["task_id"] for r in _load_existing_trace(trace_path)}
    pending = [t for t in tasks if t.task_id not in completed]
    print(f"pending calls: {len(pending)}/{len(tasks)}", flush=True)

    # run_task expects ProbeTask — check fields
    # From run.py ProbeTask: task_id, concentration, offset_index, offset_radians,
    # repetition, sample_seed, prompt, prompt_sha256
    # Representation adds representation, profile, seed_block_id
    # Our ReplayTask has all of these.

    n_done = 0
    n_valid = 0
    tin = tout = 0
    t0 = time.perf_counter()
    with trace_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    run_task, task, backend, max_attempts=args.max_attempts
                ): task
                for task in pending
            }
            for fut in as_completed(futures):
                record = fut.result()
                # Enrich with profile/block if missing
                task = futures[fut]
                record["profile"] = task.profile
                record["representation"] = task.representation
                record["seed_block_id"] = task.seed_block_id
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
                        f"valid_rate={n_valid/max(n_done,1):.3f}",
                        flush=True,
                    )

    cost = (tin * args.input_price_per_million + tout * args.output_price_per_million) / 1e6
    summary = {
        "run_dir": str(run_dir),
        "n_planned": len(tasks),
        "n_executed": n_done,
        "n_valid": n_valid,
        "valid_rate": n_valid / max(n_done, 1),
        "actual_input_tokens": tin,
        "actual_output_tokens": tout,
        "actual_cost_usd": cost,
        "elapsed_seconds": time.perf_counter() - t0,
        "smoke_pass": True,
        "v1_predictions_locked": str(pred_path.relative_to(ROOT).as_posix()),
    }
    (run_dir / "session_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
