"""R2 Claude macroscopic matched-representation collective runner.

Round-robin acquisition across three representations for each seed×K cell.
Estimate-only is allowed without paid auth; paid runs require r2 auth_go + --yes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from analysis.stage3.collective_scan import _histogram_from_phases  # noqa: E402
from circlemap.backends import (  # noqa: E402
    AnthropicResponseBackend,
    MockResponseBackend,
    QuotaError,
)
from circlemap.costing import estimate_cost, format_cost_estimate  # noqa: E402
from circlemap.engine import step  # noqa: E402
from circlemap.observation import wrap_phase  # noqa: E402
from circlemap.representations import (  # noqa: E402
    build_representation_prompt_from_histogram,
)
from circlemap.response import ACTION_VALUES, parse_social_action  # noqa: E402
from experiments.stage_c.run_collective import _omega  # noqa: E402

DEFAULT_PROTOCOL = Path(__file__).with_name(
    "protocol_matched_rep_collective_r2_claude_v0_1.json"
)
ARTIFACT_DIR = ROOT / "analysis" / "matched_rep_collective" / "r2_macro"
REPS = ("moments_m1_m3", "centers_24_standard", "intervals_24_decimal6")


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _order_parameter(phases: np.ndarray, m: int = 1) -> float:
    return float(np.abs(np.mean(np.exp(1j * m * wrap_phase(phases)))))


def _sample_seed(
    *,
    family: str,
    model_id: str,
    representation: str,
    init_seed: int,
    coupling: float,
    agent: int,
    t: int,
) -> int:
    payload = (
        f"{family}|{model_id}|{representation}|{init_seed}|"
        f"{float(coupling):.12g}|{agent}|{t}"
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % 2_147_483_647


def _deterministic_rescue(raw_text: str | None):
    if not raw_text:
        return None
    text = raw_text.strip()
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


def _authorized(protocol: dict, *, backend: str) -> tuple[bool, str]:
    auth_path = ROOT / protocol["auth_go_sidecar"]
    if not auth_path.exists():
        return False, f"missing_auth_sidecar:{auth_path.as_posix()}"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    if auth.get("experiment_family") != protocol.get("experiment_family"):
        return False, "auth_go_wrong_experiment_family"
    if auth.get("protocol_version") != protocol.get("protocol_version"):
        return False, "auth_go_protocol_version_mismatch"
    if auth.get("paid_authorized"):
        return True, "r2_auth_go_paid"
    if backend == "mock" and auth.get("mock_smoke_authorized"):
        return True, "r2_mock_smoke_authorized"
    return False, "auth_go_paid_authorized_false"


def _make_backend(args: argparse.Namespace, protocol: dict):
    backend = args.backend
    model = args.model or protocol["model"]
    max_out = int(args.max_output_tokens or protocol["max_output_tokens"])
    if backend == "mock":
        return MockResponseBackend(), "mock"
    if backend == "anthropic":
        if args.env_file is not None:
            from dotenv import load_dotenv

            load_dotenv(args.env_file, encoding="utf-8-sig", override=False)
        return (
            AnthropicResponseBackend(
                model=model,
                temperature=float(args.temperature),
                max_output_tokens=max_out,
                timeout_seconds=float(args.timeout),
            ),
            model,
        )
    raise SystemExit(f"unsupported backend for R2: {backend}")


def _call_one(call: dict, backend, max_attempts: int) -> dict:
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
    started = time.perf_counter()

    for _ in range(1, max_attempts + 1):
        try:
            response = backend.call(call["prompt"], seed=int(call["sample_seed"]))
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

    return {
        **{k: call[k] for k in (
            "cell_id",
            "representation",
            "t",
            "agent",
            "sample_seed",
            "prompt_sha256",
            "init_seed",
            "coupling",
            "seed_index",
            "panel",
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
        "elapsed_seconds": time.perf_counter() - started,
    }


def _estimate(protocol: dict, args: argparse.Namespace) -> dict:
    physical = json.loads(
        (ARTIFACT_DIR / "physical_seed_manifest.json").read_text(encoding="utf-8")
    )
    tasks = json.loads(
        (ARTIFACT_DIR / "task_manifest.json").read_text(encoding="utf-8")
    )
    n = int(protocol["n_agents"])
    n_bins = int(protocol["n_bins"])
    # One sample prompt per representation (character lengths differ).
    phases = np.linspace(-np.pi, np.pi, n, endpoint=False)
    prompts: list[str] = []
    for rep in REPS:
        hist = _histogram_from_phases(phases, 0, n_bins=n_bins)
        prompts.append(build_representation_prompt_from_histogram(rep, hist))
    # Weight by equal share of total calls across three reps.
    n_calls = int(protocol["expected_calls"])
    if n_calls % 3 != 0:
        raise SystemExit("expected_calls not divisible by 3 representations")
    per_rep = n_calls // 3
    expanded = []
    for p in prompts:
        expanded.extend([p] * per_rep)

    inp = float(
        args.input_price_per_million
        if args.input_price_per_million is not None
        else protocol["input_price_per_million"]
    )
    outp = float(
        args.output_price_per_million
        if args.output_price_per_million is not None
        else protocol["output_price_per_million"]
    )
    estimate = estimate_cost(
        expanded,
        max_output_tokens_per_call=int(
            args.max_output_tokens or protocol["max_output_tokens"]
        ),
        max_attempts_per_call=int(args.max_attempts or protocol["max_attempts"]),
        input_price_per_million=inp,
        output_price_per_million=outp,
    )
    payload = {
        "status": "estimate_only_complete",
        "protocol_version": protocol["protocol_version"],
        "model": protocol["model"],
        "family": protocol["family"],
        "expected_calls": n_calls,
        "task_manifest_calls": tasks["expected_calls_total"],
        "n_physical_cells": physical["n_cells"],
        "cost_ceiling_usd": protocol["cost_ceiling_usd"],
        "estimate": estimate.as_dict(),
        "within_ceiling_base": estimate.estimated_base_cost_usd
        <= float(protocol["cost_ceiling_usd"]),
        "within_ceiling_max_retry": estimate.maximum_retry_cost_usd
        <= float(protocol["cost_ceiling_usd"]),
        "prompt_char_lengths": {rep: len(p) for rep, p in zip(REPS, prompts)},
        "generated_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    out_path = ARTIFACT_DIR / "estimate_only.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(format_cost_estimate(estimate))
    print(f"Wrote {out_path.as_posix()}")
    print(
        f"ceiling=${protocol['cost_ceiling_usd']}: "
        f"base_ok={payload['within_ceiling_base']} "
        f"max_retry_ok={payload['within_ceiling_max_retry']}"
    )
    return payload


def _run_cell(
    task: dict,
    protocol: dict,
    backend,
    *,
    session_dir: Path,
    concurrency: int,
    max_attempts: int,
) -> dict:
    n_agents = int(task["n_agents"])
    n_steps = int(task["n_steps"])
    n_bins = int(protocol["n_bins"])
    coupling = float(task["coupling"])
    init_seed = int(task["init_seed"])
    reps = list(task["representation_order"])
    cell_dir = session_dir / task["cell_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(init_seed)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_agents)
    omega = _omega(n_agents, float(protocol["omega_halfwidth"]))

    states: dict[str, dict] = {}
    for rep in reps:
        rep_dir = cell_dir / rep
        rep_dir.mkdir(parents=True, exist_ok=True)
        traj_path = rep_dir / "phases.npy"
        actions_path = rep_dir / "actions.npy"
        phases_hist = np.empty((n_steps + 1, n_agents), dtype=np.float64)
        actions_hist = np.empty((n_steps, n_agents), dtype=np.float64)
        if traj_path.exists() and actions_path.exists():
            prev_phases = np.load(traj_path)
            prev_actions = np.load(actions_path)
            start_t = int(prev_actions.shape[0])
            if start_t > n_steps:
                raise RuntimeError(
                    f"{task['cell_id']}/{rep}: resume steps exceed protocol"
                )
            phases_hist[: start_t + 1] = prev_phases[: start_t + 1]
            actions_hist[:start_t] = prev_actions[:start_t]
            phases = phases_hist[start_t].copy()
        else:
            phases_hist[0] = theta0
            phases = theta0.copy()
            start_t = 0
        states[rep] = {
            "dir": rep_dir,
            "phases_hist": phases_hist,
            "actions_hist": actions_hist,
            "phases": phases,
            "start_t": start_t,
            "trace": (rep_dir / "trace.jsonl").open("a", encoding="utf-8"),
            "total_in": 0,
            "total_out": 0,
            "n_valid": 0,
            "n_calls": 0,
            "n_rescued": 0,
            "n_unrecoverable": 0,
        }

    # Resume from min completed step across reps (keep triplet synchronized).
    start_t = min(states[r]["start_t"] for r in reps)
    for rep in reps:
        st = states[rep]
        if st["start_t"] != start_t:
            # Truncate ahead reps to common frontier (should be rare).
            st["phases"] = st["phases_hist"][start_t].copy()
            st["start_t"] = start_t

    totals = {"input_tokens": 0, "output_tokens": 0, "n_calls": 0, "n_valid": 0}

    try:
        for t in range(start_t, n_steps):
            for rep in reps:
                st = states[rep]
                calls = []
                for agent in range(n_agents):
                    hist = _histogram_from_phases(st["phases"], agent, n_bins=n_bins)
                    prompt = build_representation_prompt_from_histogram(rep, hist)
                    calls.append(
                        {
                            "cell_id": task["cell_id"],
                            "panel": task["panel"],
                            "seed_index": task["seed_index"],
                            "coupling": coupling,
                            "init_seed": init_seed,
                            "representation": rep,
                            "t": t,
                            "agent": agent,
                            "sample_seed": _sample_seed(
                                family=protocol["family"],
                                model_id=protocol["model"],
                                representation=rep,
                                init_seed=init_seed,
                                coupling=coupling,
                                agent=agent,
                                t=t,
                            ),
                            "prompt": prompt,
                            "prompt_sha256": _sha_text(prompt),
                        }
                    )
                results: list[dict | None] = [None] * n_agents
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = {
                        pool.submit(_call_one, call, backend, max_attempts): call[
                            "agent"
                        ]
                        for call in calls
                    }
                    for fut in as_completed(futures):
                        agent = futures[fut]
                        results[agent] = fut.result()

                if any(r is None for r in results):
                    raise RuntimeError(f"{task['cell_id']}/{rep} t={t}: missing agent")
                if any(bool(r["unrecoverable"]) for r in results):  # type: ignore[index]
                    # Do not write partial step; leave for same-cell reacquire.
                    for r in results:
                        assert r is not None
                        st["trace"].write(json.dumps(r, ensure_ascii=True) + "\n")
                        st["trace"].flush()
                        st["n_calls"] += 1
                        st["total_in"] += int(r["input_tokens"])
                        st["total_out"] += int(r["output_tokens"])
                        st["n_unrecoverable"] += 1
                    raise RuntimeError(
                        f"unrecoverable at {task['cell_id']}/{rep} t={t}; "
                        "cell not advanced (reacquire same cell)"
                    )

                actions = np.zeros(n_agents, dtype=np.float64)
                for agent, record in enumerate(results):
                    assert record is not None
                    st["n_calls"] += 1
                    st["total_in"] += int(record["input_tokens"])
                    st["total_out"] += int(record["output_tokens"])
                    totals["n_calls"] += 1
                    totals["input_tokens"] += int(record["input_tokens"])
                    totals["output_tokens"] += int(record["output_tokens"])
                    if record["valid"]:
                        st["n_valid"] += 1
                        totals["n_valid"] += 1
                        actions[agent] = float(record["action_value"])
                    if record["rescued_parse"]:
                        st["n_rescued"] += 1
                    st["trace"].write(json.dumps(record, ensure_ascii=True) + "\n")
                st["trace"].flush()

                st["phases"] = step(st["phases"], omega, coupling, actions)
                st["actions_hist"][t] = actions
                st["phases_hist"][t + 1] = st["phases"]
                np.save(st["dir"] / "phases.npy", st["phases_hist"][: t + 2])
                np.save(st["dir"] / "actions.npy", st["actions_hist"][: t + 1])

            if (t + 1) % 10 == 0 or t + 1 == n_steps:
                rs = {
                    rep: _order_parameter(states[rep]["phases"]) for rep in reps
                }
                print(
                    f"  {task['cell_id']}: step {t+1}/{n_steps} "
                    + " ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in rs.items())
                )
    finally:
        for rep in reps:
            states[rep]["trace"].close()

    summaries = {}
    for rep in reps:
        st = states[rep]
        r1 = [
            _order_parameter(st["phases_hist"][i], 1)
            for i in range(n_steps + 1)
        ]
        r2 = [
            _order_parameter(st["phases_hist"][i], 2)
            for i in range(n_steps + 1)
        ]
        summary = {
            "cell_id": task["cell_id"],
            "panel": task["panel"],
            "seed_index": task["seed_index"],
            "coupling": coupling,
            "init_seed": init_seed,
            "representation": rep,
            "n_steps": n_steps,
            "n_calls": st["n_calls"],
            "n_valid": st["n_valid"],
            "n_rescued": st["n_rescued"],
            "n_unrecoverable": st["n_unrecoverable"],
            "valid_rate": st["n_valid"] / max(st["n_calls"], 1),
            "actual_input_tokens": st["total_in"],
            "actual_output_tokens": st["total_out"],
            "final_r1": r1[-1],
            "final_r2": r2[-1],
            "mean_r1": float(np.mean(r1[1:])) if len(r1) > 1 else float("nan"),
            "r1_series": r1,
            "r2_series": r2,
        }
        (st["dir"] / "run_meta.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        summaries[rep] = summary

    # K=0 control within cell
    k0_ok = True
    k0_max_diff = 0.0
    if abs(coupling) < 1e-15:
        for m in (1, 2, 3):
            series = {
                rep: np.array(
                    [
                        _order_parameter(states[rep]["phases_hist"][i], m)
                        for i in range(n_steps + 1)
                    ]
                )
                for rep in reps
            }
            for a in reps:
                for b in reps:
                    diff = float(np.max(np.abs(series[a] - series[b])))
                    k0_max_diff = max(k0_max_diff, diff)
        tol = float(protocol["k0_control"]["tolerance"])
        k0_ok = k0_max_diff < tol
        if not k0_ok:
            raise RuntimeError(
                f"K=0 engine/matching failure on {task['cell_id']}: "
                f"max|Δr_m|={k0_max_diff} >= {tol}"
            )

    cell_meta = {
        "cell_id": task["cell_id"],
        "panel": task["panel"],
        "seed_index": task["seed_index"],
        "coupling": coupling,
        "init_seed": init_seed,
        "representations": reps,
        "totals": totals,
        "k0_control": {
            "applicable": abs(coupling) < 1e-15,
            "max_abs_diff_rm": k0_max_diff,
            "pass": k0_ok,
        },
        "per_representation": {
            rep: {
                "final_r1": summaries[rep]["final_r1"],
                "final_r2": summaries[rep]["final_r2"],
                "valid_rate": summaries[rep]["valid_rate"],
            }
            for rep in reps
        },
    }
    (cell_dir / "cell_meta.json").write_text(
        json.dumps(cell_meta, indent=2), encoding="utf-8"
    )
    return cell_meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--backend", choices=("mock", "anthropic"), default="mock"
    )
    parser.add_argument("--model")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--input-price-per-million", type=float, default=None)
    parser.add_argument("--output-price-per-million", type=float, default=None)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "runs" / "stage_c" / "r2_claude_macro",
    )
    parser.add_argument("--resume-session", type=Path, default=None)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--cell-id",
        action="append",
        help="optional subset of cell_id values; default = all frozen tasks",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="optional cap for mock smoke",
    )
    args = parser.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    for req in (
        ARTIFACT_DIR / "physical_seed_manifest.json",
        ARTIFACT_DIR / "task_manifest.json",
    ):
        if not req.exists():
            raise SystemExit(
                f"missing freeze artifact {req}; run "
                "py -m analysis.matched_rep_collective.freeze_r2_artifacts"
            )

    if args.estimate_only:
        print("R2 estimate-only: auth_go not required")
        _estimate(protocol, args)
        return 0

    ok, reason = _authorized(protocol, backend=args.backend)
    if not ok:
        raise SystemExit(
            "R2 Claude macro not authorized "
            f"(reason={reason}). Complete human review and set "
            f"{protocol['auth_go_sidecar']} paid_authorized=true before "
            "anthropic --yes. Use --estimate-only for costing. "
            "Mock smoke allowed when mock_smoke_authorized=true."
        )
    print(f"R2 authorization: {reason}")

    if args.backend == "anthropic" and not args.yes:
        _estimate(protocol, args)
        raise SystemExit("Refusing paid run without --yes")

    concurrency = int(
        args.concurrency
        or protocol["acquisition"]["concurrency_within_timestep_per_rep"]
    )
    max_attempts = int(args.max_attempts or protocol["max_attempts"])
    backend, model_name = _make_backend(args, protocol)

    task_manifest = json.loads(
        (ARTIFACT_DIR / "task_manifest.json").read_text(encoding="utf-8")
    )
    tasks = task_manifest["tasks"]
    if args.cell_id:
        wanted = set(args.cell_id)
        tasks = [t for t in tasks if t["cell_id"] in wanted]
    if args.max_cells is not None:
        tasks = tasks[: int(args.max_cells)]

    if args.resume_session is not None:
        session_dir = args.resume_session
        if not session_dir.is_dir():
            raise SystemExit(f"--resume-session not a directory: {session_dir}")
        print(f"Resuming session: {session_dir}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        proto_hash = _sha_file(args.protocol)[:12]
        session_dir = (
            args.out_root / f"{protocol['protocol_version']}_{proto_hash}" / stamp
        )
        session_dir.mkdir(parents=True, exist_ok=False)
        (session_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
        (session_dir / "resolved.json").write_text(
            json.dumps(
                {
                    "protocol_sha256": _sha_file(args.protocol),
                    "backend": args.backend,
                    "model": model_name,
                    "task_manifest_sha256": _sha_file(
                        ARTIFACT_DIR / "task_manifest.json"
                    ),
                    "physical_seed_manifest_sha256": _sha_file(
                        ARTIFACT_DIR / "physical_seed_manifest.json"
                    ),
                    "n_tasks": len(tasks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Record PID for ResearchBot-safe stop (PID-only kill).
    (session_dir / "pid").write_text(str(__import__("os").getpid()), encoding="utf-8")

    summaries = []
    for task in tasks:
        print(f"=== cell {task['order_index']}: {task['cell_id']} ===")
        meta = _run_cell(
            task,
            protocol,
            backend,
            session_dir=session_dir,
            concurrency=concurrency,
            max_attempts=max_attempts,
        )
        summaries.append(meta)

    (session_dir / "session_summary.json").write_text(
        json.dumps({"n_cells": len(summaries), "cells": summaries}, indent=2),
        encoding="utf-8",
    )
    print(f"Done. session={session_dir.as_posix()} n_cells={len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
