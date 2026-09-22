"""Stage C LLM collective runner (moments encoding; peer-matched N).

Synchronous barrier each timestep; concurrent API calls within a step.
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
from circlemap.engine import step  # noqa: E402
from circlemap.observation import wrap_phase  # noqa: E402
from circlemap.representations import build_representation_prompt_from_histogram  # noqa: E402
from circlemap.response import parse_social_action  # noqa: E402
from experiments.stage_b_response_law.run import _make_backend  # noqa: E402

DEFAULT_PROTOCOL = Path(__file__).with_name("protocol_stage_c_v0_1.json")
ACTION_LABEL = {-1.0: "retard", 0.0: "stay", 1.0: "advance"}


@dataclass(frozen=True)
class AgentCall:
    run_id: str
    t: int
    agent: int
    sample_seed: int
    prompt: str
    prompt_sha256: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _omega(n: int, halfwidth: float) -> np.ndarray:
    if n == 1:
        return np.zeros(1)
    return np.linspace(-halfwidth, halfwidth, n, dtype=np.float64)


def _order_parameter(phases: np.ndarray) -> float:
    return float(np.abs(np.mean(np.exp(1j * wrap_phase(phases)))))


def _resolve_bundle(protocol: dict) -> Path:
    key = "bundle_root"
    if key not in protocol and "bundle_root_prediction" in protocol:
        key = "bundle_root_prediction"
    return ROOT / protocol[key]


def _is_matched_rep_family(protocol: dict) -> bool:
    return (
        protocol.get("experiment_family") == "matched_representation_collective"
        and bool(protocol.get("surrogate_free"))
    )


def _matched_rep_auth_path() -> Path:
    return ROOT / "analysis" / "matched_rep_collective" / "auth_go.json"


def _matched_rep_authorized(
    protocol: dict, *, backend: str = "openai"
) -> tuple[bool, str]:
    """Surrogate-free matched multi-rep family: checklist sidecar only."""
    if not _is_matched_rep_family(protocol):
        return False, "not_matched_rep_family"
    if protocol.get("bundle_root") or protocol.get("bundle_root_prediction"):
        return False, "matched_rep_must_not_set_bundle_root"
    if protocol.get("criteria", {}).get("compare_to_surrogate"):
        return False, "matched_rep_must_disable_surrogate_comparison"
    auth_path = _matched_rep_auth_path()
    if not auth_path.exists():
        return False, f"missing_auth_sidecar:{auth_path.as_posix()}"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    if auth.get("experiment_family") != "matched_representation_collective":
        return False, "auth_go_wrong_experiment_family"
    if auth.get("match_group") and auth.get("match_group") != protocol.get(
        "match_group"
    ):
        return False, "auth_go_match_group_mismatch"
    if auth.get("paid_authorized"):
        return True, "matched_rep_auth_go_sidecar"
    if backend == "mock" and auth.get("mock_smoke_authorized"):
        return True, "matched_rep_mock_smoke_authorized"
    return False, "auth_go_paid_authorized_false"


def _stage_c_authorized(bundle: Path, protocol: dict) -> tuple[bool, str]:
    """Return (ok, reason). Hash-locked files are never mutated here."""
    if _is_matched_rep_family(protocol):
        return _matched_rep_authorized(protocol)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    ood = json.loads((bundle / "ood_policy.json").read_text(encoding="utf-8"))
    if manifest.get("stage_c_authorized") or ood.get("stage_c", {}).get("authorized"):
        return True, "bundle_manifest_or_ood"
    # v2: checklist authorization sidecar (keeps moments_bundle_v2 hashes intact)
    auth_path = (
        ROOT / "analysis" / "stage3a_artifacts" / "stage3b_v2_freeze_authorization.json"
    )
    version = str(protocol.get("protocol_version", ""))
    if version.startswith("stage-c-v0.2") and auth_path.exists():
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
        if (
            auth.get("checklist_all_gates_pass")
            and auth.get("stage_c_v0_2_paid_authorized_by_checklist")
            and manifest.get("freeze_ready")
            and protocol.get("bundle_root_prediction", "").endswith("moments_bundle_v2")
        ):
            return True, "v2_checklist_authorization_sidecar"
    return False, "not_authorized"


def _derive_seed(
    base: int,
    run_id: str,
    t: int,
    agent: int,
    *,
    protocol_version: str,
    representation: str = "",
) -> int:
    # Representation in the payload keeps API draws independent across encodings
    # even if protocol_version were accidentally shared.
    payload = f"{protocol_version}|{representation}|{base}|{run_id}|{t}|{agent}"
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % 2_147_483_647


def _run_specs(protocol: dict) -> list[dict]:
    specs = []
    couplings_by_n = protocol.get("couplings_by_n")
    for n_agents in protocol["n_agents"]:
        if couplings_by_n is not None:
            couplings = couplings_by_n.get(str(int(n_agents)))
            if couplings is None:
                couplings = couplings_by_n.get(int(n_agents))
            if couplings is None:
                raise KeyError(
                    f"couplings_by_n missing entry for n_agents={n_agents}"
                )
        else:
            couplings = protocol["couplings"]
        for coupling in couplings:
            for seed_i in range(int(protocol["n_seeds"])):
                run_id = (
                    f"N{int(n_agents)}_K{float(coupling):+g}_s{int(seed_i)}"
                )
                seed = (
                    int(protocol["base_seed"])
                    + 1000 * int(n_agents)
                    + 10 * int(seed_i)
                    + int(round(abs(float(coupling)) * 1000))
                )
                specs.append(
                    {
                        "run_id": run_id,
                        "n_agents": int(n_agents),
                        "coupling": float(coupling),
                        "seed_index": int(seed_i),
                        "init_seed": seed,
                    }
                )
    return specs


def _sample_prompt(protocol: dict) -> str:
    edges = np.linspace(-np.pi, np.pi, int(protocol["n_bins"]) + 1)
    fractions = np.ones(int(protocol["n_bins"])) / int(protocol["n_bins"])
    from circlemap.observation import RelativePhaseHistogram

    hist = RelativePhaseHistogram(
        edges=edges,
        fractions=fractions,
        peer_count=int(protocol["n_agents"][0]) - 1,
    )
    return build_representation_prompt_from_histogram(
        protocol["representation"], hist
    )


def _estimate(protocol: dict, args: argparse.Namespace) -> None:
    prompt = _sample_prompt(protocol)
    n_calls = int(protocol["expected_calls"])
    prompts = [prompt] * n_calls
    estimate = estimate_cost(
        prompts,
        max_output_tokens_per_call=int(
            args.max_output_tokens or protocol["max_output_tokens"]
        ),
        max_attempts_per_call=int(args.max_attempts or protocol["max_attempts"]),
        input_price_per_million=float(args.input_price_per_million),
        output_price_per_million=float(args.output_price_per_million),
    )
    print(format_cost_estimate(estimate).replace("Stage B", "Stage C"))
    print(f"  protocol expected_calls: {n_calls}")
    print(f"  sample prompt chars: {len(prompt)}")


def _call_one(call: AgentCall, backend, max_attempts: int) -> dict:
    errors = []
    in_tok = 0
    out_tok = 0
    for attempt in range(1, max_attempts + 1):
        try:
            response = backend.call(call.prompt, seed=call.sample_seed)
            in_tok += response.input_tokens
            out_tok += response.output_tokens
            action = parse_social_action(response.raw_text)
            return {
                **asdict(call),
                "prompt": None,
                "valid": True,
                "action_label": action.label,
                "action_value": action.value,
                "raw_response": response.raw_text,
                "attempts": attempt,
                "errors": errors,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
        except QuotaError:
            raise
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        **asdict(call),
        "prompt": None,
        "valid": False,
        "action_label": None,
        "action_value": None,
        "raw_response": None,
        "attempts": max_attempts,
        "errors": errors,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def _run_trajectory(
    spec: dict,
    protocol: dict,
    backend,
    *,
    run_dir: Path,
    concurrency: int,
    max_attempts: int,
) -> dict:
    n_agents = spec["n_agents"]
    n_steps = int(protocol["n_steps"])
    n_bins = int(protocol["n_bins"])
    coupling = spec["coupling"]
    rng = np.random.default_rng(spec["init_seed"])
    phases = rng.uniform(-np.pi, np.pi, size=n_agents)
    omega = _omega(n_agents, float(protocol["omega_halfwidth"]))
    representation = protocol["representation"]

    traj_path = run_dir / "phases.npy"
    actions_path = run_dir / "actions.npy"
    trace_path = run_dir / "trace.jsonl"
    meta_path = run_dir / "run_meta.json"

    phases_hist = np.empty((n_steps + 1, n_agents), dtype=np.float64)
    actions_hist = np.empty((n_steps, n_agents), dtype=np.float64)
    start_t = 0
    if traj_path.exists() and actions_path.exists():
        prev_phases = np.load(traj_path)
        prev_actions = np.load(actions_path)
        start_t = int(prev_actions.shape[0])
        if start_t > n_steps:
            raise RuntimeError(f"{spec['run_id']}: resume steps exceed protocol")
        phases_hist[: start_t + 1] = prev_phases[: start_t + 1]
        actions_hist[:start_t] = prev_actions[:start_t]
        phases = phases_hist[start_t].copy()
    else:
        phases_hist[0] = phases

    total_in = 0
    total_out = 0
    n_valid = 0
    n_calls = 0

    with trace_path.open("a", encoding="utf-8") as trace_handle:
        for t in range(start_t, n_steps):
            calls = []
            for agent in range(n_agents):
                hist = _histogram_from_phases(phases, agent, n_bins=n_bins)
                prompt = build_representation_prompt_from_histogram(
                    representation, hist
                )
                calls.append(
                    AgentCall(
                        run_id=spec["run_id"],
                        t=t,
                        agent=agent,
                        sample_seed=_derive_seed(
                            int(protocol["base_seed"]),
                            spec["run_id"],
                            t,
                            agent,
                            protocol_version=str(
                                protocol.get("protocol_version", "stage-c")
                            ),
                            representation=str(representation),
                        ),
                        prompt=prompt,
                        prompt_sha256=_sha256_text(prompt),
                    )
                )
            results: list[dict | None] = [None] * n_agents
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(_call_one, call, backend, max_attempts): call.agent
                    for call in calls
                }
                for fut in as_completed(futures):
                    agent = futures[fut]
                    results[agent] = fut.result()

            actions = np.zeros(n_agents, dtype=np.float64)
            for agent, record in enumerate(results):
                assert record is not None
                n_calls += 1
                total_in += int(record["input_tokens"])
                total_out += int(record["output_tokens"])
                if record["valid"]:
                    n_valid += 1
                    actions[agent] = float(record["action_value"])
                else:
                    actions[agent] = 0.0
                trace_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            trace_handle.flush()

            phases = step(phases, omega, coupling, actions)
            actions_hist[t] = actions
            phases_hist[t + 1] = phases
            np.save(traj_path, phases_hist[: t + 2])
            np.save(actions_path, actions_hist[: t + 1])
            if (t + 1) % 10 == 0 or t + 1 == n_steps:
                print(
                    f"  {spec['run_id']}: step {t+1}/{n_steps} "
                    f"r={_order_parameter(phases):.3f}"
                )

    r_series = [
        _order_parameter(phases_hist[t]) for t in range(n_steps + 1)
    ]
    summary = {
        **spec,
        "n_steps": n_steps,
        "n_calls": n_calls,
        "n_valid": n_valid,
        "valid_rate": n_valid / max(n_calls, 1),
        "actual_input_tokens": total_in,
        "actual_output_tokens": total_out,
        "final_r": r_series[-1],
        "mean_r": float(np.mean(r_series[1:])) if len(r_series) > 1 else float("nan"),
        "r_series": r_series,
    }
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--backend", choices=("mock", "openai"), default="mock")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--input-price-per-million", type=float, default=0.75)
    parser.add_argument("--output-price-per-million", type=float, default=4.5)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "runs" / "stage_c",
    )
    parser.add_argument(
        "--resume-session",
        type=Path,
        default=None,
        help="continue an existing session directory instead of creating a new stamp",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--run-id",
        action="append",
        help="optional subset of run_id values; default = all",
    )
    args = parser.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.backend == "openai" and not args.model:
        parser.error("--model is required for openai backend")

    matched = _is_matched_rep_family(protocol)
    bundle: Path | None = None

    # Estimate-only for matched-rep family is allowed without paid auth
    # (needed to fill the cost card before flipping auth_go.json).
    if args.estimate_only and matched:
        print(
            "Matched-rep estimate-only: auth_go not required "
            f"(family={protocol.get('experiment_family')})"
        )
        _estimate(protocol, args)
        return 0

    if matched:
        ok, auth_reason = _matched_rep_authorized(protocol, backend=args.backend)
        if not ok:
            raise SystemExit(
                "Matched representation collective not authorized "
                f"(reason={auth_reason}). Complete human review and set "
                "analysis/matched_rep_collective/auth_go.json "
                "paid_authorized=true before OpenAI --yes. "
                "Use --estimate-only without paid auth for costing. "
                "Mock smoke is allowed when mock_smoke_authorized=true."
            )
        print(f"Stage C authorization: {auth_reason} (surrogate_free)")
    else:
        # Authorization gate (never mutates hash-locked bundle files)
        bundle = _resolve_bundle(protocol)
        ok, auth_reason = _stage_c_authorized(bundle, protocol)
        if not ok:
            raise SystemExit(
                "Stage C not authorized for this protocol/bundle "
                f"(reason={auth_reason}). For v0.2a, require checklist sidecar + "
                "freeze_ready; for v0.1, require stage_c_authorized in bundle."
            )
        print(f"Stage C authorization: {auth_reason} ({bundle.as_posix()})")

    if args.estimate_only:
        _estimate(protocol, args)
        return 0

    if args.backend == "openai" and not args.yes:
        _estimate(protocol, args)
        raise SystemExit("Refusing paid run without --yes (re-run with --yes)")

    concurrency = int(
        args.concurrency or protocol["concurrency_within_timestep"]
    )
    max_attempts = int(args.max_attempts or protocol["max_attempts"])
    if args.max_output_tokens is None:
        args.max_output_tokens = int(protocol["max_output_tokens"])
    backend = _make_backend(args)

    if args.resume_session is not None:
        session_dir = args.resume_session
        if not session_dir.is_dir():
            raise SystemExit(f"--resume-session not a directory: {session_dir}")
        saved_protocol = session_dir / "protocol.json"
        if saved_protocol.exists():
            prev = json.loads(saved_protocol.read_text(encoding="utf-8"))
            if prev.get("protocol_version") != protocol.get("protocol_version"):
                raise SystemExit(
                    "resume-session protocol_version mismatch: "
                    f"{prev.get('protocol_version')} vs {protocol.get('protocol_version')}"
                )
        print(f"Resuming session: {session_dir}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        proto_hash = _sha256_file(args.protocol)[:12]
        session_dir = (
            args.out_root
            / f"{protocol['protocol_version']}_{proto_hash}"
            / stamp
        )
        session_dir.mkdir(parents=True, exist_ok=False)
        (session_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
        resolved = {
            "protocol_sha256": _sha256_file(args.protocol),
            "backend": args.backend,
            "model": args.model if args.backend == "openai" else "mock",
            "concurrency": concurrency,
            "experiment_family": protocol.get("experiment_family"),
            "surrogate_free": bool(protocol.get("surrogate_free")),
            "representation": protocol.get("representation"),
            "match_group": protocol.get("match_group"),
        }
        if bundle is not None:
            resolved["bundle_manifest_sha256"] = _sha256_file(bundle / "manifest.json")
        else:
            auth_path = _matched_rep_auth_path()
            resolved["auth_go_sha256"] = (
                _sha256_file(auth_path) if auth_path.exists() else None
            )
        (session_dir / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2),
            encoding="utf-8",
        )

    specs = _run_specs(protocol)
    if args.run_id:
        wanted = set(args.run_id)
        specs = [s for s in specs if s["run_id"] in wanted]
        if not specs:
            raise SystemExit(f"no matching run_id in {wanted}")

    summaries = []
    t0 = time.perf_counter()
    for spec in specs:
        run_dir = session_dir / spec["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_existing = run_dir / "run_meta.json"
        actions_existing = run_dir / "actions.npy"
        if meta_existing.exists() and actions_existing.exists():
            prev_meta = json.loads(meta_existing.read_text(encoding="utf-8"))
            prev_actions = np.load(actions_existing)
            if (
                int(prev_meta.get("n_steps", -1)) == int(protocol["n_steps"])
                and int(prev_actions.shape[0]) >= int(protocol["n_steps"])
            ):
                print(
                    f"Skipping complete {spec['run_id']} "
                    f"(valid_rate={prev_meta.get('valid_rate')})"
                )
                summaries.append(prev_meta)
                continue
        print(f"Starting {spec['run_id']} ...")
        summary = _run_trajectory(
            spec,
            protocol,
            backend,
            run_dir=run_dir,
            concurrency=concurrency,
            max_attempts=max_attempts,
        )
        summaries.append(summary)
        print(
            f"  done valid_rate={summary['valid_rate']:.3f} "
            f"mean_r={summary['mean_r']:.3f}"
        )

    report = {
        "session_dir": str(session_dir),
        "n_runs": len(summaries),
        "total_calls": sum(s["n_calls"] for s in summaries),
        "total_valid": sum(s["n_valid"] for s in summaries),
        "elapsed_seconds": time.perf_counter() - t0,
        "runs": [
            {k: v for k, v in s.items() if k != "r_series"} for s in summaries
        ],
    }
    (session_dir / "session_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in (
        "session_dir", "n_runs", "total_calls", "total_valid", "elapsed_seconds"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
