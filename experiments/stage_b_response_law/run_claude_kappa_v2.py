"""GPT-matched Claude kappa acquisition. Default: offline plan, no API calls.

Run as python -m experiments.stage_b_response_law.run_claude_kappa.
The journal records each dispatch before it is sent and each result on arrival.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from circlemap.backends import AnthropicResponseBackend, MockResponseBackend
from circlemap.costing import estimate_cost
from circlemap.response import parse_social_action as parse_strict_social_action
import re

def parse_social_action(raw_text):
    text = raw_text.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    return parse_strict_social_action(match.group(1) if match else raw_text)

from experiments.stage_b_response_law.run_representation import build_tasks

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = Path(__file__).with_name('protocol_claude_kappa_v2.json')

def utc():
    return datetime.now(timezone.utc).isoformat()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    replace_file(temp, path)

def replace_file(source, target):
    # Windows sync/indexing services can briefly hold the destination open.
    for attempt_index in range(6):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt_index == 5:
                raise
            time.sleep(.2 * 2**attempt_index)

@contextmanager
def run_lock(folder):
    """OS lock releases on process exit; a surviving lock file is harmless."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.acquisition.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def make_plan(spec, block, backend):
    source = ROOT / spec['gpt_sources'][block]
    source_config = read(source/'resolved_config.json')
    protocol = source_config['protocol']
    assert protocol['seed_block_id'] == block
    assert list(protocol['concentration_profiles'].values()) == [2.,4.,6.,9.,12.]
    assert protocol['repetitions_per_condition'] == 24 and len(protocol['offset_radians']) == 36
    tasks = build_tasks(protocol)
    assert len(tasks) == 12960 and len({t.task_id for t in tasks}) == 12960
    stimuli = {}
    for line in (source/'stimuli.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        key = (row['representation'], row['profile'], row['offset_index'])
        if key in stimuli:
            raise ValueError('duplicate source stimulus')
        stimuli[key] = row
    for task in tasks:
        row = stimuli[task.representation, task.profile, task.offset_index]
        assert row['serialized_prompt'] == task.prompt
        assert row['prompt_sha256'] == task.prompt_sha256
        assert row['offset_radians'] == task.offset_radians
    assert len(stimuli) == 540
    # The original seed/order is retained for traceability, not an API seed claim.
    protocol = dict(protocol)
    protocol.pop('phenotype_rules', None)
    protocol.pop('replication_rules', None)
    protocol['protocol_name'] = spec['experiment_id']
    protocol['protocol_version'] = spec['experiment_id'] + '-' + block
    protocol['bootstrap_samples'] = spec['analysis']['bootstrap_samples']
    frozen = {
        'spec': spec, 'protocol': protocol, 'backend': backend,
        'model': spec['model'] if backend == 'anthropic' else 'attractive-v1',
        'temperature': spec['temperature'], 'max_output_tokens': spec['max_output_tokens'],
        'max_attempts': spec['max_attempts'],
        'source_config_sha256': hashlib.sha256((source/'resolved_config.json').read_bytes()).hexdigest(),
        'source_stimuli_sha256': hashlib.sha256((source/'stimuli.jsonl').read_bytes()).hexdigest(),
        'tasks_sha256': digest([asdict(t) for t in tasks]),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'parser_sha256': hashlib.sha256((ROOT/'circlemap/response.py').read_bytes()).hexdigest(),
    }
    return tasks, frozen

def initialize(folder, frozen, tasks):
    path = folder/'resolved_config.json'
    if path.exists():
        if read(path)['frozen_sha256'] != digest(frozen):
            journal = folder/'attempts.jsonl'
            if journal.exists() and journal.stat().st_size:
                raise ValueError('Resume refused: protocol/model/prompts/code changed; use a new run root.')
            # A plan with zero dispatched requests may be refreshed before acquisition.
            atomic_json(path, {**frozen, 'frozen_sha256':digest(frozen), 'created_utc':utc()})
    else:
        if (folder/'attempts.jsonl').exists():
            raise ValueError('Journal exists without its frozen config')
        atomic_json(path, {**frozen, 'frozen_sha256': digest(frozen), 'created_utc': utc()})
    # Input manifests are reconstructible; response data are never overwritten here.
    atomic_json(folder/'task_manifest.json', [asdict(t) for t in tasks])
    unique = {}
    for t in tasks:
        unique.setdefault(t.prompt_sha256, {
            'representation':t.representation, 'profile':t.profile,
            'concentration':t.concentration, 'offset_index':t.offset_index,
            'offset_radians':t.offset_radians, 'seed_block_id':t.seed_block_id,
            'prompt_sha256':t.prompt_sha256, 'serialized_prompt':t.prompt})
    temp = folder/'stimuli.jsonl.tmp'
    temp.write_text(''.join(json.dumps(s)+'\n' for s in unique.values()), encoding='utf-8')
    replace_file(temp, folder/'stimuli.jsonl')

def load_journal(path, tasks):
    known = {t.task_id:t for t in tasks}
    starts, results, accepted = {}, {}, {}
    dispatch_counts = Counter()
    if path.exists():
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'Journal corruption at line {number}; preserve and repair before resuming.') from exc
            key = (event['task_id'], event['attempt'])
            if event['task_id'] not in known:
                raise ValueError('Unknown task in journal')
            if event['event'] == 'start':
                if key in starts or event['attempt'] != 1 + dispatch_counts[key[0]]:
                    raise ValueError('Duplicate/nonconsecutive dispatch')
                if event['task_id'] in accepted:
                    raise ValueError('Dispatch after accepted response')
                if event['prompt_sha256'] != known[key[0]].prompt_sha256:
                    raise ValueError('Prompt hash mismatch in journal')
                starts[key] = event
                dispatch_counts[key[0]] += 1
            elif event['event'] == 'result':
                if key not in starts or key in results:
                    raise ValueError('Result without unique dispatch')
                if event['valid']:
                    parsed = parse_social_action(event['raw_response'])
                    if parsed.label != event['action_label'] or parsed.value != event['action_value']:
                        raise ValueError('Journal action does not match raw response')
                    if key[0] in accepted:
                        raise ValueError('Duplicate accepted task')
                    accepted[key[0]] = event
                results[key] = event
            else:
                raise ValueError('Unknown journal event')
    return starts, results, accepted

class ClaudeBackend(AnthropicResponseBackend):
    """Same Messages request as the macro study, with explicit retry accounting."""
    def call(self, prompt, *, seed):
        import anthropic
        # No SDK-internal retries: each journal attempt represents one SDK request.
        with anthropic.Anthropic(api_key=os.environ[self.api_key_env].strip(),
                                 timeout=self.timeout_seconds, max_retries=0) as client:
            response = client.messages.create(model=self.model, temperature=self.temperature,
                max_tokens=self.max_output_tokens, messages=[{'role':'user','content':prompt}])
        return {
            'raw_response': ''.join(b.text for b in response.content if b.type == 'text'),
            'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens,
            'response_id': response.id, 'returned_model': response.model,
            'stop_reason': response.stop_reason,
        }

def attempt(task, backend):
    started = time.perf_counter()
    result = {'valid':False, 'action_label':None, 'action_value':None,
              'input_tokens':0, 'output_tokens':0, 'raw_response':None, 'fatal':False}
    try:
        response = backend.call(task.prompt, seed=task.sample_seed)
        if isinstance(response, dict):
            result.update(response)
            if response['returned_model'] != backend.model:
                result['fatal'] = True
                raise ValueError('Provider returned a different model identifier')
        else:
            result.update(raw_response=response.raw_text, input_tokens=response.input_tokens,
                          output_tokens=response.output_tokens)
        try:
            parse_strict_social_action(result['raw_response'])
            result['strict_valid'] = True
        except (ValueError, TypeError):
            result['strict_valid'] = False
        action = parse_social_action(result['raw_response'])
        result['fence_removed'] = not result['strict_valid']
        if result.get('stop_reason') == 'max_tokens':
            raise ValueError('Truncated response')
        result.update(valid=True, action_label=action.label, action_value=action.value)
    except Exception as exc:
        # Do not persist exception bodies, which may contain credentials/request data.
        status = getattr(exc, 'status_code', None)
        result.update(error_type=type(exc).__name__, http_status=status)
        result['fatal'] = result['fatal'] or status in (400,401,403,404,402)
    result['elapsed_seconds'] = time.perf_counter() - started
    return result

def append(handle, event):
    handle.write(json.dumps(event, ensure_ascii=False) + '\n')
    handle.flush()
    os.fsync(handle.fileno())

def export(folder, tasks, frozen):
    starts, results, accepted = load_journal(folder/'attempts.jsonl', tasks)
    # Export exactly one accepted sample per task for existing Fourier analysis.
    temporary = folder/'trace.jsonl.tmp'
    with temporary.open('w', encoding='utf-8') as handle:
        for task in tasks:
            if task.task_id in accepted:
                result = accepted[task.task_id]
                row = {**asdict(task), **result, 'prompt':None,
                       'backend':frozen['backend'], 'model':frozen['model'],
                       'attempts':result['attempt']}
                handle.write(json.dumps(row)+'\n')
    replace_file(temporary, folder/'trace.jsonl')
    first_valid = sum(r['valid'] for (_, a), r in results.items() if a == 1)
    uncertain = sorted(set(starts)-set(results))
    unresolved_tasks = sorted({key[0] for key in uncertain if key[0] not in accepted})
    counts = Counter(t.task_id for t in tasks if t.task_id in accepted)
    complete = len(counts) == len(tasks)
    summary = {
        'backend':frozen['backend'], 'model':frozen['model'], 'updated_utc':utc(),
        'planned_samples':len(tasks), 'accepted_samples':len(accepted),
        'dispatched_attempts':len(starts), 'returned_attempts':len(results),
        'uncertain_dispatches':[list(k) for k in uncertain],
        'unresolved_tasks':unresolved_tasks,
        'invalid_responses':sum(not r['valid'] and r['raw_response'] is not None for r in results.values()),
        'first_attempt_valid_rate':first_valid/len(tasks),
        'accepted_with_fence_removed':sum(bool(r.get('fence_removed')) for r in accepted.values()),
        'first_attempt_strict_valid_rate':sum(bool(r.get('strict_valid')) for (_,a),r in results.items() if a==1)/len(tasks),
        'complete':complete,
        'quality_pass':complete and not unresolved_tasks and first_valid/len(tasks) >= frozen['spec']['invalid_policy']['minimum_first_attempt_valid_rate'],
        'input_tokens_recorded':sum(r['input_tokens'] for r in results.values()),
        'output_tokens_recorded':sum(r['output_tokens'] for r in results.values()),
        'usage_caveat':'Recorded usage excludes uncertain dispatches and errors without usage.',
    }
    atomic_json(folder/'run_summary.json', summary)
    return summary

def acquire(folder, tasks, frozen, backend, concurrency=4, limit=None, retry_uncertain=False):
    starts, results, accepted = load_journal(folder/'attempts.jsonl', tasks)
    unresolved = {k for k in set(starts)-set(results) if k[0] not in accepted}
    if unresolved and not retry_uncertain:
        raise ValueError('Uncertain dispatches found. Inspect journal; --retry-uncertain allows a new paid attempt and consumes the same 3-attempt budget.')
    maximum = frozen['max_attempts']
    used = Counter(k[0] for k in starts)
    candidates = [t for t in tasks if t.task_id not in accepted]
    if limit is not None:
        candidates = candidates[:limit]
    with (folder/'attempts.jsonl').open('a', encoding='utf-8') as journal:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            # One serial preflight on every invocation; subsequent bounded batches.
            first = True
            while candidates:
                exhausted = [t for t in candidates if used[t.task_id] >= maximum]
                if exhausted:
                    break
                batch = candidates[:1 if first else concurrency]
                first = False
                futures = {}
                for task in batch:
                    used[task.task_id] += 1
                    n = used[task.task_id]
                    append(journal, {'event':'start', 'task_id':task.task_id, 'attempt':n,
                        'prompt_sha256':task.prompt_sha256, 'utc':utc(),
                        'retry_after_uncertainty':any(k[0]==task.task_id for k in unresolved)})
                    futures[pool.submit(attempt, task, backend)] = (task,n)
                failed, fatal = [], False
                for future in as_completed(futures):
                    task,n = futures[future]
                    result = future.result()
                    append(journal, {'event':'result', 'task_id':task.task_id, 'attempt':n,
                                     'utc':utc(), **result})
                    if not result['valid']:
                        failed.append(task)
                    fatal |= result['fatal']
                candidates = failed + candidates[len(batch):]
                if fatal:
                    break
                if failed:
                    time.sleep(min(2**max(used[t.task_id] for t in failed), 30))
                if sum(used.values()) % 100 < concurrency:
                    print(f'Dispatched attempts: {sum(used.values())}', flush=True)
    return export(folder, tasks, frozen)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['plan','mock','run','status','analyze'], default='plan')
    parser.add_argument('--spec', type=Path, default=DEFAULT_SPEC)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--block', choices=['block_1','block_2','both'], default='both')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Maximum pending samples this invocation, per block; resumes the same plan.')
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--retry-uncertain', action='store_true')
    parser.add_argument('--input-price-per-million', type=float)
    parser.add_argument('--output-price-per-million', type=float)
    args = parser.parse_args(argv)
    if args.concurrency < 1 or (args.limit is not None and args.limit < 1):
        parser.error('concurrency/limit must be positive')
    if (args.input_price_per_million is None) != (args.output_price_per_million is None):
        parser.error('provide both token prices, or neither')
    if args.mode == 'run' and not args.yes:
        parser.error('Live API acquisition requires --mode run --yes; default plan is offline.')
    if args.retry_uncertain and args.mode not in ('run','mock'):
        parser.error('--retry-uncertain applies only to acquisition')
    spec = read(args.spec)
    backend_name = 'mock' if args.mode == 'mock' else 'anthropic'
    root = (args.run_root or ROOT/'runs/claude_kappa_v2'/backend_name).resolve()
    # Status/analysis infer the existing backend, never silently mix mock and real.
    if args.mode in ('status','analyze'):
        existing = next((root/b/'resolved_config.json' for b in ['block_1','block_2']
                         if (root/b/'resolved_config.json').exists()), None)
        if existing is not None:
            backend_name = read(existing)['backend']
    blocks = ['block_1','block_2'] if args.block == 'both' else [args.block]
    with run_lock(root):
        plans = {b:make_plan(spec,b,backend_name) for b in blocks}
        if args.mode == 'analyze':
            if blocks != ['block_1','block_2']:
                parser.error('analysis requires both blocks')
            for b,(tasks,frozen) in plans.items():
                if not (root/b/'resolved_config.json').exists():
                    raise ValueError('Missing acquisition config')
                if read(root/b/'resolved_config.json')['frozen_sha256'] != digest(frozen):
                    raise ValueError('Acquisition configuration changed')
                summary = export(root/b,tasks,frozen)
                if not summary['quality_pass']:
                    raise ValueError(f'{b}: incomplete or failed quality gate; inspect run_summary.json')
            from analysis.analyze_complex_kernel import analyze_transmutation
            output = root/'analysis'
            output.mkdir(exist_ok=True)
            analyze_transmutation([root/b for b in blocks], output,
                                  bootstrap_samples=spec['analysis']['bootstrap_samples'])
            atomic_json(output/'ACQUISITION_PROVENANCE.json', {
                'backend':backend_name, 'synthetic':backend_name=='mock', 'spec':spec,
                'note':'Descriptive cross-model follow-up; GPT phenotype gates are not Claude hypotheses.'})
            print(f'Analysis written to {output}')
            return 0
        for block,(tasks,frozen) in plans.items():
            folder = root/block
            folder.mkdir(exist_ok=True)
            initialize(folder,frozen,tasks)
            summary = export(folder,tasks,frozen)
            if args.mode in ('plan','status'):
                if args.input_price_per_million is not None:
                    _,_,accepted = load_journal(folder/'attempts.jsonl',tasks)
                    pending = [t.prompt for t in tasks if t.task_id not in accepted]
                    if pending:
                        estimate = estimate_cost(pending, max_output_tokens_per_call=spec['max_output_tokens'],
                            max_attempts_per_call=spec['max_attempts'],
                            input_price_per_million=args.input_price_per_million,
                            output_price_per_million=args.output_price_per_million).as_dict()
                        atomic_json(folder/'cost_estimate.json',estimate)
                        print(json.dumps(estimate))
                print(block, json.dumps(summary))
                continue
            backend = MockResponseBackend() if backend_name == 'mock' else ClaudeBackend(
                model=spec['model'],temperature=spec['temperature'],
                max_output_tokens=spec['max_output_tokens'],timeout_seconds=spec['timeout_seconds'])
            summary = acquire(folder,tasks,frozen,backend,args.concurrency,args.limit,args.retry_uncertain)
            print(block,json.dumps(summary))
            if not summary['complete']:
                return 3
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
