"""Check protocol.py against the recorded experiments. No API calls.

For every collective experiment the script rebuilds each prompt from the stored state and the stored
intervention schedule, compares it with the prompt that was sent, re-parses every returned response,
maps the actions back and replays every state transition.

A moment that is zero to six decimals can print as +0.000000 or -0.000000 depending on the platform's
floating-point summation. Prompts that differ from the stored text only in that sign are counted in the
column "zero-sign"; every other difference stops the script.

Run from the repository root:  python -B collective_forecast/verify_protocol.py
"""
from pathlib import Path
import json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P                                                                        # noqa: E402

ST = P.ROOT/'independent_studies'
PHASE_COHORTS = ['relative_dynamics_20261005', 'reflection_confirmation_20261005', 'mechanism_controls_20261031',
                 'prospective_closure_validation_20261031', 'phase_persistence_validation_20261101']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def zero_sign(text):
    return text.replace('-0.000000', '+0.000000')


def stored_prompts(folder):
    """{task_id: {'prompt', 'sha256', 'mirrored'}} for either storage layout."""
    single = folder/'runs/real/prompts.json'
    if single.exists():
        data = read(single)
        return data.get('prompts', data)
    out = {}
    for part in sorted((folder/'runs/real/prompts').glob('step*.json')):
        out.update(read(part))
    return out


def check_phase(name):
    folder = ST/name
    plan, design = read(folder/'protocol.json'), read(folder/'design.json')
    omega, coupling = np.array(plan['omega']), plan['coupling']
    traj = read(folder/'runs/real/checkpoint.json')['trajectories']
    prompts = stored_prompts(folder)
    results = P.terminal_results(folder/'runs/real/attempts.jsonl')
    sent = {r['task_id']: r['prompt_sha256'] for r in P.read_journal(folder/'runs/real/attempts.jsonl') if r.get('event') == 'dispatch'}
    tasks = {}
    for t in design['tasks']:
        tasks[(t['run_id'], t['block'], t['sample'])] = t
    n_prompt = n_resp = n_step = n_zero = 0
    for run in plan['runs']:
        rid = run['run_id']; s = traj[rid]
        phases = np.array(s['phases']); steps = len(s['raw_values'])
        for step in range(steps):
            h = P.peer_histograms(phases[step]); n = len(h)
            mirrored = np.array([tasks[(rid, step, i)]['mirrored'] for i in range(n)])
            raw = np.zeros(n, int)
            for i in range(n):
                tid = tasks[(rid, step, i)]['task_id']
                text = P.phase_prompt(P.mirror_histogram(h[i]) if mirrored[i] else h[i])
                stored = prompts[tid]['prompt']
                assert prompts[tid]['sha256'] == P.sha256_text(stored) == sent[tid], tid
                if stored != text:                    # a moment that is zero to rounding can print with either sign
                    assert zero_sign(stored) == zero_sign(text), tid
                    n_zero += 1
                assert bool(prompts[tid]['mirrored']) == bool(mirrored[i]), tid
                n_prompt += 1
                r = results[tid]
                assert r['valid'] and P.parse_phase_action(r['raw_response']) == r['parsed']['value'], tid
                raw[i] = r['parsed']['value']; n_resp += 1
            assert np.array_equal(raw, np.array(s['raw_values'][step])), (rid, step)
            actions = P.physical_actions(raw, mirrored)
            if 'permutations' in s and s['permutations'][step] is not None and 'shuffled' in rid:
                actions = actions[np.array(s['permutations'][step])]
            assert np.array_equal(actions, np.array(s['actions'][step])), (rid, step)
            assert np.allclose(P.phase_update(phases[step], omega, coupling, actions), phases[step + 1], rtol=0, atol=1e-12), (rid, step)
            n_step += 1
    return n_prompt, n_resp, n_step, n_zero


def check_catalog(path):
    catalog = read(path); catalog = catalog.get('prompts', catalog)
    n = 0
    for entry in catalog.values():
        text = P.resource_prompt(entry['counts'], entry['own'])
        assert text == entry['prompt'] and P.sha256_text(text) == entry.get('sha256', P.sha256_text(text))
        n += 1
    return n


def check_resource():
    folder = ST/'resource_collective_validation_20261101'
    design = read(folder/'design.json'); catalog = read(folder/'prompts.json')
    states = np.load(folder/'runs/real/states.npy'); frames = np.array(design['frames'])
    assert np.array_equal(states[0], np.array(design['initials']))
    results = P.terminal_results(folder/'runs/real/attempts.jsonl')
    by = {(r['run'], r['step'], r['agent']): r for r in results.values()}
    n_prompt = n_resp = n_step = 0
    for run in range(states.shape[1]):
        for step in range(states.shape[0] - 1):
            x = states[step, run]; new = x.copy()
            for agent in range(len(x)):
                r = by[(run, step, agent)]; perm = frames[run, step, agent]
                assert list(perm) == r['permutation']
                counts, own = P.displayed_state(x, agent, perm)
                text = P.resource_prompt(counts, own)
                assert text == catalog[r['prompt_id']]['prompt'] and P.sha256_text(text) == r['prompt_sha256'] and own == r['local_own']
                n_prompt += 1
                if r['valid']:
                    choice = P.parse_resource_choice(r['raw_response'])
                    assert choice == r['parsed']['value']
                    new[agent] = P.physical_choice(perm, choice); n_resp += 1
            assert np.array_equal(new, states[step + 1, run]), (run, step)
            n_step += 1
    return n_prompt, n_resp, n_step


def check_calibration_parsing():
    folder = ST/'resource_collective_design_20261031'
    results = P.terminal_results(folder/'runs/real/attempts.jsonl')
    n = 0
    for r in results.values():
        if r['valid']:
            assert P.parse_resource_choice(r['raw_response']) == r['parsed']['value'], r['task_id']
            n += 1
    return n


def main():
    print(f"{'experiment':44s} {'prompts':>8s} {'responses':>10s} {'transitions':>12s} {'zero-sign':>10s}")
    for name in PHASE_COHORTS:
        a, b, c, z = check_phase(name)
        print(f'{name:44s} {a:8d} {b:10d} {c:12d} {z:10d}')
    a, b, c = check_resource()
    print(f"{'resource_collective_validation_20261101':44s} {a:8d} {b:10d} {c:12d}")
    for name in ['resource_collective_design_20261031', 'resource_collective_validation_20261101']:
        print(f'{name}: {check_catalog(ST/name/"prompts.json")} catalogue prompts rebuilt from the template')
    print(f'resource_collective_design_20261031: {check_calibration_parsing()} calibration responses re-parsed')
    print('All prompts, responses and transitions are reproduced.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
