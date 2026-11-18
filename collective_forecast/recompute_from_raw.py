"""Recompute the main reported numbers from stored trajectories, raw responses and frozen forecasts. No API calls.

Every quantity is derived here from files that are byte-identical to the acquisition-time records:
stored states (verified against the raw responses by verify_protocol.py), raw response journals, designs and
forecast files frozen before the corresponding acquisition. The only simulated input is the stored set of
trajectories of the state-independent comparison model. Nothing else is read from derived analysis tables.
Each value is compared with the value reported in the text, to the number of digits reported.

Run from the repository root:  python -B collective_forecast/recompute_from_raw.py
"""
from pathlib import Path
import csv, json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P                                                                        # noqa: E402

ST = P.ROOT/'independent_studies'
ROWS = []


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def table(path):
    with open(path, encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def report(label, value, reported, tol):
    ROWS.append((label, float(value), reported, abs(float(value) - reported) <= tol))


def late_means(folder, first, last):
    """{run_id: mean r1 over post-update steps first..last}, plus the run table."""
    plan = read(folder/'protocol.json')
    traj = read(folder/'runs/real/checkpoint.json')['trajectories']
    r1 = {rid: P.order_parameter(np.array(s['phases'])) for rid, s in traj.items()}
    return {rid: float(v[first:last + 1].mean()) for rid, v in r1.items()}, plan['runs'], r1


def by_seed(means, runs, controller):
    return {r['physical_seed']: means[r['run_id']] for r in runs if r['controller'] == controller}


def paired(means, runs, a, b):
    x, y = by_seed(means, runs, a), by_seed(means, runs, b)
    return np.array([x[s] - y[s] for s in sorted(x)])


# ----------------------------------------------------------------------------- reflection interventions
def reflection_cohorts():
    m, runs, _ = late_means(ST/'reflection_confirmation_20261005', 41, 60)
    other = sorted({r['controller'] for r in runs} - {'native'})[0]
    d = paired(m, runs, other, 'native')
    report('first cohort: independent reflection minus native, steps 41-60', d.mean(), 0.158, 5e-4)
    report('first cohort: initial states with a positive difference', (d > 0).sum(), 8, 0)

    m, runs, _ = late_means(ST/'mechanism_controls_20261031', 41, 60)
    report('control: mean alignment, independent reflection', np.mean(list(by_seed(m, runs, 'independent_mirror').values())), 0.753, 5e-4)
    report('control: mean alignment after shuffling actions', np.mean(list(by_seed(m, runs, 'shuffled_mirror').values())), 0.168, 5e-4)


# ----------------------------------------------------------------------------- prospective forecasts
def forecasts_60():
    m, runs, _ = late_means(ST/'prospective_closure_validation_20261031', 41, 60)
    pred = {r['run_id']: float(r['prediction']) for r in table(ST/'closure_transfer_20261031/prospective_phase/predictions.csv') if r['model'] == 'quadratic6'}
    err = np.array([m[r['run_id']] - pred[r['run_id']] for r in runs])
    report('60-step cohort: forecast RMSE of mean alignment, steps 41-60', np.sqrt(np.mean(err**2)), 0.0254, 5e-5)
    worst = 0.0
    for a, b in [('shared_mirror', 'independent_mirror'), ('independent_mirror', 'native'), ('shared_mirror', 'native')]:
        worst = max(worst, abs(paired(m, runs, a, b).mean() - paired(pred, runs, a, b).mean()))
    ROWS.append(('60-step cohort: largest error of a mean interface contrast (reported: below 0.01)', worst, 0.01, worst < 0.01))


def forecasts_200():
    folder = ST/'phase_persistence_validation_20261101'
    m, runs, r1 = late_means(folder, 61, 200)
    report('200-step cohort: mean alignment, independent reflection, steps 61-200', np.mean(list(by_seed(m, runs, 'independent_mirror').values())), 0.803, 5e-4)
    report('200-step cohort: mean alignment, native interface, steps 61-200', np.mean(list(by_seed(m, runs, 'native').values())), 0.603, 5e-4)
    d = paired(m, runs, 'independent_mirror', 'native')
    report('200-step cohort: independent reflection minus native', d.mean(), 0.200, 5e-4)
    report('200-step cohort: initial states with a positive difference', (d > 0).sum(), 8, 0)
    d = paired(m, runs, 'shared_mirror', 'independent_mirror')
    report('200-step cohort: shared minus independent reflection', d.mean(), -0.0310, 5e-5)
    report('200-step cohort: initial states with a negative difference', (d < 0).sum(), 8, 0)
    rows = table(ST/'phase_persistence_design_20261101/predictions.csv')
    for model, reported, tol in [('quadratic6', 0.00705, 5e-6), ('four_parameter', 0.0146, 5e-5)]:
        pred = {r['run_id']: float(r['prediction']) for r in rows if r['model'] == model and r['window'] == 'primary_61_200'}
        err = np.array([m[r['run_id']] - pred[r['run_id']] for r in runs])
        report(f'200-step cohort: forecast RMSE of mean alignment, {model}', np.sqrt(np.mean(err**2)), reported, tol)
        if model == 'quadratic6':
            report('200-step cohort: forecast of shared minus independent reflection', paired(pred, runs, 'shared_mirror', 'independent_mirror').mean(), -0.0287, 5e-5)
    curve = {}
    for r in table(ST/'phase_persistence_design_20261101/predicted_trajectories.csv'):
        if r['model'] == 'quadratic6':
            curve[(r['run_id'], int(r['step']))] = float(r['r1_mean'])
    err = np.array([r1[r['run_id']][t] - curve[(r['run_id'], t)] for r in runs for t in range(1, 201)])
    report('200-step cohort: trajectory RMSE over all post-update steps', np.sqrt(np.mean(err**2)), 0.0408, 5e-5)
    base = np.load(ST/'phase_persistence_diagnostics_20261101/baseline_trajectories.npz')['r1'].mean(axis=2)     # step x run
    err = np.array([r1[r['run_id']][t] - base[t, k] for k, r in enumerate(runs) for t in range(1, 201)])
    report('200-step cohort: trajectory RMSE of a state-independent response model', np.sqrt(np.mean(err**2)), 0.5245, 5e-5)


# ----------------------------------------------------------------------------- fixed-state one-step test
def fixed_state():
    folder = ST/'same_state_validation_20261031'
    plan, design = read(folder/'protocol.json'), read(folder/'design.json')
    results = P.terminal_results(folder/'runs/real/attempts.jsonl')
    coupling = 0.1
    phi = np.array([s['theta'] for s in plan['states']]) + np.array(plan['omega'])

    def arrays(stage, rounds):
        a = np.zeros((rounds, 8, 2, 17), int); flags = np.zeros(a.shape, bool); n = 0
        for t in design['tasks']:
            if t['stage'] != stage: continue
            ix = (t['round'], t['state'], t['arm'], t['agent'])
            a[ix] = results[t['task_id']]['parsed']['value']; flags[ix] = t['mirrored']; n += 1
        return a, flags, n

    def expected(phase, chi):                      # exact E[r1^2] after one update for independent action draws
        n = phase.shape[-1]
        return np.abs(np.mean(np.exp(1j*phase)*chi, axis=-1))**2 + np.sum(1 - np.abs(chi)**2, axis=-1)/n**2

    train, _, n_train = arrays('train', 24)        # branch 0: original description, branch 1: mirrored description
    physical = train.copy(); physical[:, :, 1, :] *= -1
    chi = np.exp(1j*coupling*physical).mean(axis=0)                                           # state x branch x agent
    raw, flags, n_val = arrays('validate', 32)     # arm 0: independent coins, arm 1: shared coin
    chosen = np.where(flags, chi[None, :, 1, None, :], chi[None, :, 0, None, :])
    pred = expected(phi[None, :, None, :], chosen)
    obs = np.abs(np.exp(1j*(phi[None, :, None, :] + coupling*np.where(flags, -raw, raw))).mean(axis=-1))**2
    gap_pred = (pred[:, :, 1] - pred[:, :, 0]).mean(axis=0); gap_obs = (obs[:, :, 1] - obs[:, :, 0]).mean(axis=0)
    report('fixed states: responses used for estimation', n_train, 6528, 0)
    report('fixed states: responses used for validation', n_val, 8704, 0)
    report('fixed states: predicted shared minus independent difference in r1 squared', gap_pred.mean(), 0.00303, 5e-6)
    report('fixed states: observed shared minus independent difference in r1 squared', gap_obs.mean(), 0.00314, 5e-6)
    report('fixed states: RMSE across states', np.sqrt(np.mean((gap_obs - gap_pred)**2)), 0.000466, 5e-7)


# ----------------------------------------------------------------------------- resource task
def resource():
    folder = ST/'resource_collective_design_20261031'
    catalog = read(folder/'prompts.json'); catalog = catalog.get('prompts', catalog)
    results = P.terminal_results(folder/'runs/real/attempts.jsonl')
    valid = [r for r in results.values() if r['valid']]
    best = sum(catalog[r['prompt_id']]['counts'][r['parsed']['value']] == min(catalog[r['prompt_id']]['counts']) for r in valid)
    report('resource calibration: valid choices', len(valid), 14670, 0)
    report('resource calibration: fraction choosing a minimum-load resource', best/len(valid), 0.94, 5e-3)

    folder = ST/'resource_collective_validation_20261101'
    design = read(folder/'design.json')
    states = np.load(folder/'runs/real/states.npy')
    congestion = P.normalized_congestion(states)                                              # step x run
    means = congestion[1:].mean(axis=0)
    policy = np.array([r['policy'] for r in design['runs']]); initial = np.array([r['initial_id'] for r in design['runs']])

    def contrast(v):
        v = np.asarray(v)
        return np.array([v[(initial == k) & (policy == 'shared')][0] - v[(initial == k) & (policy == 'independent')][0] for k in sorted(set(initial))])

    d = contrast(means)
    report('resource collectives: shared minus independent normalized congestion', d.mean(), 0.699, 5e-4)
    report('resource collectives: configurations with greater congestion under shared permutations', (d > 0).sum(), 8, 0)
    full = P.normalized_congestion(np.zeros(states.shape[-1], int))
    report('resource collectives: fraction of shared-permutation steps with all agents on one resource',
           np.mean(np.isclose(congestion[1:][:, policy == 'shared'], full)), 1.0, 0)
    forecast = read(folder/'predictions/point_predictions.json')
    report('resource collectives: forecast RMSE of run means, empirical table', np.sqrt(np.mean((means - np.array(forecast['empirical_secondary']['mean']))**2)), 0.00114, 5e-6)
    report('resource collectives: forecast of the difference, empirical table', contrast(forecast['empirical_secondary']['mean']).mean(), 0.697, 5e-4)
    report('resource collectives: forecast of the difference, smoothed table', contrast(forecast['primary']['mean']).mean(), 0.480, 5e-4)


# ----------------------------------------------------------------------------- controlled inputs
def line_order():
    folder = ST/'interface_mechanism_20261031/assay'
    tasks = read(folder/'tasks.json'); prompts = read(folder/'prompts.json'); prompts = prompts.get('prompts', prompts)
    results = P.terminal_results(folder/'runs/real/attempts.jsonl')
    providers, orders = ['GPT', 'Claude'], ['cos_first', 'sin_first']
    a = np.zeros((16, 2, 2, 64, 2))
    agree = {p: [0, 0] for p in providers}
    for t in tasks:
        v = results[t['task_id']]['parsed']['value']
        a[t['round'], providers.index(t['provider']), orders.index(t['order']), t['field'], t['branch']] = v
        if t['order'] == 'cos_first':
            c1, s1 = prompts[t['prompt_id']]['moments'][:2]
            rule = np.sign(s1) if t['provider'] == 'GPT' else np.sign(c1 + s1)
            agree[t['provider']][0] += int(v == rule); agree[t['provider']][1] += 1
    first, second = a[:8].mean(axis=0), a[8:].mean(axis=0)                                    # two halves of the acquisition rounds
    cross = np.mean((first[:, 1] - first[:, 0])*(second[:, 1] - second[:, 0]), axis=(1, 2))   # cross-half estimate of the squared change
    report('line order: RMS change of mean signed action, GPT', np.sqrt(cross[0]), 0.374, 5e-4)
    report('line order: RMS change of mean signed action, Claude', np.sqrt(cross[1]), 0.474, 5e-4)
    report('recorded-state inputs: Claude responses matching the sign of C1+S1', agree['Claude'][0]/agree['Claude'][1], 0.981, 5e-4)
    report('recorded-state inputs: GPT responses matching the sign of S1', agree['GPT'][0]/agree['GPT'][1], 0.937, 5e-4)


def main():
    for part in (reflection_cohorts, forecasts_60, forecasts_200, fixed_state, resource, line_order):
        part()
    width = max(len(r[0]) for r in ROWS)
    print(f"{'quantity':{width}s}  {'recomputed':>12s}  {'reported':>10s}")
    for label, value, reported, ok in ROWS:
        print(f"{label:{width}s}  {value:12.6g}  {reported:10.6g}  {'ok' if ok else 'MISMATCH'}")
    bad = sum(not r[3] for r in ROWS)
    print(f'{len(ROWS) - bad} of {len(ROWS)} quantities agree with the reported values.')
    (HERE/'source_data').mkdir(exist_ok=True)
    with open(HERE/'source_data/recomputed_from_raw.csv', 'w', encoding='utf-8', newline='') as fh:
        out = csv.writer(fh, lineterminator='\n'); out.writerow(['quantity', 'recomputed', 'reported', 'agrees'])
        for label, value, reported, ok in ROWS: out.writerow([label, f'{value:.10g}', f'{reported:.10g}', int(ok)])
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
