"""Response models and closed-loop forecasts, with checks against the frozen files. No API calls.

Response model. The input x is the six displayed moment values (cosine and sine parts of the first three
circular moments, rounded to six decimals). The primary model is a multinomial logistic model of the raw
action (retard, stay, advance) with features 1, x_i and x_i x_j (28 columns), fitted by penalized maximum
likelihood (L2 penalty 1e-4) to the responses of the first reflection cohort. The secondary model has four
parameters: logits (-eta, lambda, eta) with eta = b0 + bc*C1 + bs*S1.

Forecast. Each language-model call is replaced by a draw from the response model. The population is updated
with the same rule and the same stored mirror schedule as the experiment.

The script
  1. refits the primary model and compares it with the frozen coefficients,
  2. regenerates the frozen forecasts of the 60-step and 200-step cohorts from the frozen coefficients, the
     stored designs and the recorded random seeds, and compares them with the frozen forecast files.

Run from the repository root:  python -B collective_forecast/response_model.py
"""
from pathlib import Path
import csv, json, sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P                                                                        # noqa: E402

ST = P.ROOT/'independent_studies'
CENTERS = (np.arange(P.BINS) + .5)*(2*np.pi/P.BINS) - np.pi
PENALTY = 1e-4


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ----------------------------------------------------------------------------- inputs and models
def displayed_moments(h):
    """Six displayed values per histogram: C1, S1, C2, S2, C3, S3, rounded as in the prompt."""
    z = np.asarray(h) @ np.exp(1j*CENTERS[:, None]*np.arange(1, 4))
    return np.round(np.stack([z.real, z.imag], axis=-1).reshape(-1, 6), 6)


def features(x):
    x = np.asarray(x)
    return np.column_stack([np.ones((len(x), 1)), x, np.column_stack([x[:, i]*x[:, j] for i in range(6) for j in range(i, 6)])])


def quadratic_probabilities(x, coef):
    return softmax(features(x) @ coef, axis=1)


def four_parameter_probabilities(x, w):
    eta = w[0] + x[:, 0]*w[1] + x[:, 1]*w[2]
    return softmax(np.column_stack([-eta, np.full(len(x), w[3]), eta]), axis=1)


def frozen_coefficients():
    quad = np.load(ST/'closure_transfer_20261031/offline/quadratic6.npz')['coef']
    v = read(ST/'closure_transfer_20261031/phase_lag/coefficients.json')
    return quad, np.array([v[k] for k in ['intercept', 'cosine', 'sine', 'stay_logit']])


def training_set():
    """Displayed inputs and raw actions (0, 1, 2) of the first reflection cohort."""
    folder = ST/'reflection_confirmation_20261005'
    plan = read(folder/'protocol.json'); traj = read(folder/'runs/real/checkpoint.json')['trajectories']
    xs, ys = [], []
    for run in plan['runs']:
        s = traj[run['run_id']]
        for step in range(len(s['raw_values'])):
            h = P.peer_histograms(np.array(s['phases'][step]))
            mirrored = np.array(s['mirror_flags'][step], bool)
            h[mirrored] = h[mirrored, ::-1]
            xs.append(displayed_moments(h)); ys.extend(np.array(s['raw_values'][step], int) + 1)
    return np.concatenate(xs), np.array(ys)


def fit_quadratic(x, y):
    f = features(x); one = np.eye(3)[y]

    def loss(w):
        w = w.reshape(f.shape[1], 3); z = f @ w; p = softmax(z, axis=1)
        return (np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]) + PENALTY/2*np.sum(w*w),
                ((f.T @ (p - one))/len(y) + PENALTY*w).ravel())

    fit = minimize(loss, np.zeros(f.shape[1]*3), jac=True, method='L-BFGS-B', options=dict(maxiter=1000, ftol=1e-10, gtol=1e-7))
    assert fit.success, fit.message
    return fit.x.reshape(f.shape[1], 3)


# ----------------------------------------------------------------------------- closed-loop forecast
def histograms_many(theta):
    """peer_histograms for a batch of populations: (batch, n) phases -> (batch*n, bins)."""
    q = np.asarray(theta); batch, n = q.shape
    delta = (q[:, None, :] - q[:, :, None] + np.pi) % (2*np.pi) - np.pi
    ids = np.floor(np.round((delta + np.pi)*P.BINS/(2*np.pi), 12)).astype(int) % P.BINS
    row = np.broadcast_to(np.arange(batch*n).reshape(batch, n, 1), ids.shape)
    mask = np.broadcast_to(~np.eye(n, dtype=bool), ids.shape)
    return np.bincount((P.BINS*row + ids)[mask], minlength=batch*n*P.BINS).reshape(batch*n, P.BINS)/(n - 1)


def rollout(kind, initial, frames, omega, coupling, draws, seed):
    """Simulated r1 of shape (steps + 1, runs, draws). frames[run, step, agent] is the stored mirror schedule."""
    quad, four = frozen_coefficients()
    frames = np.asarray(frames, bool); runs = len(initial); n = frames.shape[2]
    theta = np.repeat(np.array(initial, float), draws, axis=0)
    rng = np.random.default_rng(seed)
    out = [np.abs(np.exp(1j*theta).mean(axis=1)).reshape(runs, draws)]
    for step in range(frames.shape[1]):
        h = histograms_many(theta)
        mirrored = np.repeat(frames[:, step], draws, axis=0)
        h[mirrored.ravel()] = h[mirrored.ravel(), ::-1]
        x = displayed_moments(h)
        p = quadratic_probabilities(x, quad) if kind == 'quadratic6' else four_parameter_probabilities(x, four)
        raw = (rng.random(len(h))[:, None] > p.cumsum(axis=1)).sum(axis=1) - 1
        theta += np.array(omega) + coupling*np.where(mirrored, -raw.reshape(-1, n), raw.reshape(-1, n))
        out.append(np.abs(np.exp(1j*theta).mean(axis=1)).reshape(runs, draws))
    return np.array(out)


# ----------------------------------------------------------------------------- checks against frozen files
def table(path):
    with open(path, encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def check_refit():
    x, y = training_set()
    frozen, _ = frozen_coefficients()
    coef = fit_quadratic(x, y)
    return len(y), float(np.max(np.abs(coef - frozen))), float(np.max(np.abs(quadratic_probabilities(x, coef) - quadratic_probabilities(x, frozen))))


def check_forecast_60():
    folder = ST/'closure_transfer_20261031/prospective_phase'
    plan = read(folder/'protocol.json'); schedule = read(folder/'transformations.json')
    runs = plan['runs']
    initial = [plan['initial_phases'][str(r['physical_seed'])] for r in runs]
    frames = np.array([[schedule[f"{r['run_id']}__t{step:03d}"] for step in range(plan['steps'])] for r in runs])
    frozen = table(folder/'predictions.csv'); worst = {}
    for kind in ['quadratic6', 'four_parameter']:
        y = rollout(kind, initial, frames, plan['omega'], plan['coupling'], 64, 20261031899)
        target = {r['run_id']: float(r['prediction']) for r in frozen if r['model'] == kind}
        worst[kind] = float(np.max(np.abs(y[41:61].mean(axis=(0, 2)) - np.array([target[r['run_id']] for r in runs]))))
    return worst


def check_forecast_200():
    folder = ST/'phase_persistence_design_20261101'
    plan, design = read(folder/'protocol.json'), read(folder/'design.json')
    runs = design['runs']
    initial = [design['initial_phases'][str(r['physical_seed'])] for r in runs]
    frames = np.array(design['frames'], bool)
    frozen = table(folder/'predictions.csv'); curves = table(folder/'predicted_trajectories.csv'); worst = {}
    for kind in ['quadratic6', 'four_parameter']:
        chunks = plan['process_draws_per_run']//plan['chunk_size']
        y = np.concatenate([rollout(kind, initial, frames, plan['omega'], plan['coupling'], plan['chunk_size'], plan['simulation_seed'] + c)
                            for c in range(chunks)], axis=2)
        err = 0.0
        for name, (a, b) in plan['windows'].items():
            target = {r['run_id']: float(r['prediction']) for r in frozen if r['model'] == kind and r['window'] == name}
            err = max(err, float(np.max(np.abs(y[a:b + 1].mean(axis=(0, 2)) - np.array([target[r['run_id']] for r in runs])))))
        mean = y.mean(axis=2); index = {r['run_id']: k for k, r in enumerate(runs)}
        err_curve = max(abs(mean[int(r['step']), index[r['run_id']]] - float(r['r1_mean'])) for r in curves if r['model'] == kind)
        worst[kind] = (err, float(err_curve))
    return worst


def main():
    n, d_coef, d_prob = check_refit()
    print(f'refit of the primary model on {n} responses: largest coefficient difference {d_coef:.2e}, largest probability difference {d_prob:.2e}')
    w60 = check_forecast_60()
    for kind, v in w60.items():
        print(f'60-step forecast, {kind}: largest difference from the frozen forecast {v:.2e}')
    w200 = check_forecast_200()
    for kind, (a, b) in w200.items():
        print(f'200-step forecast, {kind}: largest difference from the frozen window means {a:.2e}, from the frozen mean trajectories {b:.2e}')
    ok = d_prob < 1e-4 and all(v < 1e-9 for v in w60.values()) and all(a < 1e-9 and b < 1e-9 for a, b in w200.values())
    print('Frozen coefficients and frozen forecasts are reproduced.' if ok else 'DIFFERENCES EXCEED THE TOLERANCE.')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
