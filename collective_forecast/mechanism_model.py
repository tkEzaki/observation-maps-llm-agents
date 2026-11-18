"""Mean-field description of the frozen quadratic response model (moment description, Claude).

Computed from preserved files only: the frozen coefficients, the 200-step design and the observed phases.
  1. mean-field map F(theta) = theta + omega + K g(theta), with g the mean action given the peers' moments;
     native rule g and symmetrized rule g_sym = [g(x) - g(Rx)]/2.
  2. Jacobian of F, its spectrum at the fixed point of the noise-free map in the co-rotating frame.
  3. linear-noise approximation around that fixed point for independent and shared reflection coins.
  4. flat-region random-walk model with walls at +-alpha.
"""
import json, sys, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('CF_ROOT', HERE.parent))/'independent_studies'


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True); e = np.exp(z); return e/e.sum(axis=axis, keepdims=True)

Q = np.load(ROOT/'closure_transfer_20261031/offline/quadratic6.npz')['coef']        # (28, 3): retard, stay, advance
design = json.load(open(ROOT/'phase_persistence_design_20261101/design.json'))
ck = json.load(open(ROOT/'phase_persistence_validation_20261101/runs/real/checkpoint.json'))['trajectories']
OMEGA = np.array([-0.05 + 0.00625*i for i in range(17)]); K = 0.1; N = 17
CENTERS = (np.arange(24) + .5) * (2*np.pi/24) - np.pi
rng = np.random.default_rng(7)


# ----------------------------------------------------------------------------- response model
def moments_binned(theta):
    q = np.asarray(theta); n = q.shape[-1]
    delta = (q[..., None, :] - q[..., :, None] + np.pi) % (2*np.pi) - np.pi      # delta[i, j] = theta_j - theta_i
    ids = np.floor(np.round((delta + np.pi)*24/(2*np.pi), 12)).astype(int) % 24
    lead = q.shape[:-1]; B = int(np.prod(lead)) if lead else 1
    ids2 = ids.reshape(B, n, n); h = np.zeros((B, n, 24)); b = np.arange(B)
    for i in range(n):
        for j in range(n):
            if i != j:
                np.add.at(h, (b, i, ids2[:, i, j]), 1.0)
    h = h.reshape(lead + (n, 24))/(n-1)
    z = h @ np.exp(1j*CENTERS[:, None]*np.arange(1, 4))
    return np.round(np.stack([z.real, z.imag], axis=-1).reshape(q.shape + (6,)), 6)


def moments_smooth(theta):
    q = np.asarray(theta); n = q.shape[-1]
    delta = q[..., None, :] - q[..., :, None]
    mask = ~np.eye(n, dtype=bool)
    out = []
    for m in (1, 2, 3):
        z = (np.exp(1j*m*delta)*mask).sum(-1)/(n-1)
        out += [z.real, z.imag]
    return np.stack(out, axis=-1)


def feats(x):
    parts = [np.ones(x.shape[:-1] + (1,)), x, np.stack([x[..., i]*x[..., j] for i in range(6) for j in range(i, 6)], axis=-1)]
    return np.concatenate(parts, axis=-1)


def probs(x): return softmax(feats(x) @ Q, axis=-1)
def mirror(x): return x * np.array([1, -1, 1, -1, 1, -1])


def g_of(theta, kind, smooth=True):
    """mean action per agent; kind in {'native', 'sym'}; also returns d = g0 - g1 (branch difference) and variances."""
    x = moments_smooth(theta) if smooth else moments_binned(theta)
    p0 = probs(x); g0 = p0[..., 2] - p0[..., 0]
    p1 = probs(mirror(x)); g1 = -(p1[..., 2] - p1[..., 0])            # decoded action under the mirrored branch
    v0 = p0[..., 2] + p0[..., 0] - g0**2; v1 = p1[..., 2] + p1[..., 0] - g1**2
    if kind == 'native':
        return g0, np.zeros_like(g0), v0
    g = (g0 + g1)/2; var = (v0 + v1)/2 + ((g0 - g1)/2)**2             # mixture variance of a_i under a fair coin
    return g, g0 - g1, var


def F(theta, kind): return theta + OMEGA + K*g_of(theta, kind)[0]


def jacobian(theta, kind, eps=1e-6):
    J = np.zeros((N, N)); base = F(theta, kind)
    for j in range(N):
        t = theta.copy(); t[j] += eps
        J[:, j] = (F(t, kind) - base)/eps
    return J


def r1(theta): return np.abs(np.exp(1j*np.asarray(theta)).mean(-1))


# ----------------------------------------------------------------------------- locked state and spectrum
P = np.eye(N) - np.ones((N, N))/N          # projector orthogonal to uniform rotation


def locked_state(theta0, kind, iters=4000):
    """Iterate the noise-free map in the co-rotating frame; return the state and whether it converged to a fixed point."""
    t = theta0.copy()
    for _ in range(iters):
        t2 = F(t, kind); t2 -= t2.mean() - t.mean()     # remove uniform rotation
        t2 = (t2 + np.pi) % (2*np.pi) - np.pi
        if np.max(np.abs(np.angle(np.exp(1j*(t2 - t))))) < 1e-10:
            return t2, True
        t = t2
    return t, False


def lna(J, Qn, iters=5000):
    """stationary covariance on the subspace orthogonal to uniform rotation."""
    Jp = P @ J @ P; Qp = P @ Qn @ P; S = np.zeros((N, N))
    for _ in range(iters):
        S_new = Jp @ S @ Jp.T + Qp
        if np.max(np.abs(S_new - S)) < 1e-13: return S_new
        S = S_new
    return S


def noise_cov(theta, kind):
    g, d, var = g_of(theta, kind)
    Qn = K**2 * np.diag(var)
    Qs = Qn + K**2 * (np.outer(d, d) - np.diag(d**2))/4
    return Qn, Qs


def gaussian_r1(theta_star, S, n=200000):
    L = np.linalg.cholesky(S + 1e-12*np.eye(N))
    xi = rng.standard_normal((n, N)) @ L.T
    return r1(theta_star + xi).mean(), r1(theta_star + xi).std()


# ----------------------------------------------------------------------------- stochastic rollouts of the frozen model
def rollout(theta0, kind, steps, draws, frames=None, seed=0, smooth=False):
    """exact frozen-model dynamics with binned payloads (same as the forecast code); kind: native / independent / shared."""
    r = np.random.default_rng(seed); th = np.repeat(theta0[None], draws, 0); out = [r1(th)]
    for s in range(steps):
        x = moments_smooth(th) if smooth else moments_binned(th)
        if kind == 'native':
            mir = np.zeros((draws, N), bool)
        elif kind == 'independent':
            mir = r.random((draws, N)) < 0.5
        else:
            mir = np.repeat(r.random((draws, 1)) < 0.5, N, 1)
        xm = np.where(mir[..., None], mirror(x), x); p = probs(xm)
        raw = (r.random((draws, N))[..., None] > p.cumsum(-1)).sum(-1) - 1
        a = np.where(mir, -raw, raw); th = th + OMEGA + K*a; out.append(r1(th))
    return np.array(out)


# ----------------------------------------------------------------------------- effective coupling function
def vm_moments(delta, kappa):
    """moments of a von Mises peer field centred at delta (relative to the focal agent), unbinned."""
    phi = np.linspace(-np.pi, np.pi, 20001); w = np.exp(kappa*np.cos(phi)); w /= np.trapezoid(w, phi); out = []
    for m in (1, 2, 3):
        R = np.trapezoid(w*np.cos(m*phi), phi); out += [R*np.cos(m*delta), R*np.sin(m*delta)]
    return np.stack(out, -1)

def g_curve(delta, kappa, kind, eps=0.0):
    x = vm_moments(delta - eps, kappa); p0 = probs(x); g0 = p0[..., 2] - p0[..., 0]
    if kind == 'native': return g0
    p1 = probs(mirror(x)); g1 = -(p1[..., 2] - p1[..., 0]); return (g0 + g1)/2


# ----------------------------------------------------------------------------- flat-region model
def box_model(alpha_deg):
    """r1 for independent coins (random walk with drift omega_i and steps +-K between walls) and shared coins (agents at the walls)."""
    al = np.deg2rad(alpha_deg); x = np.linspace(-al, al, 4001); z = []
    for w in OMEGA:
        rho = np.exp(2*w*x/K**2); rho /= np.trapezoid(rho, x); z.append(np.trapezoid(rho*np.exp(1j*x), x))
    zp = np.where(OMEGA > 0, np.exp(1j*al), np.where(OMEGA < 0, np.exp(-1j*al), 1))
    return dict(independent=float(abs(np.mean(z))), shared=float(abs(zp.mean())))
