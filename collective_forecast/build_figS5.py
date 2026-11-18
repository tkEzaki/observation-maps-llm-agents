"""Supplementary Figure S5: why shared and independent reflection differ in the fitted model.

Uses the frozen quadratic model and preserved 200-step data through mechanism_model.py (no API calls).
Run from the repository root:  python -B collective_forecast/build_figS5.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('CF_ROOT', HERE.parent))
FIG = Path(os.environ.get('CF_FIG_OUT', ROOT/'figures/collective_forecast')); FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from mechanism_model import ck, r1, OMEGA, N, P, locked_state, jacobian, g_of, noise_cov, lna, gaussian_r1, g_curve, box_model   # noqa: E402

FS, FS_SMALL, FS_TINY, FS_LABEL = 7.0, 6.3, 5.8, 8.5
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'mathtext.fontset': 'custom', 'mathtext.rm': 'sans', 'mathtext.it': 'sans:italic', 'mathtext.bf': 'sans:bold',
    'mathtext.sf': 'sans', 'mathtext.cal': 'sans', 'font.size': FS, 'axes.labelsize': FS, 'legend.fontsize': FS_SMALL,
    'xtick.labelsize': FS_SMALL, 'ytick.labelsize': FS_SMALL, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'axes.labelpad': 2.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300})
INK, MUTED, GRID = '#222222', '#6E6E6E', '#D0D0D0'
C_NAT, C_IND, C_SHA = '#8C8C8C', '#1F78B4', '#E6550D'


def plabel(fig, ax, letter, title, dx=0.30, dy=0.07):
    W, H = fig.get_size_inches(); p = ax.get_position()
    fig.text(p.x0 - dx/W, p.y1 + dy/H, letter, fontsize=FS_LABEL, fontweight='bold', ha='left', va='bottom', color=INK)
    fig.text(p.x0 - dx/W + 0.15/W, p.y1 + dy/H, title, fontsize=FS, ha='left', va='bottom', color=INK)


# ----------------------------------------------------------------------------- computations (fast parts)
seeds = [202611011501 + i for i in range(8)]
th_late = np.array(ck['Claude__independent_mirror__seed202611011501']['phases'])[200].copy()
star, ok = locked_state(th_late, 'sym'); assert ok
J = jacobian(star, 'sym'); ev = np.linalg.eigvals(P @ J @ P); mod = np.sort(np.abs(ev))[::-1]
rel = np.angle(np.exp(1j*(star - np.angle(np.exp(1j*star).mean()))))
Qn, Qs = noise_cov(star, 'sym'); Sn, Ss = lna(J, Qn), lna(J, Qs)
lna_i, _ = gaussian_r1(star, Sn); lna_s, _ = gaussian_r1(star, Ss)
box = box_model(45)
obs_i = np.mean([r1(np.array(ck[f'Claude__independent_mirror__seed{s}']['phases'])[61:201]).mean() for s in seeds])
obs_s = np.mean([r1(np.array(ck[f'Claude__shared_mirror__seed{s}']['phases'])[61:201]).mean() for s in seeds])
FROZEN_I, FROZEN_S = 0.800, 0.771     # frozen forecasts of the 200-step cohort (steps 61-200), from the prediction freeze
summary = dict(locked_r1=float(r1(star)), lam_max=float(mod[0]), relax_steps=float(-1/np.log(mod[0])),
               wall_deg=float(np.rad2deg(np.abs(rel[OMEGA != 0]).mean())), lna_independent=float(lna_i), lna_shared=float(lna_s),
               box_independent=box['independent'], box_shared=box['shared'], observed_independent=float(obs_i), observed_shared=float(obs_s))
print(json.dumps(summary, indent=1))

# ----------------------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 3, left=0.075, right=0.935, top=0.93, bottom=0.10, wspace=0.5, hspace=0.62)
d = np.linspace(-np.pi, np.pi, 721)

# a  effective coupling function
ax = fig.add_subplot(gs[0, 0])
ax.axhline(0, color=GRID, lw=.5); ax.axvline(0, color=GRID, lw=.5)
ax.axvspan(-45, 45, color='#F1F1F1', lw=0, zorder=0)
ax.plot(np.rad2deg(d), g_curve(d, 12.0, 'native'), color='#C44E52', lw=1.2, label='native rule')
ax.plot(np.rad2deg(d), g_curve(d, 12.0, 'sym'), color=C_IND, lw=1.2, label='symmetrized rule')
ax.plot(np.rad2deg(d), np.sign(np.sin(d)), color=MUTED, lw=.6, ls=':', label=r'sign(sin $\delta$)')
ax.set(xlim=(-180, 180), ylim=(-1.15, 1.15), xticks=[-180, -90, 0, 90, 180], yticks=[-1, 0, 1],
       xlabel=r'Direction of the peer field $\delta$ (deg)', ylabel='Mean action')
ax.legend(loc='lower right', fontsize=FS_TINY, handlelength=1.4, borderaxespad=0.1)
ax.text(0, -0.5, 'no mean\ndrift', ha='center', va='center', fontsize=FS_TINY, color=MUTED, linespacing=1.15)
plabel(fig, ax, 'a', r'Fitted rule for a concentrated field ($\kappa=12$)')

# b  deterministic locked state
ax = fig.add_subplot(gs[0, 1])
ax.axhspan(-45, 45, color='#F1F1F1', lw=0, zorder=0)
ax.axhline(45, color=MUTED, lw=.5, ls=(0, (2, 2))); ax.axhline(-45, color=MUTED, lw=.5, ls=(0, (2, 2)))
ax.scatter(OMEGA, np.rad2deg(rel), s=14, color=C_IND, zorder=3, lw=0)
ax.set(xlabel=r'Intrinsic increment $\omega_i$ (rad per step)', ylabel='Phase relative to the mean (deg)', ylim=(-70, 70),
       yticks=[-45, 0, 45], xticks=[-0.05, 0, 0.05])
ax.text(0.03, 0.96, f'$r_1$ = {r1(star):.3f}\nno-noise map, fixed point', transform=ax.transAxes, va='top', fontsize=FS_TINY, linespacing=1.25)
plabel(fig, ax, 'b', 'Locked state: agents sit at the edges')

# c  Jacobian spectrum
ax = fig.add_subplot(gs[0, 2])
t = np.linspace(0, 2*np.pi, 200); ax.plot(np.cos(t), np.sin(t), color=GRID, lw=.6)
ax.scatter(ev.real, ev.imag, s=12, color=C_IND, zorder=3, lw=0)
ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), xticks=[-1, 0, 1], yticks=[-1, 0, 1], xlabel=r'Re $\lambda$', ylabel=r'Im $\lambda$')
ax.set_aspect('equal')
ax.text(0.03, 0.96, f'$|\\lambda|_{{\\max}}$ = {mod[0]:.3f}\nrelaxation ≈ {-1/np.log(mod[0]):.0f} steps', transform=ax.transAxes,
        va='top', fontsize=FS_TINY, linespacing=1.25)
plabel(fig, ax, 'c', 'Jacobian spectrum (rotation removed)')

# d  four levels versus data
ax = fig.add_subplot(gs[1, 0:2])
cats = ['Locked state\n(no noise)', 'Linear-noise\napproximation', 'Flat-region\nmodel', 'Frozen forecast\n(stochastic)', 'Observed']
ind = [r1(star), lna_i, box['independent'], FROZEN_I, obs_i]; sha = [r1(star), lna_s, box['shared'], FROZEN_S, obs_s]
x = np.arange(5); ax.bar(x - 0.18, ind, 0.34, color=C_IND, label='independent reflection'); ax.bar(x + 0.18, sha, 0.34, color=C_SHA, label='shared reflection')
for i, (a, b) in enumerate(zip(ind, sha)):
    ax.text(i, max(a, b) + 0.008, f'{b - a:+.3f}', ha='center', fontsize=FS_TINY)
ax.set(xticks=x, ylim=(0.6, 0.9), yticks=[0.6, 0.7, 0.8, 0.9], ylabel=r'Mean $r_1$, steps 61–200')
ax.set_xticklabels(cats, fontsize=FS_TINY); ax.legend(loc='upper left', fontsize=FS_TINY, ncol=2, columnspacing=1.0, borderaxespad=0.1)
plabel(fig, ax, 'd', 'Shared minus independent reflection at four levels of description')

# e  native rule: the slowest agent laps the group
ax = fig.add_subplot(gs[1, 2]); seed = 202611011506
th = np.array(ck[f'Claude__native__seed{seed}']['phases']); Z = np.exp(1j*th).mean(1)
relt = np.rad2deg(np.angle(np.exp(1j*(th[:, 0] - np.angle(Z)))))
ax.plot(np.arange(201), r1(th), color=C_NAT, lw=1.1)
ax.set(xlim=(0, 200), ylim=(0, 1), xticks=[0, 50, 100, 150, 200], yticks=[0, .5, 1], xlabel='Step', ylabel=r'$r_1$', )
ax2 = ax.twinx(); ax2.plot(np.arange(201), relt, color='#C44E52', lw=.8, ls=(0, (2, 1.2)))
ax2.set(ylim=(-180, 180), yticks=[-180, 0, 180]); ax2.set_ylabel('Slowest agent relative\nto the mean (deg)', color='#C44E52', fontsize=FS_TINY, labelpad=2)
ax2.tick_params(axis='y', colors='#C44E52', labelsize=FS_TINY); ax2.spines['right'].set_visible(True); ax2.spines['right'].set_color('#C44E52'); ax2.spines['top'].set_visible(False)
plabel(fig, ax, 'e', 'Native rule: the slowest agent laps the group')

fig.savefig(FIG/'figS5_mechanism.pdf'); fig.savefig(FIG/'figS5_mechanism.png', dpi=300)
(HERE/'source_data').mkdir(exist_ok=True); json.dump(summary, open(HERE/'source_data/figS5_summary.json', 'w'), indent=1)
print('written', FIG/'figS5_mechanism.pdf')
