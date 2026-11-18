"""Figures 2-5 and Supplementary Figures S1-S4 of the collective-forecast study.

Every plotted value is read from preserved observations, preserved model outputs or
frozen forecasts. There are no API calls and no refitting of response models. The only
computations are descriptive summaries (means, ranges, root-mean-square errors, the
order parameter r1 from stored phases). Lines in Fig. 2b,c join the controlled-input
means; no smoothing curve is fitted. Figure 1 is built by build_fig1.py and
Supplementary Figure S5 by build_figS5.py.

Run from the repository root:
  python -B collective_forecast/build_figures.py
Outputs: figures/collective_forecast/*.pdf, collective_forecast/source_data/*.csv and
collective_forecast/FIGURE_INPUTS.json (SHA-256 of every input file).
"""
from pathlib import Path
import os, sys, json, hashlib
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('CF_ROOT', HERE.parent))
FIG = Path(os.environ.get('CF_FIG_OUT', ROOT/'figures/collective_forecast')); FIG.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get('CF_DATA_OUT', HERE/'source_data')); DATA.mkdir(parents=True, exist_ok=True)
ST = ROOT/'independent_studies'; AUD = ST/'source_audit_20261102'
INPUTS = {}


def track(p):
    p = Path(p)
    INPUTS[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p
def csv(p): return pd.read_csv(track(p))
def js(p): return json.loads(track(p).read_text(encoding='utf-8-sig'))
def export(d, n): d.to_csv(DATA/n, index=False)

# ----------------------------------------------------------------------------- style
FS, FS_SMALL, FS_TINY, FS_LABEL = 7.0, 6.3, 5.8, 8.5
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'mathtext.fontset': 'custom', 'mathtext.rm': 'sans', 'mathtext.it': 'sans:italic',
    'mathtext.bf': 'sans:bold', 'mathtext.sf': 'sans', 'mathtext.cal': 'sans',
    'font.size': FS, 'axes.titlesize': FS, 'axes.labelsize': FS, 'legend.fontsize': FS_SMALL,
    'xtick.labelsize': FS_SMALL, 'ytick.labelsize': FS_SMALL,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'axes.labelpad': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlepad': 4, 'axes.titlelocation': 'left',
    'legend.frameon': False, 'legend.handlelength': 1.4, 'legend.borderaxespad': 0.3,
    'lines.linewidth': 1.0, 'lines.markersize': 3,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
})
INK, MUTED, GRID = '#222222', '#6E6E6E', '#D0D0D0'
REPS = ['moments_m1_m3', 'centers_24_standard', 'intervals_24_decimal6']
REP_C = dict(zip(REPS, ['#0072B2', '#E69F00', '#009E73']))       # identical to Fig. 1
REP_N = dict(zip(REPS, ['Moments', 'Centres', 'Intervals']))
REP_A = dict(zip(REPS, ['M', 'C', 'I']))
MODEL_C = {'GPT': '#3B3B3B', 'Claude': '#C44E52'}
POL = ['native', 'independent_mirror', 'shared_mirror', 'shuffled_mirror']
POL_C = dict(zip(POL, ['#8C8C8C', '#1F78B4', '#E6550D', '#7B68B0']))
POL_N = dict(zip(POL, ['Native', 'Independent reflection', 'Shared reflection', 'Shuffled actions']))
POL_A = dict(zip(POL, ['Native', 'Indep.', 'Shared', 'Shuffled']))
ACT = ['retard', 'stay', 'advance']
ACT_C = dict(zip(ACT, ['#AA3377', '#DADADA', '#004488']))
ACT_CMAP = LinearSegmentedColormap.from_list('act', [ACT_C['retard'], '#EFEFEF', ACT_C['advance']])
TICKS_PI = ([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])


def plabel(fig, ax, letter, title=None, dx=0.30, dy=0.07):
    """Bold lower-case panel letter at a fixed offset (inches) from the axes' top-left corner."""
    W, H = fig.get_size_inches(); p = ax.get_position()
    fig.text(p.x0 - dx/W, p.y1 + dy/H, letter, fontsize=FS_LABEL, fontweight='bold', ha='left', va='bottom', color=INK)
    if title:
        fig.text(p.x0 - dx/W + 0.15/W, p.y1 + dy/H, title, fontsize=FS, ha='left', va='bottom', color=INK)


def save(fig, name):
    fig.savefig(FIG/(name+'.pdf'))
    fig.savefig(FIG/(name+'.png'), dpi=300)
    plt.close(fig)


def canvas(fig, spec):
    """Axes whose data units are inches in both directions, for schematics."""
    ax = fig.add_subplot(spec); ax.axis('off')
    W, H = fig.get_size_inches(); p = ax.get_position(); w, h = p.width*W, p.height*H
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect('equal', adjustable='box')
    return ax, w, h


def box(ax, x, y, w, h, text='', fc='#F4F6F8', ec='#9AA5AE', fs=FS_SMALL, weight='normal', color=INK, lw=0.6, ha='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.05', fc=fc, ec=ec, lw=lw, clip_on=False))
    if text:
        tx = x + w/2 if ha == 'center' else x + 0.07
        ax.text(tx, y + h/2, text, ha=ha, va='center', fontsize=fs, fontweight=weight, color=color, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color='#5B6770', lw=0.8, rad=0.0, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}', clip_on=False))


def r1_of(ph):
    return np.abs(np.exp(1j*np.asarray(ph)).mean(axis=-1))


# ============================================================================ Figure 2
def fig2():
    curves = {'GPT': csv(ROOT/'results_frozen/analysis/complex_kernel_transmutation/offset_action_curves.csv'),
              'Claude': csv(ROOT/'results_frozen/analysis/claude_kappa_v2/offset_action_curves.csv')}
    endpoints = {'GPT': csv(ROOT/'results_frozen/analysis/complex_kernel_transmutation/complex_endpoints.csv'),
                 'Claude': csv(ROOT/'results_frozen/analysis/claude_kappa_v2/complex_endpoints.csv')}
    parity = csv(AUD/'kappa_parity_recomputed.csv')
    refl = csv(ST/'response_transport_20261005/reflection_assay/analysis/action_distributions.csv')
    refl_par = csv(AUD/'reflection_parity_recomputed.csv')
    resid = csv(ST/'interface_residual_analysis_20261031/residuals.csv')
    agree = csv(ST/'interface_residual_analysis_20261031/fixed_rule_agreement.csv')
    order = csv(ST/'interface_mechanism_20261031/assay/analysis_real/responses_by_input.csv')
    effects = js(ST/'interface_mechanism_20261031/assay/analysis_real/results.json')['effects']

    fig = plt.figure(figsize=(7.2, 7.25))
    gs = GridSpec(3, 3, figure=fig, left=0.075, right=0.975, top=0.957, bottom=0.115, wspace=0.42, hspace=0.6)

    # a  protocol (inch canvas: circles stay circular)
    ax, w, h = canvas(fig, gs[0, 0]); plabel(fig, ax, 'a', 'Controlled-input assay')
    cx, cy, R, delta = 0.47, h-0.64, 0.40, 1.0
    phi = np.linspace(delta-1.3, delta+1.3, 400)
    dens = np.exp(12*(np.cos(phi-delta)-1))
    rr = R + 0.04 + 0.17*dens
    ax.fill(np.r_[cx+(R+0.04)*np.cos(phi), cx+rr[::-1]*np.cos(phi[::-1])],
            np.r_[cy+(R+0.04)*np.sin(phi), cy+rr[::-1]*np.sin(phi[::-1])], color='#C9D0D6', lw=0, zorder=1)
    ax.plot(cx+rr*np.cos(phi), cy+rr*np.sin(phi), color='#7D8890', lw=.5, zorder=2)
    ax.text(cx+(R+0.15)*np.cos(delta+1.05), cy+(R+0.15)*np.sin(delta+1.05), 'peer\ndensity', fontsize=FS_TINY-0.3,
            color=MUTED, ha='center', va='center', linespacing=1.1)
    ax.add_patch(Circle((cx, cy), R, fc='none', ec='#9E9E9E', lw=.6, zorder=2))
    for q in np.linspace(-1.5, 1.5, 9):
        a_ = delta + q*0.29
        ax.add_patch(Circle((cx+R*np.cos(a_), cy+R*np.sin(a_)), 0.027, fc='#7D8890', ec='white', lw=.3, zorder=3))
    ax.add_patch(Circle((cx+R, cy), 0.036, fc='white', ec=INK, lw=.8, zorder=4))
    ax.plot([cx, cx+R], [cy, cy], color=MUTED, lw=.5, ls=(0, (1.5, 1.5)), zorder=2)
    ax.plot([cx, cx+R*np.cos(delta)], [cy, cy+R*np.sin(delta)], color=MUTED, lw=.5, ls=(0, (1.5, 1.5)), zorder=2)
    arc = np.linspace(0.08, delta-0.10, 40)
    ax.plot(cx+0.20*np.cos(arc), cy+0.20*np.sin(arc), color=INK, lw=.6, zorder=3)
    ax.annotate('', xy=(cx+0.20*np.cos(delta-0.06), cy+0.20*np.sin(delta-0.06)),
                xytext=(cx+0.20*np.cos(arc[-3]), cy+0.20*np.sin(arc[-3])),
                arrowprops=dict(arrowstyle='-|>,head_length=0.25,head_width=0.12', lw=.6, color=INK, shrinkA=0, shrinkB=0))
    ax.text(cx+0.29*np.cos(delta/2), cy+0.29*np.sin(delta/2), r'$\delta$', fontsize=FS, ha='center', va='center')
    xr = 1.02
    ax.add_patch(Circle((xr+0.03, h-0.17), 0.030, fc='white', ec=INK, lw=.7))
    ax.text(xr+0.10, h-0.17, 'focal agent', fontsize=FS_TINY, va='center', color=INK)
    ax.add_patch(Circle((xr+0.03, h-0.31), 0.025, fc='#7D8890', ec='none'))
    ax.text(xr+0.10, h-0.31, 'peers', fontsize=FS_TINY, va='center', color=INK)
    ax.text(xr, h-0.46, '36 offsets × 5 κ\n× 3 descriptions\n× 2 blocks × 24\n= 25,920 calls\nper model', fontsize=FS_TINY,
            color=MUTED, va='top', linespacing=1.25)
    ax.text(0.0, 0.12, r'Rotate the peer field by $\delta$; record' '\n' r'$g(\delta)=p(\mathrm{advance})-p(\mathrm{retard})$',
            fontsize=FS_SMALL, va='bottom', ha='left', linespacing=1.35)

    # b, c  rotation curves at kappa = 12
    target = -0.252
    for j, fam in enumerate(['GPT', 'Claude']):
        ax = fig.add_subplot(gs[0, 1+j]); d = curves[fam]; d = d[np.isclose(d.concentration, 12)]
        offs = d.groupby('offset_index').offset_radians.first(); delta_b = float(offs.iloc[(offs-target).abs().argmin()])
        ax.axhline(0, color=GRID, lw=.5, zorder=0); ax.axvline(0, color=GRID, lw=.5, zorder=0)
        for rep in REPS:
            m = d[d.representation == rep].groupby('offset_radians').g.mean().sort_index()
            ax.plot(m.index, m.values, color=REP_C[rep], lw=1.0, marker='o', ms=1.8, zorder=3, label=REP_N[rep])
        ax.set(xlim=(-np.pi, np.pi), ylim=(-1.12, 1.12), xticks=TICKS_PI[0], xlabel=r'Rotation $\delta$ (peers ahead if $>0$)',
               ylabel=r'Mean action $g(\delta)$' if j == 0 else '', yticks=[-1, 0, 1])
        ax.set_xticklabels(TICKS_PI[1])
        plabel(fig, ax, 'bc'[j], f'{fam}, $\\kappa=12$: one input, three descriptions')
        if j == 1:
            ax.legend(loc='center left', fontsize=FS_TINY, handlelength=1.1, handletextpad=0.4, borderaxespad=0.1,
                      bbox_to_anchor=(0.0, 0.5))

    # d  harmonic fingerprint of the six rules at kappa = 12
    ax = fig.add_subplot(gs[1, 0]); fp = []
    ax.axvline(0, color=GRID, lw=.5, zorder=0)
    lab_pos = {('GPT', REPS[0]): (0.06, 0.035, 'left', 'center'), ('GPT', REPS[1]): (0.0, 0.05, 'center', 'bottom'),
               ('GPT', REPS[2]): (0.0, 0.05, 'center', 'bottom'), ('Claude', REPS[0]): (-0.07, 0.0, 'right', 'center'),
               ('Claude', REPS[1]): (0.0, -0.045, 'center', 'top'), ('Claude', REPS[2]): (0.0, 0.04, 'center', 'bottom')}
    for fam, mk in [('GPT', 'o'), ('Claude', '^')]:
        e = endpoints[fam]; e = e[np.isclose(e.concentration, 12)]
        for rep in REPS:
            s_ = e[e.representation == rep]; assert len(s_) == 2
            ax.scatter(s_.a1, s_.R2, s=7, marker=mk, color=REP_C[rep], alpha=.55, lw=0, zorder=3)
            mx, my = s_.a1.mean(), s_.R2.mean(); fp.append(dict(provider=fam, description=REP_N[rep], a1=mx, R2=my))
            ax.scatter(mx, my, s=34, marker=mk, facecolor=REP_C[rep], edgecolor=MODEL_C[fam], lw=.8, zorder=4)
            dx, dy, ha, va = lab_pos[(fam, rep)]
            ax.text(mx+dx, my+dy, f'{fam} {REP_A[rep]}', color=REP_C[rep], fontsize=FS_TINY, fontweight='bold', ha=ha, va=va)
    ax.set(xlim=(-0.65, 1.5), ylim=(-0.08, 0.52), xticks=[-0.5, 0, 0.5, 1.0], yticks=[0, .2, .4],
           xlabel=r'First sine coefficient $a_1$ (odd)', ylabel=r'Second harmonic $R_2$')
    ax.text(0.98, 0.50, r'$a_1>0$: move toward peers' '\n' r'large $R_2$: two groups', transform=ax.transAxes, ha='right', va='center',
            fontsize=FS_TINY, color=MUTED, linespacing=1.25)
    hd = [Line2D([], [], marker='o', ls='', mfc='white', mec=MODEL_C['GPT'], ms=4, label='GPT'),
          Line2D([], [], marker='^', ls='', mfc='white', mec=MODEL_C['Claude'], ms=4, label='Claude')]
    ax.legend(handles=hd, loc='lower left', fontsize=FS_TINY, handletextpad=0.2, borderaxespad=0.1, ncol=2, columnspacing=0.6)
    plabel(fig, ax, 'd', r'Six rules at $\kappa=12$: harmonic fingerprint')
    export(pd.DataFrame(fp), 'fig2_harmonic_fingerprint.csv')

    # e  first-harmonic plane for moments
    ax = fig.add_subplot(gs[1, 1])
    ax.axhline(0, color=GRID, lw=.5, zorder=0); ax.axvline(0, color=GRID, lw=.5, zorder=0)
    for fam in ['GPT', 'Claude']:
        e = endpoints[fam]; e = e[e.representation == REPS[0]]
        for blk, ls in [('block_1', '-'), ('block_2', '--')]:
            s_ = e[e.block == blk].sort_values('concentration')
            ax.plot(s_.a1, s_.b1, ls=ls, color=MODEL_C[fam], lw=.8, alpha=.85, zorder=2)
            ax.scatter(s_.a1, s_.b1, s=3+2.0*s_.concentration, color=MODEL_C[fam], zorder=3, lw=0)
        q = parity[(parity.provider == fam) & (parity.representation == REPS[0]) & np.isclose(parity.kappa, 12)].sort_values('block')
        a1, b1 = q.sine_a1.mean(), q.cosine_b1.mean()
        ax.plot([0, a1], [0, b1], color=MODEL_C[fam], lw=.6, ls=(0, (1, 1.5)), zorder=1)
        angs = ' / '.join(f'{v:+.0f}°' for v in q.first_harmonic_angle_deg)
        if fam == 'Claude':
            ax.text(a1+0.07, b1-0.02, f'Claude\n{angs}', color=MODEL_C[fam], fontsize=FS_TINY+0.2, fontweight='bold',
                    ha='left', va='center', linespacing=1.15)
        else:
            ax.text(0.30, 0.07, f'GPT  {angs}', color=MODEL_C[fam], fontsize=FS_TINY+0.2, fontweight='bold', ha='left', va='bottom')
    ax.set(xlim=(-0.06, 1.48), ylim=(-0.25, 1.02), xlabel=r'$a_1$ (sine: odd under reflection)',
           ylabel=r'$b_1$ (cosine: even under reflection)', xticks=[0, .5, 1], yticks=[0, .5, 1])
    ax.text(0.05, 0.99, 'moments description\n' r'$\kappa=2\rightarrow12$: marker size' '\nsolid / dashed: two blocks\nangle: phase lag at $\\kappa=12$',
            fontsize=FS_TINY, color=MUTED, transform=ax.transAxes, va='top', ha='left', linespacing=1.25)
    plabel(fig, ax, 'e', 'First harmonic of the response')

    # f  reflection assay
    ax = fig.add_subplot(gs[1, 2]); pooled = refl[refl.block == 'pooled']; xt, xl = [], []
    for fi, fam in enumerate(['GPT', 'Claude']):
        for pair in range(3):
            for k, mir in enumerate([False, True]):
                r = pooled[(pooled.provider == fam) & (pooled.pair == pair) & (pooled.mirrored == mir)].iloc[0]
                x = fi*3.9 + pair*1.2 + (k-0.5)*0.46; bottom = 0
                for a, col in zip(ACT, ['p_retard', 'p_stay', 'p_advance']):
                    ax.bar(x, r[col], bottom=bottom, width=.42, color=ACT_C[a], edgecolor='white', lw=.3); bottom += r[col]
                xt.append(x); xl.append('om'[k])
            ge = float(refl_par[(refl_par.provider == fam) & (refl_par.pair == pair)].g_even.iloc[0])
            ax.text(fi*3.9 + pair*1.2, -0.13, f'{pair+1}', ha='center', va='top', fontsize=FS_TINY, color=MUTED)
            ax.text(fi*3.9 + pair*1.2, -0.25, f'{ge:.2f}', ha='center', va='top', fontsize=FS_TINY, color=MODEL_C[fam])
        ax.text(fi*3.9+1.2, -0.38, fam, ha='center', va='top', fontsize=FS, color=MODEL_C[fam])
    ax.text(-0.72, -0.13, 'pair', ha='right', va='top', fontsize=FS_TINY, color=MUTED)
    ax.text(-0.72, -0.25, 'even', ha='right', va='top', fontsize=FS_TINY, color=MUTED)
    ax.set(ylim=(0, 1.2), yticks=[0, .5, 1], xlim=(-0.8, 6.9), ylabel='Action probability', xticks=xt)
    ax.spines['left'].set_bounds(0, 1)
    ax.set_xticklabels(xl, fontsize=FS_TINY); ax.tick_params(axis='x', length=0, pad=1)
    ax.spines['bottom'].set_visible(False)
    hs = [Rectangle((0, 0), 1, 1, color=ACT_C[a]) for a in ACT[::-1]]
    ax.legend(hs, ['advance', 'stay', 'retard'], loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=FS_TINY,
              handlelength=0.9, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.0)
    plabel(fig, ax, 'f', 'Recorded state (o) and its mirror (m)')

    # g, h  response boundaries on the first-moment plane
    lim = 1.05
    for j, fam in enumerate(['GPT', 'Claude']):
        ax = fig.add_subplot(gs[2, j]); r = resid[(resid.provider == fam) & (resid.order == 'cos_first')]
        # wedges where sign(S1) and sign(C1+S1) disagree
        ax.add_patch(Polygon([[0, 0], [lim, 0], [lim, -lim]], closed=True, fc='#ECECEC', ec='none', zorder=0))
        ax.add_patch(Polygon([[0, 0], [-lim, 0], [-lim, lim]], closed=True, fc='#ECECEC', ec='none', zorder=0))
        sc = ax.scatter(r.c1, r.s1, c=r.observed, cmap=ACT_CMAP, vmin=-1, vmax=1, s=9, lw=.25, edgecolor='#555555', zorder=3)
        xx = np.array([-lim, lim])
        ax.plot(xx, 0*xx, color=MODEL_C['GPT'], lw=.7, zorder=2)
        ax.plot(xx, -xx, color=MODEL_C['Claude'], lw=.8, ls=(0, (3, 1.5)), zorder=2)
        ax.text(lim+0.03, 0, r'$S_1=0$', fontsize=FS_TINY, ha='left', va='center', color=MODEL_C['GPT'], clip_on=False)
        ax.text(lim+0.03, -lim, r'$C_1+S_1=0$', fontsize=FS_TINY, ha='left', va='center', color=MODEL_C['Claude'], clip_on=False)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=r'Presented $C_1$', ylabel=r'Presented $S_1$' if j == 0 else '',
               xticks=[-1, 0, 1], yticks=[-1, 0, 1]); ax.set_aspect('equal')
        a = agree[(agree.provider == fam) & (agree.order == 'cos_first') & (agree.stratum == 'all')].set_index('rule').action_agreement
        ax.text(0.03, 0.03, f'agreement with\nsign($S_1$): {100*a.sine_sign:.1f}%\nsign($C_1+S_1$): {100*a.sum_sign:.1f}%',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_TINY, color=INK, linespacing=1.25,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=.9), zorder=5)
        plabel(fig, ax, 'gh'[j], f'{fam}: 128 states from collectives')
        if j == 0:
            pg = ax.get_position()
        else:
            ph = ax.get_position(); W, H = fig.get_size_inches()
            xm = (pg.x0 + ph.x1)/2
            cax = fig.add_axes([xm - 1.35/W, 0.030, 1.05/W, 0.07/H])
            cb = fig.colorbar(sc, cax=cax, ticks=[-1, 0, 1], orientation='horizontal')
            cb.ax.tick_params(labelsize=FS_TINY-0.3, length=1.5, width=.5, pad=1); cb.outline.set_linewidth(.4)
            cb.ax.set_xticklabels(['retard', '0', 'advance'])
            fig.text(xm - 1.35/W, 0.030 + 0.12/H, 'Observed mean action', fontsize=FS_TINY, va='bottom', ha='left')
            fig.add_artist(Rectangle((xm + 0.05/W, 0.030 + 0.01/H), 0.16/W, 0.10/H, transform=fig.transFigure,
                                     fc='#ECECEC', ec='none'))
            fig.text(xm + 0.26/W, 0.030 + 0.06/H, 'the two sign rules disagree', fontsize=FS_TINY, va='center', color=INK)

    # i  information-preserving order change
    ax = fig.add_subplot(gs[2, 2]); txt = []
    for fam, mk in [('GPT', 'o'), ('Claude', '^')]:
        pv = order[order.provider == fam].pivot(index=['field', 'branch'], columns='order', values='mean')
        cnt = pv.groupby(['cos_first', 'sin_first']).size().reset_index(name='n')
        ax.scatter(cnt.cos_first, cnt.sin_first, s=5+1.5*cnt.n, marker=mk, color=MODEL_C[fam], alpha=.55, lw=0,
                   zorder=3 if fam == 'GPT' else 4)
        rms = np.sqrt(next(e['order_effect_squared'] for e in effects if e['provider'] == fam)); txt.append((fam, mk, rms))
    ax.plot([-1.1, 1.1], [-1.1, 1.1], ls=(0, (1.5, 1.5)), color=MUTED, lw=.6, zorder=1)
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), xticks=[-1, 0, 1], yticks=[-1, 0, 1],
           xlabel='Mean action, cosine rows first', ylabel='Mean action, sine rows first'); ax.set_aspect('equal')
    pi_ = ax.get_position(); W, H = fig.get_size_inches()
    for k, (fam, mk, rms) in enumerate(txt):
        yk = 0.030 + (0.26 - k*0.13)/H
        fig.lines.append(Line2D([pi_.x0 + 0.02/W], [yk], marker=mk, ls='', ms=3.6, color=MODEL_C[fam], alpha=.8,
                                transform=fig.transFigure))
        fig.text(pi_.x0 + 0.10/W, yk, f'{fam}: RMS change {rms:.3f}', va='center', fontsize=FS_TINY)
    fig.text(pi_.x0 - 0.01/W, 0.030 + 0.0/H, r'area $\propto$ number of inputs (128 per model)', va='center', fontsize=FS_TINY,
             color=MUTED, linespacing=1.2)
    plabel(fig, ax, 'i', 'Same numbers, different row order')

    save(fig, 'fig2_response_rules')
    export(pd.concat([v.assign(provider=k) for k, v in curves.items()]), 'fig2_rotation_curves.csv')
    export(pd.concat([v.assign(provider=k) for k, v in endpoints.items()]), 'fig2_first_harmonic.csv')
    export(parity, 'fig2_coefficients.csv'); export(refl, 'fig2_reflection_distributions.csv')
    export(refl_par, 'fig2_reflection_pairs.csv'); export(resid, 'fig2_boundary_inputs.csv')
    export(agree, 'fig2_boundary_rule_agreement.csv'); export(order, 'fig2_order_responses.csv')
    export(pd.DataFrame(effects), 'fig2_order_effect_estimates.csv')


# ============================================================================ Figure 3
def schematic_interfaces(ax, w, h):
    """Interface operation (top row) and forecasting protocol (bottom row); units are inches."""
    yt, ht = 0.86, 0.40
    xs = [0.02, 0.92, 2.12, 3.02, 4.22]; ws = [0.68, 0.98, 0.68, 0.98, 0.68]
    labels = ['Relative\nstate $h$', 'Coin $c$\n$c=0$: show $h$\n$c=1$: show $Rh$', 'Language\nmodel',
              'Action $a$\nif $c=1$,\nreturn $-a$', 'Physical\nengine']
    for x, ww, lab in zip(xs, ws, labels):
        box(ax, x, yt, ww, ht, lab, fc='#EEF1F4' if 'model' in lab else 'white', weight='bold' if 'model' in lab else 'normal',
            fs=FS_TINY+0.2)
    for i in range(4):
        arrow(ax, xs[i]+ws[i]+0.03, yt+ht/2, xs[i+1]-0.03, yt+ht/2)
    arrow(ax, xs[4]+ws[4]/2, yt+ht+0.02, xs[0]+ws[0]/2, yt+ht+0.02, rad=0.06, lw=.6, color='#9AA5AE', ms=6)
    ax.text((xs[0]+xs[4]+ws[4])/2, yt+ht+0.20, 'next state', fontsize=FS_TINY, color=MUTED, ha='center', va='bottom')
    # coin patterns
    x0 = 5.12; ax.text(x0, yt+ht+0.05, 'Coins of 17 agents at one step', fontsize=FS_TINY, color=MUTED, va='bottom')
    rng = np.random.default_rng(4)
    for k, (pol, coins) in enumerate([('native', np.zeros(17)), ('independent_mirror', rng.integers(0, 2, 17)),
                                       ('shared_mirror', np.ones(17))]):
        yy = yt + ht - 0.07 - k*0.14
        ax.text(x0, yy, POL_A[pol], fontsize=FS_TINY+0.2, color=POL_C[pol], fontweight='bold', va='center')
        for i, c in enumerate(coins):
            ax.add_patch(Circle((x0+0.45+i*0.056, yy), 0.021, fc=POL_C[pol] if c else 'white', ec=POL_C[pol], lw=.55))
    ax.text(x0, yt-0.06, 'filled: mirrored description', fontsize=FS_TINY-0.3, color=MUTED, va='top')
    # protocol row
    yb, hb = 0.02, 0.66
    steps = [('Earlier collectives', '8 initial states\nnative and independent\n16,320 responses'),
             ('Fit response model', r'$p(a\mid$six moment values$)$' '\nquadratic logit (primary)'),
             ('Freeze', 'coefficients and\nforecasts before\nany new call'),
             ('New collectives', '2 cohorts × 8 initial states\n× 3 interfaces\n60 steps: 24,480 responses\n200 steps: 81,600 responses'),
             ('Compare', 'window means and\ninterface contrasts')]
    wb = [1.10, 1.22, 0.92, 1.42, 0.95]; gap = (w - 0.04 - sum(wb))/4
    xb = [0.02]
    for ww in wb[:-1]: xb.append(xb[-1] + ww + gap)
    for (t, s_), x, ww in zip(steps, xb, wb):
        box(ax, x, yb, ww, hb, fc='#F4F6F8')
        ax.text(x+ww/2, yb+hb-0.06, t, ha='center', va='top', fontsize=FS_TINY+0.3, fontweight='bold')
        ax.text(x+ww/2, (yb+hb-0.17)/2 + 0.01, s_, ha='center', va='center', fontsize=FS_TINY, color=INK, linespacing=1.2)
    for i in range(4):
        arrow(ax, xb[i]+wb[i]+0.03, yb+hb/2, xb[i+1]-0.03, yb+hb/2)


def fig3():
    tra = csv(ST/'phase_persistence_diagnostics_20261101/B_trajectories.csv')
    scores = csv(ST/'phase_persistence_validation_20261101/analysis_real/prediction_scores.csv')
    old = csv(ST/'prospective_closure_validation_20261031/analysis_real/prediction_scores.csv')
    oldc = csv(ST/'prospective_closure_validation_20261031/analysis_real/contrasts.csv')
    eff = js(ST/'phase_persistence_validation_20261101/analysis_real/results.json')['effects']

    H = 7.55; fig = plt.figure(figsize=(7.2, H))
    topA = 1 - 0.20/H; botA = topA - 1.62/H
    gsA = GridSpec(1, 1, figure=fig, left=0.072, right=0.985, top=topA, bottom=botA)
    gsB = GridSpec(3, 6, figure=fig, left=0.072, right=0.985, top=botA - 0.42/H, bottom=0.055,
                   height_ratios=[1.25, 1.45, 1.05], hspace=0.50, wspace=1.15)
    ax, w, h = canvas(fig, gsA[0, 0]); schematic_interfaces(ax, w, h)
    plabel(fig, ax, 'a', 'Interfaces and the prospective forecasting protocol', dx=0.30, dy=0.0)

    q6 = tra[tra.model == 'quadratic6']; win = q6[(q6.step >= 61) & (q6.step <= 200)]
    for j, pol in enumerate(POL[:3]):
        ax = fig.add_subplot(gsB[0, 2*j:2*j+2]); d = q6[q6.controller == pol]
        ax.axvspan(41, 60, color='#F1F1F1', lw=0, zorder=0); ax.axvspan(61, 200, color='#EAF0F5', lw=0, zorder=0)
        for _, v in d.groupby('physical_seed'):
            ax.plot(v.step, v.observed, color=POL_C[pol], alpha=.22, lw=.5, zorder=1)
        m = d.groupby('step')[['observed', 'predicted']].mean()
        ax.plot(m.index, m.observed, color=POL_C[pol], lw=1.5, zorder=3, label='Observed (mean of 8)')
        ax.plot(m.index, m.predicted, color='black', ls=(0, (3, 1.6)), lw=.95, zorder=4, label='Frozen forecast')
        wv = win[win.controller == pol]
        ob = wv.groupby('physical_seed')['observed'].mean().mean(); pr = wv.groupby('physical_seed')['predicted'].mean().mean()
        ax.text(0.97, 0.05, f'steps 61–200\nobserved {ob:.3f}\nforecast {pr:.3f}', transform=ax.transAxes, ha='right',
                va='bottom', fontsize=FS_TINY, color=INK, linespacing=1.25)
        ax.set(xlim=(0, 200), ylim=(0, 1), xticks=[0, 50, 100, 150, 200], yticks=[0, .5, 1], xlabel='Step',
               ylabel=r'Alignment $r_1$' if j == 0 else '')
        plabel(fig, ax, 'bcd'[j], POL_N[pol])
        if j == 0:
            ax.legend(loc='upper right', fontsize=FS_TINY, handlelength=1.6, borderaxespad=0.1)

    # e  predicted vs observed window means
    ax = fig.add_subplot(gsB[1, 0:2])
    s2 = scores[(scores.model == 'quadratic6') & (scores.window == 'primary_61_200')]; s1 = old[old.model == 'quadratic6']
    for pol in POL[:3]:
        a = s2[s2.controller == pol]; ax.scatter(a.prediction, a.observed, color=POL_C[pol], s=13, lw=0, zorder=3)
        b = s1[s1.controller == pol]; ax.scatter(b.prediction, b.observed, facecolors='none', edgecolors=POL_C[pol], s=14, lw=.7, zorder=3)
    ax.plot([.45, .9], [.45, .9], ls=(0, (1.5, 1.5)), color=MUTED, lw=.6, zorder=1)
    rm2 = np.sqrt(np.mean((s2.observed-s2.prediction)**2)); rm1 = np.sqrt(np.mean((s1.observed-s1.prediction)**2))
    ax.set(xlim=(.45, .9), ylim=(.45, .9), xticks=[.5, .6, .7, .8, .9], yticks=[.5, .6, .7, .8, .9],
           xlabel='Forecast run mean', ylabel='Observed run mean'); ax.set_aspect('equal')
    hd = [Line2D([], [], marker='o', ls='', color=MUTED, ms=3.5, label=f'200 steps (61–200)\nRMSE {rm2:.4f}'),
          Line2D([], [], marker='o', ls='', mfc='none', mec=MUTED, ms=3.5, label=f'60 steps (41–60)\nRMSE {rm1:.4f}')]
    ax.legend(handles=hd, loc='upper left', fontsize=FS_TINY, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.5)
    ax.text(0.97, 0.03, 'colour: interface', transform=ax.transAxes, ha='right', va='bottom', fontsize=FS_TINY, color=MUTED)
    plabel(fig, ax, 'e', '48 new trajectories, no refitting')

    # f  scoreboard of frozen contrast forecasts
    ax = fig.add_subplot(gsB[1, 2:4]); rows = []
    cmap = {'independent_mirror - native': ('I − N', POL_C['independent_mirror']),
            'shared_mirror - native': ('S − N', '#555555'),
            'shared_mirror - independent_mirror': ('S − I', POL_C['shared_mirror'])}
    q = oldc[oldc.model == 'quadratic6'].set_index('contrast')
    for c in cmap:
        rows.append(('60-step cohort, steps 41–60', c, q.loc[c, 'predicted'], q.loc[c, 'observed'], q.loc[c, 'observed_ci_low'], q.loc[c, 'observed_ci_high']))
    for wkey, wname in [('early_41_60', '200-step cohort, steps 41–60'), ('primary_61_200', '200-step cohort, steps 61–200 (primary)'),
                        ('terminal_161_200', '200-step cohort, steps 161–200')]:
        for c in cmap:
            e = next(e for e in eff if e['model'] == 'quadratic6' and e['window'] == wkey and e['contrast'] == c)
            rows.append((wname, c, e['predicted'], e['observed'], *e['observed_bootstrap95']))
    sb = pd.DataFrame(rows, columns=['cohort_window', 'contrast', 'predicted', 'observed', 'observed_ci_low', 'observed_ci_high'])
    export(sb, 'fig3_contrast_scoreboard.csv')
    gl = {'60-step cohort, steps 41–60': '60-step\nsteps 41–60', '200-step cohort, steps 41–60': '200-step\nsteps 41–60',
          '200-step cohort, steps 61–200 (primary)': '200-step\nsteps 61–200\n(primary)', '200-step cohort, steps 161–200': '200-step\nsteps 161–200'}
    y = 0; XG, XC = -0.112, -0.058
    for gi, (gname, g) in enumerate(sb.groupby('cohort_window', sort=False)):
        ys = []
        if gi % 2 == 1:
            ax.axhspan(y+0.5, y-2.5, color='#F3F3F3', lw=0, zorder=0)
        for _, r in g.iterrows():
            lab, col = cmap[r.contrast]
            ax.plot([r.observed_ci_low, r.observed_ci_high], [y, y], color=col, lw=1.0, zorder=2, solid_capstyle='butt')
            ax.scatter(r.observed, y, color=col, s=12, zorder=3, lw=0)
            ax.scatter(r.predicted, y, marker='D', s=11, facecolors='white', edgecolors='black', lw=.6, zorder=4)
            ax.text(XC, y, lab, color=col, fontsize=FS_TINY, ha='right', va='center', fontweight='bold')
            if np.sign(r.observed) != np.sign(r.predicted):
                ax.text(max(r.observed_ci_high, r.predicted)+0.012, y, 'signs differ; both near 0', fontsize=FS_TINY-0.4,
                        va='center', ha='left', color=MUTED)
            ys.append(y); y -= 1
        ax.text(XG, np.mean(ys), gl[gname], fontsize=FS_TINY, ha='right', va='center', linespacing=1.15, clip_on=False,
                fontweight='bold' if 'primary' in gname else 'normal')
        y -= 0.1
    ax.axvline(0, color=GRID, lw=.6, zorder=1)
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set(xlim=(-0.10, 0.27), ylim=(y+0.35, 2.9), xticks=[0, .1, .2], xlabel=r'Difference in mean $r_1$')
    ax.spines['bottom'].set_bounds(-0.05, 0.27)
    hd = [Line2D([], [], marker='o', color=MUTED, ms=3, lw=1, label='observed, 95% CI'),
          Line2D([], [], marker='D', ls='', mfc='white', mec='black', ms=3, label='frozen forecast')]
    ax.legend(handles=hd, loc='upper right', fontsize=FS_TINY, handletextpad=0.3, borderaxespad=0.0, ncol=1, handlelength=1.2,
              labelspacing=0.25)
    plabel(fig, ax, 'f', 'All prespecified interface contrasts', dx=0.66)

    # g  error by model and interface
    ax = fig.add_subplot(gsB[1, 4:6]); post = tra[tra.step > 0]; models = ['constant_old', 'four_parameter', 'quadratic6']
    mname = {'constant_old': 'State-independent', 'four_parameter': 'Four-parameter logit', 'quadratic6': 'Quadratic logit (primary)'}
    mcol = {'constant_old': '#CFCFCF', 'four_parameter': '#9DB3C4', 'quadratic6': '#2F5D7C'}
    rec = []
    for i, pol in enumerate(POL[:3]):
        for k, mdl in enumerate(models):
            v = np.sqrt(np.mean(post[(post.model == mdl) & (post.controller == pol)].residual**2)); rec.append((pol, mdl, v))
            ax.bar(i + (k-1)*0.27, v, width=.25, color=mcol[mdl], label=mname[mdl] if i == 0 else None)
    export(pd.DataFrame(rec, columns=['interface', 'model', 'trajectory_rmse']), 'fig3_trajectory_rmse.csv')
    ax.set_yscale('log'); ax.set(ylim=(0.008, 4), xticks=range(3), ylabel='Trajectory RMSE, steps 1–200')
    ax.set_xticklabels([POL_A[p] for p in POL[:3]])
    ax.legend(loc='upper right', fontsize=FS_TINY, ncol=1, handlelength=0.9, handletextpad=0.3, borderaxespad=0.0)
    plabel(fig, ax, 'g', 'State-dependent responses are required')

    # h  time-resolved error
    ax = fig.add_subplot(gsB[2, 0:3]); ax.axvspan(61, 200, color='#EAF0F5', lw=0, zorder=0)
    for pol in POL[:3]:
        d = q6[(q6.controller == pol) & (q6.step > 0)]
        e = d.groupby('step')['residual'].apply(lambda r: np.sqrt(np.mean(r**2)))
        ax.plot(e.index, e.values, color=POL_C[pol], lw=1.0, label=POL_N[pol])
    ax.set(xlim=(0, 200), ylim=(0, 0.24), xticks=[0, 50, 100, 150, 200], yticks=[0, .1, .2], xlabel='Step',
           ylabel='RMSE over 8 states')
    ax.legend(loc='upper left', fontsize=FS_TINY, handlelength=1.4, borderaxespad=0.05)
    plabel(fig, ax, 'h', 'Errors concentrate in native-interface transients')

    # i  worst transient
    ax = fig.add_subplot(gsB[2, 3:6]); seed = 202611011506
    d = q6[(q6.controller == 'native') & (q6.physical_seed == seed)]
    ax.fill_between(d.step, d.process_p025, d.process_p975, color='#D6DEE5', lw=0, label='95% simulation band')
    ax.plot(d.step, d.observed, color=POL_C['native'], lw=1.3, label='Observed')
    ax.plot(d.step, d.predicted, color='black', ls=(0, (3, 1.6)), lw=.9, label='Frozen forecast')
    ax.set(xlim=(0, 200), ylim=(0, 1), xticks=[0, 50, 100, 150, 200], yticks=[0, .5, 1], xlabel='Step', ylabel=r'$r_1$')
    ax.legend(loc='lower right', fontsize=FS_TINY, ncol=3, handlelength=1.2, columnspacing=0.7, borderaxespad=0.05)
    plabel(fig, ax, 'i', f'Largest transient error (native interface, seed {seed})')

    save(fig, 'fig3_prospective_prediction')
    export(tra, 'fig3_all_trajectories.csv'); export(scores, 'fig3_200step_scores.csv'); export(old, 'fig3_60step_scores.csv')


# ============================================================================ Figure 4
def fig4():
    ck = js(ST/'mechanism_controls_20261031/runs/real/checkpoint.json')['trajectories']
    mech = csv(ST/'mechanism_controls_20261031/analysis_real/by_run.csv')
    md = csv(ST/'reflection_confirmation_20261005/analysis/matched_differences.csv')
    old = csv(ST/'prospective_closure_validation_20261031/analysis_real/prediction_scores.csv')
    oldc = csv(ST/'prospective_closure_validation_20261031/analysis_real/contrasts.csv')
    scores = csv(ST/'phase_persistence_validation_20261101/analysis_real/prediction_scores.csv')
    eff = js(ST/'phase_persistence_validation_20261101/analysis_real/results.json')['effects']
    fixed = csv(ST/'same_state_validation_20261031/analysis_real/by_state.csv')
    tra = csv(ST/'phase_persistence_diagnostics_20261101/B_trajectories.csv')
    rng = np.random.default_rng(1)

    fig = plt.figure(figsize=(7.2, 5.15))
    gs = GridSpec(2, 3, figure=fig, left=0.075, right=0.985, top=0.94, bottom=0.10, wspace=0.42, hspace=0.62)

    # a  four-condition time courses
    ax = fig.add_subplot(gs[0, 1]); rows = []
    for run, v in ck.items():
        pol = run.split('__')[1]; seed = int(run.split('seed')[1]); r1 = r1_of(np.array(v['phases']))
        rows += [dict(controller=pol, physical_seed=seed, step=t, r1=x) for t, x in enumerate(r1)]
    tc = pd.DataFrame(rows); export(tc, 'fig4_control_trajectories.csv')
    chk = tc[(tc.step >= 41) & (tc.step <= 60)].groupby(['controller', 'physical_seed']).r1.mean()
    ref = mech.set_index(['controller', 'physical_seed']).late_mean_r1
    assert np.allclose(chk.loc[ref.index].values, ref.values, atol=1e-9), 'r1 from stored phases must match the saved run means'
    ax.axvspan(41, 60, color='#F0F0F0', lw=0, zorder=0)
    for pol in POL:
        d = tc[tc.controller == pol]
        for _, s_ in d.groupby('physical_seed'): ax.plot(s_.step, s_.r1, color=POL_C[pol], alpha=.18, lw=.5)
        m = d.groupby('step').r1.mean(); ax.plot(m.index, m.values, color=POL_C[pol], lw=1.3, label=POL_A[pol])
        yl = m.values[-1] + {'native': 0.0, 'independent_mirror': -0.035, 'shared_mirror': 0.045, 'shuffled_mirror': 0.0}[pol]
        ax.text(61.5, yl, POL_A[pol], color=POL_C[pol], fontsize=FS_TINY, fontweight='bold', va='center', ha='left', clip_on=False)
    ax.set(xlim=(0, 60), ylim=(0, 1), xticks=[0, 20, 40, 60], yticks=[0, .5, 1], xlabel='Step', ylabel=r'Alignment $r_1$')
    ax.text(50.5, 0.03, 'window', fontsize=FS_TINY, color=MUTED, ha='center', va='bottom')
    plabel(fig, ax, 'b', 'Four interfaces, four new initial states')

    # b  per-seed window means
    ax = fig.add_subplot(gs[0, 2]); pv = mech.pivot(index='physical_seed', columns='controller', values='late_mean_r1')[POL]
    for _, row in pv.iterrows(): ax.plot(range(4), row.values, color='#C8C8C8', lw=.6, zorder=1)
    for i, pol in enumerate(POL):
        ax.scatter(np.full(len(pv), i), pv[pol], color=POL_C[pol], s=14, zorder=3, lw=0)
        ax.plot([i-.2, i+.2], [pv[pol].mean()]*2, color=POL_C[pol], lw=1.6, zorder=2)
        ax.text(i+0.24, pv[pol].mean(), f'{pv[pol].mean():.2f}', ha='left', va='center', fontsize=FS_TINY, color=POL_C[pol])
    ax.set(xlim=(-.5, 3.7), ylim=(0, 1), xticks=range(4), yticks=[0, .5, 1], ylabel=r'Mean $r_1$, steps 41–60')
    ax.set_xticklabels([POL_A[p] for p in POL], fontsize=FS_TINY)
    plabel(fig, ax, 'c', 'Shuffling actions removes order')

    # c  independent minus native across three cohorts
    ax = fig.add_subplot(gs[0, 0]); eff_sets = []
    eff_sets.append(('First 60-step\nsteps 41–60', md.delta_r1.to_numpy()))
    o = old[old.model == 'quadratic6'].pivot(index='physical_seed', columns='controller', values='observed')
    eff_sets.append(('Second 60-step\nsteps 41–60', (o.independent_mirror-o.native).to_numpy()))
    o = scores[(scores.model == 'quadratic6') & (scores.window == 'primary_61_200')].pivot(index='physical_seed', columns='controller', values='observed')
    eff_sets.append(('200-step\nsteps 61–200', (o.independent_mirror-o.native).to_numpy()))
    export(pd.DataFrame([dict(cohort=c.replace('\n', ' '), pair_index=i, independent_minus_native=v)
                         for c, vals in eff_sets for i, v in enumerate(vals)]), 'fig4_cohort_effects.csv')
    for j, (name, v) in enumerate(eff_sets):
        ax.scatter(j + rng.uniform(-.13, .13, len(v)), v, color=POL_C['independent_mirror'], s=12, lw=0, alpha=.85, zorder=3)
        ax.plot([j-.24, j+.24], [v.mean()]*2, color='black', lw=1.5, zorder=4)
        ax.text(j, 0.30, f'{(v > 0).sum()}/{len(v)} > 0\nmean {v.mean():.3f}', ha='center', va='top', fontsize=FS_TINY, linespacing=1.2)
    ax.axhline(0, color=GRID, lw=.6)
    ax.set(xlim=(-.55, 2.55), ylim=(-0.03, 0.30), xticks=range(3), yticks=[0, .1, .2], ylabel=r'Indep. − native, mean $r_1$')
    ax.set_xticklabels([n for n, _ in eff_sets], fontsize=FS_TINY)
    plabel(fig, ax, 'a', 'Independent reflection in three cohorts')

    # d  fixed-state one-step test (values in units of 1e-3)
    ax = fig.add_subplot(gs[1, 0]); sc = 1e3
    lo = fixed.predicted_gap + fixed.residual_ci_low; hi = fixed.predicted_gap + fixed.residual_ci_high
    ax.errorbar(sc*fixed.predicted_gap, sc*fixed.observed_gap,
                yerr=[sc*np.clip(fixed.observed_gap-lo, 0, None), sc*np.clip(hi-fixed.observed_gap, 0, None)], fmt='none',
                ecolor='#9DB3C4', elinewidth=.8, capsize=1.5, capthick=.6, zorder=2)
    ax.scatter(sc*fixed.predicted_gap, sc*fixed.observed_gap, color='#2F5D7C', s=14, zorder=3, lw=0)
    ax.plot([-2, 7], [-2, 7], ls=(0, (1.5, 1.5)), color=MUTED, lw=.6)
    ax.axhline(0, color=GRID, lw=.5, zorder=0); ax.axvline(0, color=GRID, lw=.5, zorder=0)
    rm = np.sqrt(np.mean((fixed.observed_gap-fixed.predicted_gap)**2))
    ax.text(0.04, 0.97, f'mean forecast {sc*fixed.predicted_gap.mean():.2f}\nmean observed {sc*fixed.observed_gap.mean():.2f}\n'
            f'RMSE {sc*rm:.2f}', transform=ax.transAxes, va='top', fontsize=FS_TINY, linespacing=1.25)
    ax.set(xlim=(-2.5, 7.5), ylim=(-3.5, 7.5), xlabel=r'Forecast one-step $\Delta r_1^2$ (×10$^{-3}$)',
           ylabel=r'Observed one-step $\Delta r_1^2$ (×10$^{-3}$)')
    plabel(fig, ax, 'd', 'Shared − independent coin at 8 fixed states')

    # e  time-resolved shared minus independent in the 200-step cohort
    ax = fig.add_subplot(gs[1, 1]); q6 = tra[tra.model == 'quadratic6']
    pv_o = q6.pivot_table(index='step', columns=['controller', 'physical_seed'], values='observed')
    pv_p = q6.pivot_table(index='step', columns=['controller', 'physical_seed'], values='predicted')
    do = pv_o['shared_mirror'] - pv_o['independent_mirror']; dp = pv_p['shared_mirror'] - pv_p['independent_mirror']
    mo, se = do.mean(axis=1), do.std(axis=1, ddof=1)/np.sqrt(do.shape[1]); mp = dp.mean(axis=1)
    sm = lambda x: x.rolling(5, center=True, min_periods=1).mean()
    ax.axvspan(41, 60, color='#F0F0F0', lw=0, zorder=0); ax.axvspan(61, 200, color='#EAF0F5', lw=0, zorder=0)
    ax.fill_between(mo.index, sm(mo-se), sm(mo+se), color=POL_C['shared_mirror'], alpha=.22, lw=0)
    ax.plot(mo.index, sm(mo), color=POL_C['shared_mirror'], lw=1.2, label='Observed, mean ± s.e.')
    ax.plot(mp.index, sm(mp), color='black', ls=(0, (3, 1.6)), lw=.9, label='Frozen forecast')
    ax.axhline(0, color='#9E9E9E', lw=.6)
    export(pd.DataFrame({'step': mo.index, 'observed_mean': mo.values, 'observed_se': se.values, 'forecast_mean': mp.values}),
           'fig4_shared_minus_independent_by_step.csv')
    ax.set(xlim=(0, 200), ylim=(-0.07, 0.15), xticks=[0, 50, 100, 150, 200], yticks=[-0.05, 0, .05, .1, .15], xlabel='Step',
           ylabel=r'Shared − indep., $r_1$')
    ax.legend(loc='upper right', fontsize=FS_TINY, handlelength=1.4, borderaxespad=0.05)
    ax.text(0.98, 0.04, '200-step cohort; 5-step moving average', transform=ax.transAxes, ha='right', fontsize=FS_TINY, color=MUTED)
    plabel(fig, ax, 'e', 'Early advantage, later deficit of the shared coin')

    # f  cohort/window summary of shared minus independent
    ax = fig.add_subplot(gs[1, 2]); sets = []
    po = old[old.model == 'quadratic6'].pivot(index='physical_seed', columns='controller', values='observed')
    qo = oldc[(oldc.model == 'quadratic6') & (oldc.contrast == 'shared_mirror - independent_mirror')].iloc[0]
    sets.append(('60-step\n41–60', (po.shared_mirror-po.independent_mirror).to_numpy(), qo.predicted, (qo.observed_ci_low, qo.observed_ci_high)))
    for wkey, wl in [('early_41_60', '200-step\n41–60'), ('primary_61_200', '200-step\n61–200'), ('terminal_161_200', '200-step\n161–200')]:
        e = next(e for e in eff if e['model'] == 'quadratic6' and e['window'] == wkey and e['contrast'] == 'shared_mirror - independent_mirror')
        sets.append((wl, np.array(e['per_seed']), e['predicted'], tuple(e['observed_bootstrap95'])))
    for j, (name, v, pred, ci) in enumerate(sets):
        ax.scatter(j + rng.uniform(-.13, .13, len(v)), v, color=POL_C['shared_mirror'], s=10, lw=0, alpha=.8, zorder=3)
        ax.plot([j+0.27, j+0.27], ci, color='black', lw=.9, zorder=3); ax.scatter(j+0.27, v.mean(), color='black', s=9, zorder=4)
        ax.scatter(j+0.27, pred, marker='D', s=13, facecolors='white', edgecolors='black', lw=.7, zorder=5)
    ax.axhline(0, color=GRID, lw=.6)
    ax.set(xlim=(-.5, 3.6), ylim=(-0.06, 0.15), xticks=range(4), ylabel=r'Shared − indep., mean $r_1$')
    ax.set_xticklabels([s_[0] for s_ in sets], fontsize=FS_TINY)
    hd = [Line2D([], [], marker='o', ls='', color=POL_C['shared_mirror'], ms=3, label='initial states'),
          Line2D([], [], marker='o', color='black', ms=2.5, lw=.9, label='mean, 95% CI'),
          Line2D([], [], marker='D', ls='', mfc='white', mec='black', ms=3, label='frozen forecast')]
    ax.legend(handles=hd, loc='upper right', fontsize=FS_TINY, handletextpad=0.2, borderaxespad=0.05)
    plabel(fig, ax, 'f', 'Shared − independent by cohort and window')

    save(fig, 'fig4_interventions')
    export(mech, 'fig4_shuffle_runs.csv'); export(fixed, 'fig4_fixed_states.csv')
    export(csv(ST/'phase_persistence_validation_20261101/analysis_real/contrasts_by_seed.csv'), 'fig4_contrasts.csv')


# ============================================================================ Figure 5
def resource_schematic(ax, w, h):
    counts = [('A', 7), ('B', 5), ('C', 4)]
    bx, by, bw, bh = 0.05, 0.30, 0.44, 0.70
    for k, (lab, n) in enumerate(counts):
        x0 = bx + k*(bw+0.10)
        ax.add_patch(Rectangle((x0, by), bw, bh, fc='#F4F6F8', ec='#9AA5AE', lw=.6))
        for i in range(n):
            ax.add_patch(Circle((x0+0.10+(i % 3)*0.12, by+0.13+(i//3)*0.15), 0.045, fc='#8C8C8C', ec='none'))
        if lab == 'A':
            ax.add_patch(Circle((x0+0.10+1*0.12, by+0.13+2*0.15), 0.048, fc='white', ec=INK, lw=.9))
            ax.text(x0+0.29, by+0.13+2*0.15, 'you', fontsize=FS_TINY, ha='left', va='center', color=INK)
        ax.text(x0+bw/2, by+bh+0.04, lab, ha='center', va='bottom', fontsize=FS, fontweight='bold')
        ax.text(x0+bw/2, by-0.05, f'{n} others', ha='center', va='top', fontsize=FS_TINY, color=MUTED)
    px, pw = 1.74, 1.50
    arrow(ax, px-0.14, by+bh*0.62, px-0.03, by+bh*0.62)
    box(ax, px, 0.68, pw, 0.60, 'Other-agent counts: A=7, B=5, C=4\nYour current resource: A\n'
        'Choose the resource with the lowest\nresulting load; no switching cost', fc='white', fs=FS_TINY, ha='left')
    ax.text(px+0.04, 1.31, 'prompt (abridged)', fontsize=FS_TINY, color=MUTED, va='bottom')
    box(ax, px, 0.06, pw, 0.46, 'All 17 agents choose at once from\nthe same counts; choices are mapped\nback to the physical resources',
        fc='#F4F6F8', fs=FS_TINY)
    arrow(ax, px+pw/2, 0.66, px+pw/2, 0.54)
    qx = px + pw + 0.16; ax.text(qx, 1.40, 'Labels shown at one step', fontsize=FS_SMALL, va='top', color=INK)
    rng = np.random.default_rng(5); perms = ['ABC', 'BCA', 'CAB', 'ACB', 'BAC', 'CBA']
    for k, (pol, name, note) in enumerate([('independent_mirror', 'Independent', 'own permutation per agent'),
                                           ('shared_mirror', 'Shared', 'one permutation for all agents')]):
        yy = 1.05 - k*0.55
        ax.text(qx, yy, name, fontsize=FS_SMALL, color=POL_C[pol], fontweight='bold', va='center')
        lab = [perms[i] for i in rng.integers(0, 6, 4)] if k == 0 else ['BCA']*4
        ax.text(qx, yy-0.17, ' '.join(lab) + ' …', fontsize=FS_TINY, family='monospace', color=POL_C[pol], va='center')
        ax.text(qx, yy-0.32, note, fontsize=FS_TINY, color=MUTED, va='center')


def fig5():
    cg = np.load(track(ST/'resource_collective_validation_20261101/analysis_real/congestion.npy'))
    rm = csv(ST/'resource_collective_validation_20261101/analysis_real/run_metrics.csv')
    comp = csv(ST/'resource_collective_validation_20261101/analysis_real/predictor_comparison.csv')
    fc = js(ST/'resource_collective_validation_20261101/predictions/point_predictions.json')
    sens = csv(ST/'resource_prediction_diagnostics_20261101/sensitivity.csv')
    base = js(ST/'resource_prediction_diagnostics_20261101/deterministic_baseline.json')
    cal = csv(ST/'resource_collective_design_20261031/analysis_real/by_input.csv')
    pc = csv(ST/'resource_update_correlation_20261101/analysis/point_contrasts.csv')
    post = csv(ST/'resource_update_correlation_20261101/analysis/posterior_intervals.csv')
    assert list(rm.policy[:2]) == ['independent', 'shared'] and cg.shape == (61, 16)
    obs = (rm[rm.policy == 'shared'].set_index('initial_id').mean_congestion
           - rm[rm.policy == 'independent'].set_index('initial_id').mean_congestion)
    obsr = rm.mean_congestion.to_numpy()
    sec_m, prim_m = np.array(fc['empirical_secondary']['mean']), np.array(fc['primary']['mean'])
    sec_c, prim_c = np.array(fc['empirical_secondary']['curve']), np.array(fc['primary']['curve'])
    assert sec_c.shape == (60, 16) and np.allclose(prim_m, rm.predicted_mean)
    rm_sec, rm_prim = np.sqrt(np.mean((sec_m-obsr)**2)), np.sqrt(np.mean((prim_m-obsr)**2))
    assert np.isclose(rm_sec, comp.set_index('model').run_mean_rmse['empirical_secondary'])
    assert np.isclose(rm_prim, comp.set_index('model').run_mean_rmse['primary'])
    PRIM_C, SEC_C = '#8A637D', '#1F78B4'

    H = 6.6; fig = plt.figure(figsize=(7.2, H))
    topA = 1 - 0.22/H; botA = topA - 1.50/H
    gsA = GridSpec(1, 2, figure=fig, left=0.075, right=0.975, top=topA, bottom=botA, width_ratios=[4.5, 1.85], wspace=0.08)
    gs = GridSpec(2, 6, figure=fig, left=0.075, right=0.975, top=botA - 0.46/H, bottom=0.07, wspace=1.35, hspace=0.62)

    # a  task schematic
    ax, w, h = canvas(fig, gsA[0, 0]); resource_schematic(ax, w, h)
    plabel(fig, ax, 'a', 'Resource allocation: 17 agents, three identical resources', dy=0.0)

    # b  calibration map
    ax = fig.add_subplot(gsA[0, 1])
    g = cal.groupby(['A', 'B', 'C'])['accuracy'].mean().reset_index(name='acc')
    X = g.B + g.C/2; Y = g.C*np.sqrt(3)/2
    cm = LinearSegmentedColormap.from_list('acc', ['#6A1B4D', '#C2577F', '#EBD9E1', '#F2F2F2'])
    sc = ax.scatter(X, Y, c=g.acc, cmap=cm, vmin=0.4, vmax=1, s=19, marker='h', lw=0)
    tri = np.array([[0, 0], [16, 0], [8, 16*np.sqrt(3)/2], [0, 0]]); ax.plot(tri[:, 0], tri[:, 1], color='#9E9E9E', lw=.5)
    ax.text(-0.4, -0.6, 'A = 16', fontsize=FS_TINY, ha='right', va='top'); ax.text(16.4, -0.6, 'B = 16', fontsize=FS_TINY, ha='left', va='top')
    ax.text(8, 14.3, 'C = 16', fontsize=FS_TINY, ha='center', va='bottom')
    for (A, B, C, xt_) in [(10, 5, 1, 2.0), (5, 10, 1, 14.0)]:
        r = cal[(cal.A == A) & (cal.B == B) & (cal.C == C) & (cal.own == 2)].iloc[0]
        x, yv = B + C/2, C*np.sqrt(3)/2
        ax.scatter(x, yv, s=40, facecolors='none', edgecolors=POL_C['independent_mirror'], lw=.9, zorder=4)
        ax.annotate(f'A={A}, B={B}, C={C}\nat C: {int(round(r.valid*r.accuracy))}/{int(r.valid)}', (x, yv), xytext=(xt_, -4.3),
                    fontsize=FS_TINY, ha='center', va='center', color=POL_C['independent_mirror'], linespacing=1.15,
                    arrowprops=dict(arrowstyle='-', lw=.5, color=POL_C['independent_mirror']))
    ax.set_xlim(-2.5, 18.5); ax.set_ylim(-6.0, 15.2); ax.set_aspect('equal'); ax.axis('off')
    cax = ax.inset_axes([0.68, 0.84, 0.30, 0.035]); cb = fig.colorbar(sc, cax=cax, ticks=[0.4, 0.7, 1.0], orientation='horizontal')
    cb.ax.tick_params(labelsize=FS_TINY-0.3, length=1.5, width=.4, pad=1); cb.outline.set_linewidth(.4)
    cb.set_label('min-load choices', fontsize=FS_TINY-0.3, labelpad=1)
    plabel(fig, ax, 'b', f'Calibration, {int(cal.valid.sum()):,} choices', dx=0.10)
    export(cal, 'fig5_calibration_by_input.csv')

    # c  observed congestion with the frozen forecasts
    ax = fig.add_subplot(gs[0, 0:4]); steps = np.arange(1, 61)
    ax.axhline(289/97, color='#BDBDBD', lw=.6, ls=(0, (2, 2))); ax.axhline(1.0, color='#BDBDBD', lw=.6, ls=(0, (2, 2)))
    ax.text(59.5, 289/97+0.05, 'all agents on one resource', fontsize=FS_TINY, color=MUTED, va='bottom', ha='right')
    ax.text(59.5, 1.0-0.05, 'balanced 6 / 6 / 5', fontsize=FS_TINY, color=MUTED, va='top', ha='right')
    for j, pol, name in [(0, 'independent_mirror', 'Independent, observed'), (1, 'shared_mirror', 'Shared, observed')]:
        for k in range(j, 16, 2): ax.plot(range(61), cg[:, k], color=POL_C[pol], alpha=.13, lw=.5)
        ax.plot(range(61), cg[:, j::2].mean(1), color=POL_C[pol], lw=1.3, label=name)
        ax.plot(steps, prim_c[:, j::2].mean(1), color=PRIM_C, lw=.8, ls=(0, (1, 1.2)), zorder=4,
                label='Forecast, smoothed table' if j == 0 else None)
        ax.plot(steps, sec_c[:, j::2].mean(1), color='black', lw=.8, ls=(0, (3, 1.6)), zorder=5,
                label='Forecast, empirical table' if j == 0 else None)
    ax.set(xlim=(0, 60), ylim=(0.6, 3.25), xticks=[0, 10, 20, 30, 40, 50, 60], yticks=[1, 1.5, 2, 2.5, 3], xlabel='Step',
           ylabel='Normalized congestion')
    ax.legend(loc='lower left', fontsize=FS_TINY, bbox_to_anchor=(0.0, 0.17), ncol=2, columnspacing=1.0, handlelength=1.8,
              borderaxespad=0.1)
    plabel(fig, ax, 'c', 'New collectives (16,320 choices) and frozen forecasts')
    export(pd.DataFrame([dict(step=t+1, run=j, policy='independent' if j % 2 == 0 else 'shared', observed=cg[t+1, j],
                              forecast_empirical=sec_c[t, j], forecast_smoothed=prim_c[t, j]) for t in range(60) for j in range(16)]),
           'fig5_congestion_and_forecasts.csv')

    # d  frozen forecasts of the 16 run means
    ax = fig.add_subplot(gs[0, 4:6])
    cols = [POL_C['independent_mirror'] if p == 'independent' else POL_C['shared_mirror'] for p in rm.policy]
    ax.plot([2.2, 3.02], [2.2, 3.02], ls=(0, (1.5, 1.5)), color=MUTED, lw=.6, zorder=1)
    ax.scatter(prim_m, obsr, s=18, facecolors='none', edgecolors=cols, lw=.8, zorder=3)
    ax.scatter(sec_m, obsr, s=14, c=cols, lw=0, zorder=4)
    ax.set(xlim=(2.2, 3.02), ylim=(2.2, 3.02), xticks=[2.25, 2.5, 2.75, 3.0], yticks=[2.25, 2.5, 2.75, 3.0],
           xlabel='Frozen forecast, run mean', ylabel='Observed run mean'); ax.set_aspect('equal')
    ax.text(0.04, 0.97, f'empirical table (filled)\nRMSE {rm_sec:.4f}\nsmoothed table (open)\nRMSE {rm_prim:.3f}',
            transform=ax.transAxes, va='top', fontsize=FS_TINY, linespacing=1.25)
    ax.annotate('', xy=(prim_m[1::2].mean()+0.01, 2.979), xytext=(sec_m[1::2].mean()-0.02, 2.979),
                arrowprops=dict(arrowstyle='-|>', lw=.5, color=PRIM_C, shrinkA=0, shrinkB=0))
    hd = [Line2D([], [], marker='o', ls='', color=POL_C['independent_mirror'], ms=3.5, label='independent'),
          Line2D([], [], marker='o', ls='', color=POL_C['shared_mirror'], ms=3.5, label='shared')]
    ax.legend(handles=hd, loc='lower right', fontsize=FS_TINY, handletextpad=0.2, borderaxespad=0.1)
    plabel(fig, ax, 'd', 'Forecasts of 16 run means')
    export(pd.DataFrame({'run': rm.run, 'policy': rm.policy, 'observed': obsr, 'forecast_empirical': sec_m,
                         'forecast_smoothed': prim_m}), 'fig5_run_forecasts.csv')

    # e  sensitivity of the forecast contrast to smoothing
    ax = fig.add_subplot(gs[1, 0:2]); al = ['alpha_0', 'alpha_0.01', 'alpha_0.05', 'alpha_0.1', 'alpha_0.5']
    sv = sens.set_index('name'); xpos = np.arange(5)
    ax.axhspan(obs.min(), obs.max(), color=POL_C['shared_mirror'], alpha=.14, lw=0)
    ax.axhline(obs.mean(), color=POL_C['shared_mirror'], lw=.9)
    ax.axhline(sv.loc['fixed_rank_greedy', 'contrast'], color='#4D4D4D', lw=.7, ls=(0, (3, 2)))
    ax.plot(xpos, sv.loc[al, 'contrast'], color='#5B6770', lw=.9, marker='o', ms=3)
    prim = float(comp[comp.model == 'primary'].predicted_contrast.iloc[0]); sec = float(comp[comp.model == 'empirical_secondary'].predicted_contrast.iloc[0])
    ax.scatter([4], [prim], marker='D', s=26, color=PRIM_C, zorder=5)
    ax.scatter([0], [sec], marker='D', s=26, color=SEC_C, zorder=5)
    rv = sv.loc['primary_revert_concentrated', 'contrast']
    ax.scatter([4], [rv], marker='^', s=24, facecolors='white', edgecolors=PRIM_C, lw=.8, zorder=5)
    ax.text(4.35, obs.max()+0.008, 'observed (range of 8 pairs)', fontsize=FS_TINY, color=POL_C['shared_mirror'], ha='right', va='bottom')
    ax.text(4.35, sv.loc['fixed_rank_greedy', 'contrast']-0.012, 'greedy rule', fontsize=FS_TINY, color='#4D4D4D', ha='right', va='top')
    ax.text(0.12, sec-0.03, 'frozen\nempirical', fontsize=FS_TINY, color=SEC_C, ha='left', va='top', linespacing=1.1)
    ax.text(3.75, prim, 'frozen\nsmoothed', fontsize=FS_TINY, color=PRIM_C, ha='right', va='center', linespacing=1.1)
    ax.text(4.0, rv-0.014, '3 rows\nreverted', fontsize=FS_TINY, color=PRIM_C, ha='center', va='top', linespacing=1.1)
    ax.text(1.9, 0.60, 'post hoc\nsimulations', fontsize=FS_TINY, color='#5B6770', ha='center', va='top', linespacing=1.1)
    ax.set(xticks=xpos, ylim=(0.42, 0.75), xlim=(-.4, 4.4), xlabel=r'Pseudo-count $\alpha$', ylabel='Shared − indep. congestion')
    ax.set_xticklabels(['0', '.01', '.05', '.1', '.5'])
    plabel(fig, ax, 'e', 'Sensitivity to smoothing')
    export(sens, 'fig5_smoothing_sensitivity.csv'); export(comp, 'fig5_resource_forecasts.csv')

    # f  deterministic comparator
    ax = fig.add_subplot(gs[1, 2:4]); bm = np.array(base['mean'])
    ax.scatter(bm, obsr, c=cols, s=14, lw=0, zorder=3)
    for i0 in range(0, len(rm), 2):
        ax.plot(bm[i0:i0+2], obsr[i0:i0+2], color='#C8C8C8', lw=.5, zorder=1)
    ax.plot([2.2, 3.02], [2.2, 3.02], ls=(0, (1.5, 1.5)), color=MUTED, lw=.6)
    ax.set(xlim=(2.2, 3.02), ylim=(2.2, 3.02), xticks=[2.25, 2.5, 2.75, 3.0], yticks=[2.25, 2.5, 2.75, 3.0],
           xlabel='Greedy rule, run mean', ylabel='Observed run mean'); ax.set_aspect('equal')
    ax.text(0.04, 0.97, f'all 16,048 choices at\nsteps 2–60 match\nRMSE {base["rmse"]:.3f}', transform=ax.transAxes,
            va='top', fontsize=FS_TINY, linespacing=1.25)
    ax.text(0.97, 0.05, f'shared − indep.\nobserved {obs.mean():.3f}\ngreedy {base["contrast"]:.3f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=FS_TINY, linespacing=1.25)
    plabel(fig, ax, 'f', 'A simple rule reproduces the crowding')

    # g  dependence on simultaneous updating (model calculations)
    ax = fig.add_subplot(gs[1, 4:6])
    ests = [('MLE', 'Empirical table', SEC_C, 'o'), ('Dirichlet_half', r'Smoothed ($\alpha=0.5$)', PRIM_C, 's'),
            ('least_load_fixed_ties', 'Greedy, fixed ties', '#4D4D4D', '^'), ('least_load_random_ties', 'Greedy, random ties', '#9E9E9E', 'v'),
            ('uniform', 'Uniform random', '#C8C8C8', 'D')]
    xs = {1: 0, 4: 1, 17: 2}; sub = pc[pc.contrast == 'shared-independent']
    for k, (key, lab, col, mk) in enumerate(ests):
        d = sub[sub.estimator == key].sort_values('m'); x = np.array([xs[m] for m in d.m]) + (k-2)*0.075
        ax.plot(x, d.mean_difference, color=col, lw=.7, marker=mk, ms=3.0, label=lab, zorder=3)
        if key == 'Dirichlet_half':
            pp = post[post.contrast == 'shared-independent'].sort_values('m')
            ax.errorbar(np.array([xs[m] for m in pp.m]) + (k-2)*0.075, pp.mean_difference_median,
                        yerr=[pp.mean_difference_median-pp.mean_difference_low, pp.mean_difference_high-pp.mean_difference_median],
                        fmt='none', ecolor=col, elinewidth=.7, capsize=1.5, zorder=2)
    ax.scatter(2.3, obs.mean(), marker='*', s=50, color=POL_C['shared_mirror'], zorder=5, label='Observed collectives', lw=0)
    ax.axhline(0, color=GRID, lw=.6)
    ax.set(xlim=(-.3, 2.5), ylim=(-0.04, 0.82), xticks=[0, 1, 2], ylabel='Shared − indep. congestion')
    ax.set_xticklabels(['1', '4', '17']); ax.set_xlabel(r'Agents updating at once, $m$')
    ax.legend(loc='upper left', fontsize=FS_TINY-0.3, handlelength=1.3, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.3)
    plabel(fig, ax, 'g', 'Updating at once (model calculations)')
    export(sub, 'fig5_update_size_contrasts.csv')

    save(fig, 'fig5_resource_allocation'); export(rm, 'fig5_resource_runs.csv')


# ============================================================================ Supplementary
def supp():
    tra = csv(ST/'phase_persistence_diagnostics_20261101/B_trajectories.csv'); q6 = tra[tra.model == 'quadratic6']
    for page, pol in enumerate(POL[:3], 1):
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.4), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.12, wspace=0.12, hspace=0.38)
        for ax, (seed, d) in zip(axes.ravel(), q6[q6.controller == pol].groupby('physical_seed')):
            ax.fill_between(d.step, d.process_p025, d.process_p975, color='#D6DEE5', lw=0)
            ax.plot(d.step, d.observed, color=POL_C[pol], lw=1.0); ax.plot(d.step, d.predicted, color='black', ls=(0, (3, 1.6)), lw=.75)
            rm = np.sqrt(np.mean(d[d.step > 0].residual**2))
            ax.set_title(f'seed {seed}', fontsize=FS_SMALL, loc='left'); ax.text(0.97, 0.05, f'RMSE {rm:.3f}', transform=ax.transAxes,
                                                                                 ha='right', fontsize=FS_TINY)
            ax.set(ylim=(0, 1), xlim=(0, 200), xticks=[0, 100, 200], yticks=[0, .5, 1])
        for ax in axes[1]: ax.set_xlabel('Step')
        for ax in axes[:, 0]: ax.set_ylabel(r'$r_1$')
        fig.suptitle(f'{POL_N[pol]}: observations (solid), frozen forecast (dashed), 95% simulation band', fontsize=FS, x=0.07, ha='left')
        save(fig, f'figS{page}_trajectories')
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6)); fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.1, wspace=0.28, hspace=0.5)
    ends = []
    for j, (fam, base) in enumerate([('GPT', ROOT/'results_frozen/analysis/complex_kernel_transmutation'), ('Claude', ROOT/'results_frozen/analysis/claude_kappa_v2')]):
        d = csv(base/'complex_endpoints.csv'); d['provider'] = fam; assert np.allclose(d.R2, np.hypot(d.a2, d.b2)); ends.append(d)
        for i, (col, yl) in enumerate([('a1', r'First sine coefficient $a_1$'), ('R2', r'Second-harmonic amplitude $R_2$')]):
            ax = axes[i, j]
            for rep in REPS:
                sub = d[d.representation == rep]
                for blk, z in sub.groupby('block'):
                    z = z.sort_values('concentration'); ax.plot(z.concentration, z[col], color=REP_C[rep], alpha=.35, lw=.6)
                m = sub.groupby('concentration')[col].mean(); ax.plot(m.index, m.values, 'o-', color=REP_C[rep], ms=3, lw=1.0, label=REP_N[rep])
            ax.axhline(0, color=GRID, lw=.5); ax.set(xlabel=r'Concentration $\kappa$', ylabel=yl, xticks=[2, 4, 6, 9, 12])
            plabel(fig, ax, 'abcd'[2*i+j], fam)
            if i == 0 and j == 0: ax.legend(fontsize=FS_TINY, loc='lower left')
    save(fig, 'figS4_concentration'); export(pd.concat(ends), 'figS4_concentration_coefficients.csv')


if __name__ == '__main__':
    fig2(); fig3(); fig4(); fig5(); supp()
    for p, h in INPUTS.items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h, p
    (HERE/'FIGURE_INPUTS.json').write_text(json.dumps(INPUTS, indent=2), encoding='utf-8')
    print('Figures 2-5 and S1-S4 (version 2) written; source hashes unchanged.')
