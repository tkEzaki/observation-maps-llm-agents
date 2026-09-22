"""Science Advances figures, version 4: Fig. 1 and Fig. 2 in the shared journal style.

Same panels and data as version 3. Changes: Arial-metric type and STIX Sans maths
shared with Figs. 3-5 (analysis.natcomm_figures_opus.style), no text below 6 pt,
native bold capital panel letters (9 pt) with regular running titles, the shared
Okabe-Ito encoding colours and action colours, and M/C/I abbreviations.

Run from the repository root:  python -m analysis.sciadv_revision.build_figures_v4

No provider calls. Every number drawn here is read from recorded trajectories,
recorded stimuli and the frozen analysis tables that the previous figure
builders already used. Panel a of Fig. 1 encodes a recorded initial state with
the experiment's own encoders (circlemap), so the printed payload lines are the
strings the models actually received for that state.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

from analysis.sciadv_revision.build_figures import (ROOT, OUT, PAIRED, R2, REPS, NAMES,
    COLORS, setup, save, load, persistence, design_matrix)
from analysis.natcomm_figures_opus import style as S

OUT4 = ROOT/'figures/sciadv_v4'


def save(fig, name):
    """Write to figures/sciadv_v4 so the version-3 exports stay untouched."""
    OUT4.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT4/(name+'.pdf'))
    fig.savefig(OUT4/(name+'.png'), dpi=300)
    plt.close(fig)
from circlemap.observation import relative_phase_histogram
from circlemap.representations import build_representation_prompt_from_histogram

GPT_SESSIONS = {
    'moments_m1_m3': ROOT/'runs/stage_c/matched-rep-collective-v0.1-moments_ed41bff41a9a/20260724T065125Z',
    'centers_24_standard': ROOT/'runs/stage_c/matched-rep-collective-v0.1-centers_65c4f06d18a1/20260724T065125Z',
    'intervals_24_decimal6': ROOT/'runs/stage_c/matched-rep-collective-v0.1-intervals_1d533f018b96/20260724T065128Z',
}
MICRO = {'GPT': ROOT/'analysis/complex_kernel_transmutation', 'Claude': ROOT/'runs/claude_kappa_v2/anthropic/analysis'}
EPS = ROOT/'analysis/complex_kernel_antipodal_weight/epsilon_trajectories.csv'
ATLAS = ROOT/'analysis/stage_b_offline_p3_p4/phenotype_atlas.csv'
BLOCKS = [('block_1', 'o', '-'), ('block_2', 's', '--')]
# Shared with Fig. 5E: magenta retard, grey stay, navy advance (no clash with the encoding greens).
ACTION_COLORS = {'retard': S.ACTION['p_minus'], 'stay': S.ACTION['p_zero'], 'advance': S.ACTION['p_plus']}
# Encoding colours exactly as in Figs. 3-5 (Okabe-Ito).
COLORS = [S.REP[r] for r in REPS]
ABBR = ['M', 'C', 'I']
MONO = 'Liberation Mono'
FS_MIN = 6.0
FS_LETTER = 9.0
INK = '#222222'
GREY = '#8A939B'
SNAP_SEED = 0
SNAP_K = 0.15
SNAP_TIMES = [0, 15, 40, 100]


def _check_sessions():
    meta = json.loads((PAIRED/'meta.json').read_text())
    for s in meta['sessions']:
        p = ROOT.joinpath(*str(s['session']).replace('\\', '/').split('/'))
        assert p == GPT_SESSIONS[s['representation']], (p, s['representation'])


def phases(family, rep, k, seed):
    if family == 'GPT':
        p = GPT_SESSIONS[rep]/f'N17_K+{k:g}_s{seed}'/'phases.npy'
    else:
        p = R2/f'N17_K+{k:g}_s{seed}'/rep/'phases.npy'
    a = np.load(p)
    assert a.shape == (101, 17), (p, a.shape)
    return a


def r1_of(ph):
    return float(np.abs(np.exp(1j*ph).mean()))


def ring(ax, ph, color, focal=None, r1_label=True, lw=.6, ms=2.6):
    ax.add_patch(Circle((0, 0), 1, fill=False, color='#B8BEC4', lw=lw))
    ax.scatter(np.cos(ph), np.sin(ph), s=ms**2, c=color, linewidths=0, zorder=3)
    if focal is not None:
        ax.plot(np.cos(ph[focal]), np.sin(ph[focal]), 'o', ms=ms*2.1, mfc='white', mec=color, mew=1.1, zorder=4)
    if r1_label:
        ax.text(0, 0, f'{r1_of(ph):.2f}', ha='center', va='center', fontsize=FS_MIN, color='#444444')
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_axis_off()


def label_at(fig, x_in, y_in, letter, text=''):
    """Bold capital letter + regular running title; x, y in inches from the top-left."""
    W, H = fig.get_size_inches()
    fig.text(x_in/W, 1-y_in/H, letter.upper(), fontsize=FS_LETTER, fontweight='bold', ha='left', va='baseline', color=S.INK)
    if text:
        fig.text((x_in+S.LETTER_ADVANCE_MM*S.MM)/W, 1-y_in/H, text, fontsize=S.FS_TITLE, ha='left', va='baseline', color='#2B2B2B')


def panel_title(ax, letter, text, **_):
    """Panel letter on the shared optical column used by Figs. 3-5 (8.6 mm left of the axes)."""
    fig = ax.figure
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    label_at(fig, pos.x0*W + S.LETTER_DX_MM*S.MM, (1-pos.y1)*H - S.LETTER_DY_MM*S.MM, letter, text)


def color_abbr_ticks(ax):
    for lab in ax.get_xticklabels():
        if lab.get_text() in ABBR:
            lab.set_color(COLORS[ABBR.index(lab.get_text())])


# ============================================================================
# Figure 1
# ============================================================================

def fig1(seeds, series, ep):
    _check_sessions()
    fig = plt.figure(figsize=(7.1, 8.9))
    W, H = fig.get_size_inches()
    def axes_in(x, y, w, h):  # inches from top-left
        return fig.add_axes([x/W, 1-(y+h)/H, w/W, h/H])

    # ---- a  schematic built from a recorded initial state ------------------
    ph0 = phases('GPT', REPS[0], SNAP_K, SNAP_SEED)[0]
    for rep in REPS[1:]:  # initial state identical across encodings
        assert np.allclose(phases('GPT', rep, SNAP_K, SNAP_SEED)[0], ph0)
    hist = relative_phase_histogram(ph0, 0)
    prompts = {rep: build_representation_prompt_from_histogram(rep, hist) for rep in REPS}
    def data_lines(rep, n):
        text = prompts[rep]
        i = text.find('Relative phase distribution')
        lines = text[i:].splitlines()[1:1+n]
        return lines
    axA = axes_in(0.05, 0.12, 7.0, 2.05); axA.set_axis_off()
    axA.set_xlim(0, 7.0); axA.set_ylim(0, 2.05)
    label_at(fig, 0.21, 0.25, 'A', 'One recorded state, three descriptions, one shared engine')
    # agents on the circle
    axR = axes_in(0.12, 0.55, 1.1, 1.1)
    ring(axR, ph0, '#555555', focal=0, r1_label=False, ms=3.0)
    axR.text(0, -1.2, 'N = 17 agents on a circle\n(recorded initial state, t = 0)', ha='center', va='top', fontsize=FS_MIN, color='#444444')
    axR.annotate('focal agent', (np.cos(ph0[0]), np.sin(ph0[0])), (-1.2, 1.28), fontsize=FS_MIN, color=INK,
                 arrowprops=dict(arrowstyle='-', lw=.5, color='#666666'), ha='left', va='bottom')
    # its relative-phase field, as the 24-bin histogram the encoders start from
    axH = fig.add_axes([1.55/W, 1-(0.6+1.0)/H, 1.0/W, 1.0/H], projection='polar')
    centers = 0.5*(hist.edges[:-1]+hist.edges[1:])
    axH.bar(centers, hist.fractions, width=2*np.pi/24, bottom=0.0, color='#7A8791', edgecolor='white', lw=.3)
    axH.set_theta_zero_location('E'); axH.set_theta_direction(1)
    axH.set_yticks([]); axH.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    axH.set_xticklabels(['', '', '', ''])
    axH.tick_params(pad=-4)
    axH.spines['polar'].set_linewidth(.5)
    axH.set_ylim(0, max(hist.fractions)*1.15)
    axH.grid(False)
    axA.text(2.05, 0.42, 'relative phases of the 16 peers\nin 24 bins (the same field\nfor all three encodings)', ha='center', va='top', fontsize=FS_MIN, color='#444444')
    # the three payloads
    x0, y_top = 2.70, 1.72
    box_h = 0.42
    for j, rep in enumerate(REPS):
        y = y_top - j*(box_h+0.05)
        axA.add_patch(FancyBboxPatch((x0, y-box_h), 2.80, box_h, boxstyle='round,pad=0.02', fc='#F6F7F9', ec=COLORS[j], lw=.8))
        axA.text(x0+0.06, y-0.09, NAMES[j], color=COLORS[j], fontsize=7.0, fontweight='bold', va='center')
        n = 2 if rep == REPS[0] else 2
        lines = data_lines(rep, n)
        axA.text(x0+0.64, y-0.09, lines[0], fontfamily=MONO, fontsize=FS_MIN, va='center', color=INK)
        axA.text(x0+0.64, y-0.21, lines[1], fontfamily=MONO, fontsize=FS_MIN, va='center', color=INK)
        axA.text(x0+0.64, y-0.33, '...  (6 values in total)' if j == 0 else '...  (24 bins in total)', fontfamily=MONO, fontsize=FS_MIN, va='center', color='#777777')
    axA.text(x0+1.40, y_top+0.04, 'the text the model reads (first two data lines)', ha='center', va='bottom', fontsize=6, color='#444444')
    # model and action
    axA.add_patch(FancyBboxPatch((5.72, 0.88), 1.2, 0.72, boxstyle='round,pad=0.02', fc='#F6F7F9', ec='#9AA4AE', lw=.8))
    axA.text(6.32, 1.42, 'GPT or Claude', ha='center', va='center', fontsize=6.8, fontweight='bold', color=INK)
    axA.text(6.32, 1.20, 'no memory, no goal', ha='center', va='center', fontsize=FS_MIN, color='#555555')
    axA.text(6.32, 1.00, r'$f_i \in \{-1,\ 0,\ +1\}$', ha='center', va='center', fontsize=7, color=INK)
    axA.text(6.32, 1.66, 'retard / stay / advance', ha='center', va='bottom', fontsize=6, color='#444444')
    # arrows
    for (xa, xb, yy) in [(1.28, 1.50, 1.10), (2.50, 2.68, 1.10), (5.52, 5.70, 1.10)]:
        axA.annotate('', (xb, yy), (xa, yy), arrowprops=dict(arrowstyle='->', lw=.9, color=INK))
    # engine loop back to the circle
    axA.plot([6.32, 6.32, 0.62, 0.62], [0.88, 0.16, 0.16, 0.30], color=INK, lw=.9)
    axA.annotate('', (0.62, 0.36), (0.62, 0.30), arrowprops=dict(arrowstyle='->', lw=.9, color=INK))
    axA.text(4.6, 0.16, r'  shared engine:  $x_i(t+1)=x_i(t)+\omega_i+K\,f_i(t)$  ', ha='center', va='center', fontsize=6.8, color=INK,
             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#9AA4AE', lw=.7), zorder=5)
    axA.text(6.95, 0.55, 'only the\ndescription\nchanges', ha='right', va='center', fontsize=6, color='#444444', style='italic')

    # ---- b, c  phase snapshots -------------------------------------------
    snap_top = 2.45
    row_h, col_w = 0.62, 0.62
    for fi, family in enumerate(['GPT', 'Claude']):
        xoff = 0.78 + fi*3.55
        label_at(fig, 0.21 + fi*3.55, snap_top - 0.04, 'BC'[fi], f'{family}, K = {SNAP_K:g}, seed {SNAP_SEED}: phase configurations')
        for j, rep in enumerate(REPS):
            P = phases(family, rep, SNAP_K, SNAP_SEED)
            assert np.isclose(r1_of(P[-1]), series[family, rep, SNAP_K, SNAP_SEED][-1], atol=1e-6)
            fig.text((xoff-0.05)/W, 1-(snap_top+0.12+j*row_h+row_h/2)/H, NAMES[j], color=COLORS[j], fontsize=7, ha='right', va='center', fontweight='bold')
            for ti, t in enumerate(SNAP_TIMES):
                ax = axes_in(xoff+ti*col_w, snap_top+0.12+j*row_h, col_w-0.04, row_h-0.04)
                ring(ax, P[t], COLORS[j], focal=None, r1_label=True, ms=2.0)
                if j == 0:
                    ax.text(0, 1.38, f't = {t}', ha='center', va='bottom', fontsize=6.2, color='#444444')
    fig.text(0.5, 1-(snap_top+0.12+3*row_h+0.02)/H, r'number in each ring: polar order $r_1$ at that step', ha='center', va='top', fontsize=FS_MIN, color='#666666')

    # ---- d, e  trajectories at K = 0.15 ------------------------------------
    traj_top = 4.75
    handles = []
    for fi, family in enumerate(['GPT', 'Claude']):
        ax = axes_in(0.55+fi*3.55, traj_top, 2.75, 1.32)
        for j, rep in enumerate(REPS):
            stack = np.array([series[family, rep, SNAP_K, s] for s in seeds])
            for y in stack:
                ax.plot(range(101), y, color=COLORS[j], alpha=.22, lw=.55)
            line, = ax.plot(range(101), stack.mean(0), color=COLORS[j], lw=1.9, label=NAMES[j])
            if fi == 0: handles.append(line)
        ax.axhline(.9, color='#777777', lw=.6, ls=':', zorder=0)
        ax.text(100, .905, r'$r_1=0.9$', ha='right', va='bottom', fontsize=FS_MIN, color='#777777')
        ax.set(xlim=(0, 100), ylim=(0, 1.04), xticks=[0, 50, 100], yticks=[0, .5, 1], xlabel='Step t')
        if fi == 0: ax.set_ylabel(r'Polar order $r_1$')
        panel_title(ax, 'de'[fi], f'{family}, K = {SNAP_K:g}: six matched seeds')
    ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=6.5, ncol=1, handlelength=1.6)

    # ---- f  endpoint polar order per seed, both couplings ---------------------
    f_top = 6.55
    ax = axes_in(0.55, f_top, 3.95, 1.55)
    groups = [(fam, k) for fam in ['GPT', 'Claude'] for k in [.08, .15]]
    xpos = {}
    for gi, (fam, k) in enumerate(groups):
        base = gi*4.0
        finals = np.array([[series[fam, rep, k, s][-1] for rep in REPS] for s in seeds])
        for si in range(len(seeds)):
            ax.plot(base+np.arange(3), finals[si], color='#C9CED3', lw=.6, zorder=1)
        for j in range(3):
            xs = base+j+np.linspace(-.18, .18, len(seeds))
            ax.scatter(xs, finals[:, j], s=14, color=COLORS[j], zorder=3, linewidths=0)
            n = int((finals[:, j] >= .9).sum())
            ax.text(base+j, 1.06, f'{n}/6', ha='center', va='bottom', fontsize=6.3, color=COLORS[j], fontweight='bold')
            xpos[(fam, k, j)] = base+j
        ax.text(base+1, -0.20, f'{fam}, K = {k:g}', ha='center', va='top', fontsize=6.4, color=INK)
    ax.axhline(.9, color='#777777', lw=.6, ls=':', zorder=0)
    for gi in range(1, 4):
        ax.axvline(gi*4.0-1.0, color='#E3E6E9', lw=.6, zorder=0)
    ax.set(xlim=(-0.7, 14.7), ylim=(0, 1.2), yticks=[0, .5, .9, 1], xticks=[xpos[g] for g in sorted(xpos)],
           xticklabels=ABBR*4, ylabel=r'Final polar order $r_1(T)$')
    ax.tick_params(axis='x', length=0, labelsize=6.5)
    color_abbr_ticks(ax)
    panel_title(ax, 'f', r'Final order per seed; counts with $r_1(T)\geq0.9$')

    # ---- g  two-cluster tendency Q2 -----------------------------------------
    ax = axes_in(5.05, f_top, 1.95, 1.55)
    q2_all = []
    for fi, fam in enumerate(['GPT', 'Claude']):
        base = fi*4.0
        q2 = np.array([[ep_q2(ep, fam, rep, SNAP_K, s, series) for rep in REPS] for s in seeds])
        q2_all.extend(q2.ravel())
        for si in range(len(seeds)):
            ax.plot(base+np.arange(3), q2[si], color='#C9CED3', lw=.6, zorder=1)
        for j in range(3):
            xs = base+j+np.linspace(-.18, .18, len(seeds))
            ax.scatter(xs, q2[:, j], s=14, color=COLORS[j], zorder=3, linewidths=0)
        ax.text(base+1, -0.165, f'{fam}, K = {SNAP_K:g}', ha='center', va='top', fontsize=6.4, color=INK, transform=ax.get_xaxis_transform())
    ax.axhline(0, color='#777777', lw=.6, ls=':', zorder=0)
    ax.set(xlim=(-0.7, 6.7), ylim=(min(q2_all)-0.06, max(q2_all)+0.06), xticks=[0, 1, 2, 4, 5, 6], xticklabels=ABBR*2, ylabel=r'$Q_2=r_2-r_1$ at $t=T$')
    ax.tick_params(axis='x', length=0, labelsize=6.5)
    color_abbr_ticks(ax)
    panel_title(ax, 'g', 'Two-cluster tendency')

    # ---- h  K = 0 control -----------------------------------------------------
    h_top = 8.0 - 0.0
    # (placed to the right of the snapshot rows would crowd; use the strip below f/g instead)
    # verify exact identity across encodings from recorded trajectories
    zero = []
    for j in range(3):
        ts = pd.read_csv(PAIRED/f'_tmp_session_{j}/trajectory_harmonics_timeseries.csv')
        a = ts[np.isclose(ts.coupling, 0) & ts.seed_index.isin(seeds)].sort_values(['seed_index', 't'])
        assert len(a) == 606
        zero.append(a[['r1', 'r2', 'r3']].to_numpy())
    assert all(np.array_equal(zero[0], z) for z in zero[1:])
    control = json.loads((R2/'r2_inference/k0_control.json').read_text())
    assert control['pass'] and control['max_abs_diff_rm'] == 0
    fig.text(0.5, 1-(f_top+1.55+0.5)/H,
             'K = 0 control: with the coupling term removed, the three encodings produced identical $r_1,r_2,r_3$ trajectories '
             'from every shared seed in both models\n(exact equality of the recorded series), although their action distributions still differed.',
             ha='center', va='top', fontsize=6.3, color='#344454')
    save(fig, 'fig1_collective_phases')


def ep_q2(ep, fam, rep, k, seed, series):
    """Final Q2 = r2 - r1 from recorded trajectories (GPT: paired tables; Claude: run_meta)."""
    if fam == 'GPT':
        j = REPS.index(rep)
        ts = pd.read_csv(PAIRED/f'_tmp_session_{j}/trajectory_harmonics_timeseries.csv')
        a = ts[np.isclose(ts.coupling, k) & (ts.seed_index == seed)].sort_values('t').iloc[-1]
        assert np.isclose(a.r1, series[fam, rep, k, seed][-1])
        return float(a.r2 - a.r1)
    meta = json.loads((R2/f'N17_K+{k:g}_s{seed}'/rep/'run_meta.json').read_text())
    assert np.isclose(meta['r1_series'][-1], series[fam, rep, k, seed][-1])
    return float(meta['r2_series'][-1] - meta['r1_series'][-1])


# ============================================================================
# Figure 2
# ============================================================================

def _curves(family):
    c = pd.read_csv(MICRO[family]/'offset_action_curves.csv')
    e = pd.read_csv(MICRO[family]/'complex_endpoints.csv')
    return c, e


def _fit(sub):
    x = sub.offset_radians.to_numpy(); y = sub.g.to_numpy()
    coef = np.linalg.lstsq(design_matrix(x), y, rcond=None)[0]
    xx = np.linspace(-np.pi, np.pi, 721)
    return x, y, xx, design_matrix(xx) @ coef, coef


def fig2():
    fig = plt.figure(figsize=(7.1, 7.55))
    W, H = fig.get_size_inches()
    def axes_in(x, y, w, h, **kw):
        return fig.add_axes([x/W, 1-(y+h)/H, w/W, h/H], **kw)
    data = {fam: _curves(fam) for fam in ['GPT', 'Claude']}
    for fam, (c, e) in data.items():
        for (block, rep, k), cells in c.groupby(['block', 'representation', 'concentration']):
            if rep not in REPS: continue
            assert len(cells) == 36
            coef = np.linalg.lstsq(design_matrix(cells.offset_radians.to_numpy()), cells.g, rcond=None)[0]
            frozen = e[(e.block == block) & (e.representation == rep) & (e.concentration == k)].iloc[0]
            assert np.allclose(coef[:5], frozen[['a0', 'a1', 'b1', 'a2', 'b2']].to_numpy(dtype=float), atol=1e-8)
    gpt_offsets = data['GPT'][0][(data['GPT'][0].concentration == 12) & (data['GPT'][0].block == 'block_1')].groupby('offset_index').offset_radians.first()
    cl_offsets = data['Claude'][0][(data['Claude'][0].concentration == 12) & (data['Claude'][0].block == 'block_1')].groupby('offset_index').offset_radians.first()
    assert np.allclose(gpt_offsets.sort_index().to_numpy(), cl_offsets.sort_index().to_numpy())

    # ---- a  protocol: one field, rotated ----------------------------------
    label_at(fig, 0.21, 0.22, 'A', 'Controlled-field protocol: one peer distribution rotated around the focal agent')
    kappa = 12.0
    for i, delta in enumerate([-np.pi/2, 0.0, np.pi/2]):
        ax = axes_in(0.3+i*1.05, 0.55, 0.8, 0.8, projection='polar')
        phi = np.linspace(-np.pi, np.pi, 361)
        dens = np.exp(kappa*np.cos(phi-delta)); dens = 0.85*dens/dens.max()
        ax.fill_between(phi, 0, dens, color='#7A8791', alpha=.55, lw=0)
        ax.plot(phi, dens, color='#5A6772', lw=.6)
        ax.plot([0], [1.0], 'o', ms=5, mfc='white', mec=INK, mew=1.0, zorder=5)
        ax.set_theta_zero_location('E'); ax.set_ylim(0, 1.05); ax.set_yticks([])
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2]); ax.set_xticklabels(['', '', '', ''])
        ax.tick_params(pad=-3); ax.grid(False); ax.spines['polar'].set_linewidth(.5)
        ax.set_title({0: r'$\delta=-\pi/2$' + '\npeers behind', 1: r'$\delta=0$' + '\npeers at the agent', 2: r'$\delta=+\pi/2$' + '\npeers ahead'}[i], fontsize=FS_MIN, pad=1)
    axT = axes_in(3.55, 0.3, 3.45, 1.05); axT.set_axis_off(); axT.set_xlim(0, 1); axT.set_ylim(0, 1)
    axT.text(0, 0.98, 'A von Mises peer field (open circle: focal agent) is rotated by $\\delta$ (36 values)\n'
             'at concentration $\\kappa$ (5 values), written under each encoding, and shown to\n'
             'the model 24 times in each of two acquisition blocks (25,920 responses per\n'
             'model family).', fontsize=6.1, va='top', color=INK, linespacing=1.35)
    axT.text(0, 0.42, 'Mean action $g(\\delta)=p(+1\\mid\\delta)-p(-1\\mid\\delta)$ is the interaction rule.\n'
             '$g>0$ with peers ahead and $g<0$ with peers behind is attraction toward\n'
             'the group; Fourier components $a_1\\sin\\delta+b_1\\cos\\delta+\\ldots$ summarize it.',
             fontsize=6.1, va='top', color=INK, linespacing=1.35)

    # ---- b  full action distributions on one rotated field ------------------
    target = -0.252
    oi = int((gpt_offsets - target).abs().idxmin()); delta_b = float(gpt_offsets.loc[oi])
    ax = axes_in(0.55, 1.65, 2.55, 1.3)
    xt = []; xl = []
    for fi, fam in enumerate(['GPT', 'Claude']):
        c, _ = data[fam]
        for j, rep in enumerate(REPS):
            sub = c[(c.representation == rep) & (c.concentration == 12) & (c.offset_index == oi)]
            assert len(sub) == 2  # two blocks
            probs = sub[['p_retard', 'p_stay', 'p_advance']].mean().to_numpy()
            x = fi*3.6 + j
            bottom = 0
            for val, key in zip(probs, ['retard', 'stay', 'advance']):
                ax.bar(x, val, bottom=bottom, width=.72, color=ACTION_COLORS[key], edgecolor='#8C8C8C', lw=.4)
                bottom += val
            xt.append(x); xl.append(ABBR[j])
        ax.text(fi*3.6+1, -0.17, fam, ha='center', va='top', fontsize=6.8, color=INK)
    ax.set(xlim=(-0.7, 6.3), ylim=(0, 1.0), yticks=[0, .5, 1], xticks=xt, xticklabels=xl, ylabel='Action probability')
    ax.tick_params(axis='x', length=0, labelsize=6.5)
    color_abbr_ticks(ax)
    ax.spines['bottom'].set_visible(False)
    panel_title(ax, 'b', f'One field ($\\kappa=12$, $\\delta={delta_b:+.2f}$), six rules', fontsize=8)
    leg = [Line2D([], [], marker='s', ls='', ms=6, color=ACTION_COLORS[k], label=k) for k in ['advance', 'stay', 'retard']]
    ax.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False, fontsize=6, handletextpad=.3, columnspacing=.8)

    # ---- c, d  rotation curves at kappa = 12, three encodings overlaid --------
    zero_cross = {}
    for fi, fam in enumerate(['GPT', 'Claude']):
        ax = axes_in(3.65+fi*1.75, 1.65, 1.55, 1.3)
        c, e = data[fam]
        for j, rep in enumerate(REPS):
            for block, marker, ls in BLOCKS:
                sub = c[(c.representation == rep) & (c.concentration == 12) & (c.block == block)].sort_values('offset_radians')
                x, y, xx, yy, coef = _fit(sub)
                ax.scatter(x, y, s=2.5, marker=marker, color=COLORS[j], alpha=.22, linewidths=0, zorder=2)
                ax.plot(xx, np.clip(yy, -1, 1), ls=ls, lw=1.0 if block == 'block_1' else .7, color=COLORS[j], zorder=3)
                if block == 'block_1':
                    # upward zero crossing nearest delta = 0 of the fitted curve
                    s = np.sign(yy); idx = np.where((s[:-1] <= 0) & (s[1:] > 0))[0]
                    if len(idx):
                        z = xx[idx[np.argmin(np.abs(xx[idx]))]]
                        zero_cross[fam, rep] = z
                        ax.plot([z], [-1.08], marker='^', ms=3.5, color=COLORS[j], clip_on=False, zorder=4)
        ax.axhline(0, color='#BBBBBB', lw=.5, zorder=0); ax.axvline(0, color='#DDDDDD', lw=.5, zorder=0)
        ax.set(xlim=(-np.pi, np.pi), ylim=(-1.12, 1.12), yticks=[-1, 0, 1], xticks=[-np.pi, 0, np.pi], xticklabels=['−π', '0', 'π'], xlabel=r'Rotation $\delta$')
        if fi == 0: ax.set_ylabel(r'Mean action $g(\delta)$')
        panel_title(ax, 'cd'[fi], f'{fam}, $\\kappa=12$', fontsize=8)
    fig.text(5.4/W, 1-3.33/H, 'triangles: where the block-1 fit switches from retard to advance nearest $\\delta=0$',
             ha='center', va='top', fontsize=FS_MIN, color='#666666')

    # ---- e, f  first-harmonic vector C1 = a1 + i b1 across kappa ---------------
    top_ef = 3.68
    for fi, fam in enumerate(['GPT', 'Claude']):
        ax = axes_in(0.55+fi*1.85, top_ef, 1.55, 1.55)
        c, e = data[fam]
        for j, rep in enumerate(REPS):
            for block, marker, ls in BLOCKS:
                sub = e[(e.representation == rep) & (e.block == block)].sort_values('concentration')
                assert len(sub) == 5
                ax.plot(sub.a1, sub.b1, ls=ls, lw=.9, color=COLORS[j], alpha=.9 if block == 'block_1' else .55, zorder=3)
                ax.scatter(sub.a1, sub.b1, s=[6, 9, 12, 16, 22], marker=marker, color=COLORS[j], zorder=4, linewidths=0)
                pass
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color='#DDDDDD', lw=.5, zorder=0)
        ax.axhline(0, color='#CCCCCC', lw=.5, zorder=0); ax.axvline(0, color='#CCCCCC', lw=.5, zorder=0)
        ax.set(xlim=(-0.7, 1.5), ylim=(-0.5, 1.1), xlabel=r'$a_1$ (odd: attraction $>0$)', ylabel=r'$b_1$ (even: shift)' if fi == 0 else '')
        ax.set_aspect('equal'); ax.set_anchor('N'); ax.apply_aspect()   # top-aligned so letters line up with G
        ax.text(-0.66, -0.46, 'attraction\nreversed', fontsize=FS_MIN, color='#888888', va='bottom')
        ax.text(1.48, 1.05, r'marker size: $\kappa=2\to12$', fontsize=FS_MIN, color='#888888', ha='right', va='top')
        panel_title(ax, 'ef'[fi], f'{fam}: $a_1+\\mathrm{{i}}\\,b_1$ across $\\kappa$', fontsize=8)

    # ---- g  second harmonic R2 across kappa, both families ---------------------
    ax = axes_in(4.55, top_ef, 2.45, 1.38)
    for fi, fam in enumerate(['GPT', 'Claude']):
        c, e = data[fam]
        for j, rep in enumerate(REPS):
            sub = e[(e.representation == rep)].groupby('concentration').R2.agg(['mean', 'min', 'max']).sort_index()
            ax.plot(sub.index, sub['mean'], ls='-' if fam == 'GPT' else '--', lw=1.1, color=COLORS[j], marker='o' if fam == 'GPT' else 's', ms=2.6)
            ax.fill_between(sub.index, sub['min'], sub['max'], color=COLORS[j], alpha=.15, lw=0)
    ax.set(xticks=[2, 4, 6, 9, 12], xlabel=r'Concentration $\kappa$', ylabel=r'Second harmonic $R_2$', ylim=(0, .7))
    ax.legend(handles=[Line2D([], [], color='#555555', ls='-', marker='o', ms=2.6, label='GPT'), Line2D([], [], color='#555555', ls='--', marker='s', ms=2.6, label='Claude')],
              frameon=False, fontsize=6, loc='upper left')
    panel_title(ax, 'g', r'Second harmonic $R_2$ across $\kappa$', fontsize=8)

    # ---- h, i  imbalance sweep (GPT) ---------------------------------------------
    top_hi = 5.62
    eps = pd.read_csv(EPS)
    for pi, (field, ylabel, title) in enumerate([('activity', r'Activity $A$', r'GPT: activity vs $\varepsilon$'), ('a1', r'$a_1$', r'GPT: attraction vs $\varepsilon$')]):
        ax = axes_in(0.55+pi*1.85, top_hi, 1.55, 1.25)
        for j, rep in enumerate(REPS):
            sub = eps[eps.representation == rep].sort_values('epsilon')
            ax.plot(sub.epsilon, sub[field], marker='o', ms=2.8, lw=1.0, color=COLORS[j])
        ax.axvline(0, color='#CCCCCC', lw=.5, zorder=0)
        if field == 'a1': ax.axhline(0, color='#CCCCCC', lw=.5, zorder=0)
        ax.set(xlabel=r'Imbalance $\varepsilon$', ylabel=ylabel, xticks=[-.1, 0, .1, .2])
        if field == 'activity': ax.set_ylim(0, 1.05)
        panel_title(ax, 'hi'[pi], title, fontsize=8)

    # ---- j  dTV between encodings across stimulus families (GPT) -----------------
    atlas = pd.read_csv(ATLAS)
    fams = ['unimodal_k9', 'antipodal_equal_k6', 'asymmetric_w075_sep2_k6', 'sparse_N8_unimodal_k6', 'sparse_N16_antipodal_k6', 'bimodal_equal_sep2_k6']
    short = ['unimodal', 'antipodal', 'asymmetric', 'sparse unimodal', 'sparse antipodal', 'two-peaked']
    def probs(rep_short, fam):
        row = atlas[(atlas.representation == rep_short) & (atlas.profile == fam)].iloc[0]
        A, a0 = float(row.activity), float(row.a0)
        p = np.clip([0.5*(A-a0), 1-A, 0.5*(A+a0)], 0, 1); return p/p.sum()
    mat = np.zeros((6, 3))
    for i, fam in enumerate(fams):
        pm, pc, pi_ = [probs(s, fam) for s in ['moments', 'centers', 'intervals']]
        mat[i] = [0.5*np.abs(pm-pc).sum(), 0.5*np.abs(pm-pi_).sum(), 0.5*np.abs(pc-pi_).sum()]
    ax = axes_in(5.15, top_hi, 1.4, 1.25)
    im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=max(.4, mat.max()), aspect='auto')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['M–C', 'M–I', 'C–I'], fontsize=S.FS_TICK)
    ax.set_yticks(range(6)); ax.set_yticklabels(short, fontsize=6)
    for (i, k), v in np.ndenumerate(mat):
        ax.text(k, i, f'{v:.2f}', ha='center', va='center', fontsize=FS_MIN, color='white' if v < .55*im.norm.vmax else 'black')
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, cax=axes_in(6.62, top_hi+0.15, 0.08, 0.95)); cb.ax.tick_params(labelsize=FS_MIN); cb.set_label(r'$d_{\mathrm{TV}}$', fontsize=6)
    panel_title(ax, 'j', r'GPT: $d_{\mathrm{TV}}$, six field families', fontsize=8)

    fig.text(0.5, 1-7.3/H, 'Fitted curves are clipped to [−1, 1] for display; all coefficients come from the original unconstrained order-6 fits. '
             'Solid/circles: block 1; dashed/squares: block 2.', ha='center', va='top', fontsize=6, color='#56616B')
    save(fig, 'fig2_microscopic_operators')
    return zero_cross


def setup_v4():
    setup()                      # output directory and base rcParams
    S.apply_style()              # Arial-metric sans + STIX Sans maths, shared with Figs. 3-5
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


if __name__ == '__main__':
    setup_v4()
    ep, rows, seeds, series = load()
    fig1(seeds, series, ep)
    zc = fig2()
    print(json.dumps({f'{k[0]}:{k[1]}': round(v, 3) for k, v in zc.items()}, indent=1))
