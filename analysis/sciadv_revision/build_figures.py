"""Science Advances revision: existing data only; no provider calls.

Run from the repository root: python -m analysis.sciadv_revision.build_figures
Figures 1/2 reorganize existing trajectories and blockwise Fourier fits.
The persistence summary is a post hoc descriptive check, not new inference.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures/sciadv'
PAIRED = ROOT / 'analysis/matched_rep_collective/paired_analysis'
R2 = ROOT / 'runs/stage_c/r2_claude_macro/matched-rep-collective-r2-claude-v0.1_ff2ef27f0dfe/20260725T050234Z'
MICRO = ROOT / 'analysis/complex_kernel_transmutation'
REPS = ['moments_m1_m3', 'centers_24_standard', 'intervals_24_decimal6']
NAMES = ['moments', 'centers', 'intervals']
COLORS = ['#0072B2', '#D88C00', '#009E73']

def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'savefig.dpi': 220})

def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'))
    fig.savefig(OUT / (name + '.png'))
    plt.close(fig)

def load():
    ep = pd.read_csv(PAIRED / 'endpoints_long.csv')
    rows = json.loads((R2 / 'r2_inference/trajectory_rows.json').read_text())
    seeds = sorted({r['seed_index'] for r in rows if r['panel'] == 'core'})
    assert len(seeds) == 6
    series = {}
    for j, rep in enumerate(REPS):
        ts = pd.read_csv(PAIRED / f'_tmp_session_{j}/trajectory_harmonics_timeseries.csv')
        for k in [.08, .15]:
            for seed in seeds:
                a = ts[np.isclose(ts.coupling, k) & (ts.seed_index == seed)].sort_values('t')
                y = a.r1.to_numpy()
                assert len(y) == 101
                series['GPT', rep, k, seed] = y
                meta = json.loads((R2 / f'N17_K+{k:g}_s{seed}' / rep / 'run_meta.json').read_text())
                series['Claude', rep, k, seed] = np.array(meta['r1_series'])
                assert len(series['Claude', rep, k, seed]) == 101
    return ep, rows, seeds, series

def persistence(seeds, series):
    records = []
    for (family, rep, k, seed), y in series.items():
        suffix = 0
        for v in y[::-1]:
            if v <= .9:
                break
            suffix += 1
        records.append(dict(family=family, representation=rep, coupling=k,
            seed_index=int(seed), final_r1=float(y[-1]),
            trailing_observations_gt_09=suffix,
            trailing_elapsed_steps_gt_09=max(0, suffix-1),
            final20_mean_r1=float(y[-20:].mean())))
    df = pd.DataFrame(records)
    df.to_csv(OUT / 'persistence_by_run.csv', index=False)
    summary = []
    for (family, rep, k), a in df.groupby(['family', 'representation', 'coupling']):
        summary.append(dict(family=family, representation=rep, coupling=k,
            n=len(a), endpoint_count=int((a.final_r1 >= .9).sum()),
            trailing20_count=int((a.trailing_observations_gt_09 >= 20).sum()),
            trailing50_count=int((a.trailing_observations_gt_09 >= 50).sum()),
            min_trailing_observations=int(a.trailing_observations_gt_09.min()),
            final20_mean_r1=float(a.final20_mean_r1.mean())))
    (OUT/'persistence_summary.json').write_text(json.dumps(summary, indent=2))
    return summary

def fig1(seeds, series):
    from analysis.natcomm_figures_opus.fig2_micro import _shared_field_records, _excerpt_lines
    records = _shared_field_records()
    # This helper checks generating parameters and the encoded field itself.
    snippets = [_excerpt_lines(records[r]['serialized_prompt']).splitlines()[0] for r in REPS]
    # Verify the zero-coupling badge from recorded data, independently of the schematic.
    zero = []
    for j in range(3):
        ts = pd.read_csv(PAIRED / f'_tmp_session_{j}/trajectory_harmonics_timeseries.csv')
        a = ts[np.isclose(ts.coupling, 0) & ts.seed_index.isin(seeds)].sort_values(['seed_index','t'])
        assert len(a) == 606
        zero.append(a[['r1','r2','r3']].to_numpy())
    assert all(np.array_equal(zero[0], z) for z in zero[1:])
    control = json.loads((R2/'r2_inference/k0_control.json').read_text())
    assert control['pass'] and control['max_abs_diff_rm'] == 0
    fig = plt.figure(figsize=(7.1, 6.65))
    ax = fig.add_axes([.055,.80,.90,.18]); ax.set_axis_off(); ax.set_xlim(-.02,1.02); ax.set_ylim(0,1.12)
    ax.text(0,1.04,'a  One physical system, three state descriptions',fontweight='bold',fontsize=10)
    boxes=[(.00,.26,.17,.57,'Same 24-bin\nrelative-phase\nfield'),
           (.24,.20,.52,.70,''),
           (.83,.26,.17,.57,'GPT or Claude\n\nAction: -1, 0, +1')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',fc='#F4F6F8',ec='#9AA4AE',lw=.7))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7)
    for j, (name, snippet) in enumerate(zip(NAMES, snippets)):
        y = .77 - j*.22
        ax.text(.255,y,name,color=COLORS[j],fontsize=6.5,va='center',fontweight='bold')
        ax.text(.375,y,snippet+' ...',fontfamily='DejaVu Sans Mono',fontsize=5.25,va='center')
    for x0,x1 in [(.175,.23),(.77,.82)]:
        ax.annotate('',(x1,.55),(x0,.55),arrowprops=dict(arrowstyle='->',lw=.8))
    ax.plot([.915,.915,.085],[.25,.12,.12],color='#222222',lw=.8)
    ax.annotate('',(.085,.25),(.085,.12),arrowprops=dict(arrowstyle='->',lw=.8))
    ax.text(.51,-.055,r'Shared engine: $x_i(t+1)=x_i(t)+\omega_i+Kf_i(t)$',ha='center',fontsize=8)
    handles=[]
    for row,family in enumerate(['GPT','Claude']):
        for col,k in enumerate([.08,.15]):
            a=fig.add_axes([.09+col*.48,.525-row*.285,.37,.19])
            for j,rep in enumerate(REPS):
                stack=np.array([series[family,rep,k,s] for s in seeds])
                for y in stack: a.plot(range(101),y,color=COLORS[j],alpha=.22,lw=.55)
                line,=a.plot(range(101),stack.mean(0),color=COLORS[j],lw=1.8,label=NAMES[j])
                if row==col==0: handles.append(line)
            a.axhline(.9,color='#777777',lw=.6,ls=':',zorder=0)
            a.set(xlim=(0,100),ylim=(0,1.04),xticks=[0,50,100],yticks=[0,.5,1],xlabel='Step',ylabel=r'Polar order $r_1$' if col==0 else '')
            a.set_title(f'{chr(98+row*2+col)}  {family}, K = {k:g}',loc='left',fontweight='bold')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.776),ncol=3,frameon=False)
    ax=fig.add_axes([.09,.033,.83,.125]); ax.set_axis_off()
    ax.text(0,1.04,'f  High polar order at the endpoint: number of runs with '+r'$r_1(T)\geq0.9$',fontsize=9,fontweight='bold')
    counts=[]
    for family in ['GPT','Claude']:
        counts.append([f"{sum(series[family,rep,k,s][-1]>=.9 for s in seeds)}/6" for k in [.08,.15] for rep in REPS])
    tab=ax.table(cellText=counts,rowLabels=['GPT','Claude'],colLabels=NAMES+NAMES,cellLoc='center',bbox=[.08,.10,.91,.72])
    tab.auto_set_font_size(False); tab.set_fontsize(8)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#D8DDE2'); cell.set_linewidth(.5)
        if r==0 and c>=0: cell.get_text().set_color(COLORS[c%3])
    ax.text(.30,.88,'K = 0.08',ha='center',fontsize=8); ax.text(.77,.88,'K = 0.15',ha='center',fontsize=8)
    fig.text(.5,.014,'K = 0 control: identical trajectories across encodings in both models.',ha='center',fontsize=7,fontweight='bold',color='#344454')
    save(fig,'fig1_collective_phases')

def design_matrix(x,M=6):
    return np.column_stack([np.ones(len(x))]+[q(m*x) for m in range(1,M+1) for q in (np.sin,np.cos)])

def _fig2_gpt_legacy():
    curves=pd.read_csv(MICRO/'offset_action_curves.csv')
    ep=pd.read_csv(MICRO/'complex_endpoints.csv')
    assert set(REPS)<=set(ep.representation)
    fig=plt.figure(figsize=(7.1,5.85))
    for j,rep in enumerate(REPS):
        a=fig.add_axes([.085+j*.315,.56,.245,.31])
        for block,marker,ls in [('block_1','o','-'),('block_2','s','--')]:
            sub=curves[(curves.representation==rep)&(curves.concentration==12)&(curves.block==block)].sort_values('offset_radians')
            assert len(sub)==36
            x=sub.offset_radians.to_numpy(); y=sub.g.to_numpy()
            coef=np.linalg.lstsq(design_matrix(x),y,rcond=None)[0]
            frozen=ep[(ep.representation==rep)&(ep.concentration==12)&(ep.block==block)].iloc[0]
            assert np.allclose(coef[:5],[frozen.a0,frozen.a1,frozen.b1,frozen.a2,frozen.b2],atol=1e-8)
            xx=np.linspace(-np.pi,np.pi,401)
            a.scatter(x,y,s=9,marker=marker,color=COLORS[j],alpha=.6)
            a.plot(xx,np.clip(design_matrix(xx)@coef,-1,1),ls=ls,lw=1.2,color=COLORS[j])
        a.axhline(0,lw=.55,color='#AAAAAA')
        a.set(xticks=[-np.pi,0,np.pi],xticklabels=[r'$-\pi$','0',r'$\pi$'],xlabel=r'Field rotation $\delta$',ylim=(-1.12,1.12),yticks=[-1,0,1],ylabel=r'Mean action $g(\delta)$' if j==0 else '')
        a.set_title(f'{chr(97+j)}  {NAMES[j]}',loc='left',fontweight='bold',color=COLORS[j])
    fig.text(.5,.958,r'GPT responses to rotated unimodal fields ($\kappa=12$)',ha='center',fontsize=11,fontweight='bold')
    fig.text(.5,.915,'Points: empirical mean action; lines: existing M = 6 Fourier fits; solid/circles and dashed/squares: two acquisition blocks.',ha='center',fontsize=6.7)
    for j,(field,title) in enumerate([('a1',r'Odd first harmonic $a_1$'),('R2',r'Second-harmonic amplitude $R_2$'),('a0',r'Offset-independent bias $a_0$')]):
        a=fig.add_axes([.085+j*.315,.16,.245,.255])
        for ri,rep in enumerate(REPS):
            for block,marker,ls in [('block_1','o','-'),('block_2','s','--')]:
                sub=ep[(ep.representation==rep)&(ep.block==block)].sort_values('concentration')
                assert len(sub)==5
                a.plot(sub.concentration,sub[field],marker=marker,ms=3,lw=1,ls=ls,color=COLORS[ri],alpha=.8)
        a.axhline(0,lw=.55,color='#AAAAAA'); a.set(xticks=[2,4,6,9,12],xlabel=r'Concentration $\kappa$')
        a.set_title(f'{chr(100+j)}  {title}',loc='left',fontsize=8,fontweight='bold')
    fig.text(.5,.065,'Each block: 24 responses per field and rotation. Both blocks are shown separately; lines across concentrations guide the eye.',ha='center',fontsize=6.7)
    fig.text(.5,.025,'Fourier curves clipped to [-1, 1] for display only; all coefficients are from the original unconstrained fits.',ha='center',fontsize=7,color='#56616B')
    save(fig,'fig2_microscopic_operators')

def fig2():
    # Use the same two-family figure in both offline entry points.
    from analysis.sciadv_revision.integrate_claude_figures import fig2 as comparison
    comparison()

if __name__=='__main__':
    setup()
    ep,rows,seeds,series=load()
    summary=persistence(seeds,series)
    fig1(seeds,series); fig2()
    for f in ['fig3_replay_operator.pdf','fig4_cross_family.pdf','fig5_observation_map_control.pdf']:
        shutil.copyfile(ROOT/'figures/natcomm_opus'/f,OUT/f)
    print(json.dumps(summary,indent=2))
