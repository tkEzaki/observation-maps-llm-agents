"""Create separate GPT/Claude comparison drafts after the acquisition quality gate."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.analyze_transmutation import design_matrix

ROOT=Path(__file__).resolve().parents[2]
REPS=['moments_m1_m3','centers_24_standard','intervals_24_decimal6']
NAMES=['moments','centers','intervals']
COLORS=['#0072B2','#D88C00','#009E73']

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/claude_kappa_v1/anthropic')
    parser.add_argument('--allow-mock',action='store_true')
    args=parser.parse_args(argv)
    provenance=json.loads((args.run_root/'analysis/ACQUISITION_PROVENANCE.json').read_text())
    mock=provenance['synthetic']
    if mock and not args.allow_mock:
        parser.error('Mock data are not Claude measurements; use --allow-mock for pipeline QA only.')
    for block in ['block_1','block_2']:
        summary=json.loads((args.run_root/block/'run_summary.json').read_text())
        if not summary['quality_pass']:
            raise ValueError('Both acquisition blocks must pass quality checks')
    sources=[ROOT/'analysis/complex_kernel_transmutation',args.run_root/'analysis']
    families=['GPT','MOCK - NOT CLAUDE DATA' if mock else 'Claude']
    output=args.run_root/'comparison_figures'; output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig, axes=plt.subplots(2,3,figsize=(10,6),sharex=True,sharey=True)
    for row,source in enumerate(sources):
        curves=pd.read_csv(source/'offset_action_curves.csv')
        for col,rep in enumerate(REPS):
            ax=axes[row,col]
            for block,marker,style in [('block_1','o','-'),('block_2','s','--')]:
                sub=curves[(curves.representation==rep)&(curves.concentration==12)&(curves.block==block)].sort_values('offset_radians')
                assert len(sub)==36
                x=sub.offset_radians.to_numpy(); y=sub.g.to_numpy()
                coefficients=np.linalg.lstsq(design_matrix(x,6),y,rcond=None)[0]
                xx=np.linspace(-np.pi,np.pi,401)
                ax.scatter(x,y,s=10,marker=marker,color=COLORS[col],alpha=.6)
                ax.plot(xx,np.clip(design_matrix(xx,6)@coefficients,-1,1),style,color=COLORS[col])
            ax.axhline(0,color='.7',lw=.5)
            ax.set_title(f'{families[row]}\n{NAMES[col]}',fontsize=9)
            ax.set(ylim=(-1.12,1.12),xticks=[-np.pi,0,np.pi],xticklabels=['-π','0','π'])
            if row==1:ax.set_xlabel('Field rotation δ')
            if col==0:ax.set_ylabel('Mean action')
    fig.suptitle('Controlled-field rotation responses (κ = 12)')
    fig.text(.5,.015,'Blocks shown separately; M=6 fits clipped to [-1,1] for display only.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.95))
    for ext in ['pdf','png']:fig.savefig(output/f'rotation_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,6),sharex=True,sharey='col')
    for row,source in enumerate(sources):
        endpoints=pd.read_csv(source/'complex_endpoints.csv')
        for col,(field,title) in enumerate([('a1','Odd first harmonic a1'),('R2','Second harmonic R2'),('a0','Offset-independent bias a0')]):
            ax=axes[row,col]
            for rep,name,color in zip(REPS,NAMES,COLORS):
                for block,marker,style in [('block_1','o','-'),('block_2','s','--')]:
                    sub=endpoints[(endpoints.representation==rep)&(endpoints.block==block)].sort_values('concentration')
                    assert len(sub)==5
                    ax.plot(sub.concentration,sub[field],style,marker=marker,ms=3,color=color,label=name if block=='block_1' else None)
            ax.axhline(0,color='.7',lw=.5);ax.set_title(f'{families[row]}\n{title}',fontsize=8)
            ax.set_xticks([2,4,6,9,12])
            if row==1:ax.set_xlabel('Concentration κ')
    axes[0,0].legend(frameon=False,fontsize=7)
    fig.suptitle('Concentration dependence of controlled-field responses')
    fig.text(.5,.015,'Unconstrained M=6 coefficients; solid/circles: block 1, dashed/squares: block 2. Lines guide the eye.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.95))
    for ext in ['pdf','png']:fig.savefig(output/f'kappa_comparison.{ext}',dpi=180)
    plt.close(fig)
    print(output)

if __name__=='__main__':main()
