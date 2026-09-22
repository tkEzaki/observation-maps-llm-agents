"""Offline manuscript figures and an acquisition audit; never calls a provider."""
from pathlib import Path
from collections import Counter
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.publication.build_figures import ROOT, OUT, REPS, NAMES, COLORS, setup, save, design_matrix

RUN = ROOT/'runs/claude_kappa_v2/anthropic'
SOURCES = [ROOT/'analysis/complex_kernel_transmutation', RUN/'analysis']
BLOCKS = [('block_1', 'o', '-'), ('block_2', 's', '--')]

def audit():
    report = {'blocks': {}, 'source_hashes': {}}
    all_ep = []
    for family, source in zip(['GPT','Claude'], SOURCES):
        ep = pd.read_csv(source/'complex_endpoints.csv')
        all_ep.append(ep.assign(family=family))
        curves = pd.read_csv(source/'offset_action_curves.csv')
        for (block, rep, k), cells in curves.groupby(['block','representation','concentration']):
            if rep not in REPS: continue
            assert len(cells)==36
            coef = np.linalg.lstsq(design_matrix(cells.offset_radians.to_numpy()),cells.g,rcond=None)[0]
            frozen=ep[(ep.block==block)&(ep.representation==rep)&(ep.concentration==k)].iloc[0]
            assert np.allclose(coef[:5], frozen[['a0','a1','b1','a2','b2']].to_numpy(dtype=float),atol=1e-8)
        for name in ['complex_endpoints.csv','offset_action_curves.csv']:
            p=source/name
            report['source_hashes'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for block,_,_ in BLOCKS:
        folder=RUN/block
        rows=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
        events=[json.loads(s) for s in (folder/'attempts.jsonl').read_text().splitlines()]
        counts=Counter((r['representation'],r['concentration'],r['offset_index']) for r in rows)
        assert len(rows)==len({r['task_id'] for r in rows})==12960
        assert len(counts)==540 and set(counts.values())=={24}
        assert all(r['valid'] and r['fence_removed'] and not r['strict_valid'] for r in rows)
        assert all(r['returned_model']=='claude-haiku-4-5-20251001' for r in rows)
        # Recompute every plotted probability directly from retained actions.
        raw=pd.DataFrame(rows)
        key=['representation','concentration','offset_index']
        derived=pd.DataFrame({'g':raw.groupby(key).action_value.mean()})
        for value,field in [(-1,'p_retard'),(0,'p_stay'),(1,'p_advance')]:
            raw[field]=(raw.action_value==value).astype(float)
            derived[field]=raw.groupby(key)[field].mean()
        frozen=pd.read_csv(RUN/'analysis/offset_action_curves.csv')
        frozen=frozen[frozen.block==block].set_index(key)
        assert np.allclose(derived.sort_index(),frozen.loc[derived.index,derived.columns].sort_index(),atol=1e-12)
        stimuli=[json.loads(line) for line in (folder/'stimuli.jsonl').read_text().splitlines()]
        lookup={(r['representation'],r['concentration'],r['offset_index']):r for r in stimuli}
        for r in rows:
            stimulus=lookup[tuple(r[k] for k in key)]
            assert r['prompt_sha256']==stimulus['prompt_sha256']==hashlib.sha256(stimulus['serialized_prompt'].encode()).hexdigest()
        starts={(e['task_id'],e['attempt']) for e in events if e['event']=='start'}
        results=[e for e in events if e['event']=='result']
        summary=json.loads((folder/'run_summary.json').read_text())
        assert summary['quality_pass'] and not summary['unresolved_tasks']
        report['blocks'][block]={'accepted':len(rows),'cells':len(counts),'responses_per_cell':24,
            'dispatches':len(starts),'results':len(results),
            'errors':sum(not e['valid'] for e in results),
            'unreturned_dispatches':len(starts-{(e['task_id'],e['attempt']) for e in results}),
            'first_attempt_valid_rate':summary['first_attempt_valid_rate']}
        report['blocks'][block]['raw_probabilities_match_figure_data']=True
        report['blocks'][block]['retained_prompt_hashes_verified']=True
        # The four imported responses must be the first received response for each task.
        original=[json.loads(s) for s in (ROOT/'runs/claude_kappa_v1/anthropic/block_1/attempts.jsonl').read_text().splitlines()]
        first={}
        for e in original:
            if e['event']=='result': first.setdefault(e['task_id'],e)
        imported=[r for r in rows if r.get('imported_from_v1')]
        for r in imported:
            assert r['raw_response']==first[r['task_id']]['raw_response']
            assert r['response_id']==first[r['task_id']]['response_id']
        report['blocks'][block]['imported_first_responses']=len(imported)
    report['excluded_preflight_duplicate_responses']=5
    report['unique_dispatches_including_preflight']=sum(b['dispatches'] for b in report['blocks'].values())+5
    pd.concat(all_ep).to_csv(OUT/'claude_integration_endpoints.csv',index=False)
    (OUT/'claude_integration_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def fig2():
    setup()
    fig, axes=plt.subplots(4,3,figsize=(7.1,6.8))
    fig.subplots_adjust(left=.09,right=.985,bottom=.13,top=.94,wspace=.36,hspace=.91)
    for family_i,(family,source) in enumerate(zip(['GPT','Claude'],SOURCES)):
        curves=pd.read_csv(source/'offset_action_curves.csv'); ep=pd.read_csv(source/'complex_endpoints.csv')
        row=family_i*2
        for col,rep in enumerate(REPS):
            ax=axes[row,col]
            for block,marker,style in BLOCKS:
                sub=curves[(curves.representation==rep)&(curves.concentration==12)&(curves.block==block)].sort_values('offset_radians')
                x=sub.offset_radians.to_numpy(); y=sub.g.to_numpy()
                coef=np.linalg.lstsq(design_matrix(x),y,rcond=None)[0]
                xx=np.linspace(-np.pi,np.pi,401)
                ax.scatter(x,y,s=7,marker=marker,color=COLORS[col],alpha=.65,zorder=3)
                ax.plot(xx,np.clip(design_matrix(xx)@coef,-1,1),style,lw=.9,color=COLORS[col])
            ax.axhline(0,color='.75',lw=.5); ax.axvline(0,color='.85',lw=.5)
            ax.set(ylim=(-1.13,1.13),yticks=[-1,0,1],xticks=[-np.pi,0,np.pi],xticklabels=['−π','0','π'],xlabel=r'Rotation $\delta$')
            ax.set_title(f'{chr(97+row*3+col)}  {family}: {NAMES[col]}',loc='left',fontsize=8,fontweight='bold')
            if col==0: ax.set_ylabel('Mean action')
        for col,(field,title,limits) in enumerate([('a1',r'Odd first harmonic $a_1$',(-.55,1.4)),('b1',r'Even first harmonic $b_1$',(-.65,.95)),('R2',r'Second harmonic $R_2$',(-.025,.7))]):
            ax=axes[row+1,col]
            for rep,name,color in zip(REPS,NAMES,COLORS):
                for block,marker,style in BLOCKS:
                    sub=ep[(ep.representation==rep)&(ep.block==block)].sort_values('concentration')
                    ax.plot(sub.concentration,sub[field],style,marker=marker,ms=2.5,lw=.9,color=color,label=name if block=='block_1' else None)
            ax.axhline(0,color='.75',lw=.5)
            ax.set(ylim=limits,xticks=[2,4,6,9,12],xlabel=r'Concentration $\kappa$')
            ax.set_title(f'{chr(97+(row+1)*3+col)}  {title}',loc='left',fontsize=8,fontweight='bold')
    fig.text(.5,.98,r'Rotated fields ($\kappa=12$) and concentration dependence',ha='center',fontsize=10,fontweight='bold')
    fig.legend(*axes[1,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.002),ncol=3,frameon=False,fontsize=7)
    fig.text(.5,.047,'Block 1: solid / circles; block 2: dashed / squares',ha='center',fontsize=6.5)
    save(fig,'fig2_microscopic_operators')

def supplement():
    setup(); source=SOURCES[1]
    ep=pd.read_csv(source/'complex_endpoints.csv')
    analysis=json.loads((source/'complex_kernel_analysis.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(7.1,4.9)); fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.88,wspace=.4,hspace=.6)
    for col,(field,title) in enumerate([('a0',r'Offset-independent bias $a_0$'),('R2',r'Second harmonic $R_2$'),('activity','Mean activity')]):
        ax=axes[0,col]
        for rep,name,color in zip(REPS,NAMES,COLORS):
            for block,marker,style in BLOCKS:
                sub=ep[(ep.representation==rep)&(ep.block==block)].sort_values('concentration')
                ax.plot(sub.concentration,sub[field],style,marker=marker,ms=3,color=color,label=name if block=='block_1' else None)
        ax.set(xticks=[2,4,6,9,12],xlabel=r'Concentration $\kappa$'); ax.set_title(f'{chr(97+col)}  {title}',fontsize=8,loc='left',fontweight='bold')
        if field=='activity':ax.set_ylim(.9,1.005)
    sensitivity=[]
    for rep,name,color in zip(REPS,NAMES,COLORS):
        for block,marker,style in BLOCKS:
            c=analysis['blocks'][block]['representations'][rep]['conditions']['kappa_12']
            orders=[2,4,6,8,12]; records=[]
            for order in orders:
                h=c['harmonics'] if order==6 else c['sensitivity_orders'][str(order)]
                a=h['a_sin']['1']; b=h['b_cos']['1']
                records.append({'block':block,'representation':rep,'M':order,'a1':a,'b1':b,'phi1_raw_degrees':np.degrees(np.arctan2(b,a))})
            sensitivity.extend(records)
            for col,field in enumerate(['a1','b1','phi1_raw_degrees']):axes[1,col].plot(orders,[r[field] for r in records],style,marker=marker,ms=3,color=color)
    for col,title in enumerate([r'Odd first harmonic $a_1$',r'Even first harmonic $b_1$',r'Phase angle (degrees)']):
        axes[1,col].set(xticks=[2,4,6,8,12],xlabel='Fourier order M');axes[1,col].set_title(f'{chr(100+col)}  {title}',fontsize=8,loc='left',fontweight='bold')
    fig.suptitle('Claude: additional endpoints and fit-order sensitivity',fontsize=11,fontweight='bold')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    save(fig,'figS26_claude_sensitivity')
    pd.DataFrame(sensitivity).to_csv(OUT/'claude_kappa12_order_sensitivity.csv',index=False)
    curves=pd.read_csv(source/'offset_action_curves.csv')
    fig,axes=plt.subplots(2,3,figsize=(7.1,4.6),sharey=True)
    fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.88,wspace=.22,hspace=.43)
    for row,(block,_,_) in enumerate(BLOCKS):
        for col,rep in enumerate(REPS):
            ax=axes[row,col];sub=curves[(curves.block==block)&(curves.representation==rep)&(curves.concentration==12)].sort_values('offset_radians')
            bottom=np.zeros(36)
            for field,color,label in [('p_retard','#8C6BB1','retard'),('p_stay','#D9D9D9','stay'),('p_advance','#31A354','advance')]:
                ax.bar(sub.offset_radians,sub[field],width=.115,bottom=bottom,color=color,label=label);bottom+=sub[field].to_numpy()
            ax.set(ylim=(0,1),xticks=[-np.pi,0,np.pi],xticklabels=['−π','0','π'],xlabel=r'Rotation $\delta$')
            ax.set_title(f'{chr(97+row*3+col)}  {NAMES[col]}, block {row+1}',loc='left',fontsize=8,fontweight='bold')
            if col==0:ax.set_ylabel('Action probability')
    fig.suptitle(r'Claude: full action distributions at $\kappa=12$',fontweight='bold',fontsize=11)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    save(fig,'figS27_claude_actions')

if __name__=='__main__':
    setup(); print(json.dumps(audit(),indent=2)); fig2(); supplement()
