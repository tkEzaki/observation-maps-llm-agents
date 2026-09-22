"""Count valid real-model responses in reported analyses; no API calls.

Population responses are counted from retained action arrays, cross-checked
against traces. This excludes valid calls in aborted/repeated rounds.
Mock acquisitions, superseded collective sessions, the exploratory Stage C
v0.1 panel, transport errors and duplicate preflight texts are excluded.
"""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
files=[]; groups=Counter(); exclusions=Counter()
def rows(path):
    files.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def collective(root,group):
    n=0
    for ap in sorted(root.rglob('actions.npy')):
        a=np.load(ap); trace=ap.parent/'trace.jsonl'
        assert a.ndim==2 and np.isin(a,[-1,0,1]).all()
        valid={}
        for r in rows(trace):
            if r.get('valid') is True:
                key=(r['t'],r['agent'])
                if key in valid: exclusions['superseded_valid_collective_records']+=1
                valid[key]=r['action_value']
            else: exclusions['invalid_collective_records']+=1
        assert len(valid)==a.size,(trace,len(valid),a.shape)
        assert all(valid[t,i]==a[t,i] for t in range(a.shape[0]) for i in range(a.shape[1])),trace
        n+=a.size
    groups[group]+=int(n)
meta=json.loads((ROOT/'analysis/matched_rep_collective/paired_analysis/meta.json').read_text())
for x in meta['sessions']: collective(ROOT/x['session'].replace('\\','/'),'GPT matched populations')
collective(ROOT/'runs/stage_c/r2_claude_macro/matched-rep-collective-r2-claude-v0.1_ff2ef27f0dfe/20260725T050234Z','Claude retained populations')
for p in ['stage-c-v0.2a_ec22717957a1/20260724T014052Z','stage-c-v0.2b_47d97bc0acc3/20260724T023438Z']:
    collective(ROOT/'runs/stage_c'/p,'GPT prospective surrogate validation')
micro=['antipodal_weight','centers_cent3','collective_replay','feature_sweep','matched_rep_collective_omap','matched_rep_collective_replay','matched_rep_collective_sbc','matched_rep_collective_slc','response_law','response_law_representation','sparse_peer','stimulus_manifold','transmutation']
for folder in micro+['matched_rep_collective_r1','claude_kappa_v2']:
    for p in sorted((ROOT/'runs'/folder).rglob('trace.jsonl')):
        for r in rows(p):
            if r.get('backend')=='mock': exclusions['mock_records']+=1;continue
            if r.get('valid') is not True: exclusions['invalid_microscopic_records']+=1;continue
            if folder=='matched_rep_collective_r1': group='Claude replay' if 'claude_' in str(p) else 'Gemini replay'
            elif folder=='claude_kappa_v2': group='Claude concentration follow-up'
            else: group='GPT microscopic acquisitions'
            groups[group]+=1
expected={'GPT matched populations':122400,'Claude retained populations':132600,'GPT prospective surrogate validation':55600,'GPT microscopic acquisitions':166007,'Claude replay':2304,'Gemini replay':2281,'Claude concentration follow-up':25920}
assert dict(groups)==expected,(groups,expected)
total=sum(groups.values());assert total==507112
result={'definition':'Valid real-model responses in reported analyses, including prospective surrogate validation; retained population actions counted once per run/time/agent.','total':total,'groups':dict(groups),'excluded_encountered_records':dict(exclusions),'other_exclusions':['Superseded GPT initial acquisition: 2,652 valid records','Separate earlier Claude K=0 acquisition: 5,100 valid records; retained-session K=0 runs already included','Exploratory Stage C v0.1: 6,600 valid records','Claude preflight duplicates (5), transport errors (44), dispatches without results (4)','Separate *_mock folders and duplicate archive copies'],'files':files}
(ROOT/'docs/response_count_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
