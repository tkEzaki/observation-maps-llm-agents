"""One user-authorized transport recovery; preserve frozen acquisition and journal."""
import argparse
from collections import Counter
import hashlib
import time
from experiments.stage_b_response_law import run_claude_kappa_v2 as r

TRANSPORT={'APIConnectionError','APITimeoutError'}

def eligible(starts,results,accepted):
    counts=Counter(k[0] for k in starts)
    selected=[]
    for task_id,n in counts.items():
        if task_id in accepted or n!=3:continue
        history=[results.get((task_id,i)) for i in range(1,4)]
        if all(x is not None and not x['valid'] and x['raw_response'] is None
               and x.get('error_type') in TRANSPORT for x in history):
            selected.append(task_id)
    return sorted(selected)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yes',action='store_true')
    parser.add_argument('--block',choices=['block_1','block_2'],default='block_1')
    parser.add_argument('--expected-tasks',type=int,default=4)
    args=parser.parse_args(argv)
    if not args.yes:parser.error('This recovery sends paid requests; --yes is required.')
    root=r.ROOT/'runs/claude_kappa_v2/anthropic'
    with r.run_lock(root):
        spec=r.read(r.DEFAULT_SPEC)
        tasks,frozen=r.make_plan(spec,args.block,'anthropic')
        folder=root/args.block
        if r.read(folder/'resolved_config.json')['frozen_sha256']!=r.digest(frozen):
            raise ValueError('Frozen configuration changed')
        starts,results,accepted=r.load_journal(folder/'attempts.jsonl',tasks)
        manifest=root/('transport_recovery_20260918.json' if args.block=='block_1' else 'transport_recovery_block_2_20260918.json')
        if manifest.exists():
            recovery=r.read(manifest)
        else:
            ids=eligible(starts,results,accepted)
            if len(ids)!=args.expected_tasks:raise ValueError('Transport-exhausted task count differs from the expected count')
            recovery={'authorized_by':'User requested resume after the reported transport stop',
                'created_utc':r.utc(),'task_ids':ids,'additional_attempts_per_task':3,
                'maximum_total_attempts_per_selected_task':6,
                'journal_sha256_before':hashlib.sha256((folder/'attempts.jsonl').read_bytes()).hexdigest(),
                'accepted_before':len(accepted),
                'reason':'Three transport failures per task; no response text received. No action-dependent selection.',
                'usage_caveat':'A transport failure may have been processed or billed remotely; unavailable usage is not zero cost.'}
            r.atomic_json(manifest,recovery)
        selected=[t for t in tasks if t.task_id in recovery['task_ids']]
        if len(selected)!=args.expected_tasks:raise ValueError('Recovery manifest does not match task plan')
        backend=r.ClaudeBackend(model=spec['model'],temperature=spec['temperature'],
                               max_output_tokens=spec['max_output_tokens'],timeout_seconds=spec['timeout_seconds'])
        # The exception applies only to the named transport failures in the manifest.
        # Subsequent original-runner acquisition retains its original three-attempt cap.
        used=Counter(k[0] for k in starts)
        with (folder/'attempts.jsonl').open('a',encoding='utf-8') as journal:
            for task in selected:
                if task.task_id in accepted:continue
                if used[task.task_id]>=6:break
                for n in range(used[task.task_id]+1,7):
                    r.append(journal,{'event':'start','task_id':task.task_id,'attempt':n,
                        'prompt_sha256':task.prompt_sha256,'utc':r.utc(),
                        'recovery_manifest':manifest.name})
                    response=r.attempt(task,backend)
                    r.append(journal,{'event':'result','task_id':task.task_id,'attempt':n,
                        'utc':r.utc(),**response})
                    if response['valid'] or response['fatal']:break
                    if n<6:time.sleep(10)
                if not response['valid']:break
        summary=r.export(folder,tasks,frozen)
        _,_,now=r.load_journal(folder/'attempts.jsonl',tasks)
        recovery['accepted_recovered']=sum(t.task_id in now for t in selected)
        recovery['checked_utc']=r.utc()
        r.atomic_json(manifest,recovery)
        print('Recovered:',recovery['accepted_recovered'],'/',len(selected),'; block accepted:',summary['accepted_samples'])
        return 0 if recovery['accepted_recovered']==len(selected) else 3

if __name__=='__main__':raise SystemExit(main())
