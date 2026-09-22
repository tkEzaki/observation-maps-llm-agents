"""Finite background execution: acquire both blocks, quality-check, analyze, plot."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import argparse
from experiments.stage_b_response_law.run_claude_kappa_v2 import run_lock, atomic_json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/claude_kappa_v2/execution_20260918'
RUN=ROOT/'runs/claude_kappa_v2/anthropic'

def build_stages(concurrency):
    if not 1 <= concurrency <= 8:
        raise ValueError('Use 1 to 8 workers for the staged concurrency increase')
    return [
        ('acquisition',['experiments.stage_b_response_law.run_claude_kappa_v2','--mode','run','--yes',
                        '--concurrency',str(concurrency)]),
        ('analysis',['experiments.stage_b_response_law.run_claude_kappa_v2','--mode','analyze']),
        ('figures',['analysis.sciadv_revision.plot_claude_kappa','--run-root',str(RUN)])]

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concurrency',type=int,choices=range(1,9),default=4)
    parser.add_argument('--dry-run',action='store_true',help='Print commands only; no API or file mutations')
    parser.add_argument('--retry-uncertain',action='store_true',help='Explicitly allow retry of interrupted dispatches within the original attempt budget')
    args=parser.parse_args(argv)
    stages=build_stages(args.concurrency)
    if args.retry_uncertain:
        stages[0][1].append('--retry-uncertain')
    if args.dry_run:
        print(json.dumps(stages,indent=2))
        return 0
    OUT.mkdir(parents=True,exist_ok=True)
    with run_lock(OUT):
        if (OUT/'execution_status.json').exists():
            previous=json.loads((OUT/'execution_status.json').read_text(encoding='utf-8'))
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            atomic_json(OUT/f'execution_status_before_{stamp}.json',previous)
        status={'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
                'state':'running','run_root':str(RUN),'concurrency':args.concurrency,'stages':[]}
        for name,args in stages:
            status['current_stage']=name
            atomic_json(OUT/'execution_status.json',status)
            with (OUT/(name+'.log')).open('a',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-u','-m',*args],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1'})
            status['stages'].append({'stage':name,'exit_code':result.returncode,
                                    'finished_utc':datetime.now(timezone.utc).isoformat()})
            if result.returncode:
                status['state']='stopped';atomic_json(OUT/'execution_status.json',status)
                return result.returncode
        status['state']='complete';status['finished_utc']=datetime.now(timezone.utc).isoformat()
        atomic_json(OUT/'execution_status.json',status)
    return 0

if __name__=='__main__':raise SystemExit(main())
