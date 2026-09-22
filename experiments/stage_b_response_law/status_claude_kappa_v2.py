"""Read-only live progress; does not acquire the writer lock or invoke an API."""
from pathlib import Path
import json
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]

def main():
    output={'checked_utc':datetime.now(timezone.utc).isoformat(),'blocks':{}}
    for block in ['block_1','block_2']:
        path=ROOT/'runs/claude_kappa_v2/anthropic'/block/'attempts.jsonl'
        accepted=set();starts=results=invalid=fences=0; recent=[]
        if path.exists():
            # Ignore only an incomplete final write; do not repair or edit the journal.
            with path.open(encoding='utf-8') as handle:
                for line in handle:
                    if not line.endswith('\n'):break
                    row=json.loads(line)
                    if row['event']=='start':starts+=1
                    else:
                        results+=1;recent.append(row['utc']);recent=recent[-200:]
                        if row['valid']:
                            accepted.add(row['task_id']);fences+=bool(row.get('fence_removed'))
                        else:invalid+=1
        output['blocks'][block]={'accepted':len(accepted),'planned':12960,
            'dispatched_attempts':starts,'returned_attempts':results,
            'in_flight_or_uncertain':starts-results,'invalid_attempts':invalid,
            'accepted_with_fence_removed':fences}
        if len(recent)>=100:
            seconds=(datetime.fromisoformat(recent[-1])-datetime.fromisoformat(recent[0])).total_seconds()
            if seconds>0:output['blocks'][block]['recent_responses_per_second']=(len(recent)-1)/seconds
    output['accepted_total']=sum(b['accepted'] for b in output['blocks'].values())
    output['planned_total']=25920
    execution=ROOT/'runs/claude_kappa_v2/execution_20260918/execution_status.json'
    if execution.exists():output['execution']=json.loads(execution.read_text(encoding='utf-8'))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
