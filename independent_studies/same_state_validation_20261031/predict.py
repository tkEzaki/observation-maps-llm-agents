"""Freeze empirical predictions strictly before any held-out validation dispatch."""
from lib import *
import argparse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['mock','real'],default='mock');args=parser.parse_args()
    check();out=LOCAL/f'predictions_{args.mode}'
    if (out/'freeze.json').exists():
        f=read(out/'freeze.json');assert all(digest(LOCAL/p)==h for p,h in f['files'].items());print('Existing prediction freeze retained.');return
    history=journal_rows(LOCAL/f'runs/{args.mode}/attempts.jsonl')
    assert not any(r['event']=='dispatch' and r['stage']=='validate' for r in history)
    a,_=arrays(args.mode,'train');plan=read(LOCAL/'protocol.json');flags=validation_flags()
    pred,ideal,chi=predict_from_training(a,plan['states'],flags);out.mkdir(exist_ok=True)
    rows=[]
    for i,s in enumerate(plan['states']):
        rows.append(dict(state=i,source_run=s['source_run'],source_step=s['source_step'],r1=s['r1'],
            predicted_independent=float(pred[:,i,0].mean()),predicted_shared=float(pred[:,i,1].mean()),
            predicted_gap_schedule=float((pred[:,i,1]-pred[:,i,0]).mean()),predicted_gap_ideal_coins=float(ideal[i])))
    table(out/'by_state.csv',rows);save(out/'per_round.json',dict(predicted=pred,chi_real=chi.real,chi_imag=chi.imag))
    train_results=[r for r in history if r['event']=='result' and r['stage']=='train']
    files=[out/'by_state.csv',out/'per_round.json']
    save(out/'freeze.json',dict(created_at_utc=now(),mode=args.mode,validation_dispatches_before_prediction=0,
        training_results_sha256=sha(json.dumps(train_results,sort_keys=True)),
        files={p.relative_to(LOCAL).as_posix():digest(p) for p in files},
        ))
    print(pd.DataFrame(rows).to_string(index=False));check()

if __name__=='__main__':main()
