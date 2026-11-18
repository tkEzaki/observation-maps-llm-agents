"""Recount valid responses in the raw journals and compare with dataset_index.json. No API calls.

A response is counted when its journal has a terminal, valid result for a task.
Run from the repository root:  python -B collective_forecast/verify_counts.py
"""
from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import read_journal
def count(path, where=None):
    tasks = set()
    for d in read_journal(path):
        if d.get('event') != 'result' or not d.get('terminal') or not d.get('valid'): continue
        if where and any(d.get(k) != v for k, v in where.items()): continue
        tasks.add(d['task_id'])
    return len(tasks)
def main():
    index = json.loads((ROOT/'collective_forecast/dataset_index.json').read_text())
    bad = 0
    for d in index['datasets']:
        if 'journal' not in d: continue
        n = count(ROOT/d['journal'], d.get('where'))
        ok = n == d['valid_responses']; bad += not ok
        print(f"{d['dataset']:40s} recorded {d['valid_responses']:6d}  recounted {n:6d}  {'ok' if ok else 'MISMATCH'}")
    return 1 if bad else 0
if __name__ == '__main__': raise SystemExit(main())
