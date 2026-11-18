"""Utilities used only by this isolated exploratory study."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'
TABLES = RESULTS / 'tables'
FIGURES = RESULTS / 'figures'
MODELS = RESULTS / 'models'
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
REPS = ['moments_m1_m3', 'centers_24_standard', 'intervals_24_decimal6']
NAMES = dict(zip(REPS, ['moments', 'centers', 'intervals']))
ACTIONS = np.array([-1., 0., 1.])
CENTERS = (np.arange(24) + .5) * (2*np.pi/24) - np.pi

def mkdirs():
    for path in [RESULTS, TABLES, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def record_inputs(paths):
    path=RESULTS/'input_hashes.json'
    data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for p in paths:
        p=Path(p)
        data[p.relative_to(ROOT).as_posix()]={'sha256':digest(p),'bytes':p.stat().st_size}
    write_json(path,data)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                  allow_nan=False, default=json_default) + '\n', encoding='utf-8')

def json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Not JSON serializable: {type(value)}')

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'Empty table: {path}')
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def histogram_batch(phases, bins=24):
    """Exactly match the frozen encoder's bin coordinates, without imports."""
    p = np.asarray(phases)
    n = p.size
    delta = wrap(p[None, :] - p[:, None])
    ids = np.floor(np.round((delta+np.pi)*bins/(2*np.pi), 12)).astype(int) % bins
    mask = ~np.eye(n, dtype=bool)
    row = np.broadcast_to(np.arange(n)[:, None], (n, n))
    h = np.bincount((row*bins+ids)[mask], minlength=n*bins).reshape(n, bins)/(n-1)
    return h

def moments(h):
    z = np.asarray(h) @ np.exp(1j*CENTERS[:, None]*np.arange(1,4))
    return np.stack([z.real, z.imag], axis=-1).reshape(-1, 6)

def payload(h, rep):
    # Respect the numbers actually supplied, including moments serialization.
    if rep == REPS[0]:
        return np.round(moments(h), 6)
    return np.round(np.asarray(h), 6)

def probs_from_ga(g, a):
    return np.stack([(a-g)/2, 1-a, (a+g)/2], axis=-1)

def chi(p, k, m=1):
    return np.asarray(p) @ np.exp(1j*m*k*ACTIONS)

def exact_order_squared(phases, omega, k, p, m=1):
    """Conditional independent draws; finite-N variance correction is retained."""
    c = chi(p, k, m)
    w = np.exp(1j*m*(phases+omega))*c
    return float(abs(np.mean(w))**2 + np.sum(1-abs(c)**2)/len(phases)**2)

def bootstrap_mean(values, seed=0):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'mean': None, 'ci_low': None, 'ci_high': None, 'n_runs': 0}
    rng = np.random.default_rng(CONFIG['random_seed']+seed)
    means = a[rng.integers(0, len(a), (CONFIG['bootstrap_replicates'], len(a)))].mean(1)
    return dict(mean=float(a.mean()), ci_low=float(np.quantile(means,.025)),
                ci_high=float(np.quantile(means,.975)), n_runs=int(len(a)))

def protected_snapshot():
    """Metadata for the earlier pipeline and hashes for its code."""
    out = {}
    for prefix in ['analysis','experiments','circlemap','figures','runs']:
        base = ROOT/prefix
        for p in base.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or 'legacy' in p.parts:
                continue
            s=p.stat()
            rel=p.relative_to(ROOT).as_posix()
            row={'size':s.st_size,'mtime_ns':s.st_mtime_ns}
            if p.suffix=='.py':
                row['sha256']=digest(p)
            out[rel]=row
    return out

def discover_runs():
    rows=[]
    gpt_bases=[
        ROOT/'runs/stage_c/matched-rep-collective-v0.1-moments_ed41bff41a9a/20260724T065125Z',
        ROOT/'runs/stage_c/matched-rep-collective-v0.1-centers_65c4f06d18a1/20260724T065125Z',
        ROOT/'runs/stage_c/matched-rep-collective-v0.1-intervals_1d533f018b96/20260724T065128Z']
    for rep, base in zip(REPS,gpt_bases):
        for path in sorted(base.glob('N*/phases.npy')):
            rows.append(run_record(path.parent, 'GPT', rep, 'core'))
    base=ROOT/'runs/stage_c/r2_claude_macro/matched-rep-collective-r2-claude-v0.1_ff2ef27f0dfe/20260725T050234Z'
    for path in sorted(base.glob('N*/*/phases.npy')):
        record=run_record(path.parent,'Claude',path.parent.name,'core')
        if record['seed_index']>=6:
            record['panel']='existing_heldout'
        rows.append(record)
    for stem in ['stage-c-v0.2a_ec22717957a1','stage-c-v0.2b_47d97bc0acc3']:
        for path in sorted((ROOT/'runs/stage_c'/stem).glob('*/N*/phases.npy')):
            rows.append(run_record(path.parent,'GPT',REPS[0],'existing_validation'))
    expected={'core':126,'existing_heldout':24,'existing_validation':44}
    actual={key:sum(r['panel']==key for r in rows) for key in expected}
    if actual!=expected:
        raise ValueError(f'Unexpected run selection: {actual}, expected {expected}')
    return rows

def run_record(path, family, rep, panel):
    cell=path.name if family=='GPT' else path.parent.name
    n,k,s=cell.split('_')
    return dict(run_key=path.relative_to(ROOT).as_posix(), family=family, representation=rep,
                panel=panel, n_agents=int(n[1:]), coupling=float(k[1:]), seed_index=int(s[1:]),
                phases_path=(path/'phases.npy').relative_to(ROOT).as_posix(),
                actions_path=(path/'actions.npy').relative_to(ROOT).as_posix())

def load_run(row):
    th=np.load(ROOT/row['phases_path'],allow_pickle=False)
    f=np.load(ROOT/row['actions_path'],allow_pickle=False)
    n=row['n_agents'];omega=np.linspace(-.05,.05,n)
    assert th.shape==(101,n) and f.shape==(100,n)
    assert np.isfinite(th).all() and np.isin(f,ACTIONS).all()
    assert np.allclose(th[1:],th[:-1]+omega+row['coupling']*f,atol=1e-10,rtol=0)
    return th,f,omega

def log(message):
    print(message,flush=True)
