"""Exact finite-state kernel for memoryless random-sequential resource choices."""
import itertools
import numpy as np
from math import factorial
PERMS=list(itertools.permutations(range(3)))
def compositions(n):return np.array([(a,b,n-a-b) for a in range(n+1) for b in range(n-a+1)],int)
def inputs(n=17):return [(tuple(c),own) for c in compositions(n-1) for own in range(3)]
def normalize(p):
    p=np.asarray(p,float);assert p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all() and p.min()>=0 and np.allclose(p.sum(1),1,atol=1e-12);return p
def symmetric(p,n=17):
    p=normalize(p);keys=inputs(n);index={k:i for i,k in enumerate(keys)};out=np.zeros_like(p)
    for row,(c,own) in enumerate(keys):
        for perm in PERMS:
            displayed=tuple(c[j] for j in perm);local_own=perm.index(own);j=index[(displayed,local_own)]
            out[row,list(perm)]+=p[j]/6
    return normalize(out)
def transition(p,n=17):
    p=normalize(p);keys=inputs(n);ix={k:i for i,k in enumerate(keys)};states=compositions(n);si={tuple(v):i for i,v in enumerate(states)};out=np.zeros((len(states),len(states)))
    assert len(p)==len(keys)
    for row,count in enumerate(states):
        for own in range(3):
            if count[own]==0:continue
            peers=count.copy();peers[own]-=1;prob=p[ix[(tuple(peers),own)]]
            for chosen in range(3):
                dest=peers.copy();dest[chosen]+=1;out[row,si[tuple(dest)]]+=count[own]/n*prob[chosen]
    assert np.allclose(out.sum(1),1,atol=1e-12) and out.min()>=0;return out
def cost(n=17):
    q,r=divmod(n,3);minimum=r*(q+1)**2+(3-r)*q*q
    return (compositions(n)**2).sum(1)/minimum
def synthetic(name,n=17):
    keys=inputs(n);c=np.array([v[0] for v in keys],float)
    if name=='uniform':return np.full((len(keys),3),1/3)
    if name=='least_load':
        p=(c==c.min(1,keepdims=True)).astype(float);return p/p.sum(1,keepdims=True)
    assert name=='label_bias';z=-.7*c+np.array([2.5,0,0]);z-=z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
def trajectories(T,n=17,steps=1020):
    # Column t gives E[c(X_t)|X_0=n] for all initial count states.
    c=cost(n);v=c.copy();curves=[v.copy()]
    for _ in range(steps):v=T@v;curves.append(v.copy())
    return np.array(curves).T
def window_moments(T,n=17,start=1,end=170):
    # Exact first and second moments of the time-averaged post-update cost,
    # conditional on the estimated kernel; no independent-time approximation.
    assert 1<=start<=end;c=cost(n);f=np.zeros(len(c));g=f.copy()
    for step in range(end,0,-1):
        w=1. if step>=start else 0.;g=T@(w*w*c*c+2*w*c*f+g);f=T@(w*c+f)
    length=end-start+1;mean=f/length;variance=np.maximum(g/length**2-mean**2,0)
    return mean,variance
def uniform_stationary(n=17):
    return np.array([factorial(n)/np.prod([factorial(int(z)) for z in c])/3**n for c in compositions(n)])
