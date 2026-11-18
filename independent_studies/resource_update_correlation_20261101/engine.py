"""Exact occupancy kernels for uniformly selected simultaneous updates.

All selected agents observe the SAME pre-update population. Conditional API
draws are independent given the chosen label frame. No APIs are used here.
"""
import itertools
from functools import lru_cache
from math import comb,factorial
import numpy as np
PERMS=list(itertools.permutations(range(3)))

@lru_cache(None)
def states(n):return np.array([(a,b,n-a-b) for a in range(n+1) for b in range(n-a+1)],int)

@lru_cache(None)
def inputs(n):return [(tuple(c),a) for c in states(n-1) for a in range(3)]

def validate(p,n):
    p=np.asarray(p,float);assert p.shape==(len(inputs(n)),3) and np.isfinite(p).all() and p.min()>=0 and np.allclose(p.sum(1),1,atol=1e-12);return p

@lru_cache(None)
def geometry(n,m):
    assert 1<=m<=n
    ns=states(n);nsindex={tuple(c):i for i,c in enumerate(ns)};ix={v:i for i,v in enumerate(inputs(n))}
    lookup=np.zeros((len(ns),3),int)
    for i,c in enumerate(ns):
        for a in range(3):
            if c[a]:peer=c.copy();peer[a]-=1;lookup[i,a]=ix[(tuple(peer),a)]
    parents=[];selected=[];weights=[];dest=[];s=states(m)
    for i,c in enumerate(ns):
        for chosen in s:
            if np.any(chosen>c):continue
            parents.append(i);selected.append(chosen);weights.append(np.prod([comb(int(c[a]),int(chosen[a])) for a in range(3)])/comb(n,m))
            remaining=c-chosen;dest.append([nsindex[tuple(remaining+z)] for z in s])
    return lookup,np.array(parents),np.array(selected),np.array(weights),np.array(dest)

def independent_kernel(p,m,n=17):
    p=validate(p,n);lookup,parents,selected,weights,dest=geometry(n,m)
    probabilities=p[lookup[parents]];nr=len(parents);r=np.arange(nr)
    dist=np.zeros((nr,m+1,m+1));dist[:,0,0]=1
    for step in range(m):
        own=np.where(step<selected[:,0],0,np.where(step<selected[:,:2].sum(1),1,2))
        pr=probabilities[r,own];new=dist*pr[:,2,None,None]
        new[:,1:,:]+=dist[:,:-1,:]*pr[:,0,None,None]
        new[:,:,1:]+=dist[:,:,:-1]*pr[:,1,None,None]
        dist=new
    output=dist[:,states(m)[:,0],states(m)[:,1]]*weights[:,None]
    T=np.zeros((len(states(n)),len(states(n))))
    np.add.at(T,(np.broadcast_to(parents[:,None],dest.shape).ravel(),dest.ravel()),output.ravel())
    assert T.min()>=0 and np.max(abs(T.sum(1)-1))<1e-11
    return T

@lru_cache(None)
def permutation_indices(n):
    ix={v:i for i,v in enumerate(inputs(n))};si={tuple(c):i for i,c in enumerate(states(n))}
    inp=[];macro=[]
    for perm in PERMS:
        inp.append([ix[(tuple(c[a] for a in perm),perm.index(own))] for c,own in inputs(n)])
        macro.append([si[tuple(c[list(perm)])] for c in states(n)])
    return np.array(inp),np.array(macro)

def frames(p,n=17):
    p=validate(p,n);idx,_=permutation_indices(n);branches=[]
    for g,perm in enumerate(PERMS):
        physical=np.zeros_like(p);physical[:,list(perm)]=p[idx[g]];branches.append(physical)
    return np.array(branches)

def kernels(p,m,n=17):
    native=independent_kernel(p,m,n);branches=frames(p,n)
    individual=independent_kernel(branches.mean(0),m,n)
    _,macro=permutation_indices(n)
    shared=sum(native[np.ix_(idx,idx)] for idx in macro)/6
    assert np.allclose(shared.sum(1),1)
    return dict(native=native,independent=individual,shared=shared)

def cost(n=17):
    q,r=divmod(n,3);den=r*(q+1)**2+(3-r)*q*q
    return (states(n)**2).sum(1)/den

def initial(n=17):return np.array([factorial(n)/np.prod([factorial(int(a)) for a in c])/3**n for c in states(n)])

def baseline(name,n=17):
    c=np.array([v[0] for v in inputs(n)])
    if name=='uniform':return np.full((len(c),3),1/3)
    best=c==c.min(1,keepdims=True)
    if name=='least_load_random_ties':return best/best.sum(1,keepdims=True)
    assert name=='least_load_fixed_ties';return np.eye(3)[c.argmin(1)]

def forecast(T,m,n=17,full_states=False):
    assert 1020%m==0 and 680%m==0
    steps=1020//m;late_start=680//m;c=cost(n);w=initial(n);f=c.copy();curves=[float(w@f)];auc=np.zeros(len(c));late=auc.copy()
    for step in range(1,steps+1):
        f=T@f;curves.append(float(w@f));auc+=f
        if step>late_start:late+=f
    answer=dict(mean_congestion=float(w@auc/steps),late_congestion=float(w@late/(steps-late_start)),curves=np.array(curves))
    if full_states:answer['by_state_mean']=auc/steps;answer['by_state_late']=late/(steps-late_start)
    return answer

def covariance_difference(p,m,n=17):
    """Analytic one-step shared-minus-independent normalized quadratic cost."""
    branches=frames(p,n);lookup,parents,selected,weight,_=geometry(n,m)
    values=branches[:,lookup[parents]] # frame, selection, own resource, chosen resource
    means=values.mean(0);var=values.var(0)
    total=(values*selected[None,:,:,None]).sum(2)
    covsum=total.var(0).sum(1)-(var*selected[:,:,None]).sum((1,2))
    out=np.zeros(len(states(n)));np.add.at(out,parents,weight*covsum)
    q,r=divmod(n,3);return out/(r*(q+1)**2+(3-r)*q*q)

def jump_diagnostics(T,m,n=17):
    # Mean squared change of resource fractions at one and two batch lags.
    xyz=states(n)/n;distance=((xyz[:,None,:]-xyz[None,:,:])**2).sum(2)
    d1=(T*distance).sum(1);d2=((T@T)*distance).sum(1);dist=initial(n);a=[];b=[]
    for step in range(1020//m):
        if step>=680//m:a.append(float(dist@d1));b.append(float(dist@d2))
        dist=dist@T
    return dict(late_lag1_squared_fraction_change=float(np.mean(a)),late_lag2_squared_fraction_change=float(np.mean(b)),)
