"""Live training and inference on synthetic records only.

This reproduces the default native AURA path for a selected benchmark seed,
including the original random draws before the gate/calibration/ranker splits.
The public benchmark runner remains the authority for manuscript results.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from .core import (S, names, gen, acquire, lr, gf, fuse, lgt, cal, ef, explain, pred)

MODALITIES = ['Clinical', 'Laboratory', 'Imaging']

@dataclass
class SyntheticAURA:
    experts: list
    gates: list
    calibrator: object
    no_disagreement_ranker: object
    full_ranker: object
    threshold: float

    def predict(self, features, mask, quality):
        """Return model scores, not clinical probabilities. Empty inputs defer."""
        x = np.asarray(features, dtype=float).reshape(-1, 10)
        m = np.asarray(mask, dtype=bool).reshape(-1, 3)
        q = np.asarray(quality, dtype=float).reshape(-1, 3)
        if not (len(x) == len(m) == len(q)):
            raise ValueError('Input batch sizes differ')
        if not np.isfinite(x).all() or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
            raise ValueError('Finite features and quality in [0,1] required')
        output = []
        for xi, mi, qi in zip(x, m, q):
            if not mi.any():
                output.append({'synthetic_score': None, 'decision': 'defer: no evidence'})
                continue
            p, v = pred(self.experts, xi[None], qi[None])
            raw, w = self._fusion(p, mi[None], qi[None])
            pr = cal(self.calibrator, raw)
            base, full = ef(pr, raw, p, v, mi[None], qi[None], w)
            u = self.full_ranker.predict_proba(full)[0, 1]
            phi = explain(p, w, mi[None], self.calibrator)[0]
            output.append({'synthetic_score': float(pr[0]), 'error_rank_score': float(u),
                'error_rank_without_D': float(self.no_disagreement_ranker.predict_proba(base)[0,1]),
                'disagreement_4D': float(full[0,-1]), 'development_cutoff': self.threshold,
                'decision': 'defer' if u > self.threshold else f'illustrative class {int(pr[0]>=.5)}',
                **{f'{n.lower()}_weight':float(w[0,k]) for k,n in enumerate(MODALITIES)},
                **{f'{n.lower()}_expert_score':float(p[0,k]) if mi[k] else None for k,n in enumerate(MODALITIES)},
                **{f'{n.lower()}_attribution':float(phi[2+k]) for k,n in enumerate(MODALITIES)},
                'attribution_reference':float(phi[0])})
        return pd.DataFrame(output)

    def _fusion(self,p,m,q):
        estimated = np.column_stack([g.predict(gf(p,q)[:,k]) for k,g in enumerate(self.gates)])
        return fuse(p,m,1/np.maximum(.03,estimated))


def fit_demo(seed=0):
    """Train default seed model and return untouched reference test observations."""
    r = np.random.default_rng(20261124 + seed)
    X,y,_ = gen(10000,r)
    tr,ga,ca,va,te = np.split(np.arange(10000),[5000,6500,7500,8500])
    ta,tm,tq = acquire(X[tr],y[tr],r,'train')
    experts = []
    for k,sl in enumerate(S):
        members=[]
        for _ in range(5):
            i=r.integers(0,len(tr),len(tr))
            members.append(lr().fit(np.c_[ta[i,sl],tq[i,k]],y[tr][i]))
        experts.append(members)
    def part(ids):
        return (*acquire(X[ids],y[ids],r,'train'),y[ids])
    a,m,q,yg=part(ga)
    p,v=pred(experts,a,q)
    z=gf(p,q)
    gates=[Ridge(alpha=5).fit(z[m[:,k],k],(p[m[:,k],k]-yg[m[:,k]])**2) for k in range(3)]
    acquire(X[tr],y[tr],r,'train')  # early-baseline acquisition in benchmark RNG sequence
    def fused(ids):
        a,m,q,y=part(ids)
        p,v=pred(experts,a,q)
        g=np.column_stack([gate.predict(gf(p,q)[:,k]) for k,gate in enumerate(gates)])
        raw,w=fuse(p,m,1/np.maximum(.03,g))
        return raw,w,p,v,m,q,y
    raw,w,p,v,m,q,yc=fused(ca)
    calibrator=LogisticRegression(C=100).fit(lgt(raw),yc)
    raw,w,p,v,m,q,yv=fused(va)
    pr=cal(calibrator,raw)
    b,e=ef(pr,raw,p,v,m,q,w)
    err=(pr>=.5)!=yv
    rb=lr().fit(b,err); rd=lr().fit(e,err)
    threshold=float(np.quantile(rd.predict_proba(e)[:,1],.8))
    model=SyntheticAURA(experts,gates,calibrator,rb,rd,threshold)
    a,m,q=acquire(X[te],y[te],r,'complete')
    return model,a,m,q,y[te]

if __name__ == '__main__':
    model,a,m,q,y=fit_demo()
    print('SYNTHETIC DEMO — not a clinical diagnostic tool')
    masks=np.array([[1,1,1],[1,0,1],[1,0,0],[0,0,0]])
    print(model.predict(np.repeat(a[:1],4,axis=0),masks,np.repeat(q[:1],4,axis=0)).to_string(index=False))
