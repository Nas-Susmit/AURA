"""Shared simulator and numerical primitives for the revised AURA benchmark.
The historical experiment.py is retained separately.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit, logit
from scipy.stats import t
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from math import factorial
S = [slice(0, 4), slice(4, 7), slice(7, 10)]
names = ['stiffness', 'movement_relief', 'night_pain', 'extra_articular', 'hla_b27', 'crp_z', 'esr_z', 'bme', 'erosion', 'structural_score']
scenarios = ['complete', 'mcar30', 'mcar50', 'quality_shift', 'selective_acquisition', 'mimic_shift', 'combined_shift']

def gen(n, r, shift=False):
    """Generate illustrative latent classes and ten correlated synthetic features."""
    y = r.binomial(1, 0.25 if shift else 0.4, n)
    z = r.normal(size=n) + 0.8 * y
    h = r.binomial(1, np.where(y == 0, 0.48 if shift else 0.17, 0.08))
    g = r.normal(size=n)
    b = lambda a: r.binomial(1, expit(a), n)
    X = np.column_stack([b(-0.9 + 1.1 * y + 0.55 * z + 0.8 * h), b(-0.9 + 1.5 * y + 0.3 * z + 0.3 * h), b(-1 + 1.1 * y + 0.5 * z + 0.7 * h), b(-2 + 1.35 * y + 0.2 * g), b(-2.2 + 2.15 * y + 0.3 * g), 0.55 * y + 0.48 * z + 0.22 * h + r.normal(size=n), 0.35 * y + 0.52 * z + 0.22 * h + r.normal(size=n), b(-1.8 + 1.65 * y + 0.65 * z + 1.9 * h), b(-2.2 + 1.8 * y + 0.42 * z + 0.35 * h), 0.55 * y + 0.35 * z + 0.45 * h + r.normal(size=n)])
    return (X, y, h)

def acquire(X, y, r, s):
    """Apply quality-dependent noise and scenario-dependent modality masks."""
    X = X.copy()
    n = len(y)
    q = r.beta(2, 4, (n, 3)) if s in ['quality_shift', 'combined_shift'] else r.beta(5, 2, (n, 3))
    m = np.ones((n, 3), bool)
    rates = {'train': 0.22, 'mcar30': 0.3, 'mcar50': 0.5, 'combined_shift': 0.4}
    if s in rates:
        m = r.random((n, 3)) >= rates[s]
    if s == 'selective_acquisition':
        m[:, 1] = r.random(n) < expit(-0.7 + 1.15 * X[:, 0] + 0.55 * y)
        m[:, 2] = r.random(n) < expit(-1.1 + 1.4 * X[:, 1] + 0.7 * y)
    empty = ~m.any(1)
    m[empty, r.integers(0, 3, empty.sum())] = True
    for k, sl in enumerate(S):
        for j in range(sl.start, sl.stop):
            if j in [0, 1, 2, 3, 4, 7, 8]:
                X[:, j] = np.where(r.random(n) < 0.36 * (1 - q[:, k]), 1 - X[:, j], X[:, j])
            else:
                X[:, j] += r.normal(size=n) * 1.2 * (1 - q[:, k])
    return (X, m, q)

def feat(X, m, q):
    """Build early-fusion features; zero unavailable features and quality."""
    X = X.copy()
    for k, s in enumerate(S):
        X[~m[:, k], s] = 0
    return np.c_[X, m, q * m]

def lr():
    """Return a standardized logistic regression pipeline."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=400))

def fit_ex(X, y, r):
    a, m, q = acquire(X, y, r, 'train')
    out = []
    for k, s in enumerate(S):
        ens = []
        for _ in range(5):
            i = r.integers(0, len(y), len(y))
            ens.append(lr().fit(np.c_[a[i, s], q[i, k]], y[i]))
        out.append(ens)
    return out

def pred(ex, a, q):
    """Return ensemble mean probability and population variance per modality."""
    p = np.stack([np.stack([e.predict_proba(np.c_[a[:, s], q[:, k]])[:, 1] for e in ex[k]], 1) for k, s in enumerate(S)], 1)
    return (p.mean(2), p.var(2))

def fuse(p, m, w):
    """Normalize nonnegative weights across observed modalities and combine scores."""
    w = w * m
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-09)
    return ((w * p).sum(1), w)

def gf(p, q):
    """Construct four features for reliability-gate regression."""
    return np.stack([q, abs(p - 0.5), q * abs(p - 0.5), q * q], 2)

def gate(g, p, q):
    """Estimate inverse clipped squared-error weights."""
    a = gf(p, q)
    return 1 / np.maximum(0.03, np.column_stack([g[k].predict(a[:, k]) for k in range(3)]))

def lgt(p):
    """Transform clipped probabilities to logits for recalibration."""
    return logit(np.clip(p, 1e-06, 1 - 1e-06)).reshape(-1, 1)

def cal(c, p):
    """Apply a fitted logistic recalibrator."""
    return c.predict_proba(lgt(p))[:, 1]

def ent(p):
    """Compute binary entropy in bits."""
    p = np.clip(p, 1e-06, 1 - 1e-06)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def ef(p, raw, pe, v, m, q, w):
    """Construct base error-ranking features and append scaled disagreement 4D."""
    d = (w * (pe - raw[:, None]) ** 2).sum(1)
    base = np.c_[ent(p), (w * v).sum(1), m.sum(1) / 3, (q * m).sum(1) / m.sum(1), m]
    return (base, np.c_[base, 4 * d])

def rc(y, p, u):
    """Expected retained risk at each rank, averaging errors within exact score ties."""
    e = ((p >= 0.5) != y).astype(float)
    i = np.argsort(u, kind='stable')
    s = u[i]
    e = e[i]
    a = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    b = np.r_[a[1:], len(s)]
    return np.cumsum(np.repeat(np.add.reduceat(e, a) / (b - a), b - a)) / np.arange(1, len(y) + 1)

def met(y, p, u):
    """Compute discrimination, calibration and selective prediction metrics."""
    a = rc(y, p, u)
    e = (p >= 0.5) != y
    b = np.minimum((p * 10).astype(int), 9)
    ece = sum(((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(10) if (b == k).any()))
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p), brier=brier_score_loss(y, p), ece=ece, logloss=log_loss(y, p), accuracy=1 - e.mean(), aurc=a.mean(), risk80=a[int(0.8 * len(y)) - 1], risk50=a[int(0.5 * len(y)) - 1], error_auc=roc_auc_score(e, u))

def explain(p, w, m, c):
    """Exact modality coalition attribution with frozen weights and recalibrator."""
    out = []
    for i in range(len(p)):
        A = list(np.flatnonzero(m[i]))
        k = len(A)
        v = lambda ids: cal(c, np.array([np.average(p[i, list(ids)], weights=w[i, list(ids)]) if ids else 0.5]))[0]
        phi = np.zeros(3)
        for j in A:
            for s in range(k):
                for B in combinations([a for a in A if a != j], s):
                    phi[j] += factorial(s) * factorial(k - s - 1) / factorial(k) * (v(B + (j,)) - v(B))
        assert abs(phi.sum() - (v(A) - v(()))) < 1e-09
        out.append([v(()), v(A), *phi])
    return out
