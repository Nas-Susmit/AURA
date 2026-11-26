"""Revised experiment; original generator and model primitives retained."""
from pathlib import Path
from aura.core import np, pd, t, combinations, LogisticRegression, Ridge, RandomForestClassifier, HistGradientBoostingClassifier, StandardScaler, make_pipeline, roc_auc_score, S, scenarios, gen, acquire, lr, feat, gf, fuse, lgt, cal, ent, ef, pred, explain, met, rc
ROOT = Path(__file__).resolve().parent
R = ROOT / 'revision_results'
F = ROOT / 'revision_figures'
R.mkdir(exist_ok=True)
F.mkdir(exist_ok=True)

def rank(C=1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400))

def shared(pr, p, v, m, q):
    z, w = fuse(p, m, np.ones_like(p))
    return ef(pr, z, p, v, m, q, w)

def gate2(g, p, q, clip):
    return 1 / np.maximum(clip, np.column_stack([g[k].predict(gf(p, q)[:, k]) for k in range(3)]))

def main():
    rows = []
    curves = []
    sens = []
    experts = []
    coef = []
    perm = []
    thresh = []
    st = []
    fs = []
    variants = [('default', 5, 0.03, 5, 1), ('B=3', 3, 0.03, 5, 1), ('B=10', 10, 0.03, 5, 1), ('clip=.01', 5, 0.01, 5, 1), ('clip=.10', 5, 0.1, 5, 1), ('ridge=1', 5, 0.03, 1, 1), ('ridge=20', 5, 0.03, 20, 1), ('C=.1', 5, 0.03, 5, 0.1), ('C=10', 5, 0.03, 5, 10)]
    ar = np.random.default_rng(84419)
    ax, ay, _ = gen(150, ar)
    aa, am, aq = acquire(ax, ay, ar, 'complete')
    for seed in range(10):
        r = np.random.default_rng(20261124 + seed)
        aux = np.random.default_rng(42000 + seed)
        X, y, h = gen(10000, r)
        tr, ga, ca, va, te = np.split(np.arange(10000), [5000, 6500, 7500, 8500])
        assert len(set(np.r_[tr, ga, ca, va, te])) == 10000
        ta, tm, tq = acquire(X[tr], y[tr], r, 'train')
        ex = []
        for k, sl in enumerate(S):
            ens = []
            for j in range(5):
                i = r.integers(0, len(tr), len(tr))
                ens.append(lr().fit(np.c_[ta[i, sl], tq[i, k]], y[tr][i]))
            ex.append(ens)
        for k, sl in enumerate(S):
            for j in range(5):
                i = aux.integers(0, len(tr), len(tr))
                ex[k].append(lr().fit(np.c_[ta[i, sl], tq[i, k]], y[tr][i]))

        def part(i):
            return (*acquire(X[i], y[i], r, 'train'), y[i])
        gd = part(ga)
        a, m, q = acquire(X[tr], y[tr], r, 'train')
        Z = feat(a, m, q)
        bs = {'Early logistic': lr().fit(Z, y[tr]), 'Early RF': RandomForestClassifier(n_estimators=80, min_samples_leaf=8, n_jobs=2, random_state=seed).fit(Z, y[tr]), 'Early HGB': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=seed).fit(Z, y[tr])}
        cd = part(ca)
        vd = part(va)
        td = {}
        for s in scenarios:
            xx, yy, hh = gen(len(te), r, True) if s in ['mimic_shift', 'combined_shift'] else (X[te], y[te], h[te])
            td[s] = (*acquire(xx, yy, r, s), yy)

        def cache(dat):
            a, m, q, y = dat
            pp = np.stack([np.stack([e.predict_proba(np.c_[a[:, sl], q[:, k]])[:, 1] for e in ex[k]], 1) for k, sl in enumerate(S)], 1)
            return (a, m, q, y, pp)
        gd, cd, vd = map(cache, [gd, cd, vd])
        td = {s: cache(d) for s, d in td.items()}

        def unpack(dat, B):
            a, m, q, y, pp = dat
            return (a, m, q, y, pp[:, :, :B].mean(2), pp[:, :, :B].var(2))
        for tag, B, clip, alpha, C in variants:
            a, m, q, yg, p, v = unpack(gd, B)
            z = gf(p, q)
            g = [Ridge(alpha=alpha).fit(z[m[:, k], k], (p[m[:, k], k] - yg[m[:, k]]) ** 2) for k in range(3)]

            def get(dat):
                a, m, q, y, p, v = unpack(dat, B)
                raw, w = fuse(p, m, gate2(g, p, q, clip))
                return (a, m, q, y, p, v, raw, w)
            ac0, mc, qc, yc, pc, vc, rawc, wc = get(cd)
            ac = LogisticRegression(C=100).fit(lgt(rawc), yc)
            a, m, q, yv, p, v, raw, w = get(vd)
            pr = cal(ac, raw)
            b, e = ef(pr, raw, p, v, m, q, w)
            err = (pr >= 0.5) != yv
            rb = rank(C).fit(b, err)
            rd = rank(C).fit(e, err)
            if tag == 'default':
                c = rd[-1].coef_[0, -1]
                coef.append([seed, c, np.exp(c), rd[0].scale_[-1]])
                tau = np.quantile(rd.predict_proba(e)[:, 1], 0.8)

                def rawall(a, m, q, p, raw):
                    d = {k: b.predict_proba(feat(a, m, q))[:, 1] for k, b in bs.items()}
                    d.update({'Equal fusion': fuse(p, m, np.ones_like(p))[0], 'Quality fusion': fuse(p, m, q)[0], 'AURA': raw})
                    return d
                cs = {k: LogisticRegression(C=100).fit(lgt(z), yc) for k, z in rawall(ac0, mc, qc, pc, rawc).items()}
                ec = [LogisticRegression(C=100).fit(lgt(pc[mc[:, k], k]), yc[mc[:, k]]) for k in range(3)]
                rr = {}
                for k, z in rawall(a, m, q, p, raw).items():
                    pr0 = cal(cs[k], z)
                    b0, e0 = shared(pr0, p, v, m, q)
                    er = (pr0 >= 0.5) != yv
                    rr[k] = (rank().fit(b0, er), rank().fit(e0, er))
                ap, av = pred([e[:5] for e in ex], aa, aq)
                _, aw = fuse(ap, am, gate2(g, ap, aq, clip))
                st.append(np.array(explain(ap, aw, am, ac))[:, 2:])
                for k, sl in enumerate(S):
                    sc, clf = (ex[k][0].steps[0][1], ex[k][0].steps[1][1])
                    zz = sc.transform(np.c_[aa[:, sl], aq[:, k]])
                    co = zz * clf.coef_[0]
                    assert np.allclose(co.sum(1) + clf.intercept_[0], clf.decision_function(zz))
                    fs.append((seed, k, co))
            for s, dat in td.items():
                a, m, q, yt, p, v, raw, w = get(dat)
                pr = cal(ac, raw)
                b, e = ef(pr, raw, p, v, m, q, w)
                u0 = rb.predict_proba(b)[:, 1]
                u1 = rd.predict_proba(e)[:, 1]
                a0 = rc(yt, pr, u0).mean()
                a1 = rc(yt, pr, u1).mean()
                sens.append([seed, s, tag, a0, a1, a1 - a0])
                if tag != 'default':
                    continue
                assert np.allclose(w.sum(1), 1) and np.all(w[~m] == 0) and np.isfinite(pr).all()
                for k, z in rawall(a, m, q, p, raw).items():
                    pr0 = cal(cs[k], z)
                    b0, e0 = shared(pr0, p, v, m, q)
                    r0, r1 = rr[k]
                    for mode, u in [('entropy', ent(pr0)), ('matched no D', r0.predict_proba(b0)[:, 1]), ('matched full', r1.predict_proba(e0)[:, 1])]:
                        rows.append(dict(seed=seed, scenario=s, model=k, ranking=mode, **met(yt, pr0, u)))
                        if mode == 'matched full':
                            risk = rc(yt, pr0, u)
                            for cv in np.arange(0.1, 1.001, 0.05):
                                curves.append([seed, s, k, round(cv, 2), risk[round(cv * len(risk)) - 1]])
                for mode, u in [('native no D', u0), ('native full', u1)]:
                    rows.append(dict(seed=seed, scenario=s, model='AURA', ranking=mode, **met(yt, pr, u)))
                keep = u1 <= tau
                thresh.append([seed, s, keep.mean(), ((pr[keep] >= 0.5) != yt[keep]).mean() if keep.any() else np.nan])
                err = (pr >= 0.5) != yt
                auc = roc_auc_score(err, u1)
                for rep in range(10):
                    ep = e.copy()
                    ep[:, -1] = aux.permutation(ep[:, -1])
                    up = rd.predict_proba(ep)[:, 1]
                    perm.append([seed, s, rep, rc(yt, pr, up).mean() - a1, auc - roc_auc_score(err, up)])
                for k in range(3):
                    ix = m[:, k]
                    pk = cal(ec[k], p[ix, k])
                    py = yt[ix]
                    mp = met(py, pk, ent(pk))
                    mr = met(py, p[ix, k], ent(p[ix, k]))
                    experts.append([seed, s, k, ix.sum(), mp['auroc'], mp['auprc'], mp['brier'], mp['ece'], mr['brier'], mr['ece']])
        print('seed', seed, 'complete', flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(R / 'metrics.csv', index=False)
    for name, data, cols in [('curves', curves, ['seed', 'scenario', 'model', 'coverage', 'risk']), ('sensitivity', sens, ['seed', 'scenario', 'variant', 'noD', 'full', 'delta']), ('experts', experts, ['seed', 'scenario', 'modality', 'n', 'auroc', 'auprc', 'brier', 'ece', 'raw_brier', 'raw_ece']), ('coefficients', coef, ['seed', 'standardized_coef', 'odds_ratio_per_sd', 'D_scale']), ('permutation', perm, ['seed', 'scenario', 'repeat', 'delta_aurc', 'delta_error_auc']), ('thresholds', thresh, ['seed', 'scenario', 'coverage', 'risk'])]:
        pd.DataFrame(data, columns=cols).to_csv(R / (name + '.csv'), index=False)

    def ci(a):
        a = np.array(a)
        h = t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        return [a.mean(), a.mean() - h, a.mean() + h]
    M = D.groupby(['seed', 'model', 'ranking']).mean(numeric_only=True)
    out = []
    for k in ['Equal fusion', 'Quality fusion', 'Early logistic', 'Early RF', 'Early HGB']:
        for metric in ['auroc', 'auprc', 'brier', 'ece', 'aurc']:
            for mode in ['entropy', 'matched no D', 'matched full'] if metric == 'aurc' else ['entropy']:
                z = M.xs(('AURA', mode), level=('model', 'ranking'))[metric] - M.xs((k, mode), level=('model', 'ranking'))[metric]
                out.append([k, mode, metric, *ci(z)])
    for k in D.model.unique():
        for mode0, mode1 in [('entropy', 'matched no D'), ('matched no D', 'matched full')] + ([('entropy', 'native no D'), ('native no D', 'native full'), ('entropy', 'native full')] if k == 'AURA' else []):
            z = M.xs((k, mode1), level=('model', 'ranking')).aurc - M.xs((k, mode0), level=('model', 'ranking')).aurc
            out.append([k, mode1 + ' minus ' + mode0, 'aurc', *ci(z)])
    pd.DataFrame(out, columns=['comparator', 'ranking', 'metric', 'delta', 'low', 'high']).to_csv(R / 'paired.csv', index=False)

    def stab(z):
        return [[i, j, np.abs(z[i] - z[j]).mean(), (np.sign(z[i]) == np.sign(z[j])).mean(), (np.abs(z[i]).argmax(1) == np.abs(z[j]).argmax(1)).mean()] for i, j in combinations(range(10), 2)]
    pd.DataFrame(stab(st), columns=['seed1', 'seed2', 'mae', 'sign_agreement', 'top_agreement']).to_csv(R / 'explanation_stability.csv', index=False)
    pd.DataFrame([[k, *row] for k in range(3) for row in stab([a for _, kk, a in fs if kk == k])], columns=['modality', 'seed1', 'seed2', 'mae_logit', 'sign_agreement', 'top_agreement']).to_csv(R / 'feature_stability.csv', index=False)
    print(D.groupby(['model', 'ranking'])[['auroc', 'auprc', 'brier', 'ece', 'aurc', 'risk80']].mean().round(5).to_string())
if __name__ == '__main__':
    main()
