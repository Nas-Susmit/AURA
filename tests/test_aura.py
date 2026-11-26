import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from aura.core import gen, acquire, fuse, ef, cal, explain, lgt, pred, met
from aura.demo import fit_demo
from revision_experiment import rc
from sklearn.linear_model import LogisticRegression

class TestAURA(unittest.TestCase):
    def test_generator(self):
        x,y,h=gen(1000,np.random.default_rng(10))
        xx,yy,hh=gen(1000,np.random.default_rng(10))
        np.testing.assert_array_equal(x,xx)
        np.testing.assert_array_equal(y,yy)
        self.assertEqual(x.shape,(1000,10))
        for condition in ['complete','mcar30','mcar50','quality_shift','selective_acquisition','mimic_shift','combined_shift']:
            a,m,q=acquire(x,y,np.random.default_rng(20),condition)
            self.assertTrue(m.any(1).all())
            self.assertTrue(((q>=0)&(q<=1)).all())
    def test_masked_fusion(self):
        p=np.array([[.2,.9,.6],[.1,.3,.8]])
        m=np.array([[1,0,1],[0,1,0]],bool)
        z,w=fuse(p,m,np.ones_like(p))
        np.testing.assert_allclose(z,[.4,.3]); np.testing.assert_allclose(w.sum(1),1)
        self.assertTrue((w[~m]==0).all())
        b,e=ef(z,z,p,np.zeros_like(p),m,np.ones_like(p),w)
        np.testing.assert_allclose(e[:,-1],[.16,0])
    def test_ties(self):
        np.testing.assert_allclose(rc(np.array([0,1]),np.array([.9,.9]),np.array([.5,.5])),[.5,.5])
    def test_explanations(self):
        c=LogisticRegression().fit(lgt(np.array([.1,.2,.8,.9])),[0,0,1,1])
        p=np.array([[.2,.9,.6]])
        m=np.array([[1,0,1]],bool)
        raw,w=fuse(p,m,np.ones_like(p))
        ref,out,*phi=explain(p,w,m,c)[0]
        self.assertAlmostEqual(out,ref+sum(phi));self.assertEqual(phi[1],0)
    def test_demo_matches_archived_seed(self):
        model,x,m,q,y=fit_demo()
        p,v=pred(model.experts,x,q);raw,w=model._fusion(p,m,q)
        pr=cal(model.calibrator,raw);_,e=ef(pr,raw,p,v,m,q,w)
        u=model.full_ranker.predict_proba(e)[:,1]
        actual=met(y,pr,u)
        path=Path(__file__).resolve().parents[1]/'reference_results'/'metrics.csv'
        row=pd.read_csv(path).query("seed == 0 and scenario == 'complete' and model == 'AURA' and ranking == 'native full'").iloc[0]
        for key,value in actual.items(): self.assertAlmostEqual(value,float(row[key]),places=8)
    def test_demo(self):
        model,a,m,q,y=fit_demo()
        d=model.predict(a[:3],m[:3],q[:3]);self.assertTrue(d.synthetic_score.between(0,1).all())
        zero=model.predict(a[:1],[[0,0,0]],q[:1]);self.assertEqual(zero.decision.iloc[0],'defer: no evidence')
        x=a[:1].copy(); x[:,4:7]=10000
        d1=model.predict(a[:1],[[1,0,1]],q[:1]);d2=model.predict(x,[[1,0,1]],q[:1]);self.assertAlmostEqual(d1.synthetic_score.iloc[0],d2.synthetic_score.iloc[0])
        self.assertAlmostEqual(d1.laboratory_weight.iloc[0],0)
        self.assertAlmostEqual(d1.synthetic_score.iloc[0], d1.attribution_reference.iloc[0]+d1[['clinical_attribution','laboratory_attribution','imaging_attribution']].sum(axis=1).iloc[0])
if __name__=='__main__': unittest.main()
