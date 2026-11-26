import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.stats import t
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
from math import factorial
ROOT=Path(__file__).resolve().parent; R=ROOT/'results'; F=ROOT/'figures';R.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
S=[slice(0,4),slice(4,7),slice(7,10)]; names=['stiffness','movement_relief','night_pain','extra_articular','hla_b27','crp_z','esr_z','bme','erosion','structural_score']; scenarios=['complete','mcar30','mcar50','quality_shift','selective_acquisition','mimic_shift','combined_shift']
def gen(n,r,shift=False):
 y=r.binomial(1,.25 if shift else .4,n); z=r.normal(size=n)+.8*y; h=r.binomial(1,np.where(y==0,.48 if shift else .17,.08)); g=r.normal(size=n); b=lambda a:r.binomial(1,expit(a),n)
 X=np.column_stack([b(-.9+1.1*y+.55*z+.8*h),b(-.9+1.5*y+.3*z+.3*h),b(-1+1.1*y+.5*z+.7*h),b(-2+1.35*y+.2*g),b(-2.2+2.15*y+.3*g),.55*y+.48*z+.22*h+r.normal(size=n),.35*y+.52*z+.22*h+r.normal(size=n),b(-1.8+1.65*y+.65*z+1.9*h),b(-2.2+1.8*y+.42*z+.35*h),.55*y+.35*z+.45*h+r.normal(size=n)])
 return X,y,h
def acquire(X,y,r,s):
 X=X.copy(); n=len(y); q=r.beta(2,4,(n,3)) if s in ['quality_shift','combined_shift'] else r.beta(5,2,(n,3)); m=np.ones((n,3),bool)
 rates={'train':.22,'mcar30':.3,'mcar50':.5,'combined_shift':.4}
 if s in rates:m=r.random((n,3))>=rates[s]
 if s=='selective_acquisition':
  m[:,1]=r.random(n)<expit(-.7+1.15*X[:,0]+.55*y);m[:,2]=r.random(n)<expit(-1.1+1.4*X[:,1]+.7*y)
 empty=~m.any(1);m[empty,r.integers(0,3,empty.sum())]=True
 for k,sl in enumerate(S):
  for j in range(sl.start,sl.stop):
   if j in [0,1,2,3,4,7,8]:X[:,j]=np.where(r.random(n)<.36*(1-q[:,k]),1-X[:,j],X[:,j])
   else:X[:,j]+=r.normal(size=n)*1.2*(1-q[:,k])
 return X,m,q
def feat(X,m,q):
 X=X.copy()
 for k,s in enumerate(S):X[~m[:,k],s]=0
 return np.c_[X,m,q*m]
def lr():return make_pipeline(StandardScaler(),LogisticRegression(max_iter=400))
def fit_ex(X,y,r):
 a,m,q=acquire(X,y,r,'train');out=[]
 for k,s in enumerate(S):
  ens=[]
  for _ in range(5):
   i=r.integers(0,len(y),len(y));ens.append(lr().fit(np.c_[a[i,s],q[i,k]],y[i]))
  out.append(ens)
 return out
def pred(ex,a,q):
 p=np.stack([np.stack([e.predict_proba(np.c_[a[:,s],q[:,k]])[:,1] for e in ex[k]],1) for k,s in enumerate(S)],1);return p.mean(2),p.var(2)
def fuse(p,m,w):
 w=w*m;w=w/np.maximum(w.sum(1,keepdims=True),1e-9);return (w*p).sum(1),w
def gf(p,q):return np.stack([q,abs(p-.5),q*abs(p-.5),q*q],2)
def gate(g,p,q):
 a=gf(p,q);return 1/np.maximum(.03,np.column_stack([g[k].predict(a[:,k]) for k in range(3)]))
def lgt(p):return logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1)
def cal(c,p):return c.predict_proba(lgt(p))[:,1]
def ent(p):
 p=np.clip(p,1e-6,1-1e-6);return -p*np.log2(p)-(1-p)*np.log2(1-p)
def ef(p,raw,pe,v,m,q,w):
 d=(w*(pe-raw[:,None])**2).sum(1);base=np.c_[ent(p),(w*v).sum(1),m.sum(1)/3,(q*m).sum(1)/m.sum(1),m];return base,np.c_[base,4*d]
def rc(y,p,u):
 e=(p>=.5)!=y;idx=np.argsort(u,kind='stable');return np.cumsum(e[idx])/np.arange(1,len(y)+1)
def met(y,p,u):
 a=rc(y,p,u);e=(p>=.5)!=y;b=np.minimum((p*10).astype(int),9);ece=sum((b==k).mean()*abs(p[b==k].mean()-y[b==k].mean()) for k in range(10) if (b==k).any());return dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),brier=brier_score_loss(y,p),ece=ece,logloss=log_loss(y,p),accuracy=1-e.mean(),aurc=a.mean(),risk80=a[int(.8*len(y))-1],risk50=a[int(.5*len(y))-1],error_auc=roc_auc_score(e,u))
def explain(p,w,m,c):
 out=[]
 for i in range(len(p)):
  A=list(np.flatnonzero(m[i]));k=len(A);v=lambda ids:cal(c,np.array([np.average(p[i,list(ids)],weights=w[i,list(ids)]) if ids else .5]))[0];phi=np.zeros(3)
  for j in A:
   for s in range(k):
    for B in combinations([a for a in A if a!=j],s):phi[j]+=factorial(s)*factorial(k-s-1)/factorial(k)*(v(B+(j,))-v(B))
  assert abs(phi.sum()-(v(A)-v(())))<1e-9;out.append([v(()),v(A),*phi])
 return out
rows=[];curves=[];aud=[];thresholds=[]
for seed in range(10):
 r=np.random.default_rng(20261124+seed);X,y,h=gen(10000,r);tr,ga,ca,va,te=np.split(np.arange(10000),[5000,6500,7500,8500]);assert len(set(np.r_[tr,ga,ca,va,te]))==10000
 ex=fit_ex(X[tr],y[tr],r)
 def part(i):
  a,m,q=acquire(X[i],y[i],r,'train');p,v=pred(ex,a,q);return a,m,q,p,v
 a,m,q,p,v=part(ga);g0=gf(p,q);g=[Ridge(alpha=5).fit(g0[m[:,k],k],(p[m[:,k],k]-y[ga][m[:,k]])**2) for k in range(3)]
 a,m,q=acquire(X[tr],y[tr],r,'train');Z=feat(a,m,q);bs={'Early logistic':lr().fit(Z,y[tr]),'Early RF':RandomForestClassifier(n_estimators=80,min_samples_leaf=8,n_jobs=2,random_state=seed).fit(Z,y[tr]),'Early HGB':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,random_state=seed).fit(Z,y[tr])}
 def raw(a,m,q,p):
  w=gate(g,p,q);d={k:b.predict_proba(feat(a,m,q))[:,1] for k,b in bs.items()};d.update({'Equal fusion':fuse(p,m,np.ones_like(p))[0],'Quality fusion':fuse(p,m,q)[0],'AURA':fuse(p,m,w)[0]});return d,w
 a,m,q,p,v=part(ca);d,w=raw(a,m,q,p);cs={k:LogisticRegression(C=100).fit(lgt(z),y[ca]) for k,z in d.items()}
 a,m,q,p,v=part(va);d,w=raw(a,m,q,p);z,w0=fuse(p,m,w);ap=cal(cs['AURA'],z);b,e=ef(ap,z,p,v,m,q,w0);err=(ap>=.5)!=y[va];eb=lr().fit(b,err);ed=lr().fit(e,err);tau=np.quantile(ed.predict_proba(e)[:,1],.8)
 for s in scenarios:
  xx,yy,hh=gen(len(te),r,True) if s in ['mimic_shift','combined_shift'] else (X[te],y[te],h[te]);a,m,q=acquire(xx,yy,r,s);p,v=pred(ex,a,q);d,w=raw(a,m,q,p);z,w0=fuse(p,m,w);ap=cal(cs['AURA'],z);b,e=ef(ap,z,p,v,m,q,w0)
  models={k:(cal(cs[k],z0),ent(cal(cs[k],z0))) for k,z0 in d.items() if k!='AURA'};models.update({'AURA entropy':(ap,ent(ap)),'AURA no disagreement':(ap,eb.predict_proba(b)[:,1]),'AURA full':(ap,ed.predict_proba(e)[:,1])})
  for k,(pr,u) in models.items():
   rows.append(dict(seed=seed,scenario=s,model=k,**met(yy,pr,u)));a0=rc(yy,pr,u)
   for cv in np.arange(.1,1.001,.05):curves.append([seed,s,k,round(cv,2),a0[min(len(a0)-1,int(cv*len(a0))-1)]])
  keep=models['AURA full'][1]<=tau;thresholds.append([seed,s,keep.mean(),((ap[keep]>=.5)!=yy[keep]).mean() if keep.any() else np.nan]);aud.append([seed,s,len(yy),yy.mean(),m.sum(1).mean(),(q*m).sum()/m.sum(),hh.mean()]);assert np.allclose(w0.sum(1),1) and np.all(w0[~m]==0) and np.isfinite(ap).all()
  if seed==0 and s=='complete':
   pd.DataFrame(explain(p[:100],w[:100],m[:100],cs['AURA']),columns=['reference','prediction','clinical','laboratory','imaging']).to_csv(R/'explanation_examples.csv',index=False)
   rr=[]
   for k,sl in enumerate(S):
    sc,clf=ex[k][0].steps[0][1],ex[k][0].steps[1][1];zz=sc.transform(np.c_[a[:100,sl],q[:100,k]]);co=zz*clf.coef_[0];assert np.allclose(co.sum(1)+clf.intercept_[0],clf.decision_function(zz))
    for i in range(100):
     for j,n in enumerate(names[sl]+['quality']):rr.append([i,k,n,co[i,j]])
   pd.DataFrame(rr,columns=['row','modality','feature','first_expert_logit_contribution']).to_csv(R/'feature_explanations.csv',index=False)
 if seed==0:
  df=pd.DataFrame(X,columns=names);df['synthetic_label']=y;df['latent_mimic_audit_only']=h;df['split']=np.repeat(['train','gate','calibration','uncertainty','test'],[5000,1500,1000,1000,1500]);df.to_csv(ROOT/'synthetic_seed0.csv',index=False)
 print('seed',seed,'complete',flush=True)
df=pd.DataFrame(rows);df.to_csv(R/'metrics.csv',index=False);pd.DataFrame(aud,columns=['seed','scenario','n','positive_rate','mean_modalities','mean_quality','mimic_rate']).to_csv(R/'data_audit.csv',index=False);pd.DataFrame(thresholds,columns=['seed','scenario','coverage','risk']).to_csv(R/'fixed_threshold.csv',index=False)
df.groupby(['scenario','model']).agg({k:['mean','std'] for k in ['auroc','brier','ece','aurc','risk80','risk50','error_auc']}).to_csv(R/'summary.csv');piv=df.pivot(index=['seed','scenario'],columns='model',values='aurc');out=[]
for s in scenarios+['macro']:
 a=piv.groupby('seed').mean() if s=='macro' else piv.xs(s,level='scenario')
 for k in ['AURA no disagreement','AURA entropy','Early HGB','Equal fusion']:
  z=a['AURA full']-a[k];h=t.ppf(.975,len(z)-1)*z.std(ddof=1)/np.sqrt(len(z));out.append([s,k,z.mean(),z.mean()-h,z.mean()+h])
pd.DataFrame(out,columns=['scenario','comparator','delta_aurc','ci_low','ci_high']).to_csv(R/'paired_differences.csv',index=False)
cv=pd.DataFrame(curves,columns=['seed','scenario','model','coverage','risk']);cv.to_csv(R/'risk_curves.csv',index=False)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180});fig,ax=plt.subplots(figsize=(3.45,2.45))
for k in ['Early logistic','Early HGB','Equal fusion','AURA entropy','AURA full']:
 g0=cv[cv.model==k].groupby('coverage').risk.mean();ax.plot(g0.index,g0.values,label=k,linewidth=1.3)
ax.set(xlabel='Coverage',ylabel='Selective error rate');ax.legend(fontsize=6,frameon=False);fig.tight_layout();fig.savefig(F/'risk_coverage.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(3.45,2.5));a=df.groupby(['scenario','model']).auroc.mean().unstack().loc[scenarios,['Equal fusion','Quality fusion','Early HGB','AURA full']];a.plot(ax=ax,marker='o',markersize=3,linewidth=1);ax.set_xticks(range(len(a)));ax.set_xticklabels(['Full','MC30','MC50','Quality','MNAR','Mimic','Joint'],rotation=25,ha='right');ax.set_ylabel('AUROC');ax.legend(fontsize=6,frameon=False);fig.tight_layout();fig.savefig(F/'robustness.pdf');plt.close(fig)
print(df.groupby('model')[['auroc','brier','aurc','risk80','error_auc']].mean().round(4));print(pd.read_csv(R/'paired_differences.csv').tail(4).to_string(index=False))
