import pandas as pd,numpy as np,itertools
from scipy.stats import binomtest,wilcoxon
A=pd.read_csv('../data/scored_ieee_v1.csv')
A['comp']=((A.dir_acc_prompt==1)&(A.n_unprovided_features==0)&(A.untraceable_strict==0)).astype(float).where(A.dir_acc_prompt.notna())
models=list(A.model.unique())
def holm(p):
    p=np.array(p);o=np.argsort(p);m=len(p);adj=np.empty(m);run=0
    for r,i in enumerate(o):
        run=max(run,(m-r)*p[i]);adj[i]=min(1,run)
    return adj
def pv(d,c,col): return d[d.condition==c].set_index('TransactionID')[col]
out={}
# A: C1 vs C2 composite McNemar + Wilcoxon direction
rows=[]
for k in models:
    d=A[A.model==k]
    a,b=pv(d,'with_shap','comp'),pv(d,'no_shap','comp');x=pd.concat([a,b],axis=1,keys=['a','b']).dropna()
    n10=int(((x.a==1)&(x.b==0)).sum());n01=int(((x.a==0)&(x.b==1)).sum())
    p_mc=binomtest(n10,n10+n01,0.5).pvalue if n10+n01 else 1
    a,b=pv(d,'with_shap','dir_acc_prompt'),pv(d,'no_shap','dir_acc_prompt');y=pd.concat([a,b],axis=1,keys=['a','b']).dropna()
    w=wilcoxon(y.a,y.b).pvalue
    rows.append((k,len(x),n10,n01,p_mc,len(y),w))
adj_mc=holm([r[4] for r in rows]);adj_w=holm([r[6] for r in rows])
print('H1 C1 vs C2');
for r,am,aw in zip(rows,adj_mc,adj_w): print(r[0],'comp pairs',r[1],'C1only',r[2],'C2only',r[3],'McNemar p=%.2g holm=%.2g'%(r[4],am),'| dir pairs',r[5],'Wilcoxon p=%.2g holm=%.2g'%(r[6],aw))
# B: C3 prompt vs true
rows=[]
for k in models:
    d=A[(A.model==k)&(A.condition=='flipped')].dropna(subset=['dir_acc_prompt'])
    w=wilcoxon(d.dir_acc_prompt,d.dir_acc_true).pvalue;rows.append((k,len(d),w))
adj=holm([r[2] for r in rows]);print('H2 C3 prompt vs true')
for r,a in zip(rows,adj): print(r[0],r[1],'Wilcoxon p=%.2g holm=%.2g'%(r[2],a))
# E: C1 vs C3(prompt) direction
rows=[]
for k in models:
    d=A[A.model==k];a,b=pv(d,'with_shap','dir_acc_prompt'),pv(d,'flipped','dir_acc_prompt');y=pd.concat([a,b],axis=1,keys=['a','b']).dropna()
    rows.append((k,len(y),(y.a-y.b).mean(),wilcoxon(y.a,y.b).pvalue if (y.a!=y.b).any() else 1))
adj=holm([r[3] for r in rows]);print('C1 vs C3 prompt-following (difference, Wilcoxon)')
for r,a in zip(rows,adj): print(r[0],r[1],'mean diff %.3f'%r[2],'p=%.2g holm=%.2g'%(r[3],a))
# C: H4 cluster bootstrap C1 composite wrong-prediction vs right
d=A[A.condition=='with_shap'].dropna(subset=['comp']).copy();d['err']=d.ty.isin(['FN','FP'])
tx=d.TransactionID.unique();rng=np.random.default_rng(3);g={t:v for t,v in d.groupby('TransactionID')}
diff=d[d.err].comp.mean()-d[~d.err].comp.mean();bs=[]
for _ in range(4000):
    s=rng.choice(tx,len(tx));x=pd.concat([g[t] for t in s]);
    if x.err.any() and (~x.err).any(): bs.append(x[x.err].comp.mean()-x[~x.err].comp.mean())
print('H4 composite C1 error-pred minus correct-pred: diff %.3f CI [%.3f, %.3f]'%(diff,np.percentile(bs,2.5),np.percentile(bs,97.5)),'n err',int(d.err.sum()),'n ok',int((~d.err).sum()),'p(bootstrap, 2-sided)=%.3f'%min(1,2*min(np.mean(np.array(bs)>=0),np.mean(np.array(bs)<=0))))
# D: model comparison C1 composite pairwise
rows=[]
for a,b in itertools.combinations(models,2):
    x=pd.concat([pv(A[A.model==a],'with_shap','comp'),pv(A[A.model==b],'with_shap','comp')],axis=1,keys=['a','b']).dropna()
    n10=int(((x.a==1)&(x.b==0)).sum());n01=int(((x.a==0)&(x.b==1)).sum())
    rows.append((a,b,len(x),n10,n01,binomtest(n10,n10+n01,.5).pvalue if n10+n01 else 1))
adj=holm([r[5] for r in rows]);print('Model comparison C1 composite (McNemar)')
for r,a in zip(rows,adj): print(r[0],'vs',r[1],'pairs',r[2],r[3],r[4],'p=%.2g holm=%.2g'%(r[5],a))
