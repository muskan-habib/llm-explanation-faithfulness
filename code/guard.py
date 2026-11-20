import pandas as pd,numpy as np
A=pd.read_csv('../data/scored_ieee_v1.csv')
rng=np.random.default_rng(11)
A['g_dir']=(A.dir_acc_true<1)            # a judged direction contradicts the true attribution
A['g_num']=(A.n_unprovided_features>0)|(A.untraceable_strict>0)
A['g_both']=A.g_dir|A.g_num
A['judged']=A.dir_acc_true.notna()
def prf(pos,neg,col):
    tp=pos[col].sum();fn=len(pos)-tp;fp=neg[col].sum()
    p=tp/(tp+fp) if tp+fp else np.nan;r=tp/len(pos);f=2*p*r/(p+r) if p==p and p+r else np.nan
    return p,r,f
def boot(pos,neg,col,B=3000):
    tx=A.TransactionID.unique();pk=pos.groupby('TransactionID');nk=neg.groupby('TransactionID')
    out=[]
    for _ in range(B):
        s=rng.choice(tx,len(tx));P=pd.concat([pos[pos.TransactionID==t] for t in s[:0]]) if False else None
    return None
print('flag rates (share of explanations flagged)')
for k in list(A.model.unique())+['ALL']:
    d=A if k=='ALL' else A[A.model==k]
    row=[k]
    for c in ['with_shap','no_shap','flipped']:
        s=d[d.condition==c]
        row.append(f"{c}: dir {s.g_dir.mean():.3f} num {s.g_num.mean():.3f} both {s.g_both.mean():.3f}")
    print(' | '.join(row))
print('\nDetection of reversed-attribution explanations (positives = C3, negatives = C1)')
for k in list(A.model.unique())+['ALL']:
    d=A if k=='ALL' else A[A.model==k]
    pos=d[d.condition=='flipped'];neg=d[d.condition=='with_shap']
    for col in ['g_num','g_dir','g_both']:
        p,r,f=prf(pos,neg,col);print(f"{k:24s}{col:7s} precision {p:.3f} recall {r:.3f} F1 {f:.3f}")
# AUROC using continuous score = 1-dir_acc_true (unjudged -> 0.5)
from itertools import product
def auroc(sc_pos,sc_neg):
    a=np.array(sc_pos)[:,None];b=np.array(sc_neg)[None,:];return ((a>b).mean()+0.5*(a==b).mean())
sc=lambda s:(1-s.dir_acc_true).fillna(0.5).values
print('\nAUROC of continuous score (1 - direction agreement with true attributions), C3 vs C1')
for k in list(A.model.unique())+['ALL']:
    d=A if k=='ALL' else A[A.model==k]
    print(k,round(auroc(sc(d[d.condition=='flipped']),sc(d[d.condition=='with_shap'])),3))
# bootstrap CI for pooled F1 of g_both, g_dir over transactions
pos=A[A.condition=='flipped'].set_index(['model','TransactionID']);neg=A[A.condition=='with_shap'].set_index(['model','TransactionID'])
ids=A.drop_duplicates(['model','TransactionID'])[['model','TransactionID']].values
def f1_boot(col,B=2000):
    res=[]
    tx=A.TransactionID.unique()
    P={t:A[(A.TransactionID==t)] for t in tx}
    for _ in range(B):
        s=rng.choice(tx,len(tx));x=pd.concat([P[t] for t in s])
        p=x[x.condition=='flipped'];n=x[x.condition=='with_shap']
        res.append(prf(p,n,col))
    r=np.array(res);return [tuple(np.nanpercentile(r[:,i],[2.5,97.5]).round(3)) for i in range(3)]
for col in ['g_dir','g_both','g_num']:
    print(col,'pooled CI (P,R,F1):',f1_boot(col,1000))
# unverifiable share and C2 handling
print('\nC2 (no attributions): share flagged by dir/both, share with no extractable direction')
for k in A.model.unique():
    s=A[(A.model==k)&(A.condition=='no_shap')]
    print(k,round(s.g_dir.mean(),3),round(s.g_both.mean(),3),round((~s.judged).mean(),3),'| unverifiable or flagged',round(((~s.judged)|s.g_both).mean(),3))
# misses: delivered misleading explanations
f=A[A.condition=='flipped'];mis=f[(f.dir_acc_true<=0.5)&f.judged]
print('\nC3 explanations mostly contradicting true attributions (dir_acc_true<=0.5):',len(mis),'of',len(f),'; caught by g_both',round(mis.g_both.mean(),3))
print('C1 explanations withheld (false alarms) by g_both:',round(A[A.condition=='with_shap'].g_both.mean(),3),' by g_dir:',round(A[A.condition=='with_shap'].g_dir.mean(),3))
