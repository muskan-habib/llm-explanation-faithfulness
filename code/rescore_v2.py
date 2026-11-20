import pandas as pd,numpy as np,re,importlib
import step2_llm_faithfulness as m, scorer_v2 as V
m.direction_of=V.direction_of_v2
K=m.TOP_K
def make_row(row, kind):
    r = row.copy()
    for k in range(1, K+1): r[f'shap{k}'] = -row[f'shap{k}'] if kind == 'flipped' else row[f'shap{k}']
    return r
SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻', '0123456789-')
def normalize(t):
    t = re.sub(r'\((?:e\.g\.|for example|such as)[^)]*\)', '', t)
    t = re.sub(r'(?<![\w.])(\d+\.?\d*)[eE]([+-]?\d+)(?!\w)', lambda mm: repr(float(mm.group(1)) * 10**int(mm.group(2))), t)
    def sci(mm):
        return repr(float(mm.group(1)) * 10**int(mm.group(2).translate(SUP).replace('−', '-')))
    return re.sub(r'(\d+\.?\d*)\s*[×x*]\s*10\^?\{?([⁰¹²³⁴⁵⁶⁷⁸⁹⁻−+\d-]+)\}?', sci, t)

_orig_allowed = importlib.import_module('step2_llm_faithfulness').allowed_numbers
def make_allowed(lenient):
    def f(row):
        base = list(_orig_allowed(row)); p = row['fraud_prob']; extra = []
        for q in (p, 1-p):
            for k in (0, 1, 2): extra += [round(q*100, k), round(q, k+2)]
        s = np.array([row[f'shap{k}'] for k in range(1, K+1)])
        extra += list(s) + list(-s) + list(np.abs(s))
        if lenient:
            d = [s.sum(), -s.sum(), np.abs(s.sum()), s[s>0].sum(), s[s<0].sum(), -s[s<0].sum()]
            extra += d + [round(x, 1) for x in d] + [round(x, 2) for x in d]
        return base + extra
    return f


sh=pd.read_csv('shap.csv');idx=sh.set_index('TransactionID')
M={'llama3.1:8b':'llb.csv','gpt-oss-20b':'gf.csv','Qwen 27B':'q.csv','Gemini 3.5 Flash-Lite':'flb2.csv'}
out=[]
for k,f in M.items():
    d=pd.read_csv(f)
    for _,r in d.iterrows():
        row=idx.loc[r.TransactionID].copy();row['TransactionID']=r.TransactionID
        prow=make_row(row,'flipped' if r.condition=='flipped' else 'x')
        text=str(r.explanation)
        sp,st=m.score(prow,text),m.score(row,text)
        rec=dict(model=k,TransactionID=r.TransactionID,true_label=r.true_label,condition=r.condition,coverage_top5=sp['coverage_top5'],dir_acc_prompt=sp['direction_acc'],dir_acc_true=st['direction_acc'],n_unprovided_features=sp['n_unprovided_features'],explanation=text,fraud_prob=row['fraud_prob'])
        for mode in ('strict','lenient'):
            m.allowed_numbers=make_allowed(mode=='lenient')
            rec['untraceable_'+mode]=m.score(prow,normalize(text))['n_untraceable_numbers']
        m.allowed_numbers=_orig_allowed
        out.append(rec)
pd.DataFrame(out).to_pickle('rescored_v2.pkl');print(len(out))
