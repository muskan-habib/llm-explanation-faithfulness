"""
Second dataset: ULB credit-card fraud (Kaggle: mlg-ulb/creditcardfraud). Same pipeline as step1_xgb_shap.py:
chronological 80/20 split, XGBoost, TreeSHAP top-5 per transaction (computed with XGBoost's own TreeSHAP, no 'shap' package needed).
Output: shap_top5_test_sample.csv (balanced, shuffled), metrics printed.
Run (Kaggle notebook, dataset 'Credit Card Fraud Detection' attached):  %run step1_ulb.py
"""
import glob, os, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

N_PER_CLASS = 50          # 50 fraud + 50 legitimate = 100 explained transactions (all test frauds if fewer)
TOP_K, SEED = 5, 0
OUT = 'shap_top5_test_sample.csv'

paths = glob.glob('/kaggle/input/**/creditcard.csv', recursive=True) + glob.glob('creditcard.csv') + glob.glob('/content/creditcard.csv')
assert paths, 'creditcard.csv not found: attach the Kaggle dataset "Credit Card Fraud Detection" (mlg-ulb) or upload the file'
df = pd.read_csv(paths[0])
df = df.rename(columns={'Time': 'TxnTime', 'Amount': 'TxnAmount'})      # unambiguous tokens for the scorer
df = df.sort_values('TxnTime', kind='stable').reset_index()               # 'index' = original row id -> TransactionID
y = df['Class'].values.copy(); ids = df['index'].values.copy()
X = df.drop(columns=['Class', 'index']).astype('float32')
print('rows', len(df), 'fraud rate %.3f%%' % (y.mean() * 100))

cut = int(len(y) * 0.8)
Xtr, Xte, ytr, yte, idte = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:], ids[cut:]
print('train', Xtr.shape, 'test', Xte.shape, 'fraud: train %.3f%% test %.3f%% (test frauds: %d)' % (ytr.mean()*100, yte.mean()*100, yte.sum()))

model = xgb.XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                          scale_pos_weight=(1 - ytr.mean()) / ytr.mean(), eval_metric='aucpr',
                          tree_method='hist', random_state=SEED, n_jobs=-1)
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]
print('ROC-AUC %.4f | PR-AUC %.4f' % (roc_auc_score(yte, p), average_precision_score(yte, p)))
for th in (0.3, 0.5, 0.7):
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p >= th, average='binary', zero_division=0)
    print('threshold %.1f: precision %.3f recall %.3f F1 %.3f accuracy %.4f' % (th, pr, rc, f1, ((p >= th) == yte).mean()))
print('accuracy of always predicting legitimate: %.4f' % (1 - yte.mean()))

rng = np.random.default_rng(SEED)
fr, lg = np.where(yte == 1)[0], np.where(yte == 0)[0]
sel = np.concatenate([rng.choice(fr, min(N_PER_CLASS, len(fr)), replace=False), rng.choice(lg, N_PER_CLASS, replace=False)])
sel = rng.permutation(sel)                                                # shuffled: head(N) is then balanced too
contrib = model.get_booster().predict(xgb.DMatrix(Xte.iloc[sel]), pred_contribs=True)   # TreeSHAP in log-odds; last column = bias
sv = contrib[:, :-1]
rows = []
for r, i in enumerate(sel):
    order = np.argsort(-np.abs(sv[r]))[:TOP_K]
    row = {'TransactionID': int(idte[i]), 'true_label': int(yte[i]), 'fraud_prob': float(p[i])}
    for k, j in enumerate(order, 1):
        row[f'feat{k}'] = Xte.columns[j]; row[f'value{k}'] = float(Xte.iloc[i, j]); row[f'shap{k}'] = float(sv[r][j])
    rows.append(row)
out = pd.DataFrame(rows); out.to_csv(OUT, index=False)
print('saved', OUT, len(out), 'transactions | fraud share', out.true_label.mean().round(2))
print('sanity: sum of SHAP + bias reproduces the margin ->', np.allclose(contrib.sum(1), model.predict(Xte.iloc[sel], output_margin=True), atol=1e-3))
