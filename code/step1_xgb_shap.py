"""
Step 1: rebuild the fraud model (XGBoost + SHAP) on IEEE-CIS with a time-based split.
Run on your computer:  pip install xgboost shap scikit-learn pandas numpy
Put train_transaction.csv and train_identity.csv in the same folder, then:  python step1_xgb_shap.py
Outputs: metrics printed, model_xgb.json, shap_top5_test_sample.csv (input for the LLM-faithfulness step).
NOTE: written and syntax-checked, but xgboost/shap could not be installed where it was drafted, so run it
and send back any error text.
"""
import pandas as pd, numpy as np, shap, xgboost as xgb, json
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)

USE_V_COLUMNS = False      # True = use the 339 anonymous V features (needs more memory; usually a bit better)
N_EXPLAIN = 500            # transactions to explain with SHAP (used later for the LLM evaluation)
TOP_K = 5
SEED = 0

tr = pd.read_csv('train_transaction.csv')
idn = pd.read_csv('train_identity.csv')
df = tr.merge(idn, on='TransactionID', how='left')
del tr, idn
df = df.sort_values('TransactionDT').reset_index(drop=True)      # time order

y = df['isFraud'].values.copy()
drop = ['isFraud', 'TransactionID']
if not USE_V_COLUMNS:
    drop += [c for c in df.columns if c.startswith('V')]
X = df.drop(columns=drop)
ids = df['TransactionID'].values.copy()
del df

for c in X.select_dtypes(['object', 'string']).columns:           # categorical -> integer codes
    X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
X = X.astype('float32')

cut = int(len(y) * 0.8)                                            # first 80% train, last 20% test (time split)
Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
idte = ids[cut:]
print('train', Xtr.shape, 'test', Xte.shape,
      'fraud: train %.2f%% test %.2f%%' % (ytr.mean() * 100, yte.mean() * 100))

model = xgb.XGBClassifier(
    n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    scale_pos_weight=(1 - ytr.mean()) / ytr.mean(),                # class-imbalance handling
    eval_metric='aucpr', tree_method='hist', random_state=SEED, n_jobs=-1)
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]

print('ROC-AUC %.4f | PR-AUC %.4f' % (roc_auc_score(yte, p), average_precision_score(yte, p)))
for th in (0.3, 0.5, 0.7):
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p >= th, average='binary', zero_division=0)
    print('threshold %.1f: precision %.3f recall %.3f F1 %.3f accuracy %.4f'
          % (th, pr, rc, f1, ((p >= th) == yte).mean()))
print('accuracy of always predicting "legitimate": %.4f' % (1 - yte.mean()))
model.save_model('model_xgb.json')

# --- SHAP on a reproducible sample of test transactions (half fraud, half legitimate) ---
rng = np.random.default_rng(SEED)
fraud_idx = np.where(yte == 1)[0]
legit_idx = np.where(yte == 0)[0]
sel = np.concatenate([rng.choice(fraud_idx, N_EXPLAIN // 2, replace=False),
                      rng.choice(legit_idx, N_EXPLAIN // 2, replace=False)])
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(Xte.iloc[sel])

rows = []
for r, i in enumerate(sel):
    order = np.argsort(-np.abs(sv[r]))[:TOP_K]
    row = {'TransactionID': int(idte[i]), 'true_label': int(yte[i]), 'fraud_prob': float(p[i])}
    for k, j in enumerate(order, 1):
        row[f'feat{k}'] = Xte.columns[j]
        row[f'value{k}'] = None if pd.isna(Xte.iloc[i, j]) else float(Xte.iloc[i, j])
        row[f'shap{k}'] = float(sv[r][j])
    rows.append(row)
pd.DataFrame(rows).to_csv('shap_top5_test_sample.csv', index=False)
print('saved shap_top5_test_sample.csv (', len(rows), 'transactions )')
