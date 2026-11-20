"""
Step 2: ask a local LLM (Ollama) to explain each SHAP-explained transaction, then score the explanations.
Needs: shap_top5_test_sample.csv (from step 1), Ollama running, and:  pip install requests pandas numpy
Run:  python step2_llm_faithfulness.py            (full run, 2 conditions x 500 transactions)
      python step2_llm_faithfulness.py --n 20     (quick test on 20 transactions first!)
Outputs: llm_explanations.csv (raw texts + scores), summary printed at the end.
"""
import re, sys, json, argparse, requests, pandas as pd, numpy as np

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL = 'gpt-oss:20b-cloud'       # change to another local model, e.g. 'llama3.1:8b', to compare models
TOP_K = 5

# ---------------- prompts ----------------
def build_prompt(row, with_shap):
    lines = []
    for k in range(1, TOP_K + 1):
        f, v, s = row[f'feat{k}'], row[f'value{k}'], row[f'shap{k}']
        v_txt = 'missing' if pd.isna(v) else f'{v:.4g}'
        if with_shap:
            direction = 'pushes toward FRAUD' if s > 0 else 'pushes toward LEGITIMATE'
            lines.append(f'- {f} = {v_txt}; SHAP = {s:+.3f} ({direction})')
        else:
            lines.append(f'- {f} = {v_txt}')
    label = 'FRAUD' if row['fraud_prob'] >= 0.5 else 'LEGITIMATE'
    return (
        'You are helping a fraud analyst. A model classified a transaction.\n'
        f'Model output: {label} (fraud probability {row["fraud_prob"]:.3f}).\n'
        'Most influential features:\n' + '\n'.join(lines) + '\n\n'
        'Write an explanation with four short sections: Prediction Summary, SHAP Interpretation, '
        'Risk Assessment, Limitations. Mention only the features listed above.')

def ask_llm(prompt):
    r = requests.post(OLLAMA_URL, json={'model': MODEL, 'prompt': prompt, 'stream': False,
                                        'options': {'temperature': 0}}, timeout=600)
    r.raise_for_status()
    return r.json()['response']

# ---------------- scoring ----------------
FRAUD_CUES = ['toward fraud', 'towards fraud', 'increase', 'raises', 'raised', 'higher risk', 'suspicious',
              'indicative of fraud', 'contributes to fraud', 'elevat', 'heightens', 'pushes toward fraud', 'risk factor']
LEGIT_CUES = ['toward legit', 'towards legit', 'decrease', 'lowers', 'lowered', 'reduces', 'reduced', 'lower risk',
              'reassuring', 'legitimate', 'benign', 'mitigat', 'protective', 'safe']
FEATURE_PATTERN = re.compile(r'\b(TransactionAmt|TransactionDT|ProductCD|card[1-6]|addr[12]|dist[12]|'
                             r'[PR]_emaildomain|C\d{1,2}|D\d{1,2}|M[1-9]|id[_-]\d{2}|DeviceType|DeviceInfo)\b')

def sentences(text):
    return re.split(r'(?<=[.!?\n])\s+', text)

def mentions(text, feat):
    return re.search(r'\b' + re.escape(feat) + r'\b', text, flags=re.I) is not None

def direction_of(text, feat):
    """Return +1 (fraud), -1 (legit) or 0 (unclear) from sentences that name the feature."""
    sc = 0
    for s in sentences(text):
        if mentions(s, feat):
            low = s.lower()
            f = sum(c in low for c in FRAUD_CUES)
            l = sum(c in low for c in LEGIT_CUES)
            sc += (f > l) - (l > f)
    return int(np.sign(sc))

def numbers_in(text):
    return [float(x.replace(',', '')) for x in re.findall(r'(?<![\w.])-?\d[\d,]*\.?\d*', text) if x.strip('-.,')]

def allowed_numbers(row):
    nums = [row['fraud_prob'], row['fraud_prob'] * 100, 0.5, 50, TOP_K]
    for k in range(1, TOP_K + 1):
        nums += [row[f'value{k}'], row[f'shap{k}'], abs(row[f'shap{k}'])]
        m = re.search(r'\d+$', str(row[f'feat{k}']))      # numbers inside names like card1, C13
        if m: nums.append(float(m.group()))
    return [x for x in nums if not pd.isna(x)]

def close(a, b):
    return abs(a - b) <= max(0.0015, 0.01 * abs(b))        # tolerate rounding

def score(row, text):
    feats = [row[f'feat{k}'] for k in range(1, TOP_K + 1)]
    signs = [np.sign(row[f'shap{k}']) for k in range(1, TOP_K + 1)]
    ment = [mentions(text, f) for f in feats]
    # direction agreement among mentioned features
    dirs = [direction_of(text, f) if m else None for f, m in zip(feats, ment)]
    judged = [(d, s) for d, s in zip(dirs, signs) if d not in (None, 0)]
    dir_acc = np.mean([d == s for d, s in judged]) if judged else np.nan
    # features named that were NOT provided
    named = set(m.group(1).lower().replace('-', '_') for m in FEATURE_PATTERN.finditer(text))
    given = set(f.lower() for f in feats)
    extra_feats = sorted(named - given)
    # numbers not traceable to the prompt (strip section numbering like "1." at line start)
    clean = re.sub(r'(?m)^\s*\d+[.)]\s', '', text)
    allowed = allowed_numbers(row)
    bad_nums = [n for n in numbers_in(clean) if not any(close(n, a) for a in allowed)]
    return {'coverage_top5': float(np.mean(ment)), 'direction_acc': dir_acc,
            'n_unprovided_features': len(extra_feats), 'unprovided_features': ';'.join(extra_feats),
            'n_untraceable_numbers': len(bad_nums), 'untraceable_numbers': ';'.join(map(str, bad_nums[:10]))}

# ---------------- main ----------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=None); a = ap.parse_args()
    df = pd.read_csv('shap_top5_test_sample.csv')
    if a.n: df = df.sample(a.n, random_state=0)
    out = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        for cond, with_shap in (('with_shap', True), ('no_shap', False)):
            text = ask_llm(build_prompt(row, with_shap))
            rec = {'TransactionID': row['TransactionID'], 'true_label': row['true_label'],
                   'condition': cond, 'explanation': text}
            rec.update(score(row, text)); out.append(rec)
        if i % 10 == 0: print(i, 'transactions done', flush=True)
    res = pd.DataFrame(out); res.to_csv('llm_explanations.csv', index=False)
    print('\n=== SUMMARY (mean per condition) ===')
    print(res.groupby('condition')[['coverage_top5', 'direction_acc', 'n_unprovided_features',
                                    'n_untraceable_numbers']].mean().round(3))
    print('\nShare of explanations with at least one unprovided feature:')
    print(res.groupby('condition').apply(lambda g: (g.n_unprovided_features > 0).mean()).round(3))
    print('Share with at least one untraceable number:')
    print(res.groupby('condition').apply(lambda g: (g.n_untraceable_numbers > 0).mean()).round(3))
