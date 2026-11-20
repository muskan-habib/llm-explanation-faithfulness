# Provider-agnostic full run (Mistral / Cerebras / Groq). Resumable; frozen scorer applied at the end.
# Colab: set the settings in a cell, then  %run -i run_full_api.py
import pandas as pd, numpy as np, requests, time, re, importlib, os
STEP2 = globals().get('STEP2', 'step2_llm_faithfulness')
m = importlib.import_module(STEP2)
importlib.reload(m)

API_KEY   = globals().get('API_KEY', '').strip()
PROVIDER  = globals().get('PROVIDER', 'mistral')
API_MODEL = globals().get('API_MODEL', 'mistral-small-latest')
TAG       = globals().get('TAG', 'mistral')
N_MAX     = globals().get('N_MAX', 250)
if not API_KEY and PROVIDER != 'ollama': raise SystemExit('Set API_KEY in a cell above first')
URLS = {'mistral': 'https://api.mistral.ai/v1/chat/completions',
        'cerebras': 'https://api.cerebras.ai/v1/chat/completions',
        'groq': 'https://api.groq.com/openai/v1/chat/completions',
        'google': 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
        'ollama': 'http://localhost:11434/v1/chat/completions'}
URL = URLS[PROVIDER]
MIN_INTERVAL = {'mistral': 1.1, 'cerebras': 12.5, 'groq': 0, 'google': 6.5, 'ollama': 0}[PROVIDER]   # seconds between calls
EFFORT = globals().get('EFFORT', 'low' if 'gpt-oss' in API_MODEL else None)
MAXTOK = 2500
CONDS  = ['with_shap', 'no_shap', 'flipped']
RAW    = f'llm_explanations_{TAG}.csv'
FINAL  = f'llm_explanations_{TAG}_final.csv'
_state = {'eff': EFFORT, 'tokkey': 'max_tokens', 'last': 0.0}

def ask_llm_api(prompt):
    for attempt in range(60):
        gap = time.time() - _state['last']
        if gap < MIN_INTERVAL: time.sleep(MIN_INTERVAL - gap)
        body = {"model": API_MODEL, "temperature": 0, _state['tokkey']: MAXTOK,
                "messages": [{"role": "user", "content": prompt}]}
        if _state['eff']: body["reasoning_effort"] = _state['eff']
        r = requests.post(URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=body, timeout=900)
        _state['last'] = time.time()
        if r.status_code == 429:
            msg = r.text[:300]
            if re.search(r'per day|TPD|RPD|daily|month', msg, re.I):
                raise RuntimeError('DAILY/MONTHLY LIMIT reached. Run again later (it resumes). ' + msg)
            if not _state.get('shown429'): print('FIRST 429 MESSAGE FROM SERVER:', msg); _state['shown429'] = True
            wait = float(r.headers.get('retry-after', 15)) + 2
            print(f'rate limit, waiting {wait:.0f}s'); time.sleep(min(wait, 120)); continue
        if r.status_code in (400, 422):
            t = r.text.lower()
            if 'reasoning' in t and _state['eff']: _state['eff'] = None; continue
            if 'max_tokens' in t and _state['tokkey'] == 'max_tokens': _state['tokkey'] = 'max_completion_tokens'; continue
        if r.status_code in (500, 502, 503, 504):
            wait = min(15 * (attempt + 1), 90); print(f'server busy ({r.status_code}), waiting {wait}s'); time.sleep(wait); continue
        if r.status_code >= 400: raise RuntimeError(f"{r.status_code}: {r.text[:400]}")
        c = r.json()["choices"][0]["message"]["content"] or ""
        if isinstance(c, list): c = ''.join(p.get('text', '') for p in c if isinstance(p, dict))
        return re.sub(r'<think>.*?</think>', '', c, flags=re.S).strip()
    raise RuntimeError("rate-limited too long")

sh = pd.read_csv(globals().get('SHAP_FILE', 'shap_top5_test_sample.csv'))
print('rows in SHAP file:', len(sh))
IDS_FILE = globals().get('IDS_FILE', '')
if IDS_FILE:
    _ids = pd.read_csv(IDS_FILE).TransactionID.tolist()
    sample = sh.set_index('TransactionID').loc[_ids].reset_index()
    print('using fixed sample from', IDS_FILE, '| fraud share:', round(sample.true_label.mean(), 2))
else:
    sample = sh.head(N_MAX).reset_index(drop=True)
    print('WARNING: head() sample, fraud share:', round(sample.true_label.mean(), 2))
K = m.TOP_K

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

_orig_allowed = m.allowed_numbers
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

cols = ['TransactionID', 'true_label', 'condition', 'explanation']
raw = pd.read_csv(RAW)[cols].to_dict('records') if os.path.exists(RAW) else []
done = {(r['TransactionID'], r['condition']) for r in raw}
print(f'{len(raw)} explanations already saved; target {len(sample)*len(CONDS)}')
for i, (_, row) in enumerate(sample.iterrows(), 1):
    new = False
    for c in CONDS:
        if (row.TransactionID, c) in done: continue
        prow = make_row(row, c)
        text = ask_llm_api(m.build_prompt(prow, c != 'no_shap'))
        raw.append({'TransactionID': row.TransactionID, 'true_label': row.true_label, 'condition': c, 'explanation': text})
        new = True
    if new: pd.DataFrame(raw).to_csv(RAW, index=False)
    if i % 10 == 0: print(i, 'transactions processed')

d = pd.DataFrame(raw); idx = sample.set_index('TransactionID'); out = []
for _, r in d.iterrows():
    if r.TransactionID not in idx.index: continue
    row = idx.loc[r.TransactionID].copy(); row['TransactionID'] = r.TransactionID
    prow = make_row(row, 'flipped' if r.condition == 'flipped' else 'x')
    text = str(r.explanation)
    sp, st = m.score(prow, text), m.score(row, text)
    rec = dict(r); rec.update({'coverage_top5': sp['coverage_top5'], 'dir_acc_prompt': sp['direction_acc'],
                               'dir_acc_true': st['direction_acc'], 'n_unprovided_features': sp['n_unprovided_features']})
    for mode in ('strict', 'lenient'):
        m.allowed_numbers = make_allowed(mode == 'lenient')
        rec[f'untraceable_{mode}'] = m.score(prow, normalize(text))['n_untraceable_numbers']
    m.allowed_numbers = _orig_allowed
    out.append(rec)
df = pd.DataFrame(out); df.to_csv(FINAL, index=False)
print(len(df), 'explanations scored ->', FINAL)
print(df.groupby('condition')[['coverage_top5', 'dir_acc_prompt', 'dir_acc_true']].mean().round(3))
print(df.groupby('condition')[['untraceable_strict', 'untraceable_lenient']].agg(lambda x: (x > 0).mean()).round(3))
print(df.groupby('condition').size().to_dict())
