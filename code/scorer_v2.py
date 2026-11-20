import re, numpy as np
import step2_llm_faithfulness as m
FR_RE=[r'toward[s]?\s+(?:classifying\s+)?(?:the\s+transaction\s+)?(?:as\s+)?(?:a\s+)?fraud', r'increas', r'rais(?:es|ed|ing)', r'higher risk', r'suspicio', r'indicative of fraud',
 r'contribut\w*\s+to\s+fraud', r'elevat', r'heighten', r'risk factor', r'higher than expected', r'associated with fraud']
LG_RE=[r'toward[s]?\s+(?:classifying\s+)?(?:the\s+transaction\s+)?(?:as\s+)?(?:a\s+)?legit', r'decreas', r'lower(?:s|ed)\b', r'reduc', r'lower risk', r'reassur', r'legitimate', r'benign', r'mitigat', r'protective', r'\bsafe',
 r'low (?:propensity for )?fraud', r'lower than expected']
# phrases that describe the overall decision or a comparison, not a feature's push
DROP=[r'(?:decision|conclusion|prediction)\s+to\s+(?:classify|label)[^.;]*?(?:legit\w*|fraud\w*)', r'(?:than|as)\s+expected\s+for\s+an?\s+(?:fraudulent|legitimate)\s+transaction',
      r'(?:classif\w+|predict\w+|label\w*)\s+(?:the\s+)?(?:transaction\s+)?as\s+(?:legit\w*|fraud\w*)(?=\s*[,.;]?\s*(?:and|but|$))(?<!toward classifying the transaction as fraud)']
NEG=r'\b(?:not|no|never|n\'t|isn\'t|doesn\'t|does not|is not|are not)\b[^.;]{0,45}?'
def _cues(s):
    low=s.lower()
    # keep directional "toward(s) classifying ... as X" phrases: protect them
    prot=re.findall(r'toward[s]?\s+(?:classifying\s+)?(?:the\s+transaction\s+)?as\s+(?:a\s+)?(?:fraud|legit)\w*|toward[s]?\s+classifying\s+(?:the\s+transaction\s+)?as\s+(?:a\s+)?(?:fraud|legit)\w*',low)
    f=l=0
    for p in prot:
        if 'fraud' in p: f+=2
        else: l+=2
    low2=low
    for p in prot: low2=low2.replace(p,' ')
    for d in DROP[:2]: low2=re.sub(d,' ',low2)
    low2=re.sub(r'(?:decision|conclusion|prediction)\s+to\s+classify[^.;]*',' ',low2)
    # negated fraud cues count as legit; negated legit cues as fraud
    for pat in (r'indicative of (?:fraud|fraudulent)',r'associated with (?:high |increased )?fraud(?:ulent)?',r'(?:high|higher) (?:risk|fraud)',r'suspicio\w*',r'a risk factor'):
        for mm in re.finditer(NEG+pat,low2):
            l+=1; low2=low2.replace(mm.group(0),' ',1)
    for pat in FR_RE: f+=len(re.findall(pat,low2))
    for pat in LG_RE: l+=len(re.findall(pat,low2))
    return f,l
def direction_of_v2(text,feat):
    sc=0
    for s in m.sentences(text):
        if m.mentions(s,feat):
            f,l=_cues(s); sc+=(f>l)-(l>f)
    return int(np.sign(sc))
