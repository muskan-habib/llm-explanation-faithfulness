# Faithful or Merely Fluent? Benchmarking whether LLM-written explanations stay faithful to feature attributions in fraud detection

Code, prompts, raw model outputs and human labels for the paper of the same name (Muskan Habib).
Release status: draft, prepared for a Zenodo/GitHub deposit.

## What the benchmark does
An XGBoost fraud classifier is trained on public fraud data and explained with TreeSHAP (top five features per transaction).
A language model then writes an analyst-facing explanation under three conditions:
- **C1** true SHAP values in the prompt
- **C2** no SHAP values
- **C3** every SHAP sign reversed

A keyword/number scorer checks each explanation against the true attributions (coverage, direction agreement, unprovided features, untraceable numbers) and a rule-based guard flags explanations that contradict the true attributions.

## Contents
| Folder | What is in it |
|---|---|
| `code/` | `step1_*` classifier + SHAP, `step2_*` prompts and scorer v1, `scorer_v2.py` (revised direction rules), `run_full_api.py` (generation runner), `rescore*.py`, `stats.py` (CIs, significance tests), `guard.py` (verification guard) |
| `prompts/` | The three prompts for one example transaction, generated from the code |
| `data/` | `sample_ids_100.csv` (the 100 shared IEEE-CIS test transactions), `ulb_shap_top5_test_sample.csv` (ULB SHAP file), `scored_ieee_v1.csv` / `scored_ieee_v2.csv` (all IEEE-CIS explanations scored by scorer v1 / v2) |
| `outputs/` | Raw explanations, 300 per model (100 transactions x 3 conditions): four models on IEEE-CIS, llama3.1:8b on ULB |
| `human_labels/` | Independent annotator's 50 labels, the key mapping items to transactions and conditions, and the blind sheet |

## Models
llama3.1:8b (local, Ollama), gpt-oss-20b and a Qwen 27B model (hosted API, Groq), Gemini 3.5 Flash-Lite (Google API). Temperature 0, one generation per transaction and condition.

## Reproducing the tables
```
pip install -r requirements.txt
cd code
python stats.py     # significance tests (reads ../data/scored_ieee_v1.csv)
python guard.py     # verification guard results
```
Regenerating explanations needs your own API keys and Ollama; the keys are set as variables in the runner, and none are included here.

## Data not redistributed
- **IEEE-CIS Fraud Detection**: raw transaction features are not included, because of the competition's data terms. `step1_xgb_shap.py` regenerates the model and SHAP file from the original Kaggle data. The explanation files do quote some top-five feature values; check the competition rules before depositing them publicly.
- **ULB credit-card fraud**: the SHAP file for the 100-transaction sample is included; see the Kaggle dataset page (mlg-ulb/creditcardfraud) for its licence and cite its authors.

## Known limitations (see the paper)
Small paired sample (100 transactions per condition), one model on the second dataset, keyword-based scoring (scorer v2 was tuned on the independent labels it is compared with), one independent annotator, no mixed-effects model fitted.

## Licence
Code: MIT. Explanations, prompts and labels: CC BY 4.0. Both can be changed by the author before release.
