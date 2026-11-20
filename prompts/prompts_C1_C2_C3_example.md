# Prompts used in the benchmark

Generated from the benchmark code (`build_prompt`) for one ULB fraud transaction. The three conditions differ only in the SHAP block: C1 true attributions, C2 none, C3 every sign reversed. The model name is not shown in the prompt. Decoding: temperature 0, one generation per transaction and condition.

## C1: true SHAP

```
You are helping a fraud analyst. A model classified a transaction.
Model output: FRAUD (fraud probability 1.000).
Most influential features:
- V14 = -4.691; SHAP = +4.231 (pushes toward FRAUD)
- V10 = -4.485; SHAP = +1.882 (pushes toward FRAUD)
- V4 = 0.1195; SHAP = -1.147 (pushes toward LEGITIMATE)
- V12 = -2.644; SHAP = +0.885 (pushes toward FRAUD)
- V17 = -3.483; SHAP = +0.710 (pushes toward FRAUD)

Write an explanation with four short sections: Prediction Summary, SHAP Interpretation, Risk Assessment, Limitations. Mention only the features listed above.
```

## C2: no SHAP

```
You are helping a fraud analyst. A model classified a transaction.
Model output: FRAUD (fraud probability 1.000).
Most influential features:
- V14 = -4.691
- V10 = -4.485
- V4 = 0.1195
- V12 = -2.644
- V17 = -3.483

Write an explanation with four short sections: Prediction Summary, SHAP Interpretation, Risk Assessment, Limitations. Mention only the features listed above.
```

## C3: reversed SHAP

```
You are helping a fraud analyst. A model classified a transaction.
Model output: FRAUD (fraud probability 1.000).
Most influential features:
- V14 = -4.691; SHAP = -4.231 (pushes toward LEGITIMATE)
- V10 = -4.485; SHAP = -1.882 (pushes toward LEGITIMATE)
- V4 = 0.1195; SHAP = +1.147 (pushes toward FRAUD)
- V12 = -2.644; SHAP = -0.885 (pushes toward LEGITIMATE)
- V17 = -3.483; SHAP = -0.710 (pushes toward LEGITIMATE)

Write an explanation with four short sections: Prediction Summary, SHAP Interpretation, Risk Assessment, Limitations. Mention only the features listed above.
```
