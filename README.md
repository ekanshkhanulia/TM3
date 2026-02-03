## Report Preview

<p align="center">
  <img src="assets/Screenshot%20(556).png" width="32%" alt="Report page 1">
  <img src="assets/Screenshot%20(557).png" width="32%" alt="Report page 2">
  <img src="assets/Screenshot%20(558).png" width="32%" alt="Report page 3">
</p>

# Sentence-Level Bias Detection in News

This project studies sentence-level media bias and separates emotion-driven bias
from framing-driven bias. It compares two signals for perceived bias: emotional
tone (sentiment-based features) and textual framing/content (TF-IDF features).
By contrasting their predictions, it identifies sentences that appear biased
through framing even when tone is neutral.

## Contents

- `processed.py`: cleans raw CSV, aggregates annotations, writes
  `processed/sentences_processed.csv` and `results/eda_summary.json`.
- `eda.py`: event-wise label distribution, position bias summary,
  and annotator pre-knowledge analysis.
- `baseline.py`: TF-IDF baselines (LogReg, Linear SVM, Naive Bayes) using
  conservative vs inclusive label mappings plus class-weighted variants.
- `baselineSOTA.py`: RoBERTa baseline for inclusive binary labels.
- `bias_intensity.py`: ridge regression for bias intensity with TF-IDF vs tone
  features, plus a framing-proxy based on the prediction gap.

## Data

Expected input file (raw dataset):
- `Sora_LREC2020_biasedsentences.csv`

Generated outputs:
- `processed/sentences_processed.csv`
- `results/eda_summary.json`

## Label mappings

Binary bias is evaluated under two mappings:
- Conservative: labels 1–2 -> 0 (not biased), 3–4 -> 1 (biased)
- Inclusive: label 1 -> 0 (not biased), 2–4 -> 1 (biased)

## Usage

Create the processed dataset:

```bash
python processed.py --in_csv Sora_LREC2020_biasedsentences.csv --strip_html
```

Run baselines (binary bias):

```bash
python baseline.py
python baselineSOTA.py
```

Run bias intensity and framing-proxy analysis:

```bash
python bias_intensity.py
```

Run EDA:

```bash
python eda.py
```
