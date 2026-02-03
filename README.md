# Text Mining Assignment 3

Scripts for preprocessing, EDA, and modeling bias in news sentences.

## Contents

- `processed.py`: cleans raw CSV, aggregates annotations, writes `processed/sentences_processed.csv` and `results/eda_summary.json`.
- `eda.py`: exploratory analysis and tables/plots on the raw dataset.
- `baseline.py`: TF-IDF baselines (LogReg, Linear SVM, Naive Bayes) for binary bias.
- `baselineSOTA.py`: RoBERTa baseline for inclusive binary labels.
- `bias_intensity.py`: ridge regression for bias intensity with TF-IDF vs tone features.

## Data

Expected input file (raw dataset):
- `Sora_LREC2020_biasedsentences.csv`

Generated outputs:
- `processed/sentences_processed.csv`
- `results/eda_summary.json`

## Usage

Create the processed dataset:

```bash
python processed.py --in_csv Sora_LREC2020_biasedsentences.csv --strip_html
```

Run baselines:

```bash
python baseline.py
python baselineSOTA.py
```

Run bias intensity experiments:

```bash
python bias_intensity.py
```

Run EDA:

```bash
python eda.py
```
