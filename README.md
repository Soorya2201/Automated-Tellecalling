# Automated Telecalling: Lead Conversion Scoring

**Ranks incoming sales leads by their probability of converting, so a telecalling team calls the most promising leads first.**

An early project (2023). It trains a logistic-regression classifier on historical CRM leads, reports held-out accuracy and a confusion matrix, and writes a ranked call list for new leads.

## What it does

1. **Cleans CRM data:** drops all-empty columns, fills missing categoricals with `Unknown` and missing numerics with training means, and label-encodes categorical fields.
2. **Trains and evaluates** a logistic regression on a stratified 80/20 split: accuracy, classification report, and `confusion_matrix.png`.
3. **Scores new leads** with the *same* fitted preprocessing (training means and encoders, with unseen categories handled) and writes `scored_leads.csv`, sorted by conversion probability.

`question dataset.csv` is a small paraphrase-to-answer table (question variants mapped to one canonical answer), the start of a scripted-answer lookup for callers.

## Run it

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
python "Automated Tellecalling.py" historical_leads.csv new_leads.csv
```

The historical file needs a binary `Converted` column; both files may include a `Lead Number` ID column. The original CRM data is private and not included.

## Next steps

Gradient-boosted trees with probability calibration, class-imbalance handling, and threshold selection based on call capacity rather than 0.5.
