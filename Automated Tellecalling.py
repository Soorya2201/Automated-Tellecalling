"""Lead-conversion scoring for a telecalling team.

Trains a logistic-regression classifier on historical CRM leads (target column
`Converted`), reports held-out accuracy, then scores a file of new leads so
callers can prioritise the ones most likely to convert.

Usage:
    python "Automated Tellecalling.py" historical_leads.csv new_leads.csv
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "Converted"
ID_COLUMN = "Lead Number"

train_path = sys.argv[1] if len(sys.argv) > 1 else "xy.csv"
new_path = sys.argv[2] if len(sys.argv) > 2 else "xyz.csv"

# Load and clean historical leads
df = pd.read_csv(train_path)

# Drop columns that are entirely empty
empty_columns = df.columns[df.isnull().all()]
df = df.drop(columns=empty_columns)

string_columns = df.select_dtypes(include=["object"]).columns
numeric_columns = df.select_dtypes(include=["number"]).columns.drop([TARGET, ID_COLUMN], errors="ignore")

df[string_columns] = df[string_columns].fillna("Unknown")
numeric_means = df[numeric_columns].mean()
df[numeric_columns] = df[numeric_columns].fillna(numeric_means)

# Encode categorical columns; keep the encoders to apply to new leads
label_encoders = {}
for column in string_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Train / evaluate
y = df[TARGET]
X = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Held-out accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm / cm.sum(), annot=True, fmt=".2%", cmap="Blues").set(
    title="Confusion matrix: logistic regression"
)
plt.savefig("confusion_matrix.png", bbox_inches="tight")

# Score new leads with the same preprocessing
new_leads = pd.read_csv(new_path).drop(columns=empty_columns, errors="ignore")
lead_ids = new_leads[ID_COLUMN] if ID_COLUMN in new_leads else pd.Series(range(len(new_leads)))

new_leads[string_columns] = new_leads[string_columns].fillna("Unknown")
new_leads[numeric_columns] = new_leads[numeric_columns].fillna(numeric_means)
for column in string_columns:
    known = set(label_encoders[column].classes_)
    # Unseen categories fall back to "Unknown" if it was seen in training, else the most common class
    fallback = "Unknown" if "Unknown" in known else label_encoders[column].classes_[0]
    new_leads[column] = new_leads[column].where(new_leads[column].isin(known), fallback)
    new_leads[column] = label_encoders[column].transform(new_leads[column])

features = new_leads[X.columns]
scored = pd.DataFrame({
    ID_COLUMN: lead_ids,
    "conversion_probability": model.predict_proba(features)[:, 1],
    "predicted_converted": model.predict(features),
}).sort_values("conversion_probability", ascending=False)

scored.to_csv("scored_leads.csv", index=False)
print("\nTop leads to call first:\n", scored.head(20).to_string(index=False))
print(f"\n{int(scored['predicted_converted'].sum())} of {len(scored)} new leads predicted to convert.")
