"""
train_model.py
End-to-end training pipeline for the Loan Approval Prediction System.

Steps:
1. Load the loan dataset
2. Handle missing values
3. Remove duplicate records
4. Encode categorical variables into numerical format
5. Split the dataset into training and testing data
6. Train a Random Forest Classifier
7. Evaluate (Accuracy, Confusion Matrix, Classification Report)
8. Save the trained model + encoders + dashboard stats with Pickle/JSON
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "/home/claude/loan_app/data/loan_data.csv"
MODEL_PATH = "/home/claude/loan_app/model/model.pkl"
ENCODERS_PATH = "/home/claude/loan_app/model/encoders.pkl"
STATS_PATH = "/home/claude/loan_app/model/stats.json"

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed"]
NUMERIC_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
TARGET_COL = "Loan_Status"

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
total_raw = len(df)

# ---------------------------------------------------------------------------
# 2. Handle missing values
#    - categorical -> fill with mode
#    - numeric -> fill with median
# ---------------------------------------------------------------------------
for col in CATEGORICAL_COLS:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# ---------------------------------------------------------------------------
# 3. Remove duplicate records
# ---------------------------------------------------------------------------
before_dedup = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "Loan_ID"])
df = df.reset_index(drop=True)
removed_dupes = before_dedup - len(df)

# ---------------------------------------------------------------------------
# 4. Encode categorical variables into numerical format
# ---------------------------------------------------------------------------
encoders = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
}
for col, mapping in encoders.items():
    df[col] = df[col].map(mapping)

target_map = {"Y": 1, "N": 0}
df[TARGET_COL] = df[TARGET_COL].map(target_map)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ---------------------------------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 6. Train Random Forest Classifier
# ---------------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=1,
    random_state=42,
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred).tolist()
report = classification_report(y_test, y_pred, target_names=["Rejected", "Approved"])

print("=" * 60)
print(f"Rows loaded:            {total_raw}")
print(f"Duplicate rows removed: {removed_dupes}")
print(f"Final dataset size:     {len(df)}")
print("-" * 60)
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(np.array(cm))
print("Classification Report:")
print(report)
print("=" * 60)

# ---------------------------------------------------------------------------
# 8. Save model, encoders, and dashboard stats
# ---------------------------------------------------------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(ENCODERS_PATH, "wb") as f:
    pickle.dump(
        {
            "categorical_encoders": encoders,
            "target_map": target_map,
            "feature_order": FEATURE_COLS,
        },
        f,
    )

total_apps = len(df)
approved = int((df[TARGET_COL] == 1).sum())
rejected = int((df[TARGET_COL] == 0).sum())

stats = {
    "total_applications": total_apps,
    "approved_loans": approved,
    "rejected_loans": rejected,
    "approval_rate": round(approved / total_apps * 100, 2),
    "rejection_rate": round(rejected / total_apps * 100, 2),
    "model_accuracy": round(accuracy * 100, 2),
    "confusion_matrix": cm,
    "duplicates_removed": int(removed_dupes),
}

with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved model to:     {MODEL_PATH}")
print(f"Saved encoders to:  {ENCODERS_PATH}")
print(f"Saved stats to:     {STATS_PATH}")
print(json.dumps(stats, indent=2))
