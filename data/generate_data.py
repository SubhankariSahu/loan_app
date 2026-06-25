"""
generate_data.py
Generates a realistic synthetic loan-application dataset (614 records) that
mirrors the classic Loan Prediction dataset structure described in the
project spec, calibrated to land close to the example statistics:
  Total: 614 | Approved: 422 | Rejected: 192 | Approval Rate: 68.73%
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 614

genders = rng.choice(["Male", "Female"], size=N, p=[0.81, 0.19])
married = rng.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = rng.choice(["0", "1", "2", "3+"], size=N, p=[0.58, 0.17, 0.17, 0.08])
education = rng.choice(["Graduate", "Not Graduate"], size=N, p=[0.78, 0.22])
self_employed = rng.choice(["Yes", "No"], size=N, p=[0.14, 0.86])

applicant_income = np.round(rng.gamma(shape=4.5, scale=1200, size=N) + 1500)
coapplicant_income = np.round(
    np.where(married == "Yes", rng.gamma(shape=2.0, scale=900, size=N), 0)
)
loan_amount = np.round(
    rng.gamma(shape=5.0, scale=28, size=N) + 40
)
loan_amount_term = rng.choice(
    [360, 180, 120, 60, 84, 300, 240, 36],
    size=N,
    p=[0.73, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02],
)
credit_history = rng.choice([1.0, 0.0], size=N, p=[0.84, 0.16])

# Inject a small amount of realistic missingness
def inject_missing(arr, frac, fill_val=np.nan):
    arr = arr.astype(object)
    idx = rng.choice(len(arr), size=int(len(arr) * frac), replace=False)
    arr[idx] = fill_val
    return arr

gender_m = inject_missing(genders, 0.02)
dependents_m = inject_missing(dependents, 0.025)
self_employed_m = inject_missing(self_employed, 0.03)
loan_amount_m = inject_missing(loan_amount.astype(object), 0.035)
loan_term_m = inject_missing(loan_amount_term.astype(object), 0.02)
credit_history_m = inject_missing(credit_history.astype(object), 0.08)

# ---- Underlying "true" approval logic (what the bank actually cares about) ----
score = (
    2.6 * (credit_history == 1.0).astype(float)
    + 0.00035 * (applicant_income + coapplicant_income)
    - 0.012 * loan_amount
    + 0.4 * (education == "Graduate").astype(float)
    + 0.15 * (married == "Yes").astype(float)
    - 0.25 * (self_employed == "Yes").astype(float)
    - 0.15 * (dependents == "3+").astype(float)
)
score = score + rng.normal(0, 0.22, size=N)  # mild noise so it isn't trivially separable
prob_approve = 1 / (1 + np.exp(-(score - 2.3) * 2.4))
loan_status = np.where(rng.random(N) < prob_approve, "Y", "N")

# Calibrate to land near 422 approved / 192 rejected
target_approved = 422
current_approved = (loan_status == "Y").sum()
diff = target_approved - current_approved
if diff != 0:
    flip_pool = np.where(loan_status == ("N" if diff > 0 else "Y"))[0]
    # flip the ones with prob_approve closest to 0.5 (most "borderline") first
    order = np.argsort(np.abs(prob_approve[flip_pool] - 0.5))
    flip_idx = flip_pool[order[: abs(diff)]]
    loan_status[flip_idx] = "Y" if diff > 0 else "N"

loan_id = [f"LP{str(i).zfill(6)}" for i in range(1, N + 1)]

df = pd.DataFrame(
    {
        "Loan_ID": loan_id,
        "Gender": gender_m,
        "Married": married,
        "Dependents": dependents_m,
        "Education": education,
        "Self_Employed": self_employed_m,
        "ApplicantIncome": applicant_income.astype(int),
        "CoapplicantIncome": coapplicant_income.astype(int),
        "LoanAmount": loan_amount_m,
        "Loan_Amount_Term": loan_term_m,
        "Credit_History": credit_history_m,
        "Loan_Status": loan_status,
    }
)

# A handful of exact-duplicate rows, to give the preprocessing step something to remove
dup_rows = df.sample(n=6, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)

out_path = "/home/claude/loan_app/data/loan_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows (incl. {len(dup_rows)} duplicates) to {out_path}")
print(df["Loan_Status"].value_counts())
