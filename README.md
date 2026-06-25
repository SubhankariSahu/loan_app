# Loan Approval Prediction System

A Machine Learning–powered web app that predicts whether a loan application
will be **Approved** or **Rejected**, using a Random Forest Classifier
trained on applicant data. Built with Flask, Bootstrap, and scikit-learn.

## Features

- **Dashboard** with live portfolio stats: total applications, approved /
  rejected counts, approval & rejection rate, and model accuracy
- **Approved vs Rejected pie chart** (Chart.js)
- **Loan Prediction Form** covering all 10 applicant features
- **Instant prediction** with an approval/rejection "stamp" and model
  confidence score

## Project Structure

```
loan_app/
├── app.py                  # Flask application (routes, prediction API)
├── requirements.txt
├── data/
│   ├── generate_data.py    # Generates the synthetic loan dataset
│   └── loan_data.csv       # 614-record loan application dataset
├── model/
│   ├── train_model.py      # Preprocessing + Random Forest training/eval
│   ├── model.pkl           # Trained model (generated)
│   ├── encoders.pkl        # Categorical encoders + feature order (generated)
│   └── stats.json          # Dashboard stats (generated)
├── templates/
│   └── index.html          # Dashboard page
└── static/
    ├── css/style.css
    └── js/script.js
```

## Setup

```bash
cd loan_app
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Generate the dataset (optional — already included)

```bash
python3 data/generate_data.py
```

This creates `data/loan_data.csv`: 614 applications (plus a few intentional
duplicates/missing values for the preprocessing step to clean), calibrated
to ~68.7% historical approval rate.

## 2. Train the model

```bash
python3 model/train_model.py
```

This will:
1. Load the dataset
2. Handle missing values (mode for categoricals, median for numerics)
3. Remove duplicate records
4. Encode categorical variables
5. Split into train/test sets (80/20, stratified)
6. Train a `RandomForestClassifier`
7. Print Accuracy, Confusion Matrix, and Classification Report
8. Save `model.pkl`, `encoders.pkl`, and `stats.json`

Expected output: **~87–88% accuracy**, 422 approved / 192 rejected out of
614 total applications.

## 3. Run the web app

```bash
python3 app.py
```

Then open **http://localhost:5000** in your browser.

## How prediction works

The `/predict` endpoint accepts the 10 applicant fields as JSON, encodes
them with the same mappings used during training, runs them through the
saved Random Forest model, and returns the predicted status (`Approved` /
`Rejected`) along with the model's confidence and class probabilities.

## Retraining on your own data

Replace `data/loan_data.csv` with your own file (same column names:
`Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
Loan_Status`), then re-run `python3 model/train_model.py` and restart the
app — it will pick up the newly saved model and stats automatically.

## Future Enhancements

- User authentication
- Database integration (store applications + predictions)
- Prediction history per user
- Downloadable PDF reports
- Cloud deployment
- Real-time banking system integration
- Explainable AI (e.g. SHAP feature contributions per prediction)
