"""
app.py
Flask backend for the Loan Approval Prediction System.

Routes:
  GET  /            -> Dashboard (stats, chart, prediction form)
  POST /predict      -> Runs the Random Forest model on submitted form data
  GET  /api/stats    -> JSON stats (used to redraw the chart, if needed)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
ENCODERS_PATH = BASE_DIR / "model" / "encoders.pkl"
STATS_PATH = BASE_DIR / "model" / "stats.json"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    artifacts = pickle.load(f)
    CATEGORICAL_ENCODERS = artifacts["categorical_encoders"]
    FEATURE_ORDER = artifacts["feature_order"]

with open(STATS_PATH, "r") as f:
    STATS = json.load(f)


def encode_form(form):
    """Turn raw form values into the numeric feature vector the model expects."""
    row = {}
    for col, mapping in CATEGORICAL_ENCODERS.items():
        row[col] = mapping[form[col]]

    row["ApplicantIncome"] = float(form["ApplicantIncome"])
    row["CoapplicantIncome"] = float(form.get("CoapplicantIncome") or 0)
    row["LoanAmount"] = float(form["LoanAmount"])
    row["Loan_Amount_Term"] = float(form["Loan_Amount_Term"])
    row["Credit_History"] = float(form["Credit_History"])

    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)


@app.route("/")
def index():
    return render_template("index.html", stats=STATS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or request.form
        X = encode_form(payload)

        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = round(float(max(proba)) * 100, 1)

        result = {
            "prediction": "Approved" if pred == 1 else "Rejected",
            "approved": pred == 1,
            "confidence": confidence,
            "probability_approved": round(float(proba[1]) * 100, 1),
            "probability_rejected": round(float(proba[0]) * 100, 1),
        }
        return jsonify(result)
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400


@app.route("/api/stats")
def api_stats():
    return jsonify(STATS)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
