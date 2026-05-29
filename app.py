"""
Flask REST API for real-time churn risk scoring.

Endpoints:
    GET  /health         → service health check
    POST /predict        → single customer churn prediction
    POST /predict/batch  → batch predictions (list of customers)

Usage:
    python app.py
"""
from __future__ import annotations
import os, json
from pathlib import Path
from flask import Flask, request, jsonify
import joblib
import pandas as pd

from src.preprocessing import FeatureEngineer, ENGINEERED_NUM, CAT_FEATURES

app    = Flask(__name__)
MODEL  = None
ENGINR = FeatureEngineer()

MODEL_PATH = Path(__file__).parent / "models/churn_pipeline.pkl"


def load_model():
    global MODEL
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print("⚠  No trained model found. Run `python src/train.py` first.")


def _predict_df(df: pd.DataFrame) -> list[dict]:
    df_eng = ENGINR.transform(df)
    X      = df_eng[ENGINEERED_NUM + CAT_FEATURES]
    probs  = MODEL.predict_proba(X)[:, 1]
    preds  = MODEL.predict(X)

    results = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        cid = df.get("customer_id", pd.Series([f"row_{i}"]))[i] if "customer_id" in df.columns else f"row_{i}"
        results.append({
            "customer_id"   : cid,
            "churn_risk"    : round(float(prob), 4),
            "churn_predicted": bool(pred),
            "risk_tier"     : "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW"),
        })
    return results


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Train first."}), 503
    data = request.get_json(force=True)
    try:
        df     = pd.DataFrame([data])
        result = _predict_df(df)
        return jsonify(result[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Train first."}), 503
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array of customer objects"}), 400
    try:
        df      = pd.DataFrame(data)
        results = _predict_df(df)
        return jsonify({"predictions": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
