"""
Train XGBoost churn classifier with cross-validation, SHAP explainability,
and MLflow-style artifact logging.

Usage:
    python src/train.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
)
import xgboost as xgb
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import build_preprocessor, load_and_prepare, FeatureEngineer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_TRAIN = ROOT / "data/raw/train.csv"
DATA_TEST  = ROOT / "data/raw/test.csv"
MODEL_DIR  = ROOT / "models"
REPORT_DIR = ROOT / "reports/figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ─────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators"      : 400,
    "max_depth"         : 5,
    "learning_rate"     : 0.05,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "min_child_weight"  : 3,
    "gamma"             : 0.1,
    "reg_alpha"         : 0.1,
    "reg_lambda"        : 1.0,
    "scale_pos_weight"  : 2,        # handles class imbalance
    "use_label_encoder" : False,
    "eval_metric"       : "logloss",
    "random_state"      : 42,
    "n_jobs"            : -1,
}


def evaluate(y_true, y_pred, y_prob, split="Test") -> dict:
    metrics = {
        "split"    : split,
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "roc_auc"  : round(roc_auc_score(y_true, y_prob), 4),
        "f1"       : round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall"   : round(recall_score(y_true, y_pred), 4),
    }
    print(f"\n── {split} Metrics ──────────────────────────────")
    for k, v in metrics.items():
        if k != "split":
            print(f"  {k:<12}: {v}")
    return metrics


def plot_shap(model, X_sample: pd.DataFrame, feature_names: list):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved → {REPORT_DIR / 'shap_summary.png'}")


def main():
    print("Loading data …")
    X_train, y_train = load_and_prepare(str(DATA_TRAIN))
    X_test,  y_test  = load_and_prepare(str(DATA_TEST))

    preprocessor = build_preprocessor()
    clf = xgb.XGBClassifier(**XGB_PARAMS)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])

    # ── Cross-validation ────────────────────────────────────────────────────
    print("\nRunning 5-fold stratified cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=["accuracy", "roc_auc", "f1"],
        return_train_score=True, n_jobs=-1,
    )
    print(f"  CV accuracy : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"  CV ROC-AUC  : {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"  CV F1       : {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")

    # ── Final fit ───────────────────────────────────────────────────────────
    print("\nFitting on full training set …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred, y_prob, split="Test")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ── SHAP explainability ─────────────────────────────────────────────────
    print("\nGenerating SHAP explanations …")
    X_transformed = preprocessor.transform(X_train.sample(2000, random_state=42))
    feature_names = (
        preprocessor.named_transformers_["num"].get_feature_names_out().tolist()
        + preprocessor.named_transformers_["cat"].get_feature_names_out().tolist()
    )
    plot_shap(clf, X_transformed, feature_names)

    # ── Persist ─────────────────────────────────────────────────────────────
    joblib.dump(pipeline, MODEL_DIR / "churn_pipeline.pkl")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModel saved → {MODEL_DIR / 'churn_pipeline.pkl'}")
    print(f"Metrics saved → {MODEL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
