"""
Feature engineering and preprocessing pipeline for churn prediction.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

CAT_FEATURES = ["contract_type", "payment_method", "internet_service"]
NUM_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "num_products", "support_calls",
    "senior_citizen", "partner", "dependents",
]
TARGET = "churn"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive new features before encoding."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # avg monthly spend per product
        X["spend_per_product"] = X["monthly_charges"] / X["num_products"].clip(lower=1)
        # normalised tenure bucket (0-1)
        X["tenure_ratio"] = X["tenure"] / 72.0
        # interaction: high charges + short tenure = high risk signal
        X["charge_tenure_risk"] = X["monthly_charges"] / (X["tenure"] + 1)
        # log-transform right-skewed total charges
        X["log_total_charges"] = np.log1p(X["total_charges"].clip(lower=0))
        return X


ENGINEERED_NUM = NUM_FEATURES + [
    "spend_per_product", "tenure_ratio", "charge_tenure_risk", "log_total_charges"
]


def build_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, ENGINEERED_NUM),
        ("cat", cat_pipeline, CAT_FEATURES),
    ], remainder="drop")


def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    engineer = FeatureEngineer()
    df = engineer.transform(df)
    X = df[ENGINEERED_NUM + CAT_FEATURES]
    y = df[TARGET] if TARGET in df.columns else None
    return X, y
