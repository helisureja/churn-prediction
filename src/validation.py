
import pandas as pd

REQUIRED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_products",
    "contract_type",
    "payment_method",
    "support_calls",
    "internet_service",
    "senior_citizen",
    "partner",
    "dependents",
    "churn",
]

NUMERIC_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_products",
    "support_calls",
    "senior_citizen",
    "partner",
    "dependents",
    "churn",
]


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if "customer_id" in df.columns:
        df = df.drop_duplicates(subset=["customer_id"])
    else:
        df = df.drop_duplicates()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["contract_type", "payment_method",
                "internet_service"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    return df


def validate_data(df):
    errors = []

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors

    if len(df) < 30:
        errors.append("Dataset must contain at least 30 rows")

    if df[REQUIRED_COLUMNS].isna().any().any():
        errors.append("Required fields contain missing values")

    ranges = {
        "tenure": (0, 120),
        "monthly_charges": (0, 10000),
        "total_charges": (0, 1000000),
        "num_products": (1, 20),
        "support_calls": (0, 100),
    }

    for col, (minimum, maximum) in ranges.items():
        if not df[col].between(minimum, maximum).all():
            errors.append(f"Invalid values in {col}")

    for col in ["senior_citizen", "partner",
                "dependents", "churn"]:
        if not df[col].isin([0, 1]).all():
            errors.append(f"Invalid binary values in {col}")

    if df["churn"].nunique() < 2:
        errors.append("Both churn classes are required")

    return errors
