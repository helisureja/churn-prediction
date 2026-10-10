
import pandas as pd

from src.validation import clean_data, validate_data
from src.daily_pipeline import process_file


def sample_data():
    return pd.DataFrame({
        "customer_id": range(40),
        "tenure": [12] * 40,
        "monthly_charges": [75.0] * 40,
        "total_charges": [900.0] * 40,
        "num_products": [2] * 40,
        "contract_type": ["Monthly"] * 40,
        "payment_method": ["Card"] * 40,
        "support_calls": [1] * 40,
        "internet_service": ["Fiber"] * 40,
        "senior_citizen": [0] * 40,
        "partner": [1] * 40,
        "dependents": [0] * 40,
        "churn": [0, 1] * 20,
    })


def test_valid_data():
    assert validate_data(sample_data()) == []


def test_invalid_charges():
    df = sample_data()
    df.loc[0, "monthly_charges"] = -500

    errors = validate_data(df)
    assert any("monthly_charges" in e for e in errors)


def test_missing_column():
    df = sample_data().drop(columns=["churn"])
    assert validate_data(df)


def test_duplicates():
    df = sample_data()
    duplicated = pd.concat([df, df.iloc[[0]]])

    cleaned = clean_data(duplicated)
    assert len(cleaned) == 40


def test_failed_quality_blocks_training(tmp_path, monkeypatch):
    import src.daily_pipeline as pipeline

    monkeypatch.setattr(pipeline, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(pipeline, "ARCHIVE", tmp_path / "archive")
    monkeypatch.setattr(pipeline, "REPORTS", tmp_path / "reports")

    df = sample_data()
    df.loc[0, "monthly_charges"] = -500
    csv_path = tmp_path / "invalid.csv"
    df.to_csv(csv_path, index=False)

    report = process_file(
        csv_path,
        training_command=["this-command-must-not-run"]
    )

    assert report["status"] == "FAIL"
    assert report["trained"] is False
    assert not (tmp_path / "processed" / "invalid_clean.csv").exists()
