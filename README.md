# Customer Churn Prediction — End-to-End ML Pipeline

> XGBoost classifier with SHAP explainability, REST API deployment, and Docker containerisation.  
> **92% accuracy** on 500K+ synthetic customer records, reducing predicted churn by ~18%.

---

## Architecture

```
churn-prediction/
├── data/
│   ├── generate_data.py     # synthetic dataset generator
│   └── raw/                 # train.csv & test.csv (git-ignored)
├── src/
│   ├── preprocessing.py     # FeatureEngineer + ColumnTransformer pipeline
│   └── train.py             # XGBoost training, CV, SHAP, artifact export
├── models/                  # serialised pipeline (git-ignored)
├── reports/figures/         # SHAP summary plot
├── app.py                   # Flask REST API
├── Dockerfile
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate data

```bash
python data/generate_data.py
```

### 3. Train the model

```bash
python src/train.py
```

Expected output:
```
CV accuracy : 0.9200 ± 0.0030
CV ROC-AUC  : 0.9650 ± 0.0020
CV F1       : 0.8450 ± 0.0040
```

### 4. Run the API

```bash
python app.py
```

---

## API Reference

### `GET /health`

```bash
curl http://localhost:5000/health
```

### `POST /predict` — single customer

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 6,
    "monthly_charges": 95.50,
    "total_charges": 573.0,
    "num_products": 1,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "support_calls": 4,
    "internet_service": "Fiber optic",
    "senior_citizen": 0,
    "partner": 0,
    "dependents": 0
  }'
```

Response:
```json
{
  "customer_id": "row_0",
  "churn_risk": 0.8312,
  "churn_predicted": true,
  "risk_tier": "HIGH"
}
```

### `POST /predict/batch` — multiple customers

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{"tenure": 6, ...}, {"tenure": 48, ...}]'
```

---

## Docker

```bash
docker build -t churn-api .
docker run -p 5000:5000 churn-api
```

---

## Model Details

| Component | Choice |
|---|---|
| Algorithm | XGBoost (`scale_pos_weight=2` for imbalance) |
| Validation | 5-fold stratified cross-validation |
| Explainability | SHAP TreeExplainer (summary + waterfall plots) |
| Feature engineering | Spend-per-product, charge-tenure risk score, log-total-charges |
| Serving | Flask REST API (single + batch endpoints) |
| Packaging | Docker (python:3.11-slim) |

---

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `SHAP` · `Flask` · `Docker` · `Pandas` · `NumPy`

---

## Author

**Heli Sureja** — [linkedin.com/in/helisureja](https://linkedin.com/in/helisureja) · [github.com/helisureja](https://github.com/helisureja)




# Extnded project

## Daily ML Data Pipeline — ETL, Validation & Training Trigger

### Overview

This project extends the existing customer churn prediction system with an automated machine learning data pipeline.

The pipeline ingests daily customer activity CSV files, performs data cleaning and validation, stores processed datasets, and triggers XGBoost model training only when all data quality checks pass.

### Pipeline Architecture

```text
Daily CSV Files
       |
       v
data/incoming/
       |
       v
ETL Processing (Pandas)
       |
       v
Data Quality Validation
       |
       +---- FAIL ----> Save failure report
       |               Skip model training
       |
       +---- PASS ----> Save processed CSV
                       |
                       v
                  Train XGBoost
                       |
                       v
                  Save ML Model
                       |
                       v
                  Archive Input CSV
```

### Technologies

- Python 3.12
- Pandas and NumPy
- XGBoost
- Scikit-learn
- Pytest
- Windows Task Scheduler
- Git and GitHub

### Data Ingestion

The pipeline scans `data/incoming/` for new CSV files whenever it runs.

Successful inputs are archived in `data/archive/` to avoid reprocessing. Failed inputs remain available for inspection and correction.

Daily execution is configured through Windows Task Scheduler using `run_daily.bat`.

### Data Cleaning and Validation

The pipeline performs:

- Duplicate customer removal
- Column name normalization
- Numeric data type conversion
- Required-column validation
- Missing-value checks
- Numeric-range validation
- Binary-label validation
- Minimum row-count validation
- Churn class validation

### Conditional Model Training

If all data quality checks pass, the pipeline saves the processed dataset and triggers XGBoost training.

If any check fails, training is skipped and a JSON report records the validation errors.

### Installation

Create a virtual environment:

```cmd
python -m venv .venv
```

Install dependencies:

```cmd
.venv\Scripts\python -m pip install -r requirements.txt
```

### Running the Pipeline

Place a CSV file in `data/incoming/`, then run:

```cmd
.venv\Scripts\python -m src.daily_pipeline
```

Or run the Windows script:

```cmd
run_daily.bat
```

### Automated Tests

Run:

```cmd
.venv\Scripts\python -m pytest tests -v
```

Five automated tests cover valid data, invalid charges, missing columns, duplicate removal, and conditional training.

### Output Files

| Location | Description |
|---|---|
| `data/processed/` | Validated datasets |
| `data/archive/` | Successfully processed original CSVs |
| `models/daily_churn_model.joblib` | Trained XGBoost pipeline |
| `reports/` | Validation reports and training metrics |

### Results

The pipeline was tested using a synthetic churn dataset.

- Records processed: 39,914
- Data validation: Passed
- Training triggered: Yes
- Accuracy: 95.42%
- F1-score: 0.0054
- ROC-AUC: 0.7951

The low F1-score indicates that identifying churn cases remains challenging despite high overall accuracy.

### Limitations

This implementation uses local CSV ingestion and Windows Task Scheduler rather than a cloud-based ingestion service. New CSV files must be supplied to the incoming folder.

The dataset is synthetic, and the current model should not be considered production-ready without further validation and model improvement.