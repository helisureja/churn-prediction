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
