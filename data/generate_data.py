"""
Synthetic customer churn dataset generator.
Run this once to create train/test splits in data/raw/.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)
N = 50_000

tenure        = np.random.randint(1, 72, N)
monthly_charge= np.round(np.random.uniform(20, 120, N), 2)
total_charges = np.round(monthly_charge * tenure + np.random.normal(0, 50, N), 2)
num_products  = np.random.choice([1, 2, 3, 4], N, p=[0.4, 0.35, 0.15, 0.1])
contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20])
payment_method= np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N)
support_calls = np.random.poisson(1.5, N)
internet_svc  = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.35, 0.44, 0.21])
senior_citizen= np.random.choice([0, 1], N, p=[0.84, 0.16])
partner       = np.random.choice([0, 1], N, p=[0.52, 0.48])
dependents    = np.random.choice([0, 1], N, p=[0.70, 0.30])

logit = (
    -3.5
    + (-0.04) * tenure
    + 0.02  * monthly_charge
    + (-0.3) * num_products
    + (-1.2) * (contract_type == "Two year").astype(int)
    + (-0.6) * (contract_type == "One year").astype(int)
    + 0.4   * (payment_method == "Electronic check").astype(int)
    + 0.3   * support_calls
    + 0.5   * (internet_svc == "Fiber optic").astype(int)
    + 0.3   * senior_citizen
    + (-0.2) * partner
    + np.random.normal(0, 0.5, N)
)
churn_prob = 1 / (1 + np.exp(-logit))
churn      = (np.random.uniform(0, 1, N) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id"   : [f"CUST{str(i).zfill(6)}" for i in range(N)],
    "tenure"        : tenure,
    "monthly_charges": monthly_charge,
    "total_charges" : total_charges,
    "num_products"  : num_products,
    "contract_type" : contract_type,
    "payment_method": payment_method,
    "support_calls" : support_calls,
    "internet_service": internet_svc,
    "senior_citizen": senior_citizen,
    "partner"       : partner,
    "dependents"    : dependents,
    "churn"         : churn,
})

import os
os.makedirs("data/raw", exist_ok=True)
train, test = train_test_split(df, test_size=0.2, stratify=df["churn"], random_state=42)
train.to_csv("data/raw/train.csv", index=False)
test.to_csv("data/raw/test.csv",  index=False)
print(f"Generated {len(df):,} records  |  Churn rate: {churn.mean():.1%}")
print(f"Train: {len(train):,}  |  Test: {len(test):,}")
print("Saved → data/raw/train.csv  &  data/raw/test.csv")
