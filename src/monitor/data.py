"""Carga y preparación del dataset base (mismo esquema que churn-prediction-pipeline)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "telco_customer_churn.csv"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
               "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip().replace("", np.nan)).fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})
    df[TARGET] = (df["Churn"] == "Yes").astype(int)
    return df[["customerID"] + FEATURES + [TARGET]]
