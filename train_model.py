"""
Pre-train XGBoost fare prediction model and save to models/fare_model.pkl
Run once locally: python train_model.py
"""
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap

DATA_DIR = Path("data/exports")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading ML sample...")
ml = pd.read_parquet(DATA_DIR / "ml_sample.parquet")
print(f"Loaded {len(ml):,} rows")

# Clean
df = ml.dropna(subset=["fare_amount","trip_distance","passenger_count",
                        "hour","day_of_week","month","year"]).copy()
df = df[(df["fare_amount"].between(2.5, 150)) & (df["trip_distance"].between(0.1, 40))]
print(f"After cleaning: {len(df):,} rows")

# Encode boroughs
encoders = {}
for c in ["pickup_borough","dropoff_borough"]:
    df[c] = df[c].fillna("Unknown")
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    encoders[c] = le

FEATURES = ["trip_distance","passenger_count","hour","day_of_week",
            "month","year","pickup_borough","dropoff_borough"]

X = df[FEATURES]
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training XGBoost...")
model = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, n_jobs=-1, verbosity=0
)
model.fit(X_train, y_train)

r2   = model.score(X_test, y_test)
yp   = model.predict(X_test)
rmse = np.sqrt(np.mean((y_test.values - yp)**2))
print(f"R²: {r2:.4f}  RMSE: ${rmse:.2f}")

print("Computing SHAP values (2000 samples)...")
explainer  = shap.TreeExplainer(model)
sample_idx = np.random.RandomState(42).choice(len(X_test), 2000, replace=False)
X_shap     = X_test.iloc[sample_idx]
shap_vals  = explainer.shap_values(X_shap)

# Package everything
bundle = {
    "model":      model,
    "explainer":  explainer,
    "shap_vals":  shap_vals,
    "X_shap":     X_shap,
    "features":   FEATURES,
    "encoders":   encoders,
    "metrics":    {"r2": r2, "rmse": rmse},
    "y_true":     y_test.values[:600],
    "y_pred":     yp[:600],
}

out = MODEL_DIR / "fare_model.pkl"
with open(out, "wb") as f:
    pickle.dump(bundle, f)

size_mb = out.stat().st_size / 1e6
print(f"Saved {out}  ({size_mb:.1f} MB)")
print("Done.")
