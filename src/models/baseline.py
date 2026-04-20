"""
Baseline regression model for Airbnb price prediction.

Uses LinearRegression as a performance floor for comparison against
tree-based models. Price is log-transformed to handle the heavy right
skew (median $145, max $50k+). Metrics are reported in the original
dollar scale.

Usage:
    python src/models/baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)

# High-cardinality columns: keep only the top-N most frequent values,
# everything else becomes "Other". This prevents OHE from exploding
# the feature matrix while retaining the most informative categories.
TOP_N_CATEGORIES: dict[str, int] = {
    "property_type":        10,
    "host_response_rate":   10,
    "host_acceptance_rate": 10,
    "bathrooms_text":       10,
    "host_verifications":    6,   # only 6 unique anyway
}

# Columns to drop before modelling — IDs, raw date strings, free-text,
# duplicates of better-engineered features, or constants.
DROP_BEFORE_MODEL = [
    "host_name",                  # ID-like, ~1800 unique
    "host_since",                 # date string; signal captured by host_id
    "host_location",              # 194 unique, too noisy for baseline
    "host_neighbourhood",         # 213 unique; use neighbourhood features
    "amenities",                  # raw list; captured by amenities_count
    "has_availability",           # constant (all 't')
    "neighbourhood_cleansed",     # 88 unique; captured by neighbourhood_avg_price
    "neighbourhood_group_cleansed",  # duplicate of neighbourhood_group
]

# Categorical columns to one-hot encode.
OHE_COLS = [
    "neighbourhood_group",
    "room_type",
    "host_response_time",
    "host_is_superhost",
    "host_response_rate",
    "host_acceptance_rate",
    "property_type",
    "instant_bookable",
    "host_has_profile_pic",
    "host_identity_verified",
    "host_verifications",
    "bathrooms_text",
]


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #

def _bucket_high_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infrequent categories with 'Other' before OHE."""
    for col, top_n in TOP_N_CATEGORIES.items():
        if col not in df.columns:
            continue
        top_cats = df[col].value_counts().head(top_n).index
        df[col] = df[col].where(df[col].isin(top_cats), other="Other")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) ready for sklearn, where y = log(price).

    Steps:
      1. Drop uninformative columns.
      2. Bucket high-cardinality categoricals.
      3. One-hot encode all object columns.
      4. Fill any residual nulls with column median.
      5. Log-transform price → target y.
    """
    df = df.copy()

    # 1. Drop
    df = df.drop(columns=[c for c in DROP_BEFORE_MODEL if c in df.columns])

    # 2. Bucket high-cardinality categoricals
    df = _bucket_high_cardinality(df)

    # 3. One-hot encode — only encode columns present in OHE_COLS that exist
    ohe_present = [c for c in OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_present, drop_first=True, dtype=int)

    # 4. Drop any remaining object columns (high-cardinality leftovers)
    remaining_obj = df.select_dtypes(include="object").columns.tolist()
    if remaining_obj:
        print(f"  Dropping remaining object columns: {remaining_obj}")
        df = df.drop(columns=remaining_obj)

    # 5. Fill residual nulls with column median
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        print(f"  Filling {int(null_cols.sum())} residual nulls with median "
              f"across {len(null_cols)} columns")
        df = df.fillna(df.median(numeric_only=True))

    # 6. Separate target — log-transform to handle heavy right skew
    y = np.log(df["price"])
    X = df.drop(columns=["price"])

    return X, y


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #

def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    """
    Compute MAE, RMSE, and R² in the original dollar scale.
    R² is also reported on the log scale (what the model optimises).
    """
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    r2_log = r2_score(y_true_log, y_pred_log)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "R2_log": r2_log}


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    # ------------------------------------------------------------------ #
    # Load                                                                 #
    # ------------------------------------------------------------------ #
    path = DATA_DIR / "features.csv"
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ------------------------------------------------------------------ #
    # Prepare                                                              #
    # ------------------------------------------------------------------ #
    print("\nPreparing features ...")
    X, y = prepare_features(df)
    print(f"  X shape : {X.shape[0]:,} rows x {X.shape[1]} columns")
    print(f"  y range : ${np.exp(y.min()):.0f} – ${np.exp(y.max()):.0f}  "
          f"(log range: {y.min():.2f} – {y.max():.2f})")

    # ------------------------------------------------------------------ #
    # Split                                                                #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
    print("\nTraining LinearRegression baseline ...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # Evaluate                                                             #
    # ------------------------------------------------------------------ #
    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

    train_metrics = evaluate(y_train.values, y_pred_train)
    test_metrics  = evaluate(y_test.values,  y_pred_test)

    print(f"\n{'='*50}")
    print(f"  Baseline — LinearRegression")
    print(f"{'='*50}")
    print(f"  {'Metric':<12}  {'Train':>10}  {'Test':>10}")
    print(f"  {'-'*34}")
    print(f"  {'MAE ($)':<12}  {train_metrics['MAE']:>10.2f}  {test_metrics['MAE']:>10.2f}")
    print(f"  {'RMSE ($)':<12}  {train_metrics['RMSE']:>10.2f}  {test_metrics['RMSE']:>10.2f}")
    print(f"  {'R² (orig)':<12}  {train_metrics['R2']:>10.4f}  {test_metrics['R2']:>10.4f}")
    print(f"  {'R² (log)':<12}  {train_metrics['R2_log']:>10.4f}  {test_metrics['R2_log']:>10.4f}")
    print(f"{'='*50}")

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    model_path = MODELS_DIR / "baseline.pkl"
    joblib.dump({"pipeline": pipeline, "feature_cols": list(X.columns)}, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
