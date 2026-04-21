"""
Shared pytest fixtures for the Seattle Airbnb Price Prediction test suite.

All fixtures use session scope so each file is loaded once per test run,
not once per test function.
"""
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
PRICE_CAP  = 500


# ── Raw / cleaned data ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cleaned_df() -> pd.DataFrame:
    """Full cleaned.csv as loaded from disk."""
    return pd.read_csv(DATA_DIR / "cleaned.csv")


@pytest.fixture(scope="session")
def features_df() -> pd.DataFrame:
    """Engineered features.csv — the output of the full feature pipeline."""
    return pd.read_csv(DATA_DIR / "features.csv")


# ── Model ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model_artifact() -> dict:
    """
    Loaded production_model.pkl as a dict with keys:
      pipeline, feature_cols, price_cap, calendar_features, sentiment_features
    """
    return joblib.load(MODELS_DIR / "production_model.pkl")


@pytest.fixture(scope="session")
def model_median_row(model_artifact, features_df) -> pd.DataFrame:
    """
    Single-row DataFrame containing the median value for every one of the
    model's 150 input features, computed from the capped training data.

    Used as a safe, realistic input for smoke-testing predictions.
    """
    from models.baseline import prepare_features

    trained_cols = model_artifact["feature_cols"]

    df_cap = features_df[features_df["price"] <= PRICE_CAP].reset_index(drop=True)
    X, _ = prepare_features(df_cap)
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]

    # Add any OHE columns the model expects but that aren't in this split
    for col in trained_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[[col for col in trained_cols if col in X.columns]]

    return X.median().to_frame().T
