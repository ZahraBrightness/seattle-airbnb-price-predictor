"""
Prediction interface for the XGBoost price model (models/xgboost_full.pkl).

Two independent signals are returned for every prediction:

  confidence_tier  — data quality, based purely on review activity.
    HIGH   : days_since_last_review < 365  AND  number_of_reviews > 5
    MEDIUM : days_since_last_review 365–1000  OR  number_of_reviews 1–5
    LOW    : days_since_last_review > 1000  OR  number_of_reviews == 0
    Priority when rules overlap: LOW > MEDIUM > HIGH.

  price_range  — prediction precision, based on observed MAE per price bucket.
    $0–100   : predicted ± $17   (23.8% typical MAPE)
    $100–200 : predicted ± $25   (16.9% typical MAPE)
    $200–500 : predicted ± $57   (19.5% typical MAPE)

Full output per prediction:
    predicted_price       float
    confidence_tier       "HIGH" / "MEDIUM" / "LOW"
    price_range_low       predicted_price − bucket_error
    price_range_high      predicted_price + bucket_error
    confidence_explanation  plain-English string

Usage (programmatic):
    from models.predict import predict_with_confidence
    result_df = predict_with_confidence(listing_df)

Usage (CLI — validates on held-out test set):
    python src/models/predict.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baseline import prepare_features

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

PRICE_CAP    = 500
RANDOM_STATE = 42
_LOG_CLIP    = (np.log(1), np.log(PRICE_CAP * 2))

ConfidenceTier = Literal["HIGH", "MEDIUM", "LOW"]

# Bucket boundaries and their empirical MAE (from full model evaluation)
_BUCKET_EDGES  = [0,   100,  200,  500]
_BUCKET_ERRORS = [17,   25,   57]       # ± dollars per bucket
_BUCKET_LABELS = ["$0–100", "$100–200", "$200–500"]
_BUCKET_MAPES  = ["23.8%",  "16.9%",   "19.5%"]

# Plain-English fragments assembled into confidence_explanation
_TIER_PREFIX = {
    "HIGH":   "Based on active listing with recent reviews.",
    "MEDIUM": "Moderate confidence — review history is limited or dated.",
    "LOW":    "Lower confidence — this listing has minimal review history.",
}
_BUCKET_SUFFIX = {
    "$0–100":   (
        "Budget listings ($0–100) are the most predictable "
        "with a typical error of ±$17 (23.8%)."
    ),
    "$100–200": (
        "Mid-range listings ($100–200) are well-modelled "
        "with a typical error of ±$25 (16.9%)."
    ),
    "$200–500": (
        "Premium listings ($200–500) have wider ranges due to "
        "factors like views and interior quality; typical error ±$57 (19.5%)."
    ),
}


# --------------------------------------------------------------------------- #
# assign_confidence — data quality tier                                        #
# --------------------------------------------------------------------------- #

def assign_confidence(
    days_since_last_review: float | np.ndarray | pd.Series,
    number_of_reviews: float | np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Assign a data-quality confidence tier based purely on review activity.

    Parameters
    ----------
    days_since_last_review : Days since most recent review.
                             Use 9999 for never-reviewed listings.
    number_of_reviews      : Total lifetime review count.

    Returns
    -------
    np.ndarray of str — "HIGH", "MEDIUM", or "LOW" per row.

    Rules (priority: LOW overrides all, then MEDIUM, then HIGH):
      LOW    : days_since_last_review > 1000  OR  number_of_reviews == 0
      MEDIUM : days_since_last_review in [365, 1000]  OR  number_of_reviews in [1, 5]
      HIGH   : days_since_last_review < 365  AND  number_of_reviews > 5
    """
    dslr  = np.asarray(days_since_last_review, dtype=float)
    n_rev = np.asarray(number_of_reviews,      dtype=float)

    is_low    = (dslr > 1000) | (n_rev == 0)
    is_medium = ((dslr >= 365) & (dslr <= 1000)) | ((n_rev >= 1) & (n_rev <= 5))
    is_high   = (dslr < 365) & (n_rev > 5)

    # Build from least to most restrictive, then let LOW override everything
    tiers = np.where(is_high,   "HIGH",
            np.where(is_medium, "MEDIUM",
                                "LOW"))
    tiers = np.where(is_low, "LOW", tiers)
    return tiers


# --------------------------------------------------------------------------- #
# price_precision_range — bucket-based error bounds                           #
# --------------------------------------------------------------------------- #

def price_precision_range(
    predicted_price: float | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (price_range_low, price_range_high) for each prediction using
    the empirical MAE of the model within each price bucket.

    Bucket assignments:
        $0–100    → ± $17
        $100–200  → ± $25
        $200–500  → ± $57
    Prices outside $0–500 fall back to the nearest bucket's error.
    """
    price = np.asarray(predicted_price, dtype=float)

    # Bucket index: 0 = $0-100, 1 = $100-200, 2 = $200-500
    idx = np.searchsorted(_BUCKET_EDGES[1:], price, side="right")
    idx = np.clip(idx, 0, len(_BUCKET_ERRORS) - 1)

    errors = np.array(_BUCKET_ERRORS, dtype=float)[idx]
    return (price - errors).round(2), (price + errors).round(2)


def _bucket_label(predicted_price: float) -> str:
    idx = min(
        int(np.searchsorted(_BUCKET_EDGES[1:], predicted_price, side="right")),
        len(_BUCKET_LABELS) - 1,
    )
    return _BUCKET_LABELS[idx]


# --------------------------------------------------------------------------- #
# confidence_explanation — plain English per prediction                       #
# --------------------------------------------------------------------------- #

def build_explanation(
    confidence_tier: str,
    predicted_price: float,
) -> str:
    """Return a single plain-English explanation string for one prediction."""
    bucket = _bucket_label(predicted_price)
    return f"{_TIER_PREFIX[confidence_tier]} {_BUCKET_SUFFIX[bucket]}"


def build_explanations(
    confidence_tiers: np.ndarray,
    predicted_prices: np.ndarray,
) -> np.ndarray:
    """Vectorised version of build_explanation for a full column."""
    return np.array([
        build_explanation(t, p)
        for t, p in zip(confidence_tiers, predicted_prices)
    ])


# --------------------------------------------------------------------------- #
# Full prediction pipeline                                                     #
# --------------------------------------------------------------------------- #

def predict_with_confidence(
    df: pd.DataFrame,
    model_path: Path | str = MODELS_DIR / "xgboost_full.pkl",
) -> pd.DataFrame:
    """
    Run the full XGBoost model on ``df`` and return a structured results DataFrame.

    Input
    -----
    df : pd.DataFrame
        Feature rows (same schema as data/features.csv). Must include a
        ``price`` column for prepare_features; set to 0 if unknown.

    Returns
    -------
    pd.DataFrame with one row per input row and columns:
        predicted_price        dollar-scale prediction
        confidence_tier        "HIGH" / "MEDIUM" / "LOW"
        price_range_low        predicted_price − bucket_mae
        price_range_high       predicted_price + bucket_mae
        confidence_explanation plain-English string
    """
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]

    X, _ = prepare_features(df.copy())
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]

    trained_cols = artifact.get("feature_cols", list(X.columns))
    for col in trained_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[[c for c in trained_cols if c in X.columns]]

    log_pred  = pipeline.predict(X)
    predicted = np.exp(np.clip(log_pred, *_LOG_CLIP))

    dslr  = (df["days_since_last_review"].values
             if "days_since_last_review" in df.columns
             else np.full(len(df), 9999))
    n_rev = (df["number_of_reviews"].values
             if "number_of_reviews" in df.columns
             else np.zeros(len(df)))

    tiers       = assign_confidence(dslr, n_rev)
    low, high   = price_precision_range(predicted)
    explanation = build_explanations(tiers, predicted)

    return pd.DataFrame({
        "predicted_price":        predicted.round(2),
        "confidence_tier":        tiers,
        "price_range_low":        low,
        "price_range_high":       high,
        "confidence_explanation": explanation,
    }, index=df.index)


# --------------------------------------------------------------------------- #
# CLI — validate on held-out test set                                          #
# --------------------------------------------------------------------------- #

def _validate_on_test_set() -> None:
    from sklearn.model_selection import train_test_split

    print("Loading model and test set ...")
    artifact = joblib.load(MODELS_DIR / "xgboost_full.pkl")
    pipeline = artifact["pipeline"]

    df_raw = pd.read_csv(DATA_DIR / "features.csv")
    df     = df_raw[df_raw["price"] <= PRICE_CAP].reset_index(drop=True)

    X, y = prepare_features(df)
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    _, X_t, _, y_t = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
    )
    df_test = df.loc[X_t.index].copy()
    print(f"  Test set: {len(X_t):,} listings\n")

    log_pred  = pipeline.predict(X_t)
    predicted = np.exp(np.clip(log_pred, *_LOG_CLIP))
    actual    = np.exp(y_t.values)
    abs_error = np.abs(predicted - actual)
    pct_error = abs_error / actual * 100

    dslr  = (df_test["days_since_last_review"].values
             if "days_since_last_review" in df_test.columns
             else np.full(len(df_test), 9999))
    n_rev = (df_test["number_of_reviews"].values
             if "number_of_reviews" in df_test.columns
             else np.zeros(len(df_test)))

    tiers     = assign_confidence(dslr, n_rev)
    low, high = price_precision_range(predicted)
    explanation = build_explanations(tiers, predicted)

    results = pd.DataFrame({
        "actual":      actual,
        "predicted":   predicted,
        "abs_error":   abs_error,
        "pct_error":   pct_error,
        "tier":        tiers,
        "range_low":   low,
        "range_high":  high,
        "dslr":        dslr,
        "n_reviews":   n_rev,
        "explanation": explanation,
    })

    tier_order = ["HIGH", "MEDIUM", "LOW"]
    bar_width  = 30

    # ------------------------------------------------------------------ #
    # Tier distribution                                                    #
    # ------------------------------------------------------------------ #
    counts = results["tier"].value_counts().reindex(tier_order, fill_value=0)
    total  = len(results)

    print(f"{'='*65}")
    print("  Confidence Tier Distribution  (review-activity only)")
    print(f"{'='*65}")
    for tier in tier_order:
        n   = counts[tier]
        pct = n / total * 100
        bar = "█" * int(pct / 100 * bar_width)
        print(f"  {tier:<8} {n:>5,} listings  ({pct:5.1f}%)  {bar}")

    # ------------------------------------------------------------------ #
    # Error metrics by tier                                               #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*65}")
    print("  Prediction Error by Confidence Tier")
    print(f"{'='*65}")
    print(f"  {'Tier':<8}  {'N':>5}  {'MAE':>9}  {'RMSE':>9}  {'MAPE':>7}  "
          f"{'±10%':>7}  {'±20%':>7}  {'Bias':>8}")
    print(f"  {'-'*68}")
    for tier in tier_order:
        sub = results[results["tier"] == tier]
        if len(sub) == 0:
            continue
        mae  = sub["abs_error"].mean()
        rmse = np.sqrt((sub["abs_error"] ** 2).mean())
        mape = sub["pct_error"].mean()
        w10  = (sub["pct_error"] <= 10).mean() * 100
        w20  = (sub["pct_error"] <= 20).mean() * 100
        bias = (sub["predicted"] - sub["actual"]).mean()
        bstr = f"+${bias:.1f}" if bias >= 0 else f"-${abs(bias):.1f}"
        print(f"  {tier:<8}  {len(sub):>5,}  ${mae:>8.2f}  ${rmse:>8.2f}  "
              f"{mape:>6.1f}%  {w10:>6.1f}%  {w20:>6.1f}%  {bstr:>8}")

    high_mae = results[results["tier"]=="HIGH"]["abs_error"].mean()
    med_mae  = results[results["tier"]=="MEDIUM"]["abs_error"].mean()
    low_mae  = results[results["tier"]=="LOW"]["abs_error"].mean()
    ordered  = high_mae < med_mae < low_mae
    print(f"\n  {'✓' if ordered else '✗'} Tier MAE ordering: "
          f"HIGH ${high_mae:.2f} {'<' if high_mae < med_mae else '>'} "
          f"MEDIUM ${med_mae:.2f} {'<' if med_mae < low_mae else '>'} "
          f"LOW ${low_mae:.2f}")
    print(f"  MAE lift HIGH→LOW: ${low_mae - high_mae:.2f}  "
          f"({(low_mae/high_mae - 1)*100:.0f}% worse)")

    # ------------------------------------------------------------------ #
    # Price range coverage (did actual fall within predicted range?)      #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*65}")
    print("  Price Range Coverage  (actual price within predicted range?)")
    print(f"{'='*65}")
    results["in_range"] = (
        (results["actual"] >= results["range_low"]) &
        (results["actual"] <= results["range_high"])
    )
    print(f"  {'Tier':<8}  {'N':>5}  {'In range':>9}  {'Coverage':>9}  "
          f"{'Avg range width':>16}")
    print(f"  {'-'*52}")
    for tier in tier_order:
        sub = results[results["tier"] == tier]
        if len(sub) == 0:
            continue
        in_rng  = sub["in_range"].mean() * 100
        width   = (sub["range_high"] - sub["range_low"]).mean()
        print(f"  {tier:<8}  {len(sub):>5,}  "
              f"{sub['in_range'].sum():>9,}  {in_rng:>8.1f}%  ${width:>15.0f}")

    overall_cov = results["in_range"].mean() * 100
    print(f"\n  Overall coverage: {overall_cov:.1f}%  "
          f"(actual price fell within predicted range for {overall_cov:.1f}% of listings)")

    # ------------------------------------------------------------------ #
    # Sample output — one example per tier                                #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*65}")
    print("  Sample Full Output (1 example per tier)")
    print(f"{'='*65}")
    for tier in tier_order:
        sub = results[results["tier"] == tier]
        if len(sub) == 0:
            continue
        # Pick a listing whose actual price falls inside the range (best-case demo)
        in_range_sub = sub[sub["in_range"]]
        row = (in_range_sub if len(in_range_sub) else sub).iloc[0]
        print(f"\n  [{tier}]")
        print(f"    predicted_price       : ${row.predicted:.0f}")
        print(f"    actual_price          : ${row.actual:.0f}")
        print(f"    confidence_tier       : {tier}")
        print(f"    price_range_low       : ${row.range_low:.0f}")
        print(f"    price_range_high      : ${row.range_high:.0f}")
        print(f"    confidence_explanation: \"{row.explanation}\"")
        print(f"    --- signals ---")
        print(f"    days_since_last_review: {row.dslr:.0f}")
        print(f"    number_of_reviews     : {row.n_reviews:.0f}")


if __name__ == "__main__":
    _validate_on_test_set()
