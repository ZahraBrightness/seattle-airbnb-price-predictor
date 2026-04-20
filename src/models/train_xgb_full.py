"""
XGBoost retrain with NLP + calendar + sentiment features.

Applies the same preprocessing as previous runs:
  - Cap price at $500
  - log(price) as target
  - Fill bedrooms nulls with median
  - One-hot encode categorical columns
  - Exclude peak_demand_score and estimated_annual_revenue (circular)

New feature groups vs xgboost_nlp.pkl:
  Calendar (8): peak_availability_rate, off_availability_rate,
                availability_gap, pct_weekend_available,
                avg_minimum_nights_cal, has_dynamic_minimum,
                consecutive_blocked_rate  (+ 2 circular excluded)
  Sentiment (8): avg_sentiment, recent_sentiment, sentiment_trend,
                 pct_positive_reviews, pct_negative_reviews,
                 positive_keyword_count, negative_keyword_count,
                 review_velocity

Usage:
    python src/models/train_xgb_full.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baseline import prepare_features

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)

PRICE_CAP    = 500
RANDOM_STATE = 42
_LOG_CLIP    = (np.log(1), np.log(PRICE_CAP * 2))

BUCKET_BINS   = [0, 100, 200, 500]
BUCKET_LABELS = ["$0–100", "$100–200", "$200–500"]

CALENDAR_FEATURES = [
    "peak_availability_rate", "off_availability_rate", "availability_gap",
    "pct_weekend_available", "avg_minimum_nights_cal", "has_dynamic_minimum",
    "consecutive_blocked_rate",
]
SENTIMENT_FEATURES = [
    "avg_sentiment", "recent_sentiment", "sentiment_trend",
    "pct_positive_reviews", "pct_negative_reviews",
    "positive_keyword_count", "negative_keyword_count", "review_velocity",
]
NEW_FEATURES = CALENDAR_FEATURES + SENTIMENT_FEATURES


# --------------------------------------------------------------------------- #
# Scorers                                                                      #
# --------------------------------------------------------------------------- #

def _mae_d(yt, yp):
    return mean_absolute_error(np.exp(yt), np.exp(np.clip(yp, *_LOG_CLIP)))

def _rmse_d(yt, yp):
    return np.sqrt(mean_squared_error(np.exp(yt), np.exp(np.clip(yp, *_LOG_CLIP))))

def _r2_d(yt, yp):
    return r2_score(np.exp(yt), np.exp(np.clip(yp, *_LOG_CLIP)))

CV_SCORING = {
    "mae":  make_scorer(_mae_d,  greater_is_better=False),
    "rmse": make_scorer(_rmse_d, greater_is_better=False),
    "r2":   make_scorer(_r2_d),
}


# --------------------------------------------------------------------------- #
# MAE by price bucket                                                          #
# --------------------------------------------------------------------------- #

def bucket_stats(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> pd.DataFrame:
    actual    = np.exp(y_true_log)
    predicted = np.exp(np.clip(y_pred_log, *_LOG_CLIP))
    df = pd.DataFrame({
        "actual":    actual,
        "abs_error": np.abs(predicted - actual),
        "pct_error": np.abs(predicted - actual) / actual * 100,
        "error":     predicted - actual,
    })
    df["bucket"] = pd.cut(df["actual"], bins=BUCKET_BINS,
                          labels=BUCKET_LABELS, right=False)
    return (df.groupby("bucket", observed=True)
              .agg(n    =("abs_error", "count"),
                   mae  =("abs_error", "mean"),
                   mape =("pct_error", "mean"),
                   bias =("error",     "mean"))
              .reset_index())


# --------------------------------------------------------------------------- #
# SHAP with new-feature callout                                                #
# --------------------------------------------------------------------------- #

def print_shap(model: XGBRegressor,
               X_train: pd.DataFrame,
               X_test:  pd.DataFrame,
               top_n: int = 25) -> None:
    background = shap.sample(X_train, 200, random_state=RANDOM_STATE)
    explainer  = shap.TreeExplainer(model, background)
    shap_vals  = explainer.shap_values(X_test)

    shap_df  = pd.DataFrame(shap_vals, columns=X_test.columns)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    mean_dir = shap_df.mean()
    total    = mean_abs.sum()

    cal_set  = set(CALENDAR_FEATURES)
    sent_set = set(SENTIMENT_FEATURES)

    print(f"\n  {'Rank':<5}  {'Feature':<42} {'|SHAP|':>8}  {'%tot':>6}  "
          f"{'Dir':>5}  Tag")
    print(f"  {'-'*82}")
    for rank, feat in enumerate(mean_abs.head(top_n).index, 1):
        v   = mean_abs[feat]
        pct = v / total * 100
        arrow = "+up " if mean_dir[feat] > 0 else "-dn "
        tag = " CAL" if feat in cal_set else " SEN" if feat in sent_set else ""
        print(f"  {rank:<5}  {feat:<42} {v:>8.4f}  {pct:>5.1f}%  {arrow}  {tag}")

    print(f"\n  Top-2 share: {mean_abs.head(2).sum()/total*100:.1f}%")

    # Summarise new feature groups
    for label, feat_set in [("Calendar", cal_set), ("Sentiment", sent_set)]:
        in_top = [(f, int(list(mean_abs.index).index(f))+1)
                  for f in feat_set if f in mean_abs.index]
        in_top.sort(key=lambda x: x[1])
        print(f"\n  {label} features ({len(feat_set)} total):")
        for feat, rank in in_top:
            v = mean_abs.get(feat, 0)
            print(f"    #{rank:<4} {feat:<38} |SHAP|={v:.4f}  ({v/total*100:.1f}%)")

    out = MODELS_DIR / "xgboost_full_shap_values.csv"
    shap_df.to_csv(out, index=False)
    print(f"\n  Saved SHAP values → {out}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Load + cap                                                           #
    # ------------------------------------------------------------------ #
    df_raw = pd.read_csv(DATA_DIR / "features.csv")
    print(f"Loaded features.csv: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

    new_present = [f for f in NEW_FEATURES if f in df_raw.columns]
    print(f"New feature groups present: "
          f"{sum(f in df_raw.columns for f in CALENDAR_FEATURES)}/{len(CALENDAR_FEATURES)} calendar, "
          f"{sum(f in df_raw.columns for f in SENTIMENT_FEATURES)}/{len(SENTIMENT_FEATURES)} sentiment")

    df = df_raw[df_raw["price"] <= PRICE_CAP].copy()
    removed = len(df_raw) - len(df)
    print(f"Capped at ${PRICE_CAP}: removed {removed:,} ({removed/len(df_raw)*100:.1f}%), "
          f"{len(df):,} remaining")

    # ------------------------------------------------------------------ #
    # Prepare features                                                     #
    # ------------------------------------------------------------------ #
    X, y = prepare_features(df)
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]

    new_in_X = [f for f in NEW_FEATURES if f in X.columns]
    print(f"\nNew features in model matrix: {len(new_in_X)}/{len(NEW_FEATURES)}")
    print(f"log(price) skew: {y.skew():.2f}  (raw: {df['price'].skew():.2f})")

    # ------------------------------------------------------------------ #
    # Split                                                                #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {X.shape[1]}")

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
    pipeline = Pipeline([("model", XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
    ))])

    print("\nRunning 5-fold CV ...")
    cv = cross_validate(pipeline, X_train, y_train, cv=5, scoring=CV_SCORING)
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #
    y_pred    = pipeline.predict(X_test)
    test_mae  = _mae_d(y_test.values,  y_pred)
    test_rmse = _rmse_d(y_test.values, y_pred)
    test_r2   = _r2_d(y_test.values,   y_pred)
    elapsed   = time.time() - t0

    # ------------------------------------------------------------------ #
    # Comparison table                                                     #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*66}")
    print(f"  MAE Comparison — adding calendar + sentiment features")
    print(f"{'='*66}")
    print(f"  {'Metric':<14}  {'NLP only ($30.92)':>18}  {'Full (+cal+sent)':>16}")
    print(f"  {'-'*52}")
    print(f"  {'Test MAE':<14}  {'$30.92':>18}  ${test_mae:>15.2f}")
    print(f"  {'Test RMSE':<14}  {'$46.27':>18}  ${test_rmse:>15.2f}")
    print(f"  {'Test R²':<14}  {'0.7000':>18}  {test_r2:>16.4f}")
    print(f"  {'CV MAE':<14}  {'—':>18}  "
          f"${-cv['test_mae'].mean():>7.2f} ± ${cv['test_mae'].std():>5.2f}")
    print(f"  {'Features':<14}  {'137':>18}  {X.shape[1]:>16}")
    print(f"  {'Train time':<14}  {'~6s':>18}  {elapsed:>15.1f}s")

    delta = test_mae - 30.92
    print(f"\n  Δ MAE (full − NLP only): ${delta:+.2f}  "
          f"({'worse' if delta > 0 else 'better'})")
    print(f"{'='*66}")

    # ------------------------------------------------------------------ #
    # MAE by price bucket                                                  #
    # ------------------------------------------------------------------ #
    prev_mae  = {"$0–100": 15.81, "$100–200": 25.28, "$200–500": 57.00}
    prev_mape = {"$0–100": 22.6,  "$100–200": 17.5,  "$200–500": 19.6}

    bkt = bucket_stats(y_test.values, y_pred)
    print(f"\n  MAE by Price Bucket (vs NLP-only baseline):")
    print(f"  {'Bucket':<12} {'N':>5}  {'MAE':>9}  {'Prev MAE':>9}  "
          f"{'ΔMAE':>7}  {'MAPE':>7}  {'ΔMAPE':>8}  {'Bias':>8}")
    print(f"  {'-'*76}")
    for row in bkt.itertuples():
        s     = str(row.bucket)
        dmae  = row.mae  - prev_mae.get(s,  row.mae)
        dmape = row.mape - prev_mape.get(s, row.mape)
        bias  = f"+${row.bias:.0f}" if row.bias >= 0 else f"-${abs(row.bias):.0f}"
        print(f"  {s:<12} {row.n:>5,}  ${row.mae:>8.2f}  "
              f"${prev_mae.get(s,0):>8.2f}  {dmae:>+6.2f}  "
              f"{row.mape:>6.1f}%  {dmape:>+7.1f}pp  {bias:>8}")

    # ------------------------------------------------------------------ #
    # SHAP                                                                 #
    # ------------------------------------------------------------------ #
    print(f"\nSHAP Feature Importance (top 25, CAL=calendar, SEN=sentiment):")
    model = pipeline.named_steps["model"]
    print_shap(model, X_train, X_test, top_n=25)

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    model_path = MODELS_DIR / "xgboost_full.pkl"
    joblib.dump({
        "pipeline":          pipeline,
        "feature_cols":      list(X.columns),
        "price_cap":         PRICE_CAP,
        "calendar_features": CALENDAR_FEATURES,
        "sentiment_features": SENTIMENT_FEATURES,
    }, model_path)
    print(f"\nModel saved → {model_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
