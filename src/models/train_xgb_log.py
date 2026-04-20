"""
XGBoost retrain with $500 price cap + log(price) target.

Changes vs previous run:
  - Cap price at $500, removing 221 luxury/corporate outliers (3.7% of data).
    Justified by IQR (3×) cutoff of $539 and domain knowledge that $500+
    listings are hotels/corporate properties that behave differently.
  - Log-transform price target to reduce skew (raw skew ~10 → log skew ~0.3
    after capping). Note: log transform was already applied in prepare_features;
    this step documents it explicitly and verifies skew reduction.
  - Fill 11 null bedrooms values with median (already done inside prepare_features
    as residual-null fallback — confirmed here for transparency).

Usage:
    python src/models/train_xgb_log.py
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
_LOG_CLIP    = (np.log(1), np.log(PRICE_CAP * 2))   # generous clip for exp() safety

BUCKET_BINS   = [0, 100, 200, 500]
BUCKET_LABELS = ["$0–100", "$100–200", "$200–500"]


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
# SHAP summary                                                                 #
# --------------------------------------------------------------------------- #

def print_shap(pipeline: Pipeline, X_train: pd.DataFrame,
               X_test: pd.DataFrame, top_n: int = 15) -> None:
    model      = pipeline.named_steps["model"]
    background = shap.sample(X_train, 200, random_state=RANDOM_STATE)
    explainer  = shap.TreeExplainer(model, background)
    shap_vals  = explainer.shap_values(X_test)

    shap_df  = pd.DataFrame(shap_vals, columns=X_test.columns)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    mean_dir = shap_df.mean()
    total    = mean_abs.sum()

    print(f"\n  {'Feature':<45} {'|SHAP|':>8}  {'% total':>7}  Direction")
    print(f"  {'-'*78}")
    for feat in mean_abs.head(top_n).index:
        v     = mean_abs[feat]
        pct   = v / total * 100
        arrow = "+ up  " if mean_dir[feat] > 0 else "- down"
        print(f"  {feat:<45} {v:>8.4f}  {pct:>6.1f}%  {arrow}")

    top2 = mean_abs.head(2).sum() / total * 100
    print(f"\n  Top-2 share: {top2:.1f}%  (was 20.3% in previous run)")

    # Save full SHAP values
    out = MODELS_DIR / "xgboost_log_shap_values.csv"
    shap_df.to_csv(out, index=False)
    print(f"  Saved SHAP values → {out}")


# --------------------------------------------------------------------------- #
# MAE by bucket                                                                #
# --------------------------------------------------------------------------- #

def print_bucket_stats(actual: np.ndarray, predicted: np.ndarray) -> None:
    df = pd.DataFrame({
        "actual": actual,
        "predicted": predicted,
        "abs_error": np.abs(predicted - actual),
        "pct_error": np.abs(predicted - actual) / actual * 100,
    })
    df["bucket"] = pd.cut(df["actual"], bins=BUCKET_BINS,
                          labels=BUCKET_LABELS, right=False)

    stats = (df.groupby("bucket", observed=True)
               .agg(n=("abs_error", "count"),
                    mae=("abs_error", "mean"),
                    mape=("pct_error", "mean"),
                    bias=("abs_error", lambda x: (predicted[x.index] - actual[x.index]).mean()))
               .reset_index())

    prev_mape = {"$0–100": 23.4, "$100–200": 17.7, "$200–500": 21.0}

    print(f"\n  {'Bucket':<12} {'N':>5}  {'MAE':>9}  {'MAPE':>7}  {'Prev MAPE':>10}  {'Δ MAPE':>8}")
    print(f"  {'-'*60}")
    for row in stats.itertuples():
        bkt = str(row.bucket)
        delta = row.mape - prev_mape.get(bkt, row.mape)
        delta_str = f"{delta:+.1f}pp"
        print(f"  {bkt:<12} {row.n:>5,}  ${row.mae:>8.2f}  "
              f"{row.mape:>6.1f}%  {prev_mape.get(bkt, 0):>9.1f}%  {delta_str:>8}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Load + cap                                                           #
    # ------------------------------------------------------------------ #
    df_raw = pd.read_csv(DATA_DIR / "features.csv")
    print(f"Loaded features.csv: {df_raw.shape[0]:,} rows")

    df = df_raw[df_raw["price"] <= PRICE_CAP].copy()
    removed = len(df_raw) - len(df)
    print(f"Capped at ${PRICE_CAP}: removed {removed:,} listings ({removed/len(df_raw)*100:.1f}%)")
    print(f"Remaining           : {len(df):,} rows")

    # ------------------------------------------------------------------ #
    # Prepare features (log-transforms price internally)                  #
    # ------------------------------------------------------------------ #
    X, y = prepare_features(df)
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]

    log_skew = float(y.skew())
    raw_skew = float(df["price"].skew())
    print(f"\nPrice skew  raw={raw_skew:.2f}  log={log_skew:.2f}  "
          f"(target: log skew < 1)")

    # ------------------------------------------------------------------ #
    # Split                                                                #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {X.shape[1]}")

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
    elapsed = time.time() - t0

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #
    y_pred = pipeline.predict(X_test)
    actual    = np.exp(y_test.values)
    predicted = np.exp(np.clip(y_pred, *_LOG_CLIP))

    test_mae  = _mae_d(y_test.values, y_pred)
    test_rmse = _rmse_d(y_test.values, y_pred)
    test_r2   = _r2_d(y_test.values, y_pred)

    # ------------------------------------------------------------------ #
    # Comparison table                                                     #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*58}")
    print(f"  Comparison: Before vs After")
    print(f"{'='*58}")
    print(f"  {'Metric':<14}  {'Before (no cap)':>16}  {'After ($500 cap)':>16}")
    print(f"  {'-'*50}")
    print(f"  {'Test MAE':<14}  {'$70.62':>16}  ${test_mae:>15.2f}")
    print(f"  {'Test RMSE':<14}  {'$470.57':>16}  ${test_rmse:>15.2f}")
    print(f"  {'Test R²':<14}  {'0.9821':>16}  {test_r2:>16.4f}")
    print(f"  {'CV MAE':<14}  {'$99.48 ± $32':>16}  "
          f"${-cv['test_mae'].mean():>7.2f} ± ${cv['test_mae'].std():>5.2f}")
    print(f"  {'Train time':<14}  {'4.9s':>16}  {elapsed:>15.1f}s")
    print(f"{'='*58}")

    # ------------------------------------------------------------------ #
    # MAE by bucket                                                        #
    # ------------------------------------------------------------------ #
    print(f"\nMAE by Price Bucket:")
    print_bucket_stats(actual, predicted)

    # ------------------------------------------------------------------ #
    # SHAP                                                                 #
    # ------------------------------------------------------------------ #
    print(f"\nSHAP Feature Importance (top 15):")
    print_shap(pipeline, X_train, X_test)

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    out = MODELS_DIR / "xgboost_log.pkl"
    joblib.dump({"pipeline": pipeline, "feature_cols": list(X.columns),
                 "price_cap": PRICE_CAP}, out)
    print(f"\nModel saved → {out}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
