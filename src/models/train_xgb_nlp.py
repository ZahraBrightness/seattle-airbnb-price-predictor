"""
XGBoost retrain with NLP features added to features.csv.

Applies the same preprocessing as xgboost_log.pkl:
  - Cap price at $500
  - log(price) as target
  - Fill bedrooms nulls with median
  - One-hot encode categorical columns

NLP additions (19 features extracted from description + amenities):
  has_view, has_waterfront, is_downtown, has_hot_tub, has_pool,
  has_parking, has_gym, is_newly_renovated, is_luxury, is_cozy,
  has_fireplace, has_private_entrance, has_backyard, is_entire_floor,
  amenity_has_hot_tub, amenity_has_pool, amenity_has_parking,
  amenity_has_gym, amenity_has_ev_charger

Usage:
    python src/models/train_xgb_nlp.py
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

NLP_FEATURES = [
    "has_view", "has_waterfront", "is_downtown", "has_hot_tub", "has_pool",
    "has_parking", "has_gym", "is_newly_renovated", "is_luxury", "is_cozy",
    "has_fireplace", "has_private_entrance", "has_backyard", "is_entire_floor",
    "amenity_has_hot_tub", "amenity_has_pool", "amenity_has_parking",
    "amenity_has_gym", "amenity_has_ev_charger",
]


# --------------------------------------------------------------------------- #
# Scorers (dollar-scale, with log-clip to prevent exp() blowup)               #
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
        "predicted": predicted,
        "abs_error": np.abs(predicted - actual),
        "pct_error": np.abs(predicted - actual) / actual * 100,
        "error":     predicted - actual,
    })
    df["bucket"] = pd.cut(df["actual"], bins=BUCKET_BINS,
                          labels=BUCKET_LABELS, right=False)
    return (df.groupby("bucket", observed=True)
              .agg(n=("abs_error", "count"),
                   mae=("abs_error", "mean"),
                   mape=("pct_error", "mean"),
                   bias=("error", "mean"))
              .reset_index())


# --------------------------------------------------------------------------- #
# SHAP summary with NLP feature callout                                       #
# --------------------------------------------------------------------------- #

def print_shap(model: XGBRegressor, X_train: pd.DataFrame,
               X_test: pd.DataFrame, top_n: int = 20) -> None:
    background = shap.sample(X_train, 200, random_state=RANDOM_STATE)
    explainer  = shap.TreeExplainer(model, background)
    shap_vals  = explainer.shap_values(X_test)

    shap_df  = pd.DataFrame(shap_vals, columns=X_test.columns)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    mean_dir = shap_df.mean()
    total    = mean_abs.sum()

    nlp_set = set(NLP_FEATURES)

    print(f"\n  {'Rank':<5}  {'Feature':<40} {'|SHAP|':>8}  {'% total':>7}  {'Dir':>6}  NLP?")
    print(f"  {'-'*82}")
    for rank, feat in enumerate(mean_abs.head(top_n).index, 1):
        v     = mean_abs[feat]
        pct   = v / total * 100
        arrow = "+up  " if mean_dir[feat] > 0 else "-down"
        nlp_marker = " ◄" if feat in nlp_set else ""
        print(f"  {rank:<5}  {feat:<40} {v:>8.4f}  {pct:>6.1f}%  {arrow}  {nlp_marker}")

    top2_pct = mean_abs.head(2).sum() / total * 100
    print(f"\n  Top-2 share: {top2_pct:.1f}%")

    # NLP features ranked below top_n
    nlp_ranks = {feat: rank for rank, feat in enumerate(mean_abs.index, 1)
                 if feat in nlp_set}
    nlp_in_top = {f: r for f, r in nlp_ranks.items() if r <= top_n}
    nlp_outside = {f: r for f, r in nlp_ranks.items() if r > top_n}

    print(f"\n  NLP features in top {top_n}: {len(nlp_in_top)}")
    for feat, rank in sorted(nlp_in_top.items(), key=lambda x: x[1]):
        v = mean_abs[feat]
        print(f"    #{rank:<4} {feat:<35} |SHAP|={v:.4f}  ({v/total*100:.1f}%)")

    if nlp_outside:
        print(f"\n  NLP features outside top {top_n}:")
        for feat, rank in sorted(nlp_outside.items(), key=lambda x: x[1]):
            v = mean_abs[feat]
            print(f"    #{rank:<4} {feat:<35} |SHAP|={v:.4f}  ({v/total*100:.1f}%)")

    # Save full SHAP values
    out = MODELS_DIR / "xgboost_nlp_shap_values.csv"
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

    nlp_present = [f for f in NLP_FEATURES if f in df_raw.columns]
    print(f"NLP features present: {len(nlp_present)}/{len(NLP_FEATURES)}")

    df = df_raw[df_raw["price"] <= PRICE_CAP].copy()
    removed = len(df_raw) - len(df)
    print(f"Capped at ${PRICE_CAP}: removed {removed:,} ({removed/len(df_raw)*100:.1f}%), "
          f"{len(df):,} remaining")

    # ------------------------------------------------------------------ #
    # Prepare features                                                     #
    # ------------------------------------------------------------------ #
    X, y = prepare_features(df)
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]

    # Verify NLP features made it through prepare_features
    nlp_in_X = [f for f in NLP_FEATURES if f in X.columns]
    print(f"\nNLP features in model matrix: {len(nlp_in_X)}/{len(NLP_FEATURES)}")
    if len(nlp_in_X) < len(NLP_FEATURES):
        missing = set(NLP_FEATURES) - set(nlp_in_X)
        print(f"  Missing: {missing}")

    log_skew = float(y.skew())
    print(f"log(price) skew: {log_skew:.2f}  (raw: {df['price'].skew():.2f})")

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
    y_pred = pipeline.predict(X_test)
    test_mae  = _mae_d(y_test.values, y_pred)
    test_rmse = _rmse_d(y_test.values, y_pred)
    test_r2   = _r2_d(y_test.values, y_pred)

    elapsed = time.time() - t0

    # ------------------------------------------------------------------ #
    # Comparison table                                                     #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*62}")
    print(f"  MAE Comparison: No NLP vs With NLP (same XGBoost params)")
    print(f"{'='*62}")
    print(f"  {'Metric':<14}  {'No NLP (baseline)':>18}  {'With NLP':>10}")
    print(f"  {'-'*46}")
    print(f"  {'Test MAE':<14}  {'$31.64':>18}  ${test_mae:>9.2f}")
    print(f"  {'Test RMSE':<14}  {'$47.58':>18}  ${test_rmse:>9.2f}")
    print(f"  {'Test R²':<14}  {'0.6828':>18}  {test_r2:>10.4f}")
    print(f"  {'CV MAE':<14}  {'—':>18}  "
          f"${-cv['test_mae'].mean():>7.2f} ± ${cv['test_mae'].std():>5.2f}")
    print(f"  {'Features':<14}  {'105':>18}  {X.shape[1]:>10}")
    print(f"  {'Train time':<14}  {'~5s':>18}  {elapsed:>9.1f}s")

    delta_mae = test_mae - 31.64
    delta_str = f"{delta_mae:+.2f}" if delta_mae >= 0 else f"{delta_mae:.2f}"
    print(f"\n  Δ MAE (NLP − baseline): ${delta_str}  "
          f"({'worse' if delta_mae > 0 else 'better'})")
    print(f"{'='*62}")

    # ------------------------------------------------------------------ #
    # MAE by price bucket                                                  #
    # ------------------------------------------------------------------ #
    prev_mape = {"$0–100": 23.4, "$100–200": 17.7, "$200–500": 21.0}
    prev_mae  = {"$0–100": 16.59, "$100–200": 25.63, "$200–500": 62.25}

    bkt = bucket_stats(y_test.values, y_pred)

    print(f"\n  MAE by Price Bucket (vs No-NLP baseline):")
    print(f"  {'Bucket':<12} {'N':>5}  {'MAE':>9}  {'Prev MAE':>9}  "
          f"{'ΔMAE':>7}  {'MAPE':>7}  {'Prev MAPE':>10}  {'ΔMAPE':>7}  {'Bias':>8}")
    print(f"  {'-'*80}")
    for row in bkt.itertuples():
        bkt_str   = str(row.bucket)
        delta_mae_b  = row.mae  - prev_mae.get(bkt_str, row.mae)
        delta_mape_b = row.mape - prev_mape.get(bkt_str, row.mape)
        bias_str  = f"+${row.bias:.0f}" if row.bias >= 0 else f"-${abs(row.bias):.0f}"
        print(f"  {bkt_str:<12} {row.n:>5,}  ${row.mae:>8.2f}  "
              f"${prev_mae.get(bkt_str, 0):>8.2f}  "
              f"{delta_mae_b:>+6.2f}  "
              f"{row.mape:>6.1f}%  "
              f"{prev_mape.get(bkt_str, 0):>9.1f}%  "
              f"{delta_mape_b:>+6.1f}pp  "
              f"{bias_str:>8}")

    # ------------------------------------------------------------------ #
    # SHAP                                                                 #
    # ------------------------------------------------------------------ #
    print(f"\nSHAP Feature Importance (top 20, NLP features marked ◄):")
    model = pipeline.named_steps["model"]
    print_shap(model, X_train, X_test, top_n=20)

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    model_path = MODELS_DIR / "xgboost_nlp.pkl"
    joblib.dump({
        "pipeline":     pipeline,
        "feature_cols": list(X.columns),
        "price_cap":    PRICE_CAP,
        "nlp_features": nlp_in_X,
    }, model_path)
    print(f"\nModel saved → {model_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
