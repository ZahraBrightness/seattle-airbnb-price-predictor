"""
Hyperparameter tuning for XGBoost price prediction using Optuna.

Applies the same preprocessing as xgboost_log.pkl:
  - Cap price at $500
  - log(price) as target
  - Fill bedrooms nulls with median
  - One-hot encode categorical columns

Usage:
    python src/models/tuning.py [--trials N]   # default: 30
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baseline import prepare_features

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)

PRICE_CAP    = 500
RANDOM_STATE = 42
N_CV_FOLDS   = 5
_LOG_CLIP    = (np.log(1), np.log(PRICE_CAP * 2))

BUCKET_BINS   = [0, 100, 200, 500]
BUCKET_LABELS = ["$0–100", "$100–200", "$200–500"]

# Silence Optuna's default INFO logging; we print our own trial summaries.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Data preparation (mirrors train_xgb_log.py)                                 #
# --------------------------------------------------------------------------- #

def load_and_prepare() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "features.csv")
    before = len(df)
    df = df[df["price"] <= PRICE_CAP].copy()
    print(f"Loaded {before:,} listings → capped at ${PRICE_CAP}: "
          f"{before - len(df):,} removed, {len(df):,} remaining")

    X, y = prepare_features(df)
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]
    print(f"Features: {X.shape[1]}  |  "
          f"log(price) skew: {y.skew():.2f}  (raw: {df['price'].skew():.2f})")
    return X, y


# --------------------------------------------------------------------------- #
# Optuna objective                                                             #
# --------------------------------------------------------------------------- #

def make_objective(X_train: pd.DataFrame, y_train: pd.Series) -> callable:
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 10.0),
            "n_jobs":            -1,
            "random_state":      RANDOM_STATE,
            "verbosity":         0,
        }
        model = XGBRegressor(**params)

        # CV score in log space; convert fold predictions back to dollar MAE
        fold_maes = []
        for train_idx, val_idx in kf.split(X_train):
            Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
            ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(Xtr, ytr, verbose=False)
            y_pred = np.clip(model.predict(Xval), *_LOG_CLIP)
            fold_maes.append(mean_absolute_error(np.exp(yval), np.exp(y_pred)))

        return float(np.mean(fold_maes))

    return objective


# --------------------------------------------------------------------------- #
# Metrics helpers                                                              #
# --------------------------------------------------------------------------- #

def dollar_metrics(y_true_log: np.ndarray,
                   y_pred_log: np.ndarray) -> tuple[float, float, float]:
    actual    = np.exp(y_true_log)
    predicted = np.exp(np.clip(y_pred_log, *_LOG_CLIP))
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    return mae, rmse, r2


def bucket_stats(y_true_log: np.ndarray,
                 y_pred_log: np.ndarray) -> pd.DataFrame:
    actual    = np.exp(y_true_log)
    predicted = np.exp(np.clip(y_pred_log, *_LOG_CLIP))
    df = pd.DataFrame({
        "actual":    actual,
        "abs_error": np.abs(predicted - actual),
        "pct_error": np.abs(predicted - actual) / actual * 100,
    })
    df["bucket"] = pd.cut(df["actual"], bins=BUCKET_BINS,
                          labels=BUCKET_LABELS, right=False)
    return (df.groupby("bucket", observed=True)
              .agg(n=("abs_error", "count"),
                   mae=("abs_error", "mean"),
                   mape=("pct_error", "mean"))
              .reset_index())


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main(n_trials: int = 30) -> None:
    t_start = time.time()

    # ------------------------------------------------------------------ #
    # Load + split                                                         #
    # ------------------------------------------------------------------ #
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # ------------------------------------------------------------------ #
    # Optuna study                                                         #
    # ------------------------------------------------------------------ #
    print(f"Starting Optuna study — {n_trials} trials, {N_CV_FOLDS}-fold CV, "
          f"metric = dollar MAE\n")
    print(f"  {'Trial':>5}  {'MAE ($)':>9}  {'n_est':>6}  "
          f"{'depth':>5}  {'lr':>7}  {'sub':>5}  {'col':>5}  "
          f"{'mcw':>4}  {'α':>6}  {'λ':>6}  {'Best':>9}")
    print(f"  {'-'*85}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    objective = make_objective(X_train, y_train)
    best_so_far = np.inf

    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        nonlocal best_so_far
        mae = trial.value
        p   = trial.params
        if mae < best_so_far:
            best_so_far = mae
            marker = " ◄ best"
        else:
            marker = ""
        print(
            f"  {trial.number+1:>5}  {mae:>9.2f}  "
            f"{p['n_estimators']:>6}  {p['max_depth']:>5}  "
            f"{p['learning_rate']:>7.4f}  {p['subsample']:>5.2f}  "
            f"{p['colsample_bytree']:>5.2f}  {p['min_child_weight']:>4}  "
            f"{p['reg_alpha']:>6.2f}  {p['reg_lambda']:>6.2f}  "
            f"{best_so_far:>9.2f}{marker}"
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    # ------------------------------------------------------------------ #
    # Best params                                                          #
    # ------------------------------------------------------------------ #
    best_params = study.best_params
    best_cv_mae = study.best_value

    print(f"\n{'='*60}")
    print(f"  Best CV MAE : ${best_cv_mae:.2f}")
    print(f"  Best params :")
    for k, v in best_params.items():
        print(f"    {k:<22} {v}")

    params_path = MODELS_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump({"best_cv_mae": best_cv_mae, "params": best_params}, f, indent=2)
    print(f"\n  Saved → {params_path}")

    # ------------------------------------------------------------------ #
    # Final model on full training set                                     #
    # ------------------------------------------------------------------ #
    print(f"\nTraining final model on full training set ...")
    final_model = XGBRegressor(
        **best_params,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    final_model.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    y_pred = final_model.predict(X_test)
    mae, rmse, r2 = dollar_metrics(y_test.values, y_pred)
    bkt = bucket_stats(y_test.values, y_pred)

    prev_mape = {"$0–100": 22.1, "$100–200": 18.2, "$200–500": 19.8}

    print(f"\n{'='*58}")
    print(f"  Test Results — Tuned vs Baseline")
    print(f"{'='*58}")
    print(f"  {'Metric':<14}  {'Baseline ($500 cap)':>19}  {'Tuned':>10}")
    print(f"  {'-'*48}")
    print(f"  {'MAE':<14}  {'$31.64':>19}  ${mae:>9.2f}")
    print(f"  {'RMSE':<14}  {'$47.58':>19}  ${rmse:>9.2f}")
    print(f"  {'R²':<14}  {'0.6828':>19}  {r2:>10.4f}")
    print(f"{'='*58}")

    print(f"\n  MAE by Price Bucket:")
    print(f"  {'Bucket':<12} {'N':>5}  {'MAE':>9}  {'MAPE':>7}  {'Prev MAPE':>10}  {'Δ':>7}")
    print(f"  {'-'*55}")
    for row in bkt.itertuples():
        bkt_str = str(row.bucket)
        delta = row.mape - prev_mape.get(bkt_str, row.mape)
        print(f"  {bkt_str:<12} {row.n:>5,}  ${row.mae:>8.2f}  "
              f"{row.mape:>6.1f}%  {prev_mape.get(bkt_str, 0):>9.1f}%  "
              f"{delta:>+6.1f}pp")

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    model_path = MODELS_DIR / "tuned_model.pkl"
    joblib.dump({
        "model":        final_model,
        "feature_cols": list(X.columns),
        "price_cap":    PRICE_CAP,
        "best_params":  best_params,
        "best_cv_mae":  best_cv_mae,
    }, model_path)
    print(f"\n  Tuned model saved → {model_path}")
    print(f"  Total elapsed : {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    args = parser.parse_args()
    main(n_trials=args.trials)
