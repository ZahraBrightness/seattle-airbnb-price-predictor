"""
MLflow experiment tracker for all Airbnb price-prediction model runs.

Tracks three runs under the "airbnb-price-prediction" experiment:
  Run 1 — LinearRegression baseline      → models/baseline.pkl
  Run 2 — Tuned XGBoost (Optuna params)  → models/tuned_model.pkl
  Run 3 — Full XGBoost (NLP+cal+sent)    → models/production_model.pkl
                                            tagged production=true

Tracking store: <project_root>/mlflow/  (local SQLite via MLflow)

Usage:
    python src/models/run_training.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baseline import prepare_features

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MLFLOW_DIR = Path(__file__).resolve().parents[2] / "mlflow"

PRICE_CAP    = 500
RANDOM_STATE = 42
_LOG_CLIP    = (np.log(1), np.log(PRICE_CAP * 2))
EXPERIMENT   = "airbnb-price-prediction"


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
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


def dollar_metrics(y_true_log: np.ndarray,
                   y_pred_log: np.ndarray) -> dict[str, float]:
    return {
        "mae":  round(_mae_d(y_true_log,  y_pred_log), 4),
        "rmse": round(_rmse_d(y_true_log, y_pred_log), 4),
        "r2":   round(_r2_d(y_true_log,   y_pred_log), 6),
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load features.csv, apply $500 price cap, call prepare_features,
    and return the 80/20 train/test split.
    """
    df_raw = pd.read_csv(DATA_DIR / "features.csv")
    df     = df_raw[df_raw["price"] <= PRICE_CAP].reset_index(drop=True)

    X, y = prepare_features(df)
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def _align_to_artifact(X: pd.DataFrame, trained_cols: list[str]) -> pd.DataFrame:
    """Add any missing OHE columns (as 0) and reorder to match trained model."""
    for col in trained_cols:
        if col not in X.columns:
            X[col] = 0
    return X[[c for c in trained_cols if c in X.columns]]


# --------------------------------------------------------------------------- #
# Run 1 — LinearRegression baseline                                           #
# --------------------------------------------------------------------------- #

def run_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series,   y_test:  pd.Series) -> str:
    print("\n" + "─" * 55)
    print("  Run 1: LinearRegression baseline")
    print("─" * 55)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)

    train_m = dollar_metrics(y_train.values, pipeline.predict(X_train))
    test_m  = dollar_metrics(y_test.values,  pipeline.predict(X_test))

    artifact_path = MODELS_DIR / "baseline.pkl"
    joblib.dump({"pipeline": pipeline, "feature_cols": list(X_train.columns)},
                artifact_path)

    with mlflow.start_run(run_name="LinearRegression") as run:
        mlflow.log_params({"model_name": "LinearRegression"})
        mlflow.log_metrics({
            "train_mae":  train_m["mae"],  "test_mae":  test_m["mae"],
            "train_rmse": train_m["rmse"], "test_rmse": test_m["rmse"],
            "train_r2":   train_m["r2"],   "test_r2":  test_m["r2"],
        })
        mlflow.log_artifact(str(artifact_path))
        run_id = run.info.run_id

    print(f"  train_mae=${train_m['mae']:.2f}  test_mae=${test_m['mae']:.2f}  "
          f"test_r2={test_m['r2']:.4f}")
    print(f"  Artifact → {artifact_path.name}")
    print(f"  Run ID   : {run_id[:8]}...")
    return run_id


# --------------------------------------------------------------------------- #
# Run 2 — Tuned XGBoost                                                       #
# --------------------------------------------------------------------------- #

def run_tuned_xgb(X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: pd.Series,    y_test:  pd.Series) -> str:
    print("\n" + "─" * 55)
    print("  Run 2: XGBoost_tuned  (Optuna best params)")
    print("─" * 55)

    params_file = MODELS_DIR / "best_params.json"
    with open(params_file) as f:
        saved = json.load(f)
    best_params  = saved["params"]
    best_cv_mae  = saved["best_cv_mae"]

    model = XGBRegressor(
        **best_params,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    pipeline = Pipeline([("model", model)])

    print(f"  Running 5-fold CV ...")
    cv = cross_validate(pipeline, X_train, y_train, cv=5, scoring=CV_SCORING)
    pipeline.fit(X_train, y_train)

    train_m = dollar_metrics(y_train.values, pipeline.predict(X_train))
    test_m  = dollar_metrics(y_test.values,  pipeline.predict(X_test))
    cv_mae_mean = float(-cv["test_mae"].mean())
    cv_mae_std  = float(cv["test_mae"].std())

    artifact_path = MODELS_DIR / "tuned_model.pkl"
    joblib.dump({
        "model":        pipeline.named_steps["model"],
        "feature_cols": list(X_train.columns),
        "price_cap":    PRICE_CAP,
        "best_params":  best_params,
        "best_cv_mae":  best_cv_mae,
    }, artifact_path)

    log_params = {"model_name": "XGBoost_tuned"}
    log_params.update({k: round(v, 6) if isinstance(v, float) else v
                       for k, v in best_params.items()})

    with mlflow.start_run(run_name="XGBoost_tuned") as run:
        mlflow.log_params(log_params)
        mlflow.log_metrics({
            "train_mae":   train_m["mae"],   "test_mae":   test_m["mae"],
            "train_rmse":  train_m["rmse"],  "test_rmse":  test_m["rmse"],
            "train_r2":    train_m["r2"],    "test_r2":    test_m["r2"],
            "cv_mae_mean": round(cv_mae_mean, 4),
            "cv_mae_std":  round(cv_mae_std,  4),
        })
        mlflow.log_artifact(str(artifact_path))
        run_id = run.info.run_id

    print(f"  train_mae=${train_m['mae']:.2f}  test_mae=${test_m['mae']:.2f}  "
          f"cv_mae=${cv_mae_mean:.2f}±{cv_mae_std:.2f}")
    print(f"  Artifact → {artifact_path.name}")
    print(f"  Run ID   : {run_id[:8]}...")
    return run_id


# --------------------------------------------------------------------------- #
# Run 3 — Full XGBoost (production)                                           #
# --------------------------------------------------------------------------- #

def run_full_xgb(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series,    y_test:  pd.Series) -> str:
    print("\n" + "─" * 55)
    print("  Run 3: XGBoost_full  (NLP + calendar + sentiment)")
    print("─" * 55)

    artifact = joblib.load(MODELS_DIR / "xgboost_full.pkl")
    pipeline      = artifact["pipeline"]
    trained_cols  = artifact["feature_cols"]
    cal_features  = artifact.get("calendar_features",  [])
    sent_features = artifact.get("sentiment_features", [])

    # Align columns: the split used here matches training (same seed + data)
    X_tr = _align_to_artifact(X_train.copy(), trained_cols)
    X_te = _align_to_artifact(X_test.copy(),  trained_cols)

    train_m = dollar_metrics(y_train.values, pipeline.predict(X_tr))
    test_m  = dollar_metrics(y_test.values,  pipeline.predict(X_te))

    # Save as production model
    prod_path = MODELS_DIR / "production_model.pkl"
    joblib.dump(artifact, prod_path)

    log_params = {
        "model_name":         "XGBoost_full",
        "features":           len(trained_cols),
        "price_cap":          PRICE_CAP,
        "log_transform":      True,
        "nlp_features":       19,
        "calendar_features":  len(cal_features),
        "sentiment_features": len(sent_features),
    }

    with mlflow.start_run(run_name="XGBoost_full") as run:
        mlflow.log_params(log_params)
        mlflow.log_metrics({
            "train_mae":  train_m["mae"],  "test_mae":  test_m["mae"],
            "train_rmse": train_m["rmse"], "test_rmse": test_m["rmse"],
            "train_r2":   train_m["r2"],   "test_r2":  test_m["r2"],
        })
        mlflow.log_artifact(str(prod_path))
        mlflow.set_tag("production", "true")
        run_id = run.info.run_id

    print(f"  train_mae=${train_m['mae']:.2f}  test_mae=${test_m['mae']:.2f}  "
          f"test_r2={test_m['r2']:.4f}")
    print(f"  Artifact → {prod_path.name}")
    print(f"  Tagged   : production=true")
    print(f"  Run ID   : {run_id[:8]}...")
    return run_id


# --------------------------------------------------------------------------- #
# Summary                                                                      #
# --------------------------------------------------------------------------- #

def print_summary() -> None:
    print("\n" + "=" * 75)
    print("  EXPERIMENT SUMMARY  —  airbnb-price-prediction")
    print("=" * 75)

    runs_df = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        order_by=["metrics.test_mae ASC"],
    )

    if runs_df.empty:
        print("  No runs found.")
        return

    cols_to_show = {
        "tags.mlflow.runName":  "Run name",
        "metrics.test_mae":     "Test MAE",
        "metrics.test_rmse":    "Test RMSE",
        "metrics.test_r2":      "Test R²",
        "metrics.train_mae":    "Train MAE",
        "tags.production":      "Production",
    }
    present = {k: v for k, v in cols_to_show.items() if k in runs_df.columns}
    display = runs_df[list(present.keys())].copy()
    display.columns = list(present.values())

    # Format numeric columns
    for col in ["Test MAE", "Test RMSE", "Train MAE"]:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) else "—"
            )
    if "Test R²" in display.columns:
        display["Test R²"] = display["Test R²"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—"
        )
    display["Production"] = display["Production"].fillna("")

    print(display.to_string(index=False))

    # Highlight production model
    prod_runs = runs_df[runs_df.get("tags.production", pd.Series(dtype=str)) == "true"]
    if not prod_runs.empty:
        best = prod_runs.iloc[0]
        name = best.get("tags.mlflow.runName", "unknown")
        mae  = best.get("metrics.test_mae", float("nan"))
        print(f"\n  Production model : {name}  (test_mae=${mae:.2f})")

    print(f"\n  Tracking store   : sqlite:///mlflow/mlflow.db")
    print(f"  Total runs logged: {len(runs_df)}")
    print("=" * 75)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # MLflow setup                                                         #
    # ------------------------------------------------------------------ #
    MLFLOW_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    print(f"MLflow tracking URI : sqlite:///mlflow/mlflow.db")
    print(f"Experiment          : {EXPERIMENT}")

    # ------------------------------------------------------------------ #
    # Load data once — shared across all runs                             #
    # ------------------------------------------------------------------ #
    print("\nLoading and preparing data ...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  "
          f"|  Features: {X_train.shape[1]}")

    # ------------------------------------------------------------------ #
    # Runs                                                                 #
    # ------------------------------------------------------------------ #
    ids = {}
    ids["baseline"]   = run_baseline(X_train, X_test, y_train, y_test)
    ids["tuned_xgb"]  = run_tuned_xgb(X_train, X_test, y_train, y_test)
    ids["full_xgb"]   = run_full_xgb(X_train, X_test, y_train, y_test)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print_summary()
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
