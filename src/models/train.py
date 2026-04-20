"""
Model comparison for Airbnb price prediction.

Three models chosen to balance accuracy AND interpretability for a host
audience that needs to understand *why* a listing is priced a certain way:

  1. Ridge Regression
       Each coefficient is the exact log-price impact of a 1-unit feature
       change (after exponentiation → a percentage multiplier). A host can
       see "adding 1 amenity multiplies my price by X". Best for linear,
       additive explanations.

  2. Random Forest
       Non-linear ensemble; captures interactions (e.g. room_type × location
       premium). Feature importances show which variables the model splits on
       most. SHAP values give per-listing explanations. Robust to outliers.

  3. XGBoost
       Gradient-boosted trees — typically the most accurate model on tabular
       regression. TreeSHAP is exact (not sampled) and fast, making it the
       gold standard for explaining individual predictions to hosts.

All models predict log(price) to handle price's heavy right skew.
Metrics are reported back in original dollar scale.

Usage:
    python src/models/train.py
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Reuse preprocessing from baseline
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baseline import prepare_features  # noqa: E402

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Model definitions                                                            #
# --------------------------------------------------------------------------- #

def _build_models() -> dict[str, Pipeline]:
    """
    Return a dict of named sklearn-compatible pipelines.

    Ridge uses StandardScaler because coefficients are scale-sensitive.
    Tree models (RF, XGBoost) are scale-invariant — no scaler needed.
    """
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0)),
        ]),
        "RandomForest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("model", XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )),
        ]),
    }


# --------------------------------------------------------------------------- #
# Custom scorers (dollar scale)                                                #
# --------------------------------------------------------------------------- #
# CV operates on log(price) predictions. We convert back to dollars so the
# CV metrics are directly comparable to the baseline reported in dollar terms.
#
# Predictions are clipped to [log(1), log(200_000)] before exp() to prevent
# numerical explosion on Ridge folds that extrapolate to extreme log values.

_LOG_CLIP = (np.log(1), np.log(200_000))

def _mae_dollar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(np.exp(y_true), np.exp(np.clip(y_pred, *_LOG_CLIP)))

def _rmse_dollar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(np.exp(y_true), np.exp(np.clip(y_pred, *_LOG_CLIP))))

def _r2_dollar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(np.exp(y_true), np.exp(np.clip(y_pred, *_LOG_CLIP)))

CV_SCORING = {
    "mae":  make_scorer(_mae_dollar,  greater_is_better=False),
    "rmse": make_scorer(_rmse_dollar, greater_is_better=False),
    "r2":   make_scorer(_r2_dollar),
}


# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #

def train_and_evaluate(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = 5,
) -> dict[str, dict]:
    """
    For each model: run CV, refit on full train set, evaluate on held-out test.
    Returns a dict of results keyed by model name.
    """
    results = {}

    for name, pipeline in models.items():
        print(f"\n  Training {name} ...")
        t0 = time.time()

        # 5-fold cross-validation on training set
        cv = cross_validate(
            pipeline, X_train, y_train,
            cv=cv_folds,
            scoring=CV_SCORING,
            return_train_score=False,
            n_jobs=1,      # pipelines already use n_jobs=-1 internally
        )

        # Refit on full training set for test evaluation and SHAP
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred_test = pipeline.predict(X_test)

        results[name] = {
            "pipeline":    pipeline,
            "cv_mae_mean": -cv["test_mae"].mean(),
            "cv_mae_std":  cv["test_mae"].std(),
            "cv_rmse_mean":-cv["test_rmse"].mean(),
            "cv_rmse_std": cv["test_rmse"].std(),
            "cv_r2_mean":  cv["test_r2"].mean(),
            "test_mae":    _mae_dollar(y_test.values, y_pred_test),
            "test_rmse":   _rmse_dollar(y_test.values, y_pred_test),
            "test_r2":     _r2_dollar(y_test.values, y_pred_test),
            "train_time":  elapsed,
        }

        print(f"    CV  MAE : ${results[name]['cv_mae_mean']:.2f} "
              f"± {results[name]['cv_mae_std']:.2f}")
        print(f"    CV  RMSE: ${results[name]['cv_rmse_mean']:.2f}")
        print(f"    CV  R²  : {results[name]['cv_r2_mean']:.4f}")
        print(f"    Test MAE: ${results[name]['test_mae']:.2f}  "
              f"RMSE: ${results[name]['test_rmse']:.2f}  "
              f"R²: {results[name]['test_r2']:.4f}")
        print(f"    Time    : {elapsed:.1f}s")

    return results


# --------------------------------------------------------------------------- #
# Comparison table                                                             #
# --------------------------------------------------------------------------- #

def print_comparison_table(results: dict[str, dict], baseline: dict) -> str:
    rows = []

    # Include baseline for reference
    rows.append({
        "Model":         "LinearRegression (baseline)",
        "CV MAE ($)":    "—",
        "CV RMSE ($)":   "—",
        "CV Std":        "—",
        "Test MAE ($)":  f"{baseline['test_mae']:.2f}",
        "Test RMSE ($)": f"{baseline['test_rmse']:.2f}",
        "Test R²":       f"{baseline['test_r2']:.4f}",
        "Train Time":    "—",
    })

    for name, r in results.items():
        rows.append({
            "Model":         name,
            "CV MAE ($)":    f"{r['cv_mae_mean']:.2f} ± {r['cv_mae_std']:.2f}",
            "CV RMSE ($)":   f"{r['cv_rmse_mean']:.2f}",
            "CV Std":        f"{r['cv_mae_std']:.2f}",
            "Test MAE ($)":  f"{r['test_mae']:.2f}",
            "Test RMSE ($)": f"{r['test_rmse']:.2f}",
            "Test R²":       f"{r['test_r2']:.4f}",
            "Train Time":    f"{r['train_time']:.1f}s",
        })

    table = pd.DataFrame(rows).set_index("Model")
    sep = "=" * 105
    print(f"\n{sep}")
    print("  Model Comparison")
    print(sep)
    print(table.to_string())
    print(sep)
    return table.to_string()


# --------------------------------------------------------------------------- #
# SHAP analysis                                                                #
# --------------------------------------------------------------------------- #

def shap_analysis(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    top_n: int = 15,
) -> None:
    """
    Compute TreeSHAP (for tree models) or LinearSHAP (for Ridge) and print
    a text summary of the top features driving price up and down.
    """
    print(f"\n{'='*60}")
    print(f"  SHAP Feature Importance — {name}")
    print(f"{'='*60}")

    model = pipeline.named_steps["model"]

    # For Ridge, transform X through the scaler first
    if "scaler" in pipeline.named_steps:
        X_tr = pd.DataFrame(
            pipeline.named_steps["scaler"].transform(X_train),
            columns=X_train.columns,
        )
        X_te = pd.DataFrame(
            pipeline.named_steps["scaler"].transform(X_test),
            columns=X_test.columns,
        )
        explainer = shap.LinearExplainer(model, X_tr)
        shap_values = explainer.shap_values(X_te)
    else:
        # Tree models: use TreeExplainer on a background sample for speed
        background = shap.sample(X_train, 200, random_state=42)
        explainer = shap.TreeExplainer(model, background)
        shap_values = explainer.shap_values(X_test)

    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    mean_dir = shap_df.mean()   # positive → pushes price up, negative → down

    print(f"\n  {'Feature':<45} {'Mean |SHAP|':>12}  {'Direction':>12}  Interpretation")
    print(f"  {'-'*95}")

    for feat in mean_abs.head(top_n).index:
        abs_val = mean_abs[feat]
        direction = mean_dir[feat]
        arrow = "▲ price +" if direction > 0 else "▼ price −"
        # Convert log-scale SHAP to approximate % impact
        pct = (np.exp(abs_val) - 1) * 100
        print(f"  {feat:<45} {abs_val:>12.4f}  {arrow:>12}  (~{pct:.0f}% impact)")

    print(f"\n  Note: SHAP values are in log(price) space.")
    print(f"  A SHAP of +0.10 ≈ 10.5% price increase; −0.10 ≈ 9.5% decrease.")

    # Save SHAP values for further analysis
    shap_path = MODELS_DIR / f"{name.lower()}_shap_values.csv"
    shap_df.to_csv(shap_path, index=False)
    print(f"\n  Full SHAP values saved to: {shap_path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------ #
    # Load and prepare                                                     #
    # ------------------------------------------------------------------ #
    print("Loading data/features.csv ...")
    df = pd.read_csv(DATA_DIR / "features.csv")

    X, y = prepare_features(df)

    # XGBoost rejects feature names containing [, ], <, >, or ,
    # (produced by OHE of columns like host_verifications which store lists).
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows  "
          f"|  Features: {X.shape[1]}")

    # Baseline numbers for comparison table
    baseline = {"test_mae": 139.36, "test_rmse": 1233.05, "test_r2": 0.8772}

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
    print("\nTraining 3 models with 5-fold CV ...")
    models   = _build_models()
    results  = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # ------------------------------------------------------------------ #
    # Comparison table                                                     #
    # ------------------------------------------------------------------ #
    print_comparison_table(results, baseline)

    # ------------------------------------------------------------------ #
    # Identify best model by test MAE                                      #
    # ------------------------------------------------------------------ #
    best_name = min(results, key=lambda n: results[n]["test_mae"])
    best = results[best_name]

    print(f"\n  Best model: {best_name}")
    print(f"  Reason: lowest test MAE (${best['test_mae']:.2f}) with "
          f"R² = {best['test_r2']:.4f}")

    # ------------------------------------------------------------------ #
    # SHAP for best model                                                  #
    # ------------------------------------------------------------------ #
    shap_analysis(best_name, best["pipeline"], X_train, X_test)

    # ------------------------------------------------------------------ #
    # Save all models                                                      #
    # ------------------------------------------------------------------ #
    print(f"\nSaving models ...")
    for name, r in results.items():
        path = MODELS_DIR / f"{name.lower()}.pkl"
        joblib.dump(
            {"pipeline": r["pipeline"], "feature_cols": list(X.columns)},
            path,
        )
        print(f"  Saved {name} → {path}")

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
