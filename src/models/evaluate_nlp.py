"""
Thorough evaluation of the NLP XGBoost model (models/xgboost_nlp.pkl).

Produces plots saved to models/plots/nlp/:
  1. residuals_vs_predicted.png  — signed residuals + LOWESS trend
  2. actual_vs_predicted.png     — scatter with ±20% band
  3. learning_curve.png          — train/val MAE vs training set size
  4. shap_stability.png          — heatmap of top-10 SHAP ranks across seeds

Also prints:
  - 10-fold CV with per-fold scores and variance
  - Learning curve numbers
  - 10 best and 10 worst predictions with key features
  - Feature importance stability table across 5 seeds

Usage:
    python src/models/evaluate_nlp.py
"""

from __future__ import annotations

import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baseline import prepare_features

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PLOTS_DIR  = MODELS_DIR / "plots" / "nlp"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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

DISPLAY_FEATURES = [
    "accommodates", "bedrooms", "bathrooms", "amenities_count",
    "distance_to_downtown", "neighbourhood_avg_price",
    "days_since_last_review", "review_scores_rating",
    "has_view", "is_luxury", "amenity_has_hot_tub", "is_newly_renovated",
]

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _mae_d(yt, yp):
    return mean_absolute_error(np.exp(yt), np.exp(np.clip(yp, *_LOG_CLIP)))

def _rmse_d(yt, yp):
    return np.sqrt(mean_squared_error(np.exp(yt), np.exp(np.clip(yp, *_LOG_CLIP))))

CV_SCORING = {
    "mae":  make_scorer(_mae_d,  greater_is_better=False),
    "rmse": make_scorer(_rmse_d, greater_is_better=False),
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Reload features.csv, apply $500 cap, prepare features, split."""
    df_raw = pd.read_csv(DATA_DIR / "features.csv")
    df     = df_raw[df_raw["price"] <= PRICE_CAP].copy()

    X, y = prepare_features(df)
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
    )
    return df, X_train, X_test, y_train, y_test


def build_results(model, X_test: pd.DataFrame, y_test: pd.Series,
                  df_orig: pd.DataFrame) -> pd.DataFrame:
    """Combine predictions with original feature values for inspection."""
    y_pred_log = model.predict(X_test)
    actual     = np.exp(y_test.values)
    predicted  = np.exp(np.clip(y_pred_log, *_LOG_CLIP))
    error      = predicted - actual
    abs_error  = np.abs(error)
    pct_error  = abs_error / actual * 100

    results = X_test.copy()
    results["actual"]    = actual
    results["predicted"] = predicted
    results["error"]     = error
    results["abs_error"] = abs_error
    results["pct_error"] = pct_error
    results["bucket"]    = pd.cut(actual, bins=BUCKET_BINS,
                                  labels=BUCKET_LABELS, right=False)
    return results


# --------------------------------------------------------------------------- #
# 1. Residual analysis                                                         #
# --------------------------------------------------------------------------- #

def analyse_residuals(results: pd.DataFrame) -> None:
    print(f"\n{'='*65}")
    print("  1. RESIDUAL ANALYSIS")
    print(f"{'='*65}")

    err = results["error"]
    abs_err = results["abs_error"]

    print(f"\n  Overall stats (predicted − actual):")
    print(f"    Mean error (bias)   : ${err.mean():+.2f}  "
          f"({'overpredicts' if err.mean() > 0 else 'underpredicts'} on average)")
    print(f"    Median error        : ${err.median():+.2f}")
    print(f"    Std of errors       : ${err.std():.2f}")
    print(f"    Residual skew       : {stats.skew(err):.3f}")
    print(f"    Residual kurtosis   : {stats.kurtosis(err):.3f}")

    within_10 = (results["pct_error"] <= 10).mean() * 100
    within_20 = (results["pct_error"] <= 20).mean() * 100
    within_50 = (results["pct_error"] <= 50).mean() * 100
    print(f"\n  Prediction accuracy:")
    print(f"    Within ±10%  : {within_10:.1f}% of listings")
    print(f"    Within ±20%  : {within_20:.1f}% of listings")
    print(f"    Within ±50%  : {within_50:.1f}% of listings")

    corr, pval = stats.spearmanr(results["actual"], abs_err)
    print(f"\n  Heteroscedasticity check:")
    print(f"    Spearman(actual, |error|) = {corr:.3f}  (p={pval:.2e})")
    if corr > 0.3 and pval < 0.05:
        print("    → Errors grow with price (heteroscedasticity present)")
    else:
        print("    → No strong price–error relationship")

    # -- plots ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1a. Residuals vs predicted
    ax = axes[0]
    cap = np.percentile(results["predicted"], 99)
    view = results[results["predicted"] <= cap]
    ax.scatter(view["predicted"], view["error"],
               alpha=0.2, s=10, color="steelblue")
    ax.axhline(0, color="red", lw=1.5, linestyle="--")

    from statsmodels.nonparametric.smoothers_lowess import lowess
    smooth = lowess(view["error"].values, view["predicted"].values, frac=0.3)
    ax.plot(smooth[:, 0], smooth[:, 1], color="orange", lw=2, label="LOWESS")

    ax.set_xlabel("Predicted price ($)")
    ax.set_ylabel("Error ($)  [predicted − actual]")
    ax.set_title("Residuals vs Predicted\n(flat orange = random errors)")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # 1b. Actual vs predicted
    ax2 = axes[1]
    cap2 = np.percentile(results["actual"], 99)
    view2 = results[results["actual"] <= cap2]
    ax2.scatter(view2["actual"], view2["predicted"],
                alpha=0.25, s=12, color="steelblue", label="listings")
    lim = max(view2["actual"].max(), view2["predicted"].max()) * 1.05
    ax2.plot([0, lim], [0, lim], "r--", lw=1.5, label="perfect")
    ax2.fill_between([0, lim], [0, lim*0.8], [0, lim*1.2],
                     alpha=0.08, color="green", label="±20% band")
    ax2.set_xlabel("Actual price ($)")
    ax2.set_ylabel("Predicted price ($)")
    ax2.set_title(f"Actual vs Predicted\nMAE=${results['abs_error'].mean():.2f}  "
                  f"| {within_20:.1f}% within ±20%")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # 1c. Error distribution
    ax3 = axes[2]
    cap3 = np.percentile(abs_err, 99)
    clipped = err.clip(-cap3, cap3)
    ax3.hist(clipped, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax3.axvline(0,          color="red",    lw=1.5, linestyle="--", label="zero")
    ax3.axvline(err.mean(), color="orange", lw=1.5, label=f"mean={err.mean():+.1f}")
    ax3.axvline(err.median(),color="green", lw=1.5, label=f"median={err.median():+.1f}")
    ax3.set_xlabel("Prediction error ($)")
    ax3.set_ylabel("Count")
    ax3.set_title("Residual Distribution")
    ax3.legend()
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    p = PLOTS_DIR / "residuals_vs_predicted.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {p}")


# --------------------------------------------------------------------------- #
# 2. 10-fold CV deep dive                                                      #
# --------------------------------------------------------------------------- #

def cv_deep_dive(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    print(f"\n{'='*65}")
    print("  2. 10-FOLD CROSS-VALIDATION DEEP DIVE")
    print(f"{'='*65}")

    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
    )

    fold_maes, fold_rmses = [], []
    print(f"\n  {'Fold':>5}  {'N_train':>8}  {'N_val':>7}  {'MAE ($)':>9}  {'RMSE ($)':>10}")
    print(f"  {'-'*48}")

    for fold_num, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        Xtr  = X_train.iloc[tr_idx]
        Xval = X_train.iloc[val_idx]
        ytr  = y_train.iloc[tr_idx]
        yval = y_train.iloc[val_idx]

        model.fit(Xtr, ytr, verbose=False)
        yp = model.predict(Xval)
        mae  = _mae_d(yval.values, yp)
        rmse = _rmse_d(yval.values, yp)
        fold_maes.append(mae)
        fold_rmses.append(rmse)
        print(f"  {fold_num:>5}  {len(tr_idx):>8,}  {len(val_idx):>7,}  "
              f"{mae:>9.2f}  {rmse:>10.2f}")

    mae_arr  = np.array(fold_maes)
    rmse_arr = np.array(fold_rmses)
    print(f"  {'-'*48}")
    print(f"  {'Mean':>5}  {'':>8}  {'':>7}  {mae_arr.mean():>9.2f}  {rmse_arr.mean():>10.2f}")
    print(f"  {'Std':>5}  {'':>8}  {'':>7}  {mae_arr.std():>9.2f}  {rmse_arr.std():>10.2f}")
    print(f"  {'Min':>5}  {'':>8}  {'':>7}  {mae_arr.min():>9.2f}  {rmse_arr.min():>10.2f}")
    print(f"  {'Max':>5}  {'':>8}  {'':>7}  {mae_arr.max():>9.2f}  {rmse_arr.max():>10.2f}")

    spread = mae_arr.max() - mae_arr.min()
    cv_of  = mae_arr.std() / mae_arr.mean() * 100   # coefficient of variation
    print(f"\n  Fold spread (max−min) : ${spread:.2f}")
    print(f"  Coefficient of variation: {cv_of:.1f}%  "
          f"({'low — stable' if cv_of < 5 else 'moderate' if cv_of < 10 else 'high — unstable'})")

    worst_fold = int(np.argmax(mae_arr)) + 1
    best_fold  = int(np.argmin(mae_arr)) + 1
    print(f"  Best fold  : #{best_fold}  (MAE=${mae_arr.min():.2f})")
    print(f"  Worst fold : #{worst_fold}  (MAE=${mae_arr.max():.2f})")


# --------------------------------------------------------------------------- #
# 3. Learning curve                                                            #
# --------------------------------------------------------------------------- #

def learning_curve(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame,   y_val: pd.Series) -> None:
    print(f"\n{'='*65}")
    print("  3. LEARNING CURVE")
    print(f"{'='*65}")

    fractions = [0.20, 0.40, 0.60, 0.80, 1.00]
    train_maes, val_maes, sizes = [], [], []

    model_cfg = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
    )

    print(f"\n  {'Frac':>6}  {'N_train':>8}  {'Train MAE':>10}  {'Val MAE':>10}  {'Gap':>8}")
    print(f"  {'-'*48}")

    for frac in fractions:
        n = max(1, int(len(X_train) * frac))
        Xtr_sub = X_train.iloc[:n]
        ytr_sub = y_train.iloc[:n]

        m = XGBRegressor(**model_cfg)
        m.fit(Xtr_sub, ytr_sub, verbose=False)

        tr_mae  = _mae_d(ytr_sub.values, m.predict(Xtr_sub))
        val_mae = _mae_d(y_val.values,   m.predict(X_val))
        gap     = val_mae - tr_mae

        train_maes.append(tr_mae)
        val_maes.append(val_mae)
        sizes.append(n)

        print(f"  {frac:>5.0%}  {n:>8,}  ${tr_mae:>9.2f}  ${val_mae:>9.2f}  ${gap:>7.2f}")

    # Interpretation
    final_gap = val_maes[-1] - train_maes[-1]
    delta_val = val_maes[0] - val_maes[-1]
    print(f"\n  Train↔Val gap at 100%  : ${final_gap:.2f}  "
          f"({'slight overfitting' if final_gap > 5 else 'healthy — low overfit'})")
    print(f"  Val MAE improvement 20→100%: ${delta_val:.2f}  "
          f"({'more data still helps' if delta_val > 3 else 'plateau — data saturation'})")

    # -- plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, train_maes, "o-", color="steelblue", lw=2, label="Train MAE")
    ax.plot(sizes, val_maes,   "s-", color="tomato",    lw=2, label="Val MAE")
    ax.fill_between(sizes, train_maes, val_maes, alpha=0.12, color="gray")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("MAE ($)")
    ax.set_title("Learning Curve — XGBoost NLP Model")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    p = PLOTS_DIR / "learning_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {p}")


# --------------------------------------------------------------------------- #
# 4. Best and worst predictions                                                #
# --------------------------------------------------------------------------- #

def best_worst_predictions(results: pd.DataFrame) -> None:
    print(f"\n{'='*65}")
    print("  4. PREDICTION CONFIDENCE CHECK")
    print(f"{'='*65}")

    display = ["actual", "predicted", "abs_error", "pct_error", "bucket"] + [
        c for c in DISPLAY_FEATURES if c in results.columns
    ]

    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        out = df[display].copy()
        out["actual"]    = out["actual"].map("${:,.0f}".format)
        out["predicted"] = out["predicted"].map("${:,.0f}".format)
        out["abs_error"] = out["abs_error"].map("${:,.0f}".format)
        out["pct_error"] = out["pct_error"].map("{:.1f}%".format)
        return out

    worst = results.nlargest(10, "abs_error")
    best  = results.nsmallest(10, "abs_error")

    print(f"\n  ── Top 10 WORST predictions ──────────────────────────────")
    print(_fmt(worst).to_string(index=False))

    # Patterns in worst predictions
    print(f"\n  Patterns in worst 10:")
    print(f"    Actual price range : ${worst['actual'].min():.0f} – ${worst['actual'].max():.0f}")
    print(f"    Bucket breakdown   : {worst['bucket'].value_counts().to_dict()}")
    if "has_view" in worst.columns:
        print(f"    has_view=1         : {worst['has_view'].sum()}/10")
    if "is_luxury" in worst.columns:
        print(f"    is_luxury=1        : {worst['is_luxury'].sum()}/10")
    if "days_since_last_review" in worst.columns:
        avg_dslr = worst["days_since_last_review"].mean()
        print(f"    Avg days_since_review: {avg_dslr:.0f}  "
              f"({'many unreviewed — high uncertainty' if avg_dslr > 500 else 'recently reviewed'})")
    if "accommodates" in worst.columns:
        print(f"    Avg accommodates   : {worst['accommodates'].mean():.1f}")

    print(f"\n  ── Top 10 BEST predictions ───────────────────────────────")
    print(_fmt(best).to_string(index=False))

    print(f"\n  Patterns in best 10:")
    print(f"    Actual price range : ${best['actual'].min():.0f} – ${best['actual'].max():.0f}")
    print(f"    Bucket breakdown   : {best['bucket'].value_counts().to_dict()}")
    if "days_since_last_review" in best.columns:
        avg_dslr = best["days_since_last_review"].mean()
        print(f"    Avg days_since_review: {avg_dslr:.0f}")
    if "accommodates" in best.columns:
        print(f"    Avg accommodates   : {best['accommodates'].mean():.1f}")


# --------------------------------------------------------------------------- #
# 5. Feature importance stability                                              #
# --------------------------------------------------------------------------- #

def shap_stability(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series) -> None:
    print(f"\n{'='*65}")
    print("  5. FEATURE IMPORTANCE STABILITY (5 random seeds)")
    print(f"{'='*65}")

    seeds = [42, 7, 123, 2024, 999]
    all_mean_abs: list[pd.Series] = []

    model_cfg = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=-1, verbosity=0,
    )

    for seed in seeds:
        m = XGBRegressor(**model_cfg, random_state=seed)
        m.fit(X_train, y_train, verbose=False)
        bg = shap.sample(X_train, 200, random_state=seed)
        exp = shap.TreeExplainer(m, bg)
        sv  = exp.shap_values(X_test)
        mean_abs = pd.DataFrame(sv, columns=X_test.columns).abs().mean()
        all_mean_abs.append(mean_abs)
        print(f"  Seed {seed}: done")

    # Build rank matrix (rank 1 = highest importance)
    rank_df = pd.concat(all_mean_abs, axis=1)
    rank_df.columns = [f"seed_{s}" for s in seeds]

    # Get the union top-10 features across all seeds
    top_per_seed = [ma.nlargest(10).index.tolist() for ma in all_mean_abs]
    top_union = list(dict.fromkeys(f for lst in top_per_seed for f in lst))[:15]

    # Compute rank for each seed
    rank_matrix = pd.DataFrame(index=top_union)
    for seed, ma in zip(seeds, all_mean_abs):
        ranked = ma.rank(ascending=False).astype(int)
        rank_matrix[f"seed_{seed}"] = ranked.reindex(top_union)

    rank_matrix["mean_rank"] = rank_matrix.mean(axis=1)
    rank_matrix["rank_std"]  = rank_matrix[[f"seed_{s}" for s in seeds]].std(axis=1)
    rank_matrix = rank_matrix.sort_values("mean_rank")

    print(f"\n  Feature rank stability (rank 1 = most important):")
    print(f"\n  {'Feature':<40} " +
          "  ".join(f"{'s'+str(s):>7}" for s in seeds) +
          f"  {'Mean':>6}  {'Std':>5}  {'Stable?':>8}  NLP?")
    print(f"  {'-'*100}")

    nlp_set = set(NLP_FEATURES)
    for feat, row in rank_matrix.iterrows():
        rank_vals = [row[f"seed_{s}"] for s in seeds]
        stable    = "yes" if row["rank_std"] < 3 else "no"
        nlp_mark  = " ◄" if feat in nlp_set else ""
        print(f"  {str(feat):<40} " +
              "  ".join(f"{int(v):>7}" for v in rank_vals) +
              f"  {row['mean_rank']:>6.1f}  {row['rank_std']:>5.1f}"
              f"  {stable:>8}  {nlp_mark}")

    stable_count   = (rank_matrix["rank_std"] < 3).sum()
    unstable_count = (rank_matrix["rank_std"] >= 3).sum()
    print(f"\n  Stable features (rank std < 3) : {stable_count}/{len(rank_matrix)}")
    print(f"  Unstable features (rank std ≥ 3): {unstable_count}/{len(rank_matrix)}")

    nlp_in_top = [f for f in top_union if f in nlp_set]
    if nlp_in_top:
        print(f"\n  NLP features in top-15 across seeds: {nlp_in_top}")
        for feat in nlp_in_top:
            row = rank_matrix.loc[feat]
            print(f"    {feat:<35} mean rank={row['mean_rank']:.1f}  std={row['rank_std']:.1f}")
    else:
        print(f"\n  No NLP features in top-15 across seeds.")

    # -- heatmap --------------------------------------------------------------
    heat_data = rank_matrix[[f"seed_{s}" for s in seeds]].astype(float)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(heat_data, annot=True, fmt=".0f", cmap="YlOrRd_r",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Importance rank (lower = better)"})
    ax.set_title("SHAP Feature Importance Stability\n(rank 1 = most important)")
    ax.set_xlabel("Random seed")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    plt.tight_layout()
    p = PLOTS_DIR / "shap_stability.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {p}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()
    print("Loading model and rebuilding data split ...")

    artifact = joblib.load(MODELS_DIR / "xgboost_nlp.pkl")
    pipeline  = artifact["pipeline"]
    model     = pipeline.named_steps["model"]

    df, X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {X_train.shape[1]}")

    results = build_results(model, X_test, y_test, df)
    overall_mae = results["abs_error"].mean()
    print(f"  Overall test MAE: ${overall_mae:.2f}")

    # ------------------------------------------------------------------ #
    analyse_residuals(results)

    cv_deep_dive(X_train, y_train)

    # Use last 20% of X_train as a fixed held-out validation set for the curve
    split_idx = int(len(X_train) * 0.8)
    X_lc_train = X_train.iloc[:split_idx]
    y_lc_train = y_train.iloc[:split_idx]
    X_lc_val   = X_train.iloc[split_idx:]
    y_lc_val   = y_train.iloc[split_idx:]
    learning_curve(X_lc_train, y_lc_train, X_lc_val, y_lc_val)

    best_worst_predictions(results)

    shap_stability(X_train, X_test, y_train)

    print(f"\n{'='*65}")
    print(f"  All plots saved to: {PLOTS_DIR}/")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
