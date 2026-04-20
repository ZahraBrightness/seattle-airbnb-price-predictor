"""
Prediction error analysis for the XGBoost price model.

Produces four plots saved to models/plots/:
  1. actual_vs_predicted.png  — scatter of y vs ŷ with identity line
  2. error_distribution.png   — residual histogram + Q-Q plot
  3. mae_by_bucket.png        — MAE and MAPE broken down by price range
  4. residuals_vs_predicted.png — residuals vs ŷ to surface systematic bias

Also prints:
  - MAE by price bucket ($0-100, $100-200, $200-500, $500+)
  - Top-20 highest-error listings with their key features
  - Systematic bias check (over/under-prediction by segment)

Usage:
    python src/models/error_analysis.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baseline import prepare_features

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PLOTS_DIR  = MODELS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

PRICE_BUCKETS = [0, 100, 200, 500, np.inf]
BUCKET_LABELS = ["$0–100", "$100–200", "$200–500", "$500+"]

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# --------------------------------------------------------------------------- #
# Load model + rebuild test set                                                #
# --------------------------------------------------------------------------- #

def load_predictions() -> pd.DataFrame:
    """Reload features.csv, apply the same split, and return a results frame."""
    artifact = joblib.load(MODELS_DIR / "xgboost.pkl")
    pipeline  = artifact["pipeline"]

    df = pd.read_csv(DATA_DIR / "features.csv")
    X, y = prepare_features(df)
    X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred_log = pipeline.predict(X_test)
    actual     = np.exp(y_test.values)
    predicted  = np.exp(y_pred_log)
    error      = predicted - actual          # signed: + = overpredict
    abs_error  = np.abs(error)
    pct_error  = abs_error / actual * 100    # MAPE per row

    results = X_test.copy()
    results["actual"]    = actual
    results["predicted"] = predicted
    results["error"]     = error
    results["abs_error"] = abs_error
    results["pct_error"] = pct_error
    results["bucket"]    = pd.cut(
        actual, bins=PRICE_BUCKETS, labels=BUCKET_LABELS, right=False
    )
    return results


# --------------------------------------------------------------------------- #
# Plot 1 — Actual vs predicted                                                 #
# --------------------------------------------------------------------------- #

def plot_actual_vs_predicted(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    # Clip display to 99th percentile to keep chart readable
    cap = np.percentile(df["actual"], 99)
    view = df[df["actual"] <= cap].copy()

    ax.scatter(view["actual"], view["predicted"],
               alpha=0.25, s=12, color="steelblue", label="listings")
    lim = max(view["actual"].max(), view["predicted"].max()) * 1.02
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="perfect prediction")

    # Shade ±20% band
    ax.fill_between([0, lim], [0, lim * 0.8], [0, lim * 1.2],
                    alpha=0.08, color="green", label="±20% band")

    within_20 = (df["pct_error"] <= 20).mean() * 100
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.set_title(
        f"Actual vs Predicted Price\n"
        f"MAE = ${df['abs_error'].mean():.2f}  |  "
        f"{within_20:.1f}% of listings within ±20%"
    )
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    n_clipped = len(df) - len(view)
    if n_clipped:
        ax.annotate(f"({n_clipped} listings > ${cap:,.0f} not shown)",
                    xy=(0.98, 0.04), xycoords="axes fraction",
                    ha="right", fontsize=9, color="gray")

    plt.tight_layout()
    path = PLOTS_DIR / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------- #
# Plot 2 — Error distribution (histogram + Q-Q)                               #
# --------------------------------------------------------------------------- #

def plot_error_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Clip for readability
    cap = np.percentile(df["abs_error"], 99)
    errors_clipped = df["error"].clip(-cap, cap)

    # Histogram of signed error
    ax = axes[0]
    ax.hist(errors_clipped, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", lw=1.5, linestyle="--", label="zero error")
    ax.axvline(df["error"].mean(), color="orange", lw=1.5,
               linestyle="-", label=f"mean = ${df['error'].mean():.1f}")
    ax.axvline(df["error"].median(), color="green", lw=1.5,
               linestyle="-", label=f"median = ${df['error'].median():.1f}")
    ax.set_xlabel("Prediction error ($)  [predicted − actual]")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Q-Q plot to test normality of residuals
    ax2 = axes[1]
    log_errors = np.log(df["actual"]) - np.log(df["predicted"])
    (osm, osr), (slope, intercept, r) = stats.probplot(log_errors, dist="norm")
    ax2.scatter(osm, osr, alpha=0.3, s=10, color="steelblue")
    line_x = np.array([osm[0], osm[-1]])
    ax2.plot(line_x, slope * line_x + intercept, "r--", lw=1.5)
    ax2.set_xlabel("Theoretical quantiles")
    ax2.set_ylabel("Sample quantiles (log residuals)")
    ax2.set_title(f"Q-Q Plot of Log Residuals\n(R² = {r**2:.4f}  vs normal)")

    plt.tight_layout()
    path = PLOTS_DIR / "error_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------- #
# Plot 3 — MAE / MAPE by price bucket                                         #
# --------------------------------------------------------------------------- #

def plot_mae_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    bucket_stats = (
        df.groupby("bucket", observed=True)
        .agg(
            n=("abs_error", "count"),
            mae=("abs_error", "mean"),
            mape=("pct_error", "mean"),
            median_err=("abs_error", "median"),
            bias=("error", "mean"),        # + = overpredicts, − = underpredicts
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = sns.color_palette("muted", len(bucket_stats))

    # MAE bars
    ax = axes[0]
    bars = ax.bar(bucket_stats["bucket"].astype(str), bucket_stats["mae"],
                  color=colors, edgecolor="white")
    for bar, row in zip(bars, bucket_stats.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2, f"n={row.n:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Price bucket")
    ax.set_ylabel("Mean Absolute Error ($)")
    ax.set_title("MAE by Price Bucket")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # MAPE bars
    ax2 = axes[1]
    bars2 = ax2.bar(bucket_stats["bucket"].astype(str), bucket_stats["mape"],
                    color=colors, edgecolor="white")
    for bar, row in zip(bars2, bucket_stats.itertuples()):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3, f"{row.mape:.1f}%",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("Price bucket")
    ax2.set_ylabel("Mean Absolute Percentage Error (%)")
    ax2.set_title("MAPE by Price Bucket")

    plt.tight_layout()
    path = PLOTS_DIR / "mae_by_bucket.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return bucket_stats


# --------------------------------------------------------------------------- #
# Plot 4 — Residuals vs predicted (systematic bias check)                     #
# --------------------------------------------------------------------------- #

def plot_residuals_vs_predicted(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cap = np.percentile(df["predicted"], 99)
    view = df[df["predicted"] <= cap]

    # Residuals (signed error) vs predicted price
    ax = axes[0]
    ax.scatter(view["predicted"], view["error"],
               alpha=0.2, s=10, color="steelblue")
    ax.axhline(0, color="red", lw=1.5, linestyle="--")

    # Lowess smoothed trend to reveal non-linear bias
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smooth = lowess(view["error"], view["predicted"], frac=0.3)
    ax.plot(smooth[:, 0], smooth[:, 1], color="orange", lw=2, label="LOWESS trend")

    ax.set_xlabel("Predicted price ($)")
    ax.set_ylabel("Error ($)  [predicted − actual]")
    ax.set_title("Residuals vs Predicted\n(flat orange line = random errors)")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Bias by bucket (signed mean error)
    ax2 = axes[1]
    bucket_bias = df.groupby("bucket", observed=True)["error"].mean()
    colors = ["tomato" if v > 0 else "steelblue" for v in bucket_bias]
    bars = ax2.bar(bucket_bias.index.astype(str), bucket_bias.values, color=colors)
    ax2.axhline(0, color="black", lw=1)
    for bar, val in zip(bars, bucket_bias):
        label = f"+${val:.0f}" if val > 0 else f"-${abs(val):.0f}"
        ypos = val + 1 if val >= 0 else val - 4
        ax2.text(bar.get_x() + bar.get_width() / 2, ypos,
                 label, ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("Price bucket")
    ax2.set_ylabel("Mean signed error ($)  [+ = overpredict]")
    ax2.set_title("Systematic Bias by Price Bucket\n(red = overpredicts, blue = underpredicts)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    path = PLOTS_DIR / "residuals_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------- #
# Print diagnostics                                                            #
# --------------------------------------------------------------------------- #

def print_bucket_stats(bucket_stats: pd.DataFrame) -> None:
    print(f"\n{'='*65}")
    print("  MAE by Price Bucket")
    print(f"{'='*65}")
    print(f"  {'Bucket':<12} {'N':>6}  {'MAE':>9}  {'MAPE':>7}  {'Median err':>11}  {'Bias':>9}")
    print(f"  {'-'*60}")
    for row in bucket_stats.itertuples():
        bias_str = f"+${row.bias:.0f}" if row.bias >= 0 else f"-${abs(row.bias):.0f}"
        print(f"  {str(row.bucket):<12} {row.n:>6,}  "
              f"${row.mae:>8.2f}  {row.mape:>6.1f}%  "
              f"${row.median_err:>10.2f}  {bias_str:>9}")


def print_top_errors(df: pd.DataFrame, n: int = 20) -> None:
    display_cols = [
        "actual", "predicted", "abs_error", "pct_error",
        "accommodates", "beds", "amenities_count",
        "distance_to_downtown", "neighbourhood_avg_price",
        "days_since_last_review",
    ]
    present = [c for c in display_cols if c in df.columns]

    top = df.nlargest(n, "abs_error")[present].copy()
    top["actual"]    = top["actual"].map("${:,.0f}".format)
    top["predicted"] = top["predicted"].map("${:,.0f}".format)
    top["abs_error"] = top["abs_error"].map("${:,.0f}".format)
    top["pct_error"] = top["pct_error"].map("{:.1f}%".format)

    print(f"\n{'='*65}")
    print(f"  Top {n} Highest-Error Listings")
    print(f"{'='*65}")
    print(top.to_string(index=False))


def print_bias_check(df: pd.DataFrame) -> None:
    print(f"\n{'='*65}")
    print("  Systematic Bias Check")
    print(f"{'='*65}")

    overall_bias = df["error"].mean()
    sign = "overpredicts" if overall_bias > 0 else "underpredicts"
    print(f"\n  Overall mean error : ${overall_bias:+.2f}  ({sign} on average)")
    print(f"  Overall std error  : ${df['error'].std():.2f}")

    skew = stats.skew(df["error"])
    print(f"  Residual skew      : {skew:.3f}  "
          f"({'right-skewed: occasional large over-predictions' if skew > 0.5 else 'left-skewed: occasional large under-predictions' if skew < -0.5 else 'roughly symmetric'})")

    within_10  = (df["pct_error"] <= 10).mean()  * 100
    within_20  = (df["pct_error"] <= 20).mean()  * 100
    within_50  = (df["pct_error"] <= 50).mean()  * 100
    print(f"\n  Within ±10%        : {within_10:.1f}% of listings")
    print(f"  Within ±20%        : {within_20:.1f}% of listings")
    print(f"  Within ±50%        : {within_50:.1f}% of listings")

    # Spearman correlation: is error magnitude correlated with price?
    corr, pval = stats.spearmanr(df["actual"], df["abs_error"])
    print(f"\n  Spearman(actual, |error|) = {corr:.3f}  (p={pval:.2e})")
    if corr > 0.3 and pval < 0.05:
        print("  → Significant positive correlation: model errors are larger")
        print("    for higher-priced listings (heteroscedasticity).")
    else:
        print("  → No strong systematic relationship between price and error size.")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("Loading model and rebuilding test set ...")
    df = load_predictions()
    print(f"  Test set: {len(df):,} listings  |  "
          f"Overall MAE: ${df['abs_error'].mean():.2f}  |  "
          f"MAPE: {df['pct_error'].mean():.1f}%")

    print("\nGenerating plots ...")
    plot_actual_vs_predicted(df)
    plot_error_distribution(df)
    bucket_stats = plot_mae_by_bucket(df)
    plot_residuals_vs_predicted(df)

    print_bucket_stats(bucket_stats)
    print_top_errors(df)
    print_bias_check(df)

    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
