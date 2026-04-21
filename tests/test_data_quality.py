"""
Data quality tests for cleaned.csv.

The `quality_gate` function is the production-facing check; individual test
functions exercise it against known-good and known-bad inputs so regressions
are caught before the data reaches the feature pipeline.
"""
import numpy as np
import pandas as pd
import pytest

REQUIRED_COLUMNS = [
    "price",
    "bedrooms",
    "accommodates",
    "bathrooms",
    "neighbourhood_cleansed",
    "room_type",
    "number_of_reviews",
    "latitude",
    "longitude",
]
MAX_NULL_RATE = 0.40
MIN_ROWS = 1_000


def quality_gate(df: pd.DataFrame) -> list[str]:
    """
    Run data-quality checks on a cleaned listings DataFrame.

    Returns a list of human-readable failure messages.  An empty list means
    the dataset passed all checks.
    """
    failures: list[str] = []

    # 1. Required columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        failures.append(f"Missing required columns: {missing}")

    # 2. Minimum row count
    if len(df) < MIN_ROWS:
        failures.append(f"Too few rows: {len(df)} < {MIN_ROWS}")

    # 3. Price column is numeric and contains no nulls
    if "price" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["price"]):
            failures.append("Column 'price' is not numeric")
        elif df["price"].isna().any():
            failures.append("Column 'price' contains null values")

    # 4. No column exceeds the null-rate threshold
    if len(df) > 0:
        null_rates = df.isnull().mean()
        bad_cols = null_rates[null_rates > MAX_NULL_RATE].index.tolist()
        if bad_cols:
            failures.append(
                f"Columns exceed {MAX_NULL_RATE:.0%} null rate: {bad_cols}"
            )

    return failures


# ── Tests against the real dataset ───────────────────────────────────────────

class TestQualityGateRealData:
    def test_passes_quality_gate(self, cleaned_df):
        """The production cleaned.csv must pass every quality check."""
        failures = quality_gate(cleaned_df)
        assert failures == [], f"Quality gate failures: {failures}"

    def test_required_columns_present(self, cleaned_df):
        missing = [c for c in REQUIRED_COLUMNS if c not in cleaned_df.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_price_is_numeric_and_non_null(self, cleaned_df):
        assert pd.api.types.is_numeric_dtype(cleaned_df["price"]), (
            "price column should be numeric"
        )
        assert cleaned_df["price"].notna().all(), (
            "price column must not contain nulls"
        )

    def test_row_count_above_minimum(self, cleaned_df):
        assert len(cleaned_df) >= MIN_ROWS, (
            f"Expected >= {MIN_ROWS} rows, got {len(cleaned_df)}"
        )


# ── Tests against deliberately broken DataFrames ─────────────────────────────

class TestQualityGateBadInputs:
    def test_catches_missing_price_column(self):
        df = pd.DataFrame({
            "bedrooms": [1, 2],
            "accommodates": [2, 4],
            "bathrooms": [1.0, 2.0],
            "neighbourhood_cleansed": ["Fremont", "Capitol Hill"],
            "room_type": ["Entire home/apt", "Private room"],
            "number_of_reviews": [10, 20],
            "latitude": [47.65, 47.62],
            "longitude": [-122.35, -122.32],
        })
        failures = quality_gate(df)
        assert any("price" in f for f in failures), (
            "Should flag missing 'price' column"
        )

    def test_catches_non_numeric_price(self):
        df = pd.DataFrame({col: [1, 2] for col in REQUIRED_COLUMNS})
        df["price"] = ["$120", "$250"]         # strings instead of floats
        # Pad rows to satisfy MIN_ROWS
        big_df = pd.concat([df] * (MIN_ROWS // 2 + 1), ignore_index=True)
        failures = quality_gate(big_df)
        assert any("not numeric" in f for f in failures), (
            "Should flag non-numeric price column"
        )

    def test_catches_too_few_rows(self):
        df = pd.DataFrame({col: [1.0] for col in REQUIRED_COLUMNS})
        df["price"] = [150.0]
        failures = quality_gate(df)
        assert any("Too few rows" in f for f in failures), (
            "Should flag datasets smaller than MIN_ROWS"
        )

    def test_catches_high_null_rate(self):
        n = MIN_ROWS + 100
        df = pd.DataFrame({col: [1.0] * n for col in REQUIRED_COLUMNS})
        df["price"] = [150.0] * n
        # Inject a column that is almost entirely null
        df["mostly_null"] = [np.nan] * n
        failures = quality_gate(df)
        assert any("null rate" in f for f in failures), (
            "Should flag columns with null rate above threshold"
        )
