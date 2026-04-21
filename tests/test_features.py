"""
Feature-pipeline tests against features.csv.

These tests verify the output of the full feature-engineering step before
anything touches the model.  They are intentionally structural — they check
shape, column presence, and value ranges rather than exact numerical values
so they remain valid across re-runs of the pipeline.
"""
import pandas as pd
import pytest

KEY_FEATURES = [
    "bedrooms",
    "accommodates",
    "bathrooms",
    "amenities_count",
    "distance_to_downtown",
]

PRICE_MIN = 15
PRICE_MAX = 500


class TestFeaturesShape:
    def test_minimum_column_count(self, features_df):
        """Feature matrix must have at least 100 columns after engineering."""
        assert features_df.shape[1] >= 100, (
            f"Expected >= 100 columns, got {features_df.shape[1]}"
        )

    def test_no_fully_null_columns(self, features_df):
        """No column should be entirely null — that signals a broken join."""
        fully_null = [
            col for col in features_df.columns
            if features_df[col].isna().all()
        ]
        assert fully_null == [], (
            f"Fully-null columns found: {fully_null}"
        )


class TestKeyFeaturesExist:
    def test_key_features_present(self, features_df):
        missing = [f for f in KEY_FEATURES if f not in features_df.columns]
        assert missing == [], (
            f"Key features missing from features.csv: {missing}"
        )


class TestFeatureValues:
    def test_distance_to_downtown_is_positive(self, features_df):
        """All distances must be > 0 (downtown itself is excluded from listings)."""
        assert (features_df["distance_to_downtown"] > 0).all(), (
            "distance_to_downtown contains non-positive values"
        )

    def test_price_floor(self, features_df):
        """
        features.csv is pre-cap — the $500 ceiling is applied at training time,
        not in the feature pipeline.  Only the floor ($15) is enforced here.
        """
        price = features_df["price"]
        below = (price < PRICE_MIN).sum()
        assert below == 0, (
            f"{below} rows have price < ${PRICE_MIN}"
        )
