"""
Model-artifact tests for production_model.pkl.

These tests verify the artifact loads correctly, has the expected interface,
and produces sensible predictions on a realistic input.  They are fast
(no training) and deterministic (same median row → same prediction).
"""
import math

import numpy as np
import pytest

PREDICTED_PRICE_MIN = 15.0
PREDICTED_PRICE_MAX = 500.0
EXPECTED_FEATURE_COUNT = 150


class TestModelLoads:
    def test_artifact_is_dict(self, model_artifact):
        assert isinstance(model_artifact, dict), (
            "production_model.pkl should deserialize to a dict"
        )

    def test_required_keys_present(self, model_artifact):
        required = {"pipeline", "feature_cols", "price_cap"}
        missing = required - model_artifact.keys()
        assert not missing, (
            f"model_artifact missing keys: {missing}"
        )

    def test_feature_count(self, model_artifact):
        n = len(model_artifact["feature_cols"])
        assert n == EXPECTED_FEATURE_COUNT, (
            f"Expected {EXPECTED_FEATURE_COUNT} feature columns, got {n}"
        )


class TestModelInterface:
    def test_pipeline_has_predict(self, model_artifact):
        pipeline = model_artifact["pipeline"]
        assert hasattr(pipeline, "predict"), (
            "pipeline object must expose a .predict() method"
        )

    def test_pipeline_predict_is_callable(self, model_artifact):
        assert callable(model_artifact["pipeline"].predict)


class TestModelPrediction:
    def test_prediction_on_median_row(self, model_artifact, model_median_row):
        """
        A prediction on the all-median input row should be a finite scalar
        in log-price space.
        """
        pipeline = model_artifact["pipeline"]
        log_pred = pipeline.predict(model_median_row)

        assert log_pred.shape == (1,), (
            f"Expected shape (1,), got {log_pred.shape}"
        )
        assert math.isfinite(float(log_pred[0])), (
            "Prediction is not a finite number"
        )

    def test_prediction_in_dollar_range(self, model_artifact, model_median_row):
        """
        After exponentiating the log-price prediction the implied dollar
        amount must fall within the plausible listing range.
        """
        pipeline = model_artifact["pipeline"]
        log_pred = pipeline.predict(model_median_row)
        dollar_pred = float(np.expm1(log_pred[0]))

        assert dollar_pred >= PREDICTED_PRICE_MIN, (
            f"Predicted price ${dollar_pred:.2f} is below ${PREDICTED_PRICE_MIN}"
        )
        assert dollar_pred <= PREDICTED_PRICE_MAX, (
            f"Predicted price ${dollar_pred:.2f} is above ${PREDICTED_PRICE_MAX}"
        )
