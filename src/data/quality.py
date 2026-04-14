"""
Data quality gate for ML pipelines.

Usage
-----
Programmatic:
    from data.quality import check_data_quality
    result = check_data_quality(df, required_schema={"id": "int64", "date": "object"})

CLI:
    python src/data/quality.py [filename.csv] [--target <col>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

NULL_CRITICAL_PCT = 50.0   # fail if any column exceeds this
NULL_WARN_PCT = 20.0       # warn if any column exceeds this
ROW_CRITICAL = 100         # fail below this
ROW_WARN = 1_000           # warn below this
CLASS_MIN_PCT = 5.0        # warn if any class < this share of target
CLASS_MIN_COUNT = 2        # fail if fewer than this many distinct classes

# Heuristic bounds applied to numeric columns whose names match keywords.
# Format: keyword -> (min_allowed, max_allowed, description)
KEYWORD_BOUNDS: list[tuple[str, float, float, str]] = [
    ("count",    0,     1e12,  "count column must be non-negative"),
    ("rate",     0,     100,   "rate column must be 0–100"),
    ("pct",      0,     100,   "percentage column must be 0–100"),
    ("percent",  0,     100,   "percentage column must be 0–100"),
    ("ratio",    0,     1,     "ratio column must be 0–1"),
    ("price",    0,     1e9,   "price column must be non-negative"),
    ("age",      0,     150,   "age column must be 0–150"),
    ("score",    0,     1e6,   "score column must be non-negative"),
    ("lat",     -90,    90,    "latitude must be -90–90"),
    ("lon",    -180,   180,    "longitude must be -180–180"),
    ("long",   -180,   180,    "longitude must be -180–180"),
]


# --------------------------------------------------------------------------- #
# Individual checks                                                            #
# --------------------------------------------------------------------------- #

def _check_schema(
    df: pd.DataFrame,
    required_schema: dict[str, str] | None,
) -> tuple[list[str], list[str]]:
    """Check 1: required columns exist and have the expected dtype."""
    failures, warnings = [], []

    if not required_schema:
        warnings.append("Schema check skipped: no required_schema provided.")
        return failures, warnings

    for col, expected_dtype in required_schema.items():
        if col not in df.columns:
            failures.append(f"Schema: required column '{col}' is missing.")
            continue
        actual = str(df[col].dtype)
        # Allow partial matches (e.g. "int" matches "int64")
        if expected_dtype and expected_dtype not in actual:
            warnings.append(
                f"Schema: column '{col}' expected dtype '{expected_dtype}', "
                f"got '{actual}'."
            )

    return failures, warnings


def _check_row_count(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 2: dataset has enough rows."""
    failures, warnings = [], []
    n = len(df)

    if n < ROW_CRITICAL:
        failures.append(
            f"Row count: only {n:,} rows — minimum required is {ROW_CRITICAL:,}."
        )
    elif n < ROW_WARN:
        warnings.append(
            f"Row count: {n:,} rows is low — consider collecting more data "
            f"(recommended ≥ {ROW_WARN:,})."
        )

    return failures, warnings


def _check_null_rates(df: pd.DataFrame) -> tuple[list[str], list[str], dict[str, Any]]:
    """Check 3: null rate per column."""
    failures, warnings = [], []
    n = len(df)
    null_counts: dict[str, int] = {}

    for col in df.columns:
        missing = int(df[col].isnull().sum())
        null_counts[col] = missing
        if missing == 0:
            continue
        pct = missing / n * 100
        if pct > NULL_CRITICAL_PCT:
            failures.append(
                f"Nulls: column '{col}' is {pct:.1f}% null "
                f"(>{NULL_CRITICAL_PCT}% threshold)."
            )
        elif pct > NULL_WARN_PCT:
            warnings.append(
                f"Nulls: column '{col}' is {pct:.1f}% null "
                f"(>{NULL_WARN_PCT}% warning threshold)."
            )

    return failures, warnings, null_counts


def _check_value_ranges(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 4: numeric columns within sensible bounds based on name heuristics."""
    failures, warnings = [], []
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        col_lower = col.lower()
        for keyword, lo, hi, desc in KEYWORD_BOUNDS:
            if keyword not in col_lower:
                continue
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < lo or col_max > hi:
                warnings.append(
                    f"Range: column '{col}' ({desc}) has values "
                    f"[{col_min}, {col_max}] outside expected [{lo}, {hi}]."
                )
            break  # only apply the first matching keyword

    # Generic outlier check: flag columns where |z-score| > 10 exists
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        std = series.std()
        if std == 0:
            warnings.append(
                f"Variance: column '{col}' has zero variance (constant value)."
            )
            continue
        z_max = ((series - series.mean()) / std).abs().max()
        if z_max > 10:
            warnings.append(
                f"Outliers: column '{col}' has extreme values "
                f"(max |z-score| = {z_max:.1f})."
            )

    return failures, warnings


def _check_target_distribution(
    df: pd.DataFrame,
    target_col: str | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Check 5: classification target has enough classes and balance."""
    failures, warnings = [], []
    distribution: dict[str, Any] = {}

    if target_col is None:
        warnings.append("Target check skipped: no target_col provided.")
        return failures, warnings, distribution

    if target_col not in df.columns:
        failures.append(f"Target: column '{target_col}' not found in DataFrame.")
        return failures, warnings, distribution

    counts = df[target_col].value_counts(dropna=False)
    n = len(df)
    distribution = {str(k): int(v) for k, v in counts.items()}

    num_classes = len(counts)
    if num_classes < CLASS_MIN_COUNT:
        failures.append(
            f"Target: '{target_col}' has only {num_classes} distinct class(es) — "
            f"need at least {CLASS_MIN_COUNT} for classification."
        )
        return failures, warnings, distribution

    for cls, cnt in counts.items():
        pct = cnt / n * 100
        if pct < CLASS_MIN_PCT:
            warnings.append(
                f"Target: class '{cls}' in '{target_col}' represents only "
                f"{pct:.1f}% of data (threshold: {CLASS_MIN_PCT}%) — "
                "consider resampling."
            )

    return failures, warnings, distribution


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def check_data_quality(
    df: pd.DataFrame,
    required_schema: dict[str, str] | None = None,
    target_col: str | None = None,
) -> dict[str, Any]:
    """
    Run 5 data-quality checks and return a structured report.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to validate.
    required_schema : dict, optional
        Mapping of {column_name: expected_dtype_substring}.
        Example: {"age": "int", "name": "object", "score": "float"}
    target_col : str, optional
        Name of the classification target column for distribution check.

    Returns
    -------
    dict with keys:
        success    – bool, True only if no critical failures
        failures   – list of critical error strings
        warnings   – list of non-critical concern strings
        statistics – dict of counts and distributions
    """
    all_failures: list[str] = []
    all_warnings: list[str] = []

    # 1. Schema
    f, w = _check_schema(df, required_schema)
    all_failures += f
    all_warnings += w

    # 2. Row count
    f, w = _check_row_count(df)
    all_failures += f
    all_warnings += w

    # 3. Null rates
    f, w, null_counts = _check_null_rates(df)
    all_failures += f
    all_warnings += w

    # 4. Value ranges
    f, w = _check_value_ranges(df)
    all_failures += f
    all_warnings += w

    # 5. Target distribution
    f, w, target_dist = _check_target_distribution(df, target_col)
    all_failures += f
    all_warnings += w

    statistics: dict[str, Any] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nulls_by_column": null_counts,
        "total_null_cells": sum(null_counts.values()),
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "target_distribution": target_dist,
    }

    return {
        "success": len(all_failures) == 0,
        "failures": all_failures,
        "warnings": all_warnings,
        "statistics": statistics,
    }


# --------------------------------------------------------------------------- #
# Pretty printer                                                               #
# --------------------------------------------------------------------------- #

def print_report(result: dict[str, Any]) -> None:
    status = "PASSED" if result["success"] else "FAILED"
    print(f"\n{'='*60}")
    print(f"  Data Quality Gate: {status}")
    print(f"{'='*60}")

    if result["failures"]:
        print(f"\n[CRITICAL] {len(result['failures'])} failure(s):")
        for msg in result["failures"]:
            print(f"  x {msg}")

    if result["warnings"]:
        print(f"\n[WARNING] {len(result['warnings'])} warning(s):")
        for msg in result["warnings"]:
            print(f"  ! {msg}")

    if not result["failures"] and not result["warnings"]:
        print("\n  All checks passed with no warnings.")

    s = result["statistics"]
    print(f"\n[STATISTICS]")
    print(f"  Rows            : {s['total_rows']:,}")
    print(f"  Columns         : {s['total_columns']}")
    print(f"  Total null cells: {s['total_null_cells']:,}")

    nulls = {c: v for c, v in s["total_nulls_by_column"].items() if v > 0}
    if nulls:
        print("  Nulls by column :")
        for col, cnt in nulls.items():
            pct = cnt / s["total_rows"] * 100
            print(f"    {col:<30} {cnt:>8,}  ({pct:.1f}%)")

    if s["target_distribution"]:
        print("  Target classes  :")
        for cls, cnt in s["target_distribution"].items():
            pct = cnt / s["total_rows"] * 100
            print(f"    {str(cls):<20} {cnt:>8,}  ({pct:.1f}%)")

    print()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    from data.loader import load_csv, DATA_DIR

    parser = argparse.ArgumentParser(description="Run data quality checks on a CSV.")
    parser.add_argument("filename", nargs="?", help="CSV file in data/ folder")
    parser.add_argument("--target", default=None, help="Target column for distribution check")
    parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    args = parser.parse_args()

    filename = args.filename
    if filename is None:
        csv_files = list(DATA_DIR.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {DATA_DIR}")
            sys.exit(1)
        filename = csv_files[0].name
        print(f"No filename given — using: {filename}")

    df = load_csv(filename)
    result = check_data_quality(df, target_col=args.target)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)

    sys.exit(0 if result["success"] else 1)
