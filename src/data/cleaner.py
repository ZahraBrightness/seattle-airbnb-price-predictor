"""
Data cleaning pipeline for ML projects.

Steps (in order):
  1. Drop calendar_updated column (100% null)
  2. Drop rows where price is null
  3. Convert price from currency string to float
  4. Drop exact duplicate rows
  5. Save cleaned DataFrame to data/cleaned.csv
  6. Re-run quality gate

All other nulls are left intact for feature engineering.

Usage
-----
Programmatic:
    from data.cleaner import clean_data
    cleaned_df, result = clean_data(df)

CLI:
    python src/data/cleaner.py [filename.csv.gz]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CLEANED_PATH = DATA_DIR / "cleaned.csv"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _drop_calendar_updated(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Drop calendar_updated if present (known 100% null column)."""
    if "calendar_updated" in df.columns:
        return df.drop(columns=["calendar_updated"]), True
    return df, False


def _drop_null_price(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows where price is null."""
    if "price" not in df.columns:
        return df, 0
    before = len(df)
    df = df[df["price"].notna()].copy()
    return df, before - len(df)


def _parse_price(df: pd.DataFrame) -> pd.DataFrame:
    """Convert price from '$1,234.00' string format to float."""
    if "price" not in df.columns:
        return df
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )
    return df


def _drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows, keep first occurrence."""
    before = len(df)
    df = df.drop_duplicates(keep="first")
    return df, before - len(df)


def _save(df: pd.DataFrame) -> None:
    df.to_csv(CLEANED_PATH, index=False)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run the cleaning pipeline and return the cleaned DataFrame plus a
    quality-gate report on the result.

    Returns
    -------
    (cleaned_df, quality_result)
    """
    from data.quality import check_data_quality

    log: dict[str, Any] = {"steps": []}

    # 1. Drop calendar_updated
    df, dropped = _drop_calendar_updated(df)
    log["steps"].append({"step": "drop_calendar_updated", "dropped": dropped})

    # 2. Drop rows where price is null
    df, price_null_rows = _drop_null_price(df)
    log["steps"].append({"step": "drop_null_price", "rows_dropped": price_null_rows})

    # 3. Parse price to float
    df = _parse_price(df)
    log["steps"].append({"step": "parse_price", "dtype": str(df["price"].dtype)})

    # 4. Drop duplicates
    df, dup_rows = _drop_duplicates(df)
    log["steps"].append({"step": "drop_duplicates", "rows_dropped": dup_rows})

    # 5. Save
    _save(df)
    log["steps"].append({"step": "save", "path": str(CLEANED_PATH)})

    # 6. Quality gate
    quality_result = check_data_quality(df, target_col="price")
    quality_result["cleaning_log"] = log

    return df, quality_result


# --------------------------------------------------------------------------- #
# Pretty printer                                                               #
# --------------------------------------------------------------------------- #

def print_cleaning_summary(
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    result: dict[str, Any],
) -> None:
    print("\n" + "=" * 60)
    print("  Cleaning Summary")
    print("=" * 60)

    br, bc = before_shape
    ar, ac = after_shape
    print(f"\n  Rows   : {br:>8,}  →  {ar:>8,}  ({br - ar:,} removed)")
    print(f"  Columns: {bc:>8}  →  {ac:>8}  ({bc - ac} removed)")

    for step in result.get("cleaning_log", {}).get("steps", []):
        name = step["step"]
        if name == "drop_calendar_updated" and step["dropped"]:
            print("\n  Dropped column: calendar_updated (100% null)")
        elif name == "drop_null_price":
            print(f"\n  Rows dropped (null price): {step['rows_dropped']:,}")
        elif name == "parse_price":
            print(f"\n  price converted to {step['dtype']}")
        elif name == "drop_duplicates" and step["rows_dropped"]:
            print(f"\n  Duplicate rows dropped: {step['rows_dropped']:,}")
        elif name == "save":
            print(f"\n  Saved to: {step['path']}")

    status = "PASSED" if result["success"] else "FAILED"
    n_fail = len(result["failures"])
    n_warn = len(result["warnings"])
    print(f"\n  Quality gate (post-clean): {status}  "
          f"({n_fail} failure(s), {n_warn} warning(s))")
    for msg in result["failures"]:
        print(f"    x {msg}")
    skipped = {"Schema check skipped", "Target check skipped"}
    for msg in result["warnings"]:
        if not any(s in msg for s in skipped):
            print(f"    ! {msg}")
    print()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    from data.loader import load_csv, DATA_DIR as _DATA_DIR

    parser = argparse.ArgumentParser(description="Clean a CSV and run quality gate.")
    parser.add_argument("filename", nargs="?", help="File in data/ folder")
    args = parser.parse_args()

    filename = args.filename
    if filename is None:
        csv_files = [f for f in _DATA_DIR.glob("*.csv*") if "cleaned" not in f.name]
        if not csv_files:
            print(f"No CSV files found in {_DATA_DIR}")
            sys.exit(1)
        filename = csv_files[0].name
        print(f"No filename given — using: {filename}")

    raw_df = load_csv(filename)
    before = raw_df.shape
    print(f"\nLoaded: {before[0]:,} rows x {before[1]} columns")

    cleaned_df, quality_result = clean_data(raw_df)
    print_cleaning_summary(before, cleaned_df.shape, quality_result)
    sys.exit(0 if quality_result["success"] else 1)
