"""
Calendar-derived features for Airbnb price prediction.

Reads data/calendar.csv.gz (one row per listing × date), groups by
listing_id, and engineers 10 availability and demand features per listing.

Features
--------
Availability:
  overall_availability_rate    – % of all dates where available = 't'
  peak_availability_rate       – % available June–August
  off_availability_rate        – % available December–February
  availability_gap             – off_rate − peak_rate  (+ = seasonal)
  pct_weekend_available        – % of Fri/Sat dates available
  avg_minimum_nights_cal       – mean minimum nights across all calendar rows
  has_dynamic_minimum          – 1 if minimum nights varies across dates
  consecutive_blocked_rate     – avg consecutive-blocked-streak / total dates

Demand × price:
  estimated_annual_revenue     – base_price × occupancy_rate × 365
  peak_demand_score            – base_price × (1 − peak_availability_rate)

Null handling
-------------
Listings absent from calendar get numeric features filled with the column
median and has_dynamic_minimum = 0.

Usage
-----
    from features.calendar_features import create_calendar_features
    df_with_cal = create_calendar_features(df)

CLI:
    python src/features/calendar_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CAL_NUMERIC_COLS = [
    "overall_availability_rate",
    "peak_availability_rate",
    "off_availability_rate",
    "availability_gap",
    "pct_weekend_available",
    "avg_minimum_nights_cal",
    "consecutive_blocked_rate",
    "estimated_annual_revenue",
    "peak_demand_score",
]

# Month sets for seasonal splits
_PEAK_MONTHS = {6, 7, 8}       # June – August
_OFF_MONTHS  = {12, 1, 2}      # December – February
# Weekends: Friday=4, Saturday=5 (pandas dayofweek, Monday=0)
_WEEKEND_DOW = {4, 5}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _consecutive_blocked_rate(cal: pd.DataFrame) -> pd.Series:
    """
    For each listing, compute:
        mean(lengths of consecutive blocked streaks) / total calendar dates

    A high value means the listing tends to be blocked in long uninterrupted
    runs — characteristic of monthly renters or reserved corporate housing.
    A low value means blocks are short and scattered — typical tourist hosts.

    Uses a vectorised run-length approach (no groupby.apply) that scales to
    the full 2.5 M-row calendar without per-listing Python loops.
    """
    # cal must already be sorted by (listing_id, date)
    is_blocked = (cal["available"] == "f").astype(np.int8)

    # Run identifier: increments at every value change OR listing boundary
    listing_changed = (cal["listing_id"] != cal["listing_id"].shift(fill_value=-1))
    val_changed      = (is_blocked != is_blocked.shift(fill_value=-1))
    run_id = (listing_changed | val_changed).cumsum()

    # Aggregate runs: listing, whether blocked, and length
    run_info = pd.DataFrame({
        "listing_id": cal["listing_id"].values,
        "is_blocked": is_blocked.values,
        "run_id":     run_id.values,
    }).groupby("run_id", sort=False).agg(
        listing_id=("listing_id", "first"),
        is_blocked=("is_blocked", "first"),
        length=("is_blocked", "count"),
    )

    # Average blocked-streak length per listing
    blocked_runs = run_info[run_info["is_blocked"] == 1]
    avg_streak   = blocked_runs.groupby("listing_id")["length"].mean()

    # Total dates per listing
    total_dates = cal.groupby("listing_id").size()

    rate = (avg_streak / total_dates).fillna(0.0).clip(0.0, 1.0)
    rate.name = "consecutive_blocked_rate"
    return rate


# --------------------------------------------------------------------------- #
# Main function                                                                #
# --------------------------------------------------------------------------- #

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer calendar-derived features and left-join them onto ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an ``id`` column (Airbnb listing_id).

    Returns
    -------
    pd.DataFrame — same rows as ``df``, with 10 new columns appended.
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # Load and prepare calendar                                           #
    # ------------------------------------------------------------------ #
    print("  Loading calendar.csv.gz ...")
    cal = pd.read_csv(
        DATA_DIR / "calendar.csv.gz",
        usecols=["listing_id", "date", "available", "minimum_nights"],
        parse_dates=["date"],
    )
    print(f"  Calendar: {len(cal):,} rows × {cal.shape[1]} columns  "
          f"| {cal['listing_id'].nunique():,} unique listings")

    # Sort once — all downstream ops rely on this order
    cal = cal.sort_values(["listing_id", "date"]).reset_index(drop=True)

    # Binary availability flag
    cal["is_avail"] = (cal["available"] == "t").astype(np.int8)

    # Temporal masks
    month = cal["date"].dt.month
    cal["is_peak"]    = month.isin(_PEAK_MONTHS).astype(np.int8)
    cal["is_off"]     = month.isin(_OFF_MONTHS).astype(np.int8)
    cal["is_weekend"] = cal["date"].dt.dayofweek.isin(_WEEKEND_DOW).astype(np.int8)

    # ------------------------------------------------------------------ #
    # Load base listing price from cleaned.csv                           #
    # ------------------------------------------------------------------ #
    base_price = (
        pd.read_csv(DATA_DIR / "cleaned.csv", usecols=["id", "price"])
        .dropna(subset=["price"])
        .set_index("id")["price"]
    )

    # ------------------------------------------------------------------ #
    # Feature computation (all vectorised, no apply)                     #
    # ------------------------------------------------------------------ #
    grp = cal.groupby("listing_id", sort=True)

    # 1. overall_availability_rate
    overall_avail = grp["is_avail"].mean()

    # 2. peak_availability_rate
    peak_avail = (
        cal[cal["is_peak"] == 1]
        .groupby("listing_id")["is_avail"].mean()
        .reindex(overall_avail.index)       # fill listings with no peak dates below
    )

    # 3. off_availability_rate
    off_avail = (
        cal[cal["is_off"] == 1]
        .groupby("listing_id")["is_avail"].mean()
        .reindex(overall_avail.index)
    )

    # 5. pct_weekend_available
    weekend_avail = (
        cal[cal["is_weekend"] == 1]
        .groupby("listing_id")["is_avail"].mean()
        .reindex(overall_avail.index)
    )

    # 6. avg_minimum_nights_cal
    avg_min_nights = grp["minimum_nights"].mean()

    # 7. has_dynamic_minimum  (more than one unique value across dates)
    has_dynamic = (grp["minimum_nights"].nunique() > 1).astype(int)

    # 8. consecutive_blocked_rate
    consec_rate = _consecutive_blocked_rate(cal)
    consec_rate  = consec_rate.reindex(overall_avail.index, fill_value=0.0)

    # 4. availability_gap (derived after both seasonal rates are ready)
    avail_gap = (off_avail - peak_avail)

    # 9 & 10 — require base listing price
    listing_price = base_price.reindex(overall_avail.index)   # NaN if price missing
    estimated_annual_revenue = listing_price * (1.0 - overall_avail) * 365.0
    peak_demand_score        = listing_price * (1.0 - peak_avail.fillna(overall_avail))

    # ------------------------------------------------------------------ #
    # Assemble features DataFrame (indexed by listing_id)                #
    # ------------------------------------------------------------------ #
    cal_features = pd.DataFrame({
        "overall_availability_rate": overall_avail,
        "peak_availability_rate":    peak_avail,
        "off_availability_rate":     off_avail,
        "availability_gap":          avail_gap,
        "pct_weekend_available":     weekend_avail,
        "avg_minimum_nights_cal":    avg_min_nights,
        "has_dynamic_minimum":       has_dynamic,
        "consecutive_blocked_rate":  consec_rate,
        "estimated_annual_revenue":  estimated_annual_revenue,
        "peak_demand_score":         peak_demand_score,
    }, index=overall_avail.index)
    cal_features.index.name = "id"

    # Clip availability rates to [0, 1] — should already be in range, but
    # float arithmetic on filtered subsets can produce tiny violations.
    avail_cols = [
        "overall_availability_rate", "peak_availability_rate",
        "off_availability_rate", "pct_weekend_available",
    ]
    cal_features[avail_cols] = cal_features[avail_cols].clip(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Merge onto df                                                        #
    # ------------------------------------------------------------------ #
    n_before = len(df)
    df = df.join(cal_features, on="id", how="left")
    assert len(df) == n_before, "join changed row count — check for duplicate listing_ids"

    # ------------------------------------------------------------------ #
    # Null handling for listings absent from calendar                     #
    # ------------------------------------------------------------------ #
    missing_mask = df["overall_availability_rate"].isna()
    n_missing = int(missing_mask.sum())

    if n_missing > 0:
        print(f"  {n_missing:,} listings not in calendar — filling with median / 0")
        for col in CAL_NUMERIC_COLS:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        df["has_dynamic_minimum"] = df["has_dynamic_minimum"].fillna(0).astype(int)
    else:
        print("  All listings matched in calendar (0 missing)")
        df["has_dynamic_minimum"] = df["has_dynamic_minimum"].astype(int)

    return df


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print(f"Loading data/features.csv ...")
    df = pd.read_csv(DATA_DIR / "features.csv")
    print(f"  Shape before: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    print("Engineering calendar features ...")
    df_cal = create_calendar_features(df)

    new_cols = CAL_NUMERIC_COLS + ["has_dynamic_minimum"]
    print(f"\n  Shape after : {df_cal.shape[0]:,} rows × {df_cal.shape[1]} columns")
    print(f"  New columns : {len(new_cols)}")

    # ------------------------------------------------------------------ #
    # Coverage                                                            #
    # ------------------------------------------------------------------ #
    print(f"\n  {'Feature':<32}  {'Non-null':>8}  {'Coverage':>9}  "
          f"{'Min':>8}  {'Median':>8}  {'Max':>8}")
    print(f"  {'-'*74}")
    for col in new_cols:
        non_null = df_cal[col].notna().sum()
        pct      = non_null / len(df_cal) * 100
        lo       = df_cal[col].min()
        med      = df_cal[col].median()
        hi       = df_cal[col].max()
        print(f"  {col:<32}  {non_null:>8,}  {pct:>8.1f}%  "
              f"{lo:>8.3f}  {med:>8.3f}  {hi:>8.1f}")

    # ------------------------------------------------------------------ #
    # Correlation with price                                              #
    # ------------------------------------------------------------------ #
    if "price" in df_cal.columns:
        print(f"\n  {'Feature':<32}  {'Corr with price':>16}  {'Direction'}")
        print(f"  {'-'*65}")
        corrs = df_cal[new_cols + ["price"]].corr()["price"].drop("price")
        for col, r in corrs.sort_values(key=abs, ascending=False).items():
            direction = "↑ higher price" if r > 0 else "↓ lower price"
            print(f"  {col:<32}  {r:>16.4f}  {direction}")

    # ------------------------------------------------------------------ #
    # Save                                                                #
    # ------------------------------------------------------------------ #
    out = DATA_DIR / "features_cal.csv"
    df_cal.to_csv(out, index=False)
    print(f"\n  Saved → {out}")
