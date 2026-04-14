"""
Feature engineering for Airbnb price prediction (Seattle listings).

All engineered features are documented with a WHY comment explaining
their expected predictive signal. Columns irrelevant to price prediction
are dropped before returning.

Usage
-----
Programmatic:
    from features.engineer import create_features
    featured_df = create_features(df)

CLI:
    python src/features/engineer.py [input_csv]   # defaults to data/cleaned.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FEATURED_PATH = DATA_DIR / "featured.csv"

# Columns that carry no predictive signal for price and are dropped up front.
# IDs, URLs, free-text blurbs, and scrape metadata are excluded.
_DROP_COLS = [
    "id", "listing_url", "scrape_id", "source",
    "picture_url",
    "host_url", "host_thumbnail_url", "host_picture_url",
    "name", "description", "neighborhood_overview", "host_about",
    "last_scraped", "calendar_last_scraped",
    "license",
]

# All Airbnb review sub-scores used to build review_score_avg.
SEATTLE_DOWNTOWN = (47.6062, -122.3321)   # Pike Place / business district

# Numeric features that must survive select_features() regardless of variance,
# because they are used downstream to derive meaningful location signals.
_ALWAYS_KEEP = {"latitude", "longitude", "neighbourhood_group_avg_price"}

_REVIEW_SCORE_COLS = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]


# --------------------------------------------------------------------------- #
# Feature builders                                                             #
# --------------------------------------------------------------------------- #

def _domain_features(df: pd.DataFrame) -> pd.DataFrame:
    # price_per_person: value-for-money signal — guests implicitly compare
    # nightly cost per person when choosing between listings; cheaper-per-person
    # listings can justify a higher total price.
    df["price_per_person"] = df["price"] / df["accommodates"].replace(0, np.nan)

    # amenities_count: proxy for listing quality and comfort level.
    # Hosts with more amenities can credibly charge higher prices.
    def _count_amenities(val: str) -> int:
        try:
            return len(json.loads(val))
        except (ValueError, TypeError):
            # Fallback: count comma-separated tokens for non-JSON formats.
            return len(str(val).split(",")) if pd.notna(val) and str(val).strip() else 0

    df["amenities_count"] = df["amenities"].apply(_count_amenities)

    # host_experience_years: seasoned hosts understand pricing dynamics,
    # invest in their listing, and have track records that justify higher rates.
    host_since = pd.to_datetime(df["host_since"], errors="coerce")
    last_scraped = pd.to_datetime(df["last_scraped"], errors="coerce")
    df["host_experience_years"] = (last_scraped - host_since).dt.days / 365.25

    # is_superhost: Airbnb's verified quality badge. Superhosts consistently
    # receive higher ratings and bookings, allowing them to command a premium.
    df["is_superhost"] = (df["host_is_superhost"] == "t").astype(int)

    # host_response_rate_num: high responsiveness reduces guest uncertainty,
    # lowers the chance of booking abandonment, and correlates with professionalism.
    df["host_response_rate_num"] = (
        df["host_response_rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    # host_acceptance_rate_num: hosts who accept most requests keep their
    # calendars full; a high rate signals an actively managed, in-demand listing.
    df["host_acceptance_rate_num"] = (
        df["host_acceptance_rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    # is_instant_bookable: reduces friction in the booking flow, attracting
    # last-minute travellers who are often less price-sensitive.
    df["is_instant_bookable"] = (df["instant_bookable"] == "t").astype(int)

    # has_profile_pic: signals host legitimacy and trustworthiness.
    # Guests pay more when they feel safe; a profile photo is a basic trust signal.
    df["has_profile_pic"] = (df["host_has_profile_pic"] == "t").astype(int)

    # is_identity_verified: Airbnb-verified identity further reduces perceived
    # risk, allowing verified hosts to price above unverified peers.
    df["is_identity_verified"] = (df["host_identity_verified"] == "t").astype(int)

    return df


def _statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    # review_score_avg: a single composite quality signal from seven sub-scores.
    # Averaging reduces noise from any individual dimension (e.g. location isn't
    # the host's fault) while capturing overall guest satisfaction.
    present_review_cols = [c for c in _REVIEW_SCORE_COLS if c in df.columns]
    df["review_score_avg"] = df[present_review_cols].mean(axis=1)

    # occupancy_rate: fraction of the year the listing is actually booked.
    # High occupancy means real demand exists at the current price — a strong
    # signal that the listing is correctly or under-priced.
    if "estimated_occupancy_l365d" in df.columns:
        df["occupancy_rate"] = df["estimated_occupancy_l365d"] / 365
    else:
        df["occupancy_rate"] = np.nan

    # availability_rate: complement of occupancy — what fraction of the year
    # the host keeps the listing open. A very low rate may indicate exclusivity
    # or a secondary property, both associated with higher prices.
    df["availability_rate"] = df["availability_365"] / 365

    # reviews_active: binary flag for whether the listing received a review in
    # the last 30 days. Active listings are in current demand, distinguishing
    # live inventory from stale or seasonal listings.
    df["reviews_active"] = (df["number_of_reviews_l30d"] > 0).astype(int)

    # days_since_last_review: recency of social proof. A listing reviewed
    # recently has demonstrated current-market appeal; stale reviews may
    # indicate declining quality or a dormant listing.
    last_review = pd.to_datetime(df["last_review"], errors="coerce")
    last_scraped = pd.to_datetime(df["last_scraped"], errors="coerce")
    df["days_since_last_review"] = (last_scraped - last_review).dt.days

    return df


def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # size_quality_score: captures listings that are both spacious AND highly
    # rated. A large listing with poor reviews or a tiny listing with great
    # reviews each price differently — the product captures the joint premium.
    df["size_quality_score"] = df["accommodates"] * df["review_scores_rating"]

    # revenue_per_night: actual earning power normalised by availability.
    # Adding 1 to availability_365 avoids division by zero for fully-booked
    # listings. This is a model of realised yield, not just listed price.
    df["revenue_per_night"] = (
        df["estimated_revenue_l365d"] / (df["availability_365"] + 1)
    )

    # host_quality_score: interaction between superhost status and average
    # review score. A superhost with mediocre reviews or a non-superhost with
    # great reviews each price differently; the product rewards both.
    df["host_quality_score"] = df["is_superhost"] * df["review_score_avg"]

    return df


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _location_features(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------ #
    # 1. Merge neighbourhood_group from neighbourhoods.csv                #
    # ------------------------------------------------------------------ #
    # neighbourhood_group captures broad area prestige (e.g. Capitol Hill,
    # Ballard, Beacon Hill). Neighbourhoods within the same group share
    # zoning, demographics, and access to amenities — all of which influence
    # what guests are willing to pay.
    nbr_path = DATA_DIR / "neighbourhoods.csv"
    nbr = (
        pd.read_csv(nbr_path)[["neighbourhood_group", "neighbourhood"]]
        .rename(columns={"neighbourhood": "neighbourhood_cleansed"})
    )
    df = df.merge(nbr, on="neighbourhood_cleansed", how="left")

    # ------------------------------------------------------------------ #
    # 2. distance_to_downtown                                             #
    # ------------------------------------------------------------------ #
    # Proximity to Pike Place Market, the waterfront, and the CBD is a key
    # driver of short-stay demand. Tourists and business travellers pay a
    # premium to minimise commute time to Seattle's core attractions.
    dt_lat, dt_lon = SEATTLE_DOWNTOWN
    df["distance_to_downtown"] = df.apply(
        lambda row: _haversine_km(row["latitude"], row["longitude"], dt_lat, dt_lon)
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
        else np.nan,
        axis=1,
    )

    # ------------------------------------------------------------------ #
    # 3. neighbourhood_avg_price                                          #
    # ------------------------------------------------------------------ #
    # The mean price within a specific neighbourhood captures hyper-local
    # pricing norms. A listing priced below its neighbourhood average may
    # be a bargain; above it signals premium positioning.
    nbr_avg = df.groupby("neighbourhood_cleansed")["price"].transform("mean")
    df["neighbourhood_avg_price"] = nbr_avg

    # ------------------------------------------------------------------ #
    # 4. neighbourhood_group_avg_price                                    #
    # ------------------------------------------------------------------ #
    # Broader-area average smooths out the noise from small neighbourhoods
    # with few listings, providing a more stable prestige signal for the
    # wider district (e.g. Capitol Hill vs. Beacon Hill vs. Ballard).
    grp_avg = df.groupby("neighbourhood_group")["price"].transform("mean")
    df["neighbourhood_group_avg_price"] = grp_avg

    return df


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def select_features(
    df: pd.DataFrame,
    target_col: str = "price",
    corr_threshold: float = 0.95,
    variance_threshold_frac: float = 0.01,
) -> tuple[list[str], pd.DataFrame]:
    """
    Remove redundant and near-constant numeric features, then return the
    selected feature names and the reduced DataFrame.

    Two-pass filter:
      1. Variance filter  — drop columns whose variance is below
         (variance_threshold_frac * median column variance). Near-zero-variance
         features carry almost no signal and can destabilise some models.
      2. Correlation filter — for each pair of features with |r| > corr_threshold,
         drop the one that appears later in the column order (keep first).
         Highly correlated features are redundant and inflate dimensionality.

    Parameters
    ----------
    df                    : DataFrame produced by create_features().
    target_col            : Excluded from filtering (never dropped).
    corr_threshold        : Drop one of a pair when |r| exceeds this.
    variance_threshold_frac : Drop a feature when its variance is below
                              this fraction of the median column variance.

    Returns
    -------
    (selected_feature_names, reduced_df)
    """
    numeric = df.select_dtypes(include="number")
    candidates = [c for c in numeric.columns if c != target_col]

    # Columns in _ALWAYS_KEEP are excluded from both filters but still
    # included in the returned DataFrame and selected feature list.
    pinned = [c for c in candidates if c in _ALWAYS_KEEP]
    candidates = [c for c in candidates if c not in _ALWAYS_KEEP]

    dropped_low_var: list[tuple[str, float, float]] = []   # (col, var, threshold)
    dropped_corr:    list[tuple[str, str, float]]   = []   # (col, corr_with, r)

    # ------------------------------------------------------------------ #
    # Pass 1 — variance filter                                            #
    # ------------------------------------------------------------------ #
    variances = numeric[candidates].var()
    overall_threshold = variance_threshold_frac * variances.median()

    keep_after_var: list[str] = []
    for col in candidates:
        v = variances[col]
        if v < overall_threshold:
            dropped_low_var.append((col, v, overall_threshold))
        else:
            keep_after_var.append(col)

    # ------------------------------------------------------------------ #
    # Pass 2 — correlation filter                                         #
    # ------------------------------------------------------------------ #
    corr_matrix = numeric[keep_after_var].corr().abs()

    to_drop_corr: set[str] = set()
    cols = keep_after_var
    for i, col_a in enumerate(cols):
        if col_a in to_drop_corr:
            continue
        for col_b in cols[i + 1:]:
            if col_b in to_drop_corr:
                continue
            r = corr_matrix.loc[col_a, col_b]
            if r > corr_threshold:
                # Drop the later column; keep col_a (appears first)
                to_drop_corr.add(col_b)
                dropped_corr.append((col_b, col_a, round(float(r), 4)))

    selected = [c for c in keep_after_var if c not in to_drop_corr]

    # Re-add pinned features (not subject to any filter)
    selected = pinned + selected

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    if pinned:
        print(f"\n  Pinned (always kept): {pinned}")
    print(f"\n[select_features] variance threshold = {overall_threshold:.6f}  "
          f"(median var × {variance_threshold_frac})")

    if dropped_low_var:
        print(f"\n  Dropped — low variance ({len(dropped_low_var)}):")
        for col, v, thr in dropped_low_var:
            print(f"    - {col:<40}  var={v:.6f}  <  threshold={thr:.6f}")
    else:
        print("\n  Dropped — low variance: none")

    print(f"\n  Dropped — high correlation > {corr_threshold} ({len(dropped_corr)}):")
    if dropped_corr:
        for col, corr_with, r in sorted(dropped_corr):
            print(f"    - {col:<40}  |r|={r:.4f}  with  '{corr_with}'")
    else:
        print("    none")

    # Always include target and all non-numeric columns in the output frame
    non_numeric = [c for c in df.columns if c not in numeric.columns]
    final_cols = non_numeric + ([target_col] if target_col in df.columns else []) + selected
    # Preserve original column order
    final_cols = [c for c in df.columns if c in set(final_cols)]

    print(f"\n  Pinned (skip filters)     : {len(pinned)}")
    print(f"  Features before selection : {len(candidates)}")
    print(f"  Dropped (low variance)    : {len(dropped_low_var)}")
    print(f"  Dropped (high correlation): {len(dropped_corr)}")
    print(f"  Features after selection  : {len(selected)}")

    return selected, df[final_cols]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer predictive features for Airbnb price prediction and return the
    augmented DataFrame with uninformative columns removed.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned listings DataFrame (output of cleaner.clean_data).

    Returns
    -------
    pd.DataFrame
        DataFrame with all original useful columns plus engineered features,
        ready for model training.
    """
    df = df.copy()

    df = _domain_features(df)
    df = _statistical_features(df)
    df = _interaction_features(df)
    df = _location_features(df)

    # Drop columns with no predictive value.
    # Date columns used for feature computation are dropped here after use.
    to_drop = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop)

    return df


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "cleaned.csv")
    input_path = DATA_DIR / Path(input_path).name  # always resolve inside data/

    print(f"Loading: {input_path}")
    raw = pd.read_csv(input_path)
    original_shape = raw.shape
    original_cols = set(raw.columns)

    print(f"Original shape: {original_shape[0]:,} rows x {original_shape[1]} columns")

    featured = create_features(raw)

    new_cols = sorted(set(featured.columns) - original_cols)
    dropped_cols = sorted(original_cols - set(featured.columns))

    print(f"Featured shape: {featured.shape[0]:,} rows x {featured.shape[1]} columns")
    print(f"\nNew columns added ({len(new_cols)}):")
    for col in new_cols:
        non_null = featured[col].notna().sum()
        pct = non_null / len(featured) * 100
        print(f"  + {col:<35}  {non_null:,} non-null ({pct:.1f}%)")

    print(f"\nColumns dropped ({len(dropped_cols)}):")
    for col in dropped_cols:
        print(f"  - {col}")

    # ------------------------------------------------------------------ #
    # Feature selection                                                    #
    # ------------------------------------------------------------------ #
    selected_names, final_df = select_features(featured)

    print(f"\nFinal shape after selection: "
          f"{final_df.shape[0]:,} rows x {final_df.shape[1]} columns")
    print(f"\nSelected numeric features ({len(selected_names)}):")
    for col in selected_names:
        print(f"  {col}")

    final_df.to_csv(FEATURED_PATH, index=False)
    print(f"\nSaved to: {FEATURED_PATH}")
