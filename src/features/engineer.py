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
    "listing_url", "scrape_id", "source",
    "picture_url",
    "host_url", "host_thumbnail_url", "host_picture_url",
    "name", "neighborhood_overview", "host_about",
    "last_scraped", "calendar_last_scraped",
    "license",
    # Outcome variables — derived from price and actual bookings.
    # A new host setting up their listing won't have this data.
    "estimated_revenue_l365d",
    "estimated_occupancy_l365d",
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


def create_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract binary keyword features from the listing description and amenities
    columns, then drop both raw text columns.

    Description features (15): scan lowercased description text for keywords
    that signal premium or niche attributes guests pay extra for.

    Amenities features (5): scan the amenities JSON/string for specific
    high-value amenities that also appear in descriptions, providing a
    structured cross-check signal.

    Null descriptions are treated as empty strings (listing has no text).
    Null amenities are treated as empty strings (no amenities listed).
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # Description features                                                 #
    # ------------------------------------------------------------------ #
    desc = df["description"].fillna("").str.lower() if "description" in df.columns else pd.Series("", index=df.index)

    nlp_desc = {
        "has_view":              r"\bview\b|\bviews\b|\bscenic\b|\bpanoramic\b",
        "has_waterfront":        r"\bwaterfront\b|\bwater front\b|\bwaterside\b|\blakefront\b|\bbeachfront\b",
        "is_downtown":           r"\bdowntown\b|\bcity cent(?:er|re)\b|\burban core\b",
        "has_hot_tub":           r"\bhot tub\b|\bhotttub\b|\bspa\b|\bjacuzzi\b",
        "has_pool":              r"\bpool\b|\bswimming pool\b",
        "has_parking":           r"\bparking\b|\bgarage\b|\bdriveway\b",
        "has_gym":               r"\bgym\b|\bfitness\b|\bworkout\b|\bexercise room\b",
        "has_ev_charger":        r"\bev charger\b|\belectric vehicle\b|\bev charging\b|\btesla charger\b",
        "is_newly_renovated":    r"\bnew(?:ly)? renovated\b|\brecently renovated\b|\bremodeled\b|\bupdated\b|\bmodern(?:ized)?\b",
        "is_luxury":             r"\bluxury\b|\bluxurious\b|\bupscale\b|\bhigh.end\b|\bpremium\b",
        "is_cozy":               r"\bcozy\b|\bcosy\b|\bcharm(?:ing)?\b|\bintimate\b|\bquaint\b",
        "has_fireplace":         r"\bfireplace\b|\bfire place\b|\bwood.burning\b",
        "has_private_entrance":  r"\bprivate entrance\b|\bprivate entry\b|\bseparate entrance\b",
        "has_backyard":          r"\bbackyard\b|\bback yard\b|\bprivate yard\b|\bpatio\b|\bdeck\b|\bterrace\b",
        "is_entire_floor":       r"\bentire floor\b|\bwhole floor\b|\bfull floor\b|\btop floor\b|\bpenthouse\b",
    }

    for feat, pattern in nlp_desc.items():
        df[feat] = desc.str.contains(pattern, regex=True, na=False).astype(int)

    # ------------------------------------------------------------------ #
    # Amenities features                                                   #
    # ------------------------------------------------------------------ #
    amen = df["amenities"].fillna("").str.lower() if "amenities" in df.columns else pd.Series("", index=df.index)

    nlp_amen = {
        "amenity_has_hot_tub":  r"\bhot tub\b|\bhotttub\b|\bjacuzzi\b",
        "amenity_has_pool":     r"\bpool\b|\bswimming pool\b",
        "amenity_has_parking":  r"\bparking\b|\bgarage\b",
        "amenity_has_gym":      r"\bgym\b|\bfitness\b|\bworkout\b",
        "amenity_has_ev_charger": r"\bev charger\b|\belectric vehicle\b|\bev charging\b",
    }

    for feat, pattern in nlp_amen.items():
        df[feat] = amen.str.contains(pattern, regex=True, na=False).astype(int)

    # ------------------------------------------------------------------ #
    # Coverage report                                                      #
    # ------------------------------------------------------------------ #
    all_nlp_feats = list(nlp_desc) + list(nlp_amen)
    n = len(df)
    print(f"\n  NLP feature coverage ({n:,} listings):")
    for feat in all_nlp_feats:
        pct = df[feat].mean() * 100
        print(f"    {feat:<28} {df[feat].sum():>5,}  ({pct:5.1f}%)")

    # ------------------------------------------------------------------ #
    # Drop raw text columns (now encoded)                                  #
    # ------------------------------------------------------------------ #
    for col in ["description"]:
        if col in df.columns:
            df = df.drop(columns=[col])

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
    # Exclude target, 'id' (join key, not a feature), and _ALWAYS_KEEP from
    # all filtering. 'id' is passed through untouched to the output DataFrame
    # so downstream pipelines can use it as a join key.
    _NON_FEATURE_COLS = {target_col, "id"}
    candidates = [c for c in numeric.columns if c not in _NON_FEATURE_COLS]

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

    # Always include target, 'id' (join key), and all non-numeric columns
    non_numeric = [c for c in df.columns if c not in numeric.columns]
    id_col  = ["id"] if "id" in df.columns else []
    final_cols = non_numeric + id_col + ([target_col] if target_col in df.columns else []) + selected
    # Preserve original column order (dedup via set, respecting df order)
    final_cols = [c for c in df.columns if c in set(final_cols)]

    print(f"\n  Pinned (skip filters)     : {len(pinned)}")
    print(f"  Features before selection : {len(candidates)}")
    print(f"  Dropped (low variance)    : {len(dropped_low_var)}")
    print(f"  Dropped (high correlation): {len(dropped_corr)}")
    print(f"  Features after selection  : {len(selected)}")

    return selected, df[final_cols]


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve all remaining nulls after create_features() using domain-aware
    strategies grouped by the *reason* values are missing.

    Group 1 — Incomplete host profile (same 294 rows missing host_name,
    host_since, host_listings_count, etc.). These listings lack a real host
    identity and cannot be used for prediction — drop the rows entirely.

    Group 2 — Host chose not to fill in optional profile fields. Treat
    absence as a known category ("unknown") or impute with the median so
    the model can still use the column.

    Group 3 — No reviews yet. Missing review fields are not random; they
    signal a brand-new or dormant listing. Encode that signal explicitly
    (0 for counts/scores, 9999 for days-since as an "infinity" sentinel).
    Drop the raw date columns already captured by days_since_last_review.

    Group 4 — neighbourhood is 39% null and is superseded by the cleaner
    neighbourhood_cleansed and neighbourhood_group columns — drop it.

    Group 5 — Tiny null counts (<1%) on numeric/mixed columns.
    Fill with median (numeric) or mode (categorical/text).
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # Group 1: drop rows with incomplete host profile                     #
    # ------------------------------------------------------------------ #
    before = len(df)
    df = df[df["host_name"].notna()].reset_index(drop=True)
    print(f"  [G1] Dropped {before - len(df)} rows with null host_name "
          f"(incomplete host profile)")

    # ------------------------------------------------------------------ #
    # Group 2: host chose not to set optional fields                      #
    # ------------------------------------------------------------------ #
    for col in ["host_response_time", "host_location", "host_neighbourhood"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    for col in ["host_response_rate_num", "host_acceptance_rate_num"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # host_is_superhost: unknown hosts are assumed not superhosts
    if "host_is_superhost" in df.columns:
        df["host_is_superhost"] = df["host_is_superhost"].fillna("f")

    # host_response_rate / host_acceptance_rate: raw string columns —
    # host never set them, so mark as unknown rather than imputing a number
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    print(f"  [G2] Filled host optional fields with median/unknown")

    # ------------------------------------------------------------------ #
    # Group 3: no reviews yet — encode absence as a deliberate signal     #
    # ------------------------------------------------------------------ #
    if "reviews_per_month" in df.columns:
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    if "host_quality_score" in df.columns:
        df["host_quality_score"] = df["host_quality_score"].fillna(0)

    if "days_since_last_review" in df.columns:
        # 9999 acts as an "infinity" sentinel: the model learns that
        # very large values here mean the listing has never been reviewed.
        df["days_since_last_review"] = df["days_since_last_review"].fillna(9999)

    for col in ["first_review", "last_review"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    print(f"  [G3] Filled review nulls with 0/9999; dropped first_review, last_review")

    # ------------------------------------------------------------------ #
    # Group 4: neighbourhood superseded by cleaner columns                #
    # ------------------------------------------------------------------ #
    if "neighbourhood" in df.columns:
        df = df.drop(columns=["neighbourhood"])
    print(f"  [G4] Dropped neighbourhood column (39% null, "
          f"superseded by neighbourhood_cleansed + neighbourhood_group)")

    # ------------------------------------------------------------------ #
    # Group 5: tiny null counts — fill with median or mode               #
    # ------------------------------------------------------------------ #
    numeric_median_cols = [
        "beds", "minimum_minimum_nights", "maximum_minimum_nights",
        "minimum_maximum_nights", "maximum_maximum_nights",
    ]
    for col in numeric_median_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # bathrooms_text is a string like "1 bath" — use mode
    if "bathrooms_text" in df.columns:
        mode_val = df["bathrooms_text"].mode().iloc[0]
        df["bathrooms_text"] = df["bathrooms_text"].fillna(mode_val)

    # has_availability is 't'/'f' — use mode
    if "has_availability" in df.columns:
        mode_val = df["has_availability"].mode().iloc[0]
        df["has_availability"] = df["has_availability"].fillna(mode_val)

    print(f"  [G5] Filled small numeric nulls with median; "
          f"bathrooms_text/has_availability with mode")

    return df


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
    # Null handling                                                        #
    # ------------------------------------------------------------------ #
    nulls_before = featured.isnull().sum()
    total_before = int(nulls_before[nulls_before > 0].sum())
    print(f"\nNull handling ({total_before:,} null cells across "
          f"{(nulls_before > 0).sum()} columns before) ...")

    featured = handle_nulls(featured)

    nulls_after = featured.isnull().sum()
    total_after = int(nulls_after[nulls_after > 0].sum())
    remaining = nulls_after[nulls_after > 0]
    if remaining.empty:
        print(f"  All nulls resolved. ({total_before:,} → 0)")
    else:
        print(f"  Nulls remaining ({total_after:,} cells across "
              f"{len(remaining)} columns):")
        for col, cnt in remaining.items():
            pct = cnt / len(featured) * 100
            print(f"    ! {col:<40} {cnt:>5,}  ({pct:.1f}%)")

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
