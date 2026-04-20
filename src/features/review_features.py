"""
Review sentiment features for Airbnb price prediction.

Reads data/reviews.csv.gz, keeps the most recent 50 reviews per listing,
runs VADER sentiment analysis, and engineers 8 features per listing.

Features
--------
  avg_sentiment          — mean VADER compound score across all processed reviews
  recent_sentiment       — mean compound score of most recent 10 reviews
  sentiment_trend        — recent_sentiment − avg_sentiment (+ = improving)
  pct_positive_reviews   — % of reviews with compound > 0.05
  pct_negative_reviews   — % of reviews with compound < −0.05
  positive_keyword_count — count of reviews mentioning a premium keyword
  negative_keyword_count — count of reviews mentioning a quality-concern keyword
  review_velocity        — reviews in last 12 months / total reviews

Null handling
-------------
Listings with no reviews get all features filled with 0.

Usage
-----
    from features.review_features import create_review_features
    df_with_reviews = create_review_features(df)

CLI:
    python src/features/review_features.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nltk
import numpy as np
import pandas as pd

nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

REVIEWS_PER_LISTING = 50    # only score the N most recent reviews
RECENT_WINDOW       = 10    # reviews used for recent_sentiment
BATCH_SIZE          = 1000  # listings processed per batch
PROGRESS_EVERY      = 500   # print progress line every N listings

POSITIVE_KEYWORDS = {
    "clean", "spacious", "view", "location", "quiet",
    "cozy", "comfortable", "beautiful", "amazing", "perfect",
}
NEGATIVE_KEYWORDS = {
    "noisy", "dirty", "small", "outdated", "cold",
    "disappointing", "misleading", "broken", "smell", "mold",
}

# Pre-built regex patterns for fast keyword matching (word-boundary aware)
_POS_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in POSITIVE_KEYWORDS) + r")\b")
_NEG_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in NEGATIVE_KEYWORDS) + r")\b")

REVIEW_FEATURE_COLS = [
    "avg_sentiment",
    "recent_sentiment",
    "sentiment_trend",
    "pct_positive_reviews",
    "pct_negative_reviews",
    "positive_keyword_count",
    "negative_keyword_count",
    "review_velocity",
]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _score_batch(
    batch_rev: pd.DataFrame,
    sia: SentimentIntensityAnalyzer,
) -> pd.DataFrame:
    """
    Score one batch of (already filtered, date-sorted) reviews with VADER.
    Returns the same DataFrame with an added 'compound' column.
    """
    batch_rev = batch_rev.copy()
    batch_rev["compound"] = batch_rev["comments"].apply(
        lambda text: sia.polarity_scores(text)["compound"]
    )
    batch_rev["has_pos_kw"] = batch_rev["comments"].apply(
        lambda text: 1 if _POS_RE.search(text.lower()) else 0
    )
    batch_rev["has_neg_kw"] = batch_rev["comments"].apply(
        lambda text: 1 if _NEG_RE.search(text.lower()) else 0
    )
    return batch_rev


def _aggregate(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-listing sentiment features from a scored reviews DataFrame.
    ``scored`` must be sorted by (listing_id, date DESC) so that head(N)
    within each group returns the N most recent reviews.
    """
    grp = scored.groupby("listing_id", sort=False)

    per_listing = grp.agg(
        avg_sentiment          =("compound",   "mean"),
        pct_positive_reviews   =("compound",   lambda x: (x > 0.05).mean()),
        pct_negative_reviews   =("compound",   lambda x: (x < -0.05).mean()),
        positive_keyword_count =("has_pos_kw", "sum"),
        negative_keyword_count =("has_neg_kw", "sum"),
    )

    # recent_sentiment — mean compound of the first RECENT_WINDOW rows per
    # listing; data is sorted date DESC so head(N) = N most recent reviews.
    recent = (
        scored.groupby("listing_id", sort=False)
        .head(RECENT_WINDOW)
        .groupby("listing_id")["compound"]
        .mean()
        .rename("recent_sentiment")
    )

    per_listing = per_listing.join(recent)
    per_listing["sentiment_trend"] = (
        per_listing["recent_sentiment"] - per_listing["avg_sentiment"]
    )
    per_listing.index.name = "id"
    return per_listing


# --------------------------------------------------------------------------- #
# Main function                                                                #
# --------------------------------------------------------------------------- #

def create_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer review sentiment features and left-join them onto ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an ``id`` column (Airbnb listing_id).

    Returns
    -------
    pd.DataFrame — same rows as ``df``, with 8 new columns appended.
    """
    df = df.copy()
    sia = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------------ #
    # Load reviews                                                         #
    # ------------------------------------------------------------------ #
    print("  Loading reviews.csv.gz ...")
    rev = pd.read_csv(
        DATA_DIR / "reviews.csv.gz",
        usecols=["listing_id", "date", "comments"],
        parse_dates=["date"],
    )
    rev["comments"] = rev["comments"].fillna("").astype(str)
    print(f"  Reviews: {len(rev):,} rows  |  {rev['listing_id'].nunique():,} unique listings")

    # ------------------------------------------------------------------ #
    # review_velocity — computed from full review history before filtering #
    # ------------------------------------------------------------------ #
    max_date     = rev["date"].max()
    cutoff_12m   = max_date - pd.DateOffset(years=1)
    total_counts = rev.groupby("listing_id").size().rename("total_reviews")
    recent_counts = (
        rev[rev["date"] >= cutoff_12m]
        .groupby("listing_id").size()
        .rename("recent_12m")
        .reindex(total_counts.index, fill_value=0)
    )
    velocity = (recent_counts / total_counts.clip(lower=1)).rename("review_velocity")
    print(f"  Velocity window: {cutoff_12m.date()} – {max_date.date()}")

    # ------------------------------------------------------------------ #
    # Filter to most recent N reviews per listing                         #
    # ------------------------------------------------------------------ #
    rev_sorted = (
        rev.sort_values(["listing_id", "date"], ascending=[True, False])
        .reset_index(drop=True)
    )
    top_n = rev_sorted.groupby("listing_id", sort=False).head(REVIEWS_PER_LISTING)
    top_n = top_n.sort_values(["listing_id", "date"], ascending=[True, False]).reset_index(drop=True)
    print(f"  Scoring {len(top_n):,} reviews "
          f"(top {REVIEWS_PER_LISTING} per listing) ...")

    # ------------------------------------------------------------------ #
    # VADER scoring in batches of BATCH_SIZE listings                     #
    # ------------------------------------------------------------------ #
    listing_ids  = top_n["listing_id"].unique()
    n_listings   = len(listing_ids)
    scored_parts: list[pd.DataFrame] = []
    listings_done = 0

    for batch_start in range(0, n_listings, BATCH_SIZE):
        batch_ids  = set(listing_ids[batch_start : batch_start + BATCH_SIZE])
        batch_rev  = top_n[top_n["listing_id"].isin(batch_ids)]
        scored_parts.append(_score_batch(batch_rev, sia))

        listings_done += len(batch_ids)
        # Print at every PROGRESS_EVERY boundary and at the very end
        prev_done = listings_done - len(batch_ids)
        crossed   = (listings_done // PROGRESS_EVERY) > (prev_done // PROGRESS_EVERY)
        if crossed or listings_done == n_listings:
            print(f"    {listings_done:,}/{n_listings:,} listings scored ...")

    scored = pd.concat(scored_parts, ignore_index=True)
    # Restore sort order (concat may shuffle batches relative to each other)
    scored = scored.sort_values(["listing_id", "date"], ascending=[True, False]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Per-listing aggregation                                             #
    # ------------------------------------------------------------------ #
    per_listing = _aggregate(scored)
    per_listing = per_listing.join(velocity.rename_axis("id"), how="left")
    per_listing["review_velocity"] = per_listing["review_velocity"].fillna(0.0)

    # ------------------------------------------------------------------ #
    # Merge onto df                                                        #
    # ------------------------------------------------------------------ #
    n_before = len(df)
    df = df.join(per_listing, on="id", how="left")
    assert len(df) == n_before, "join changed row count"

    # ------------------------------------------------------------------ #
    # Null handling for listings with no reviews                          #
    # ------------------------------------------------------------------ #
    missing_mask = df["avg_sentiment"].isna()
    n_missing    = int(missing_mask.sum())

    if n_missing > 0:
        print(f"  {n_missing:,} listings have no reviews — filling all features with 0")
        df[REVIEW_FEATURE_COLS] = df[REVIEW_FEATURE_COLS].fillna(0.0)
    else:
        print(f"  All listings matched in reviews (0 missing)")

    # Cast count columns to int now that nulls are gone
    df["positive_keyword_count"] = df["positive_keyword_count"].astype(int)
    df["negative_keyword_count"] = df["negative_keyword_count"].astype(int)

    return df


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Loading data/features.csv ...")
    df = pd.read_csv(DATA_DIR / "features.csv")
    print(f"  Shape before: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    print("Engineering review sentiment features ...")
    df_rev = create_review_features(df)

    print(f"\n  Shape after: {df_rev.shape[0]:,} rows × {df_rev.shape[1]} columns")
    print(f"  New columns: {len(REVIEW_FEATURE_COLS)}")

    # ------------------------------------------------------------------ #
    # Coverage and distribution                                           #
    # ------------------------------------------------------------------ #
    print(f"\n  {'Feature':<28}  {'Non-null':>8}  {'Coverage':>9}  "
          f"{'Min':>7}  {'Median':>7}  {'Max':>7}")
    print(f"  {'-'*68}")
    for col in REVIEW_FEATURE_COLS:
        non_null = df_rev[col].notna().sum()
        pct      = non_null / len(df_rev) * 100
        lo       = df_rev[col].min()
        med      = df_rev[col].median()
        hi       = df_rev[col].max()
        print(f"  {col:<28}  {non_null:>8,}  {pct:>8.1f}%  "
              f"{lo:>7.3f}  {med:>7.3f}  {hi:>7.3f}")

    # ------------------------------------------------------------------ #
    # Correlation with price                                              #
    # ------------------------------------------------------------------ #
    if "price" in df_rev.columns:
        print(f"\n  {'Feature':<28}  {'Corr with price':>16}  Direction")
        print(f"  {'-'*58}")
        corrs = df_rev[REVIEW_FEATURE_COLS + ["price"]].corr()["price"].drop("price")
        for col, r in corrs.sort_values(key=abs, ascending=False).items():
            direction = "↑ higher price" if r > 0 else "↓ lower price"
            print(f"  {col:<28}  {r:>16.4f}  {direction}")

    # ------------------------------------------------------------------ #
    # Overall sentiment summary                                           #
    # ------------------------------------------------------------------ #
    reviewed = df_rev[df_rev["avg_sentiment"] != 0]
    print(f"\n  Sentiment summary (listings with reviews: {len(reviewed):,}):")
    print(f"    Mean avg_sentiment    : {reviewed['avg_sentiment'].mean():.4f}")
    print(f"    Median avg_sentiment  : {reviewed['avg_sentiment'].median():.4f}")
    print(f"    % listings positive   : "
          f"{(reviewed['avg_sentiment'] > 0.05).mean()*100:.1f}%")
    print(f"    % listings negative   : "
          f"{(reviewed['avg_sentiment'] < -0.05).mean()*100:.1f}%")

    # ------------------------------------------------------------------ #
    # Save                                                                #
    # ------------------------------------------------------------------ #
    out = DATA_DIR / "features_rev.csv"
    df_rev.to_csv(out, index=False)
    print(f"\n  Saved → {out}")
