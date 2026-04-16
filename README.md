# Airbnb Seattle Price Prediction

Regression model to predict nightly listing prices for Seattle Airbnb hosts.

## Exploratory Data Analysis

**Dataset:** 5,927 listings × 55 features after cleaning and engineering.
35 numeric features, 20 categorical. Target: `price` (nightly rate, $).

**Key findings:**

- **Price is heavily right-skewed** (skew ~9, mean $591 vs median $145). Log-transforming the target brings skew to ~0.3 — all models train on log(price) and report metrics in dollar scale.
- **Capacity dominates pricing.** `accommodates` has the strongest correlation with price (~0.5). The relationship is non-linear: the per-guest premium accelerates above 4 guests.
- **Neighbourhood prestige outweighs raw proximity.** `neighbourhood_avg_price` correlates with price more strongly than `distance_to_downtown`. Waterfront and view premiums override downtown distance for some areas (Magnolia, Portage Bay).
- **Room type creates near-disjoint price distributions.** Entire home/apt lists 2–3× higher than private rooms with minimal overlap — `room_type` is a critical categorical feature.

**Modeling implications:**

- `estimated_revenue_l365d` and `estimated_occupancy_l365d` are outcome variables (derived from actual bookings) — excluded to keep the model usable by new hosts with no booking history.
- Listings with `days_since_last_review = 9999` (never reviewed / dormant) are systematically harder to predict and should surface a lower-confidence flag in the host-facing tool.
