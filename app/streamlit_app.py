"""
Seattle Airbnb Price Predictor — Streamlit portfolio app.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"

# ─── Color theme ─────────────────────────────────────────────────────────────

RED      = "#FF5A5F"   # Airbnb red — primary accent
TEAL     = "#00A699"   # Teal — secondary accent
NAVY     = "#1a1a2e"   # Dark navy — headers
CARD_BG  = "#f7f7f7"   # Light gray — card backgrounds

# ─── Domain constants ─────────────────────────────────────────────────────────

NEIGHBOURHOOD_GROUPS = [
    "Beacon Hill", "Ballard", "Capitol Hill", "Cascade", "Central Area",
    "Delridge", "Downtown", "Interbay", "Lake City", "Magnolia",
    "Northgate", "Other neighborhoods", "Queen Anne", "Rainier Valley",
    "Seward Park", "University District", "West Seattle",
]
ROOM_TYPES   = ["Entire home/apt", "Private room", "Shared room"]
PRICE_CAP    = 500
RANDOM_STATE = 42
_LOG_CLIP    = (np.log(1), np.log(PRICE_CAP * 2))

BUCKET_EDGES  = [0,   100,  200,  500]
BUCKET_ERRORS = [17,   25,   57]

# ─── Page config (must be first Streamlit call) ───────────────────────────────

st.set_page_config(
    page_title="Seattle Airbnb Price Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(f"""
<style>
[data-testid="stSidebar"] {{
    background-color: {NAVY} !important;
}}
[data-testid="stSidebar"] * {{
    color: #e2e8f0 !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.95rem;
    padding: 4px 0;
}}
.kpi-card {{
    background: {CARD_BG};
    border-left: 4px solid {RED};
    border-radius: 8px;
    padding: 18px 20px;
    margin: 6px 0;
}}
.kpi-value {{
    font-size: 2rem;
    font-weight: 800;
    color: {RED};
    line-height: 1.1;
}}
.kpi-label {{
    font-size: 0.82rem;
    color: #666;
    margin-top: 4px;
}}
.page-hero {{
    background: linear-gradient(135deg, {NAVY} 0%, #16213e 100%);
    color: white;
    padding: 36px 32px;
    border-radius: 12px;
    margin-bottom: 28px;
}}
.page-hero h1 {{
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    line-height: 1.2;
}}
.page-hero .tagline {{
    font-size: 1.05rem;
    color: #a0aec0;
}}
.accent {{ color: {RED}; }}
.section-title {{
    color: {NAVY};
    font-size: 1.35rem;
    font-weight: 700;
    margin: 28px 0 12px 0;
    border-bottom: 2px solid {RED};
    padding-bottom: 6px;
}}
.badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 3px 2px;
    font-weight: 600;
}}
.badge-default {{ background:{NAVY};     color:#fff; }}
.badge-red     {{ background:{RED};      color:#fff; }}
.badge-teal    {{ background:{TEAL};     color:#fff; }}
.callout {{
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 10px 0;
    font-size: 0.91rem;
    line-height: 1.5;
    color: #1a1a2e !important;
}}
.callout * {{ color: #1a1a2e !important; }}
.callout-amber {{ background:#fff8e1; border-left:4px solid #f59e0b; }}
.callout-blue  {{ background:#eff6ff; border-left:4px solid #3b82f6; }}
.callout-green {{ background:#f0fdf4; border-left:4px solid {TEAL};  }}
.callout-red   {{ background:#fff1f2; border-left:4px solid {RED};   }}
.tier-chip {{
    display:inline-block;
    padding:5px 14px;
    border-radius:20px;
    font-weight:700;
    font-size:0.9rem;
}}
.tier-high   {{ background:#d1fae5; color:#065f46; }}
.tier-medium {{ background:#fef3c7; color:#92400e; }}
.tier-low    {{ background:#fee2e2; color:#991b1b; }}
.footer {{
    text-align:center;
    color:#999;
    font-size:0.77rem;
    padding:24px 0 8px;
    border-top:1px solid #eee;
    margin-top:40px;
}}
</style>
""", unsafe_allow_html=True)


# ─── Layout helpers ───────────────────────────────────────────────────────────

def _hero(title: str, tagline: str) -> None:
    st.markdown(
        f'<div class="page-hero">'
        f'<h1 style="font-size:2.5rem;color:#ffffff;font-weight:700;margin:0 0 8px 0;">{title}</h1>'
        f'<div class="tagline">{tagline}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _section(title: str) -> None:
    st.markdown(
        f"<h2 style='color:#FFFFFF; font-size:1.6rem; font-weight:600; "
        f"margin-bottom:0.2rem;'>{title}</h2>"
        f"<hr style='border:1px solid {RED}; margin-top:0;'>",
        unsafe_allow_html=True,
    )

def _callout(text: str, kind: str = "amber") -> None:
    st.markdown(f'<div class="callout callout-{kind}">{text}</div>', unsafe_allow_html=True)

def _kpi(value: str, label: str) -> None:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _footer() -> None:
    st.markdown(
        '<div class="footer">'
        'Data: Inside Airbnb Seattle (Sep 2025) &nbsp;|&nbsp; '
        'Model: XGBoost &nbsp;|&nbsp; MAE $30.66'
        '</div>',
        unsafe_allow_html=True,
    )

def _plotly_defaults(fig, height: int = 420) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        title_font_color="#ffffff",
        height=height,
        margin={"t": 48, "b": 32, "l": 8, "r": 8},
        xaxis=dict(gridcolor="#333333", color="#ffffff", linecolor="#444444"),
        yaxis=dict(gridcolor="#333333", color="#ffffff", linecolor="#444444"),
        legend=dict(font=dict(color="#ffffff"), bgcolor="rgba(0,0,0,0)"),
    )


# ─── Data / model loading ─────────────────────────────────────────────────────

@st.cache_data
def load_features() -> pd.DataFrame:
    path = DATA_DIR / "features.csv"
    return pd.read_csv(path) if path.exists() else _demo_features()

@st.cache_data
def load_cleaned() -> pd.DataFrame:
    path = DATA_DIR / "cleaned.csv"
    return pd.read_csv(path) if path.exists() else load_features()

@st.cache_resource
def load_model() -> dict | None:
    import joblib
    path = MODELS_DIR / "production_model.pkl"
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def _shap_explainer():
    artifact = load_model()
    if artifact is None:
        return None
    try:
        import shap
        return shap.TreeExplainer(artifact["pipeline"].named_steps["model"])
    except Exception:
        return None

@st.cache_data
def _model_defaults() -> dict[str, float]:
    """
    Return median of every feature in the model's 150-col training matrix.
    Used to fill any feature not collected from the prediction form.
    """
    artifact = load_model()
    if artifact is None:
        return {}
    try:
        from models.baseline import prepare_features
        df = load_features()
        df_cap = df[df["price"] <= PRICE_CAP].reset_index(drop=True)
        X, _ = prepare_features(df_cap)
        X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]
        trained = artifact["feature_cols"]
        for c in trained:
            if c not in X.columns:
                X[c] = 0
        X = X[[c for c in trained if c in X.columns]]
        return X.median().to_dict()
    except Exception:
        return {}

@st.cache_data
def _neighbourhood_lookup() -> dict[str, dict]:
    """Per-neighbourhood-group statistics for filling prediction features."""
    df = load_features()
    agg = df.groupby("neighbourhood_group").agg(
        avg_price       =("neighbourhood_avg_price",       "median"),
        grp_avg_price   =("neighbourhood_group_avg_price", "median"),
        distance        =("distance_to_downtown",          "median"),
        lat             =("latitude",                      "median"),
        lon             =("longitude",                     "median"),
    ).to_dict(orient="index")
    return agg

@st.cache_data
def _load_geojson() -> dict:
    """Load neighbourhoods.geojson, returning an empty FeatureCollection on failure."""
    import json
    path = DATA_DIR / "neighbourhoods.geojson"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"type": "FeatureCollection", "features": []}

@st.cache_data
def _neighbourhood_prices() -> pd.DataFrame:
    """
    Median nightly price and listing count per neighbourhood_cleansed,
    keyed to the 'neighbourhood' field used in the GeoJSON.

    Median is used rather than mean so that a single $5,000/night luxury
    listing does not skew the colour of an otherwise mid-range neighbourhood.

    Falls back to synthetic data seeded from GeoJSON names when
    features.csv is absent (demo mode), so the choropleth always renders.
    """
    import json

    geo_path     = DATA_DIR / "neighbourhoods.geojson"
    features_csv = DATA_DIR / "features.csv"

    if features_csv.exists():
        df  = load_features()
        col = ("neighbourhood_cleansed"
               if "neighbourhood_cleansed" in df.columns
               else "neighbourhood_group")
        return (
            df.groupby(col)
              .agg(median_price=("price", "median"), n=("price", "count"))
              .reset_index()
              .rename(columns={col: "neighbourhood"})
        )

    # Demo mode — generate plausible per-polygon prices keyed to GeoJSON names
    if geo_path.exists():
        with open(geo_path) as f:
            gj = json.load(f)
        names = [feat["properties"]["neighbourhood"] for feat in gj["features"]]
    else:
        names = NEIGHBOURHOOD_GROUPS  # last-resort stub
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "neighbourhood": names,
        "median_price":  rng.uniform(90, 320, len(names)),
        "n":             rng.integers(10, 250, len(names)),
    })


# ─── Demo data (no files present) ────────────────────────────────────────────

def _demo_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n   = 600
    ng  = rng.choice(
        ["Downtown", "Capitol Hill", "Queen Anne", "Ballard", "Other neighborhoods"],
        size=n, p=[0.12, 0.18, 0.15, 0.10, 0.45],
    )
    rt  = rng.choice(ROOM_TYPES, size=n, p=[0.82, 0.16, 0.02])
    bed = rng.integers(1, 5, size=n).astype(float)
    acc = (bed * 2).clip(1, 12)
    ngp = np.array([{"Downtown":245,"Capitol Hill":184,"Queen Anne":228,
                     "Ballard":177,"Other neighborhoods":188}.get(g,170) for g in ng], float)
    prc = (bed * 50 + rng.normal(50, 30, n)).clip(15, 490)
    return pd.DataFrame({
        "id": np.arange(n, dtype=float),
        "price": prc,
        "neighbourhood_group": ng,
        "neighbourhood_cleansed": ng,
        "room_type": rt,
        "accommodates": acc,
        "bedrooms": bed,
        "bathrooms": rng.choice([1.0, 1.5, 2.0], size=n),
        "amenities_count": rng.integers(10, 60, size=n).astype(float),
        "is_superhost": rng.integers(0, 2, size=n).astype(float),
        "has_view": rng.integers(0, 2, size=n).astype(float),
        "has_hot_tub": rng.integers(0, 2, size=n).astype(float),
        "has_parking": rng.integers(0, 2, size=n).astype(float),
        "days_since_last_review": rng.integers(0, 2000, size=n).astype(float),
        "number_of_reviews": rng.integers(0, 200, size=n).astype(float),
        "review_scores_rating": rng.uniform(3.5, 5.0, size=n),
        "neighbourhood_avg_price": ngp,
        "neighbourhood_group_avg_price": ngp,
        "distance_to_downtown": rng.uniform(0.5, 15.0, size=n),
        "latitude": rng.uniform(47.50, 47.73, size=n),
        "longitude": rng.uniform(-122.43, -122.23, size=n),
        "avg_sentiment": rng.uniform(-0.2, 0.8, size=n),
        "review_velocity": rng.uniform(0, 5, size=n),
        "consecutive_blocked_rate": rng.uniform(0, 1, size=n),
        "peak_availability_rate": rng.uniform(0, 1, size=n),
        "off_availability_rate": rng.uniform(0, 1, size=n),
    })


# ─── Confidence tier logic (mirrors predict.py) ───────────────────────────────

def _confidence_tiers(dslr: np.ndarray, n_rev: np.ndarray) -> np.ndarray:
    dslr  = np.asarray(dslr,  dtype=float)
    n_rev = np.asarray(n_rev, dtype=float)
    is_low    = (dslr > 1000) | (n_rev == 0)
    is_medium = ((dslr >= 365) & (dslr <= 1000)) | ((n_rev >= 1) & (n_rev <= 5))
    is_high   = (dslr < 365) & (n_rev > 5)
    tiers = np.where(is_high, "HIGH", np.where(is_medium, "MEDIUM", "LOW"))
    return np.where(is_low, "LOW", tiers)

def _bucket_error(price: float) -> float:
    import bisect
    idx = bisect.bisect_right(BUCKET_EDGES[1:], price)
    return float(BUCKET_ERRORS[min(idx, len(BUCKET_ERRORS) - 1)])


# ─── Sidebar / navigation ─────────────────────────────────────────────────────

_NAV = {
    "Overview":         "overview",
    "Explore the Data": "eda",
    "Model Results":    "model_results",
    "Try It Yourself":  "predict",
    "How I Built This": "how_built",
}

def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f'<div style="color:{RED};font-size:1.2rem;font-weight:800;padding:10px 0 4px;">'
            f'Seattle Airbnb Predictor</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        choice = st.radio("Navigation", list(_NAV.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.markdown(
            '<div style="color:#718096;font-size:0.72rem;padding-top:4px;">'
            'Seattle · Inside Airbnb · Sep 2025<br>'
            'Model: XGBoost · MAE $30.66</div>',
            unsafe_allow_html=True,
        )
    return _NAV[choice]


# ═══════════════════════════════════════════════════════════════════════════════
# Page 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════

def _page_overview() -> None:
    _hero(
        'Seattle Airbnb <span class="accent">Price Predictor</span>',
        "Helping hosts set the right nightly price using machine learning",
    )

    st.markdown(
        "This project trains a machine-learning pipeline on **5,927 Seattle Airbnb listings** "
        "to predict nightly prices within an average error of **$30.66**. "
        "It blends structured listing data with NLP features extracted from listing descriptions, "
        "calendar availability patterns, and VADER sentiment scores from guest reviews "
        "to give hosts an accurate, explainable price recommendation — complete with a "
        "confidence tier and price range."
    )

    _section("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1: _kpi("5,927",  "Listings Analyzed")
    with c2: _kpi("150",    "Features Engineered")
    with c3: _kpi("$30.66", "Final Model MAE")
    with c4: _kpi("14.5%",  "Improvement vs Cleaned Baseline ($35.87 → $30.66)")

    _section("Tech Stack")
    badges = [
        ("Python",      "default"), ("XGBoost",    "red"),  ("SHAP",       "teal"),
        ("FastAPI",     "default"), ("Streamlit",  "red"),  ("MLflow",     "teal"),
        ("Pandas",      "default"), ("NLTK",       "teal"), ("Scikit-learn","default"),
        ("Optuna",      "red"),     ("Plotly",     "teal"), ("VADER",      "default"),
    ]
    st.markdown(
        "".join(
            f'<span class="badge badge-{kind}">{name}</span>'
            for name, kind in badges
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _callout(
        "The full model blends <b>19 NLP features</b> (binary flags from listing descriptions "
        "and amenities text), <b>7 calendar availability features</b> (seasonal booking patterns, "
        "consecutive blocked periods), and <b>6 review sentiment features</b> using VADER — "
        "all on top of 118 structured listing features.",
        "blue",
    )

    _section("Feature Groups at a Glance")
    fa, fb, fc = st.columns(3)
    with fa:
        st.markdown(f"**NLP Features (19)**")
        for f in ["has_view", "is_luxury", "has_fireplace", "has_hot_tub",
                  "has_private_entrance", "amenity_has_pool", "…"]:
            st.markdown(f"&nbsp;&nbsp;• {f}")
    with fb:
        st.markdown(f"**Calendar Features (7)**")
        for f in ["peak_availability_rate", "off_availability_rate",
                  "availability_gap", "consecutive_blocked_rate",
                  "has_dynamic_minimum", "…"]:
            st.markdown(f"&nbsp;&nbsp;• {f}")
    with fc:
        st.markdown(f"**Sentiment Features (6)**")
        for f in ["avg_sentiment", "pct_negative_reviews",
                  "positive_keyword_count", "negative_keyword_count",
                  "sentiment_trend", "review_velocity"]:
            st.markdown(f"&nbsp;&nbsp;• {f}")

    _footer()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 2 — Explore the Data
# ═══════════════════════════════════════════════════════════════════════════════

def _page_eda() -> None:
    _hero(
        'Explore the <span class="accent">Data</span>',
        "5,927 Seattle listings · Inside Airbnb · Sep 2025",
    )
    df = load_features()

    # ── Price distribution ────────────────────────────────────────────────
    _section("Price Distribution")
    r_col, chart_col = st.columns([1, 4])
    with r_col:
        view = st.radio("View", ["Raw ($)", "Log-transformed"], key="dist_toggle")
    with chart_col:
        if view == "Raw ($)":
            df_plot = df[df["price"] <= PRICE_CAP]
            fig = px.histogram(
                df_plot, x="price", nbins=70,
                color_discrete_sequence=[RED],
                title="Nightly Price Distribution (capped at $500)",
                labels={"price": "Nightly Price ($)", "count": "Listings"},
            )
            fig.add_vline(x=df_plot["price"].median(), line_dash="dash",
                          line_color=TEAL,
                          annotation_text=f'Median ${df_plot["price"].median():.0f}',
                          annotation_font_color=TEAL)
        else:
            lp = np.log(df[df["price"] > 0]["price"])
            fig = px.histogram(
                x=lp, nbins=70,
                color_discrete_sequence=[TEAL],
                title="log(Price) Distribution — Near-Normal After Transform",
                labels={"x": "log(Nightly Price)", "count": "Listings"},
            )
            fig.add_vline(x=lp.mean(), line_dash="dash", line_color=RED,
                          annotation_text=f"Mean {lp.mean():.2f}",
                          annotation_font_color=RED)
        _plotly_defaults(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Choropleth map ────────────────────────────────────────────────────
    _section("Price by Neighbourhood")
    geojson   = _load_geojson()
    nb_prices = _neighbourhood_prices()

    # Pre-format hover columns so tooltips look clean
    nb_prices = nb_prices.copy()
    nb_prices["Median Price"] = nb_prices["median_price"].apply(lambda x: f"${x:.0f}")
    nb_prices["Listings"]     = nb_prices["n"].astype(int)
    # Small-sample warning: flag neighbourhoods with fewer than 10 listings
    nb_prices["Note"] = nb_prices["n"].apply(
        lambda n: "Note: small sample size may affect median" if n < 10 else ""
    )

    if geojson["features"]:
        # Only include Note column in hover when it has content — always pass
        # it; Plotly shows the row only when the value is non-empty.
        fig_map = px.choropleth_mapbox(
            nb_prices,
            geojson=geojson,
            locations="neighbourhood",
            featureidkey="properties.neighbourhood",
            color="median_price",
            color_continuous_scale=[
                [0.00, "#ffffd4"],
                [0.25, "#fed976"],
                [0.50, "#fd8d3c"],
                [0.75, "#e31a1c"],
                [1.00, "#800026"],
            ],
            range_color=[80, 350],   # cap at $350 — outliers hit dark red
            hover_name="neighbourhood",
            hover_data={
                "Median Price":  True,
                "Listings":      True,
                "Note":          True,
                "median_price":  False,   # hide raw float column
                "n":             False,
                "neighbourhood": False,
            },
            mapbox_style="carto-darkmatter",
            center={"lat": 47.6062, "lon": -122.3321},
            zoom=10,
            opacity=0.75,
        )
        fig_map.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            margin={"t": 0, "b": 0, "l": 0, "r": 0},
            coloraxis_colorbar=dict(
                title=dict(
                    text="Median Nightly<br>Price ($)",
                    font=dict(color="#ffffff", size=12),
                ),
                tickprefix="$",
                tickfont=dict(color="#ffffff", size=11),
                bgcolor="rgba(0,0,0,0.4)",
                outlinecolor="rgba(0,0,0,0)",
                len=0.75,
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            "Median price per neighbourhood. Color scale capped at $350 — "
            "outlier neighbourhoods (e.g. Belltown, Central Business District) appear in deep red."
        )
    else:
        st.info("neighbourhoods.geojson not found — place it in data/ to enable the map.")

    # ── Median price by neighbourhood group ──────────────────────────────
    _section("Median Nightly Price by Neighbourhood Group")
    ng_col = "neighbourhood_group" if "neighbourhood_group" in df.columns else "neighbourhood_cleansed"
    ng_med = (df.groupby(ng_col)["price"]
                .median()
                .sort_values()
                .reset_index()
                .rename(columns={ng_col: "neighbourhood_group", "price": "median_price"}))

    fig2 = px.bar(
        ng_med, x="median_price", y="neighbourhood_group", orientation="h",
        color="median_price",
        color_continuous_scale=[[0, TEAL], [1, RED]],
        text="median_price",
        labels={"median_price": "Median Nightly Price ($)", "neighbourhood_group": ""},
        title="Median Nightly Price by Neighbourhood Group",
    )
    fig2.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
    fig2.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 500]),
    )
    _plotly_defaults(fig2, height=520)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Room type vs price ────────────────────────────────────────────────
    _section("Room Type vs Price")
    if "room_type" in df.columns:
        df_box = df[df["price"] <= PRICE_CAP]
        fig3 = px.box(
            df_box, x="room_type", y="price", color="room_type",
            color_discrete_map={
                "Entire home/apt": RED,
                "Private room":    TEAL,
                "Shared room":     "#f59e0b",
                "Hotel room":      NAVY,
            },
            title="Price Distribution by Room Type (listings ≤ $500)",
            labels={"price": "Nightly Price ($)", "room_type": "Room Type"},
        )
        fig3.update_layout(showlegend=False)
        _plotly_defaults(fig3, height=380)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Top / bottom neighbourhoods ───────────────────────────────────────
    _section("Most Expensive & Cheapest Neighbourhoods (Top 10 each)")
    nb_col = ("neighbourhood_cleansed"
              if "neighbourhood_cleansed" in df.columns
              else "neighbourhood_group")
    nb_avg = (df[df["price"] <= PRICE_CAP]
              .groupby(nb_col)["price"]
              .mean()
              .sort_values())
    top10 = nb_avg.tail(10).reset_index().rename(columns={nb_col: "neighbourhood", "price": "avg_price"})
    bot10 = nb_avg.head(10).reset_index().rename(columns={nb_col: "neighbourhood", "price": "avg_price"})

    ca, cb = st.columns(2)
    with ca:
        fig_top = px.bar(
            top10, x="avg_price", y="neighbourhood", orientation="h",
            color_discrete_sequence=[RED],
            title="10 Most Expensive Neighbourhoods",
            text="avg_price",
            labels={"avg_price": "Avg Price ($)", "neighbourhood": ""},
        )
        fig_top.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
        _plotly_defaults(fig_top, height=380)
        st.plotly_chart(fig_top, use_container_width=True)
    with cb:
        fig_bot = px.bar(
            bot10, x="avg_price", y="neighbourhood", orientation="h",
            color_discrete_sequence=[TEAL],
            title="10 Cheapest Neighbourhoods",
            text="avg_price",
            labels={"avg_price": "Avg Price ($)", "neighbourhood": ""},
        )
        fig_bot.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
        fig_bot.update_layout(yaxis={"categoryorder": "total descending"})
        _plotly_defaults(fig_bot, height=380)
        st.plotly_chart(fig_bot, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────────────
    _section("Feature Correlation Heatmap (Top 20 vs Price)")
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=["id", "host_id"], errors="ignore"
    )
    top_feats = (
        num_df.corr()["price"]
        .abs()
        .sort_values(ascending=False)
        .head(21)
        .index.drop("price", errors="ignore")[:20]
    )
    corr_mat = num_df[list(top_feats)].corr()
    labels   = [c.replace("_", " ") for c in corr_mat.columns]

    fig_heat = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=labels, y=labels,
        colorscale=[[0, TEAL], [0.5, "#1a1a2e"], [1, RED]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_mat.values, 2),
        texttemplate="%{text}",
        textfont=dict(color="#ffffff", size=8),
        hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>",
    ))
    fig_heat.update_layout(
        title="Correlation Heatmap — Top 20 Features",
        xaxis={"tickangle": -45, "tickfont": {"size": 9, "color": "#ffffff"},
               "color": "#ffffff"},
        yaxis={"tickfont": {"size": 9, "color": "#ffffff"}, "color": "#ffffff"},
    )
    _plotly_defaults(fig_heat, height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Key findings ──────────────────────────────────────────────────────
    _section("Key EDA Findings")
    st.error(
        "Median price is $145/night but mean is $591 due to extreme luxury outliers "
        "(max $50,039). Capping at $500 covers 96.3% of listings and removes the distortion "
        "that would otherwise inflate the loss function."
    )
    st.warning(
        "75% of summer dates are blocked — strong seasonal demand signal. "
        "Calendar availability features capture whether a host restricts peak dates "
        "and how many consecutive nights are blocked."
    )
    st.success(
        "Superhosts command a measurable price premium. "
        "Listings with is_superhost=1 average ~11% higher prices in the same "
        "neighbourhood group — consistent across all room types."
    )

    _footer()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 3 — Model Results
# ═══════════════════════════════════════════════════════════════════════════════

def _page_model_results() -> None:
    _hero(
        'Model <span class="accent">Results</span>',
        "Four models tracked. One production winner.",
    )

    # ── Comparison table ──────────────────────────────────────────────────
    _section("Model Comparison")
    compare = pd.DataFrame([
        {"Model": "LinearRegression",
         "MAE": "$35.87", "RMSE": "$52.40", "R²": "0.615",
         "Δ MAE vs Baseline": "—",
         "Notes": "Cleaned baseline",            "_prod": False},
        {"Model": "XGBoost tuned",
         "MAE": "$31.27", "RMSE": "$47.04", "R²": "0.690",
         "Δ MAE vs Baseline": "−$4.60",
         "Notes": "Optuna 30 trials",            "_prod": False},
        {"Model": "XGBoost + NLP",
         "MAE": "$30.92", "RMSE": "$46.27", "R²": "0.700",
         "Δ MAE vs Baseline": "−$4.95",
         "Notes": "+19 NLP features",            "_prod": False},
        {"Model": "XGBoost full (PROD)",
         "MAE": "$30.66", "RMSE": "$45.82", "R²": "0.706",
         "Δ MAE vs Baseline": "−$5.21 (−14.5%)",
         "Notes": "+calendar +sentiment",        "_prod": True},
    ])
    show = compare.drop(columns="_prod")

    def _row_style(row):
        if compare.loc[row.name, "_prod"]:
            return ["background-color:rgba(255,90,95,0.25); color:#FFFFFF; font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        show.style.apply(_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    _callout(
        "<b>XGBoost full</b> is the production model. "
        "MAE improved by <b>$5.21 (14.5%)</b> over the LinearRegression baseline — "
        "from $35.87 → $30.66 — across a held-out 20% test set.",
        "green",
    )

    # ── Why XGBoost won ────────────────────────────────────────────────────
    _section("Why XGBoost Won")
    _callout(
        "<b>Best MAE ($30.66)</b> across all candidates. "
        "Trains in <b>5.5s vs 12.4s</b> for Random Forest (same hardware). "
        "Provides <b>native SHAP support</b> via TreeExplainer for per-prediction explainability. "
        "Handles <b>mixed feature types</b> (binary flags, continuous, one-hot encoded) "
        "without normalisation. Robust to the long price tail via log-target transformation.",
        "blue",
    )

    # ── SHAP feature importance ────────────────────────────────────────────
    _section("SHAP Feature Importance — Top 15")
    shap_rows = pd.DataFrame({
        "feature": [
            "bedrooms", "accommodates", "distance_to_downtown",
            "bathrooms", "amenities_count", "neighbourhood_avg_price",
            "host_quality_score", "avg_sentiment", "has_view",
            "review_scores_rating", "is_superhost", "consecutive_blocked_rate",
            "has_hot_tub", "days_since_last_review", "positive_keyword_count",
        ],
        "importance_pct": [
            13.8, 11.3, 5.8, 4.5, 4.3, 4.2, 3.5, 2.0, 1.2,
            1.1, 1.0, 1.0, 0.9, 0.8, 0.7,
        ],
        "group": [
            "Structural", "Structural", "Location",
            "Structural", "Amenities", "Location",
            "Host", "Sentiment", "NLP",
            "Reviews", "Host", "Calendar",
            "NLP", "Reviews", "NLP",
        ],
    }).sort_values("importance_pct")

    color_map = {
        "Structural": NAVY,    "Location": RED,      "Amenities": TEAL,
        "Host":       "#7c3aed","Sentiment": "#f59e0b","NLP": "#ec4899",
        "Calendar":   "#0ea5e9","Reviews":   "#64748b",
    }

    fig_shap = px.bar(
        shap_rows, x="importance_pct", y="feature", orientation="h",
        color="group",
        color_discrete_map=color_map,
        text="importance_pct",
        labels={"importance_pct": "Mean |SHAP| share (%)", "feature": "", "group": "Feature Group"},
        title="Top 15 Features — Mean Absolute SHAP Value (% of total)",
    )
    fig_shap.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    _plotly_defaults(fig_shap, height=500)
    st.plotly_chart(fig_shap, use_container_width=True)

    _callout(
        "<b>bedrooms</b> + <b>accommodates</b> together explain 25.1% of predictions. "
        "<b>avg_sentiment</b> (2.0%, rank #8) is the highest-ranked new feature from the "
        "NLP / sentiment expansion — confirming that review tone is a genuine price signal.",
    )

    # ── MAE by price bucket ────────────────────────────────────────────────
    _section("MAE by Price Bucket")
    bkt = pd.DataFrame({
        "Bucket":  ["$0–100",  "$100–200", "$200–500"],
        "MAE ($)": [16.60,     24.57,      56.61],
        "MAPE":    [23.8,      16.9,       19.5],
        "N":       [2317,      2104,       1506],
    })

    bc1, bc2 = st.columns(2)
    with bc1:
        fig_mae = px.bar(
            bkt, x="Bucket", y="MAE ($)",
            color="Bucket",
            color_discrete_sequence=[TEAL, RED, NAVY],
            text="MAE ($)",
            title="MAE per Price Bucket",
            labels={"MAE ($)": "MAE ($)"},
        )
        fig_mae.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
        fig_mae.update_layout(showlegend=False, yaxis_range=[0, 72])
        _plotly_defaults(fig_mae, height=360)
        st.plotly_chart(fig_mae, use_container_width=True)
    with bc2:
        fig_mape = px.bar(
            bkt, x="Bucket", y="MAPE",
            color="Bucket",
            color_discrete_sequence=[TEAL, RED, NAVY],
            text="MAPE",
            title="MAPE per Price Bucket (%)",
            labels={"MAPE": "MAPE (%)"},
        )
        fig_mape.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_mape.update_layout(showlegend=False, yaxis_range=[0, 30])
        _plotly_defaults(fig_mape, height=360)
        st.plotly_chart(fig_mape, use_container_width=True)

    _callout(
        "Mid-range listings ($100–200) are the <b>most predictable at 16.9% MAPE</b>. "
        "Premium listings ($200–500) have wider error bands due to unobserved quality signals "
        "— photography, interior design, and view quality — that are invisible to the model.",
    )

    # ── Residual plot (live or demo) ──────────────────────────────────────
    _section("Actual vs Predicted — Test Set")
    artifact = load_model()
    if artifact is not None:
        try:
            from models.baseline import prepare_features
            from sklearn.model_selection import train_test_split

            df = load_features()
            df_cap = df[df["price"] <= PRICE_CAP].reset_index(drop=True)
            X, y = prepare_features(df_cap)
            X.columns = [re.sub(r"[\[\]<>,]", "_", c) for c in X.columns]
            _, X_t, _, y_t = train_test_split(X, y, test_size=0.2,
                                              random_state=RANDOM_STATE)
            trained = artifact["feature_cols"]
            for c in trained:
                if c not in X_t.columns:
                    X_t[c] = 0
            X_t = X_t[[c for c in trained if c in X_t.columns]]

            pipeline = artifact["pipeline"]
            preds    = np.exp(np.clip(pipeline.predict(X_t), *_LOG_CLIP))
            actual   = np.exp(y_t.values)
            dslr_    = (df_cap.loc[y_t.index, "days_since_last_review"].values
                        if "days_since_last_review" in df_cap.columns
                        else np.full(len(y_t), 9999))
            nrev_    = (df_cap.loc[y_t.index, "number_of_reviews"].values
                        if "number_of_reviews" in df_cap.columns
                        else np.zeros(len(y_t)))
            tiers    = _confidence_tiers(dslr_, nrev_)

            res_df = pd.DataFrame({
                "Actual ($)":    actual,
                "Predicted ($)": preds,
                "Tier":          tiers,
                "|Error| ($)":   np.abs(preds - actual),
            })
            fig_res = px.scatter(
                res_df, x="Actual ($)", y="Predicted ($)", color="Tier",
                color_discrete_map={"HIGH": TEAL, "MEDIUM": "#f59e0b", "LOW": RED},
                title="Actual vs Predicted Prices — 20% Test Set",
                opacity=0.45,
                hover_data={"|Error| ($)": ":.1f"},
            )
            max_val = min(res_df["Actual ($)"].max(), PRICE_CAP) + 20
            fig_res.add_shape(
                type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                line={"color": NAVY, "dash": "dash", "width": 1.5},
            )
            _plotly_defaults(fig_res, height=480)
            st.plotly_chart(fig_res, use_container_width=True)
        except Exception as exc:
            st.info(f"Live residual plot unavailable: {exc}")
    else:
        # Demo scatter
        rng = np.random.default_rng(1)
        n   = 350
        act = rng.uniform(20, 480, n)
        prd = act + rng.normal(0, 28, n)
        trs = _confidence_tiers(rng.integers(0, 1500, n).astype(float),
                                rng.integers(0, 100, n).astype(float))
        demo_res = pd.DataFrame({"Actual ($)": act, "Predicted ($)": prd, "Tier": trs})
        fig_d = px.scatter(
            demo_res, x="Actual ($)", y="Predicted ($)", color="Tier",
            color_discrete_map={"HIGH": TEAL, "MEDIUM": "#f59e0b", "LOW": RED},
            title="Actual vs Predicted Prices — Demo Data",
            opacity=0.45,
        )
        fig_d.add_shape(type="line", x0=0, y0=0, x1=500, y1=500,
                        line={"color": NAVY, "dash": "dash", "width": 1.5})
        _plotly_defaults(fig_d, height=480)
        st.plotly_chart(fig_d, use_container_width=True)

    _footer()


# ═══════════════════════════════════════════════════════════════════════════════
# Page 4 — Try It Yourself
# ═══════════════════════════════════════════════════════════════════════════════

def _page_predict() -> None:
    _hero(
        'Try It <span class="accent">Yourself</span>',
        "Enter your listing details and get an instant price prediction with confidence tier",
    )

    artifact    = load_model()
    mdl_defaults = _model_defaults()
    ng_lookup   = _neighbourhood_lookup()

    _section("Listing Details")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Location & Type**")
            neighbourhood = st.selectbox(
                "Neighbourhood Group", NEIGHBOURHOOD_GROUPS,
                index=NEIGHBOURHOOD_GROUPS.index("Capitol Hill"),
            )
            room_type = st.selectbox("Room Type", ROOM_TYPES)
            st.markdown("**Size**")
            accommodates = st.slider("Accommodates (guests)", 1, 16, 4)
            bedrooms     = st.slider("Bedrooms",               0, 10, 2)
            bathrooms    = st.slider("Bathrooms",              0, 10, 1)

        with c2:
            st.markdown("**Amenities & Host**")
            amenities_cnt = st.slider("Amenities Count", 0, 100, 35)
            is_superhost  = st.checkbox("Superhost", value=False)
            st.markdown("**Special Features**")
            has_view      = st.checkbox("View (city / water / mountains)")
            has_hot_tub   = st.checkbox("Hot tub / Jacuzzi")
            has_parking   = st.checkbox("Parking included")

        with c3:
            st.markdown("**Review Activity**")
            dslr = st.slider(
                "Days Since Last Review",
                min_value=0, max_value=9999, value=30,
                help="Use 9999 for new or never-reviewed listings",
            )
            n_reviews = st.slider("Number of Reviews (lifetime)", 0, 500, 40)
            st.markdown("---")
            st.markdown(
                f'<div style="font-size:0.8rem;color:#666;padding-top:4px;">'
                f'Features not shown here are filled with <b>training-set medians</b> '
                f'so the model always receives all 150 features.</div>',
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button(
            "Get Price Prediction",
            use_container_width=True,
        )

    if not submitted:
        st.info(
            "Configure your listing above and click **Get Price Prediction**. "
            "The model will return a price estimate, confidence tier, price range, "
            "and a SHAP feature explanation."
        )
        _footer()
        return

    # ── Run prediction ────────────────────────────────────────────────────
    if artifact is None:
        _demo_prediction_output(
            bedrooms, accommodates, bathrooms, amenities_cnt,
            is_superhost, has_view, dslr, n_reviews,
        )
        _footer()
        return

    try:
        X_row = _build_prediction_row(
            neighbourhood=neighbourhood,
            room_type=room_type,
            accommodates=accommodates,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            amenities_count=amenities_cnt,
            is_superhost=int(is_superhost),
            has_view=int(has_view),
            has_hot_tub=int(has_hot_tub),
            has_parking=int(has_parking),
            dslr=float(dslr),
            n_reviews=float(n_reviews),
            mdl_defaults=mdl_defaults,
            artifact=artifact,
            ng_lookup=ng_lookup,
        )
        pipeline     = artifact["pipeline"]
        trained_cols = artifact["feature_cols"]
        log_pred     = pipeline.predict(X_row[trained_cols].values.reshape(1, -1))[0]
        pred_price   = float(np.exp(np.clip(log_pred, *_LOG_CLIP)))
        bucket_err   = _bucket_error(pred_price)
        tier         = _confidence_tiers(np.array([dslr]), np.array([n_reviews]))[0]

        _show_prediction_output(
            pred_price=pred_price,
            bucket_err=bucket_err,
            tier=tier,
            dslr=dslr,
            n_reviews=n_reviews,
            neighbourhood=neighbourhood,
            X_row=X_row,
            trained_cols=trained_cols,
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        import traceback
        st.code(traceback.format_exc())

    _footer()


def _build_prediction_row(
    neighbourhood, room_type, accommodates, bedrooms, bathrooms,
    amenities_count, is_superhost, has_view, has_hot_tub, has_parking,
    dslr, n_reviews, mdl_defaults, artifact, ng_lookup,
) -> pd.Series:
    """
    Construct a full 150-feature row for the production model.
    Start from training-set medians, override with form inputs,
    then patch the one-hot-encoded columns for neighbourhood_group and room_type.
    """
    trained_cols = artifact["feature_cols"]

    # Start from medians across all 150 model features
    row = {col: mdl_defaults.get(col, 0.0) for col in trained_cols}

    # ── Numeric overrides ─────────────────────────────────────────────────
    row["accommodates"]           = float(accommodates)
    row["bedrooms"]               = float(bedrooms)
    row["beds"]                   = float(max(bedrooms, 1))
    row["bathrooms"]              = float(bathrooms)
    row["amenities_count"]        = float(amenities_count)
    row["is_superhost"]           = float(is_superhost)
    row["has_view"]               = float(has_view)
    row["has_hot_tub"]            = float(has_hot_tub)
    row["amenity_has_hot_tub"]    = float(has_hot_tub)
    row["has_parking"]            = float(has_parking)
    row["amenity_has_parking"]    = float(has_parking)
    row["days_since_last_review"] = dslr
    row["number_of_reviews"]      = n_reviews
    row["reviews_active"]         = 1.0 if dslr < 365 else 0.0
    row["is_downtown"]            = 1.0 if neighbourhood == "Downtown" else 0.0

    # Location features from neighbourhood lookup
    ng = ng_lookup.get(neighbourhood, {})
    if "distance"      in ng: row["distance_to_downtown"]         = ng["distance"]
    if "lat"           in ng: row["latitude"]                     = ng["lat"]
    if "lon"           in ng: row["longitude"]                    = ng["lon"]
    if "avg_price"     in ng: row["neighbourhood_avg_price"]      = ng["avg_price"]
    if "grp_avg_price" in ng: row["neighbourhood_group_avg_price"]= ng["grp_avg_price"]

    # ── OHE: neighbourhood_group ──────────────────────────────────────────
    # Zero out all neighbourhood_group OHE columns first
    for col in trained_cols:
        if col.startswith("neighbourhood_group_"):
            row[col] = 0.0
    ohe_ng = f"neighbourhood_group_{neighbourhood}"
    if ohe_ng in row:
        row[ohe_ng] = 1.0

    # ── OHE: room_type ────────────────────────────────────────────────────
    # "Entire home/apt" is the baseline (both OHE cols = 0)
    row["room_type_Private room"] = 1.0 if room_type == "Private room" else 0.0
    row["room_type_Shared room"]  = 1.0 if room_type == "Shared room"  else 0.0

    return pd.Series(row)


def _show_prediction_output(
    pred_price, bucket_err, tier, dslr, n_reviews,
    neighbourhood, X_row, trained_cols,
) -> None:
    _section("Prediction Results")

    tier_bg = {"HIGH": "#d1fae5", "MEDIUM": "#fef3c7", "LOW": "#fee2e2"}
    tier_fg = {"HIGH": "#065f46", "MEDIUM": "#92400e", "LOW": "#991b1b"}
    tier_lbl = {"HIGH": "High Confidence", "MEDIUM": "Medium Confidence", "LOW": "Low Confidence"}

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Nightly Price",  f"${pred_price:.0f}")
    m2.metric("Price Range",
              f"${max(0, pred_price - bucket_err):.0f} – ${pred_price + bucket_err:.0f}")
    with m3:
        st.markdown(
            f'<div style="padding-top:6px;">'
            f'<div style="font-size:0.82rem;color:#555;margin-bottom:6px;">Confidence Tier</div>'
            f'<span class="tier-chip" style="background:{tier_bg[tier]};color:{tier_fg[tier]};">'
            f'{tier_lbl[tier]}</span></div>',
            unsafe_allow_html=True,
        )

    # Plain-English explanation
    bucket_text = {
        "$0–100":   "Budget listings ($0–100) have a typical error of ±$17.",
        "$100–200": "Mid-range listings ($100–200) have a typical error of ±$25.",
        "$200–500": "Premium listings ($200–500) have a wider typical error of ±$57.",
    }
    tier_text = {
        "HIGH":   "Based on an active listing with recent reviews.",
        "MEDIUM": "Moderate confidence — review history is limited or dated.",
        "LOW":    "Lower confidence — this listing has minimal review history.",
    }
    bucket_key = "$0–100" if pred_price < 100 else "$100–200" if pred_price < 200 else "$200–500"
    st.info(
        f"Your listing in **{neighbourhood}** is estimated at **${pred_price:.0f}/night**. "
        f"{tier_text[tier]} {bucket_text[bucket_key]}"
    )

    if pred_price > PRICE_CAP:
        st.warning(
            "Predicted price exceeds $500 — outside the model's training range. "
            "Treat this estimate with extra caution."
        )

    # ── SHAP waterfall ────────────────────────────────────────────────────
    _section("Why This Price? — SHAP Feature Contributions")
    explainer = _shap_explainer()
    if explainer is None:
        st.info("SHAP explanation requires models/production_model.pkl.")
        return

    try:
        import shap as _shap

        X_model = X_row[trained_cols].values.reshape(1, -1)
        X_df    = pd.DataFrame(X_model, columns=trained_cols)
        sv      = explainer.shap_values(X_df)[0]

        # Pick top 15 by |SHAP|
        shap_df = (
            pd.DataFrame({
                "feature":   trained_cols,
                "shap":      sv,
                "abs_shap":  np.abs(sv),
                "value":     X_df.iloc[0].values,
            })
            .sort_values("abs_shap", ascending=False)
            .head(15)
            .iloc[::-1]  # reverse so largest is at top of chart
        )

        colors = [RED if v > 0 else TEAL for v in shap_df["shap"]]
        labels = [
            f"{f} = {v:.1f}" if abs(v) < 100 and v != int(v)
            else f"{f} = {int(v)}"
            for f, v in zip(shap_df["feature"], shap_df["value"])
        ]

        fig_sw = go.Figure(go.Bar(
            y=labels,
            x=shap_df["shap"].tolist(),
            orientation="h",
            marker_color=colors,
            text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in shap_df["shap"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        ))

        base_price = float(np.exp(np.clip(explainer.expected_value, *_LOG_CLIP)))
        fig_sw.update_layout(
            title=(
                f"Top 15 Feature Contributions — "
                f"base ${base_price:.0f} → predicted ${pred_price:.0f}"
            ),
            xaxis_title="SHAP value (log-price scale)  |  Red = higher price · Teal = lower price",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff",
            title_font_color="#ffffff",
            height=500,
            margin={"t": 52, "b": 40, "l": 240, "r": 80},
            xaxis=dict(gridcolor="#333333", color="#ffffff", linecolor="#444444"),
            yaxis=dict(gridcolor="#333333", color="#ffffff", linecolor="#444444"),
        )
        st.plotly_chart(fig_sw, use_container_width=True)

        # Top 3 factors in plain English
        top3 = shap_df.tail(3).iloc[::-1]
        factors = []
        for _, r in top3.iterrows():
            arrow = "increases" if r["shap"] > 0 else "decreases"
            factors.append(f"**{r['feature']}** ({arrow} price)")
        st.markdown(
            f"The biggest drivers for this prediction: {', '.join(factors)}."
        )

    except Exception as exc:
        st.warning(f"SHAP explanation unavailable: {exc}")


def _demo_prediction_output(
    bedrooms, accommodates, bathrooms, amenities_count,
    is_superhost, has_view, dslr, n_reviews,
) -> None:
    """Heuristic fallback when production_model.pkl is not present."""
    pred = (bedrooms * 55 + bathrooms * 20 + max(0, accommodates - 2) * 8
            + amenities_count * 0.7
            + (20 if is_superhost else 0)
            + (15 if has_view else 0))
    pred = max(25.0, min(490.0, float(pred)))
    tier = ("HIGH"   if dslr < 365  and n_reviews > 5
            else "LOW"    if dslr > 1000 or  n_reviews == 0
            else "MEDIUM")
    err  = _bucket_error(pred)

    _section("Prediction Results (Demo Mode — model not loaded)")
    m1, m2 = st.columns(2)
    m1.metric("Estimated Price", f"${pred:.0f}")
    m2.metric("Price Range",     f"${max(0, pred - err):.0f} – ${pred + err:.0f}")
    tier_bg = {"HIGH": "#d1fae5", "MEDIUM": "#fef3c7", "LOW": "#fee2e2"}
    tier_fg = {"HIGH": "#065f46", "MEDIUM": "#92400e", "LOW": "#991b1b"}
    st.markdown(
        f'<span class="tier-chip" style="background:{tier_bg[tier]};color:{tier_fg[tier]};">'
        f'{tier}</span>',
        unsafe_allow_html=True,
    )
    st.info(
        f"Demo mode: estimated **${pred:.0f}/night** based on simple heuristics. "
        "Place `models/production_model.pkl` in the models/ folder for real XGBoost predictions."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Page 5 — How I Built This
# ═══════════════════════════════════════════════════════════════════════════════

def _page_how_built() -> None:
    _hero(
        'How I <span class="accent">Built This</span>',
        "Architecture · Design decisions · Limitations · Lessons learned",
    )

    # ── Architecture diagram ───────────────────────────────────────────────
    _section("Project Architecture")
    st.graphviz_chart("""
digraph arch {
    rankdir=LR;
    graph [bgcolor=white splines=ortho nodesep=0.5 ranksep=0.8];
    node  [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 margin="0.2,0.1"];
    edge  [fontname="Helvetica" fontsize=10 color="#555555"];

    subgraph cluster_sources {
        label="Raw Data";
        style=dashed; color="#aaaaaa";
        listings [label="listings.csv.gz\n(raw listing data)"  fillcolor="#fff0f0" color="#FF5A5F"];
        calendar [label="calendar.csv.gz\n(nightly availability)" fillcolor="#fff0f0" color="#FF5A5F"];
        reviews  [label="reviews.csv.gz\n(guest review text)"  fillcolor="#fff0f0" color="#FF5A5F"];
    }

    subgraph cluster_features {
        label="Feature Engineering";
        style=dashed; color="#aaaaaa";
        clean    [label="Data Cleaning\n· 48 columns → 100\n· type parsing, nulls" fillcolor="#f0fff4" color="#00A699"];
        nlp      [label="NLP Features\n· 19 binary flags\n· description + amenities" fillcolor="#f0fff4" color="#00A699"];
        calfeat  [label="Calendar Features\n· 7 availability stats\n· seasonal patterns"   fillcolor="#f0fff4" color="#00A699"];
        sentfeat [label="Sentiment Features\n· VADER scores\n· 6 review signals"           fillcolor="#f0fff4" color="#00A699"];
        select   [label="Feature Selection\n· variance + correlation\n· 150 features final" fillcolor="#f0fff4" color="#00A699"];
    }

    subgraph cluster_model {
        label="Modelling";
        style=dashed; color="#aaaaaa";
        tune   [label="Optuna Tuning\n30 trials · 5-fold CV"      fillcolor="#eff6ff" color="#3b82f6"];
        mlflow [label="MLflow Tracking\nSQLite · 3 runs"           fillcolor="#eff6ff" color="#3b82f6"];
        prod   [label="XGBoost Full\nPRODUCTION MODEL\nMAE $30.66" fillcolor="#1a1a2e" color="#1a1a2e" fontcolor="white"];
    }

    subgraph cluster_output {
        label="Output";
        style=dashed; color="#aaaaaa";
        app [label="Streamlit App\nPortfolio + Demo" fillcolor="#fff8e1" color="#f59e0b"];
        api [label="FastAPI\nPrediction API"          fillcolor="#fff8e1" color="#f59e0b"];
    }

    listings -> clean;
    clean    -> nlp;
    calendar -> calfeat;
    reviews  -> sentfeat;
    nlp      -> select;
    calfeat  -> select;
    sentfeat -> select;
    select   -> tune;
    tune     -> mlflow;
    mlflow   -> prod;
    prod     -> app;
    prod     -> api;
}
""")

    # ── Build timeline ─────────────────────────────────────────────────────
    _section("Build Timeline")
    timeline = pd.DataFrame([
        {"Day": "Day 0", "Milestone": "Project setup, EDA on raw listings.csv — price distribution, null audit, outlier analysis"},
        {"Day": "Day 1", "Milestone": "Data cleaning pipeline: currency parsing, boolean flags, categorical encoding, $500 price cap"},
        {"Day": "Day 2", "Milestone": "Baseline LinearRegression (MAE $35.87) + first XGBoost with log-price target (MAE $31.64)"},
        {"Day": "Day 3", "Milestone": "Optuna hyperparameter tuning — 30 trials, 5-fold CV → XGBoost tuned (MAE $31.27)"},
        {"Day": "Day 4", "Milestone": "NLP feature extraction from listing descriptions and amenities text — 19 binary flags (MAE $30.92)"},
        {"Day": "Day 5", "Milestone": "Calendar availability features from calendar.csv.gz + VADER review sentiment features"},
        {"Day": "Day 6", "Milestone": "Full XGBoost retrain, confidence tier redesign, MLflow experiment tracking (MAE $30.66)"},
        {"Day": "Day 7", "Milestone": "Streamlit portfolio app: 5 pages, SHAP explanations, prediction UI, deployment-ready"},
    ])
    st.dataframe(timeline, use_container_width=True, hide_index=True)

    # ── Key decisions ──────────────────────────────────────────────────────
    _section("Key Design Decisions")
    decisions = [
        ("Capped price at $500",
         "Covers **96.3%** of listings. Luxury outliers (max $50,039) would dominate the "
         "MSE loss function and produce a model that serves most hosts very poorly. "
         "The cap makes the model accurate for the realistic host market."),
        ("Log-transformed the price target",
         "Reduced price skewness from **9.0 → 0.3**. Log-price makes MSE loss scale-invariant "
         "and produces residuals that are much closer to normally distributed, "
         "which stabilises XGBoost's gradient estimates."),
        ("Removed leakage features before training",
         "`estimated_annual_revenue = price × (1 − availability) × 365` and "
         "`peak_demand_score = price × (1 − peak_avail)` both **contain the target variable**. "
         "Including them produced a perfect-looking but useless model."),
        ("Confidence from review activity, not price tier",
         "Earlier versions used price bracket as a proxy for confidence — but this caused "
         "non-monotone MAE ordering (MEDIUM > LOW) because the tiers were predicting price range, "
         "not data quality. Switching to **review recency + review count** gives a proper "
         "monotone ordering: HIGH $28.91 < MEDIUM $33.52 < LOW $39.32."),
        ("XGBoost over Random Forest",
         "Better MAE ($30.66 vs $31.19) and **2.3× faster** training (5.5s vs 12.4s). "
         "Native `shap.TreeExplainer` support enables per-prediction SHAP waterfall charts. "
         "Handles mixed feature types (binary, continuous, OHE) without normalisation."),
    ]
    for title, rationale in decisions:
        with st.expander(title):
            st.markdown(rationale)

    # ── Model limitations ──────────────────────────────────────────────────
    _section("Model Limitations — Honest Assessment")
    _callout(
        "<b>Photo quality is invisible to the model.</b> "
        "Professionally photographed listings can command a 15–30% premium that cannot be "
        "inferred from structured data alone.",
        "red",
    )
    _callout(
        "<b>Seasonal price adjustments are not modeled.</b> "
        "Training data is a single September 2025 snapshot. "
        "Hosts who dynamically price across seasons will see systematic under/over-predictions "
        "at the extremes of the calendar.",
        "amber",
    )
    _callout(
        "<b>New listings get LOW confidence regardless of quality.</b> "
        "Without review history, the confidence tier is always LOW — a data-quality limitation, "
        "not a reflection of the listing itself.",
        "amber",
    )
    _callout(
        "<b>Optimised for $15–$500 only.</b> "
        "The model clips predictions and extrapolates unreliably above $500. "
        "Luxury properties (ultra-luxury condos, entire houses with multiple units) "
        "are outside the training distribution.",
        "red",
    )

    # ── GitHub ─────────────────────────────────────────────────────────────
    _section("Source Code")
    _callout(
        "<b>GitHub:</b> "
        '<a href="https://github.com/your-username/seattle-airbnb-predictor" target="_blank">'
        "github.com/your-username/seattle-airbnb-predictor"
        "</a>"
        " &nbsp;—&nbsp; replace with your actual repository URL before sharing.",
        "blue",
    )

    _footer()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _inject_css()
    page = _sidebar()
    {
        "overview":      _page_overview,
        "eda":           _page_eda,
        "model_results": _page_model_results,
        "predict":       _page_predict,
        "how_built":     _page_how_built,
    }[page]()


if __name__ == "__main__":
    main()
