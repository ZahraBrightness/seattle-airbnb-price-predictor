# Seattle Airbnb Price Predictor

**Predict competitive nightly prices for Seattle Airbnb listings — with SHAP explanations and confidence tiers.**

[![CI Pipeline](https://github.com/ZahraBrightness/seattle-airbnb-price-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ZahraBrightness/seattle-airbnb-price-predictor/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF5A5F?logo=streamlit)](https://seattle-airbnb-price-predictor-ydpo9stzscb6rckukpysxh.streamlit.app/)

**Live demo:** [seattle-airbnb-price-predictor-ydpo9stzscb6rckukpysxh.streamlit.app](https://seattle-airbnb-price-predictor-ydpo9stzscb6rckukpysxh.streamlit.app/)

---

## Project Overview

### Problem

Seattle Airbnb hosts face a persistent pricing dilemma: underpricing leaves revenue on the table while overpricing kills booking rates. Most hosts rely on gut feel or manual competitor browsing — neither scales, and neither explains *why* comparable listings command higher prices.

### End User

Seattle Airbnb hosts — particularly new hosts setting an opening price and experienced hosts benchmarking against the market.

### Data

| Source | Size |
|---|---|
| Inside Airbnb Seattle — listings (Sep 2025) | 6,996 listings × 75+ raw columns |
| Inside Airbnb Seattle — reviews (Sep 2025) | 575,824 reviews |
| Inside Airbnb Seattle — calendar (Sep 2025) | 2.5M availability rows |

### Output

For any listing configuration the app produces:

- **Predicted nightly price** in dollars (XGBoost on log-transformed target, exponentiated at inference)
- **Confidence tier** — HIGH / MEDIUM / LOW based on review activity, flagging listings the model is likely to misjudge
- **SHAP waterfall** — top drivers pushing the prediction up or down from the baseline, rendered as an interactive Plotly chart

---

## Architecture

```
Raw Data                  Feature Engineering              Model & Serving
────────                  ───────────────────              ───────────────

listings.csv.gz  ──────►  Cleaning & Imputation  ──────►
                                                           XGBoost (log price)
calendar.csv.gz  ──────►  Calendar Features      ──────►     │
                          (availability, blocked             │  MLflow Tracking
reviews.csv.gz   ──────►   rate, seasonality)               │  (experiment runs,
                          Sentiment Features     ──────►     │   params, metrics)
                          (VADER per listing,                │
                           review velocity,                  ▼
                           trend)                        production_model.pkl

                                                             │
                                                             ▼
                                                    Streamlit Portfolio App
                                                    (Overview · EDA · Results
                                                     · Predict · Architecture)

                                                             │
                                                             ▼
                                                    Docker Container
                                                    Streamlit Community Cloud
```

---

## Model Results

All metrics evaluated on a held-out test set (20%). Target capped at $500/night (96.3% of listings). Models trained on log(price), metrics reported in original dollar scale.

| Model | MAE | RMSE | R² | Notes |
|---|---|---|---|---|
| Linear Regression | $35.87 | $52.40 | 0.615 | Baseline |
| XGBoost tuned | $31.27 | $47.04 | 0.690 | Optuna, 30 trials |
| XGBoost + NLP | $30.92 | $46.27 | 0.700 | +19 VADER sentiment features |
| **XGBoost full** | **$30.66** | **$45.82** | **0.706** | **PRODUCTION** |

The production model improves MAE by **$5.21 (14.5%)** over the linear baseline and **$0.26 (0.8%)** over the NLP-augmented model — diminishing returns consistent with the expressiveness ceiling finding below.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.9** | Core language |
| **XGBoost** | Gradient-boosted tree regressor — production model |
| **scikit-learn** | Preprocessing pipelines, train/test split, linear baseline |
| **SHAP** | Per-prediction feature attribution via TreeExplainer |
| **Optuna** | Bayesian hyperparameter tuning (30 trials, TPE sampler) |
| **MLflow** | Experiment tracking — params, metrics, artifact logging |
| **VADER (NLTK)** | Rule-based sentiment analysis on 575K+ review texts |
| **Pandas / NumPy** | Data wrangling, feature engineering |
| **Geopy** | Geodesic distance from each listing to downtown Seattle |
| **Plotly** | Interactive charts — choropleth, SHAP waterfall, scatter |
| **Streamlit** | 5-page portfolio web app |
| **Docker** | Containerised runtime (`python:3.9-slim` + `libgomp1`) |
| **GitHub Actions** | CI pipeline — pytest (required) + ruff lint (non-blocking) |

---

## Setup & Installation

**Requirements:** Python 3.9+, Git

```bash
# 1. Clone the repository
git clone https://github.com/ZahraBrightness/seattle-airbnb-price-predictor.git
cd seattle-airbnb-price-predictor

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install the local src/ package in editable mode
pip install -e .
```

Data files (`data/cleaned.csv`, `data/features.csv`, `data/neighbourhoods.geojson`) and the production model (`models/production_model.pkl`) are committed to the repository and will be available after cloning.

---

## How to Run

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

### Docker

```bash
# Build the image
docker build -t seattle-airbnb-predictor .

# Run the container
docker run -p 8501:8501 seattle-airbnb-predictor
```

Or with Docker Compose (bind-mounts `data/` and `models/` so you can update them without a rebuild):

```bash
docker compose up
```

### Tests

```bash
pytest tests/ -v
```

20 tests across data quality, feature pipeline, and model artifact checks. All pass in under 5 seconds.

---

## Top 10 Features by SHAP Importance

Computed via `shap.TreeExplainer` on the full test set. Importance is mean absolute SHAP value as a percentage of total model output magnitude.

| Rank | Feature | SHAP Importance | Why It Matters |
|---|---|---|---|
| 1 | `bedrooms` | 13.8% | Strongest capacity signal — room count sets listing tier |
| 2 | `accommodates` | 11.3% | Per-guest premium accelerates non-linearly above 4 guests |
| 3 | `distance_to_downtown` | 5.8% | Proximity to Pike Place / Capitol Hill commands a premium |
| 4 | `bathrooms` | 4.5% | Correlated with bedrooms but adds independent signal |
| 5 | `amenities_count` | 4.3% | Hosts with more amenities skew toward premium positioning |
| 6 | `neighbourhood_avg_price` | 4.2% | Encodes neighbourhood prestige more reliably than geography alone |
| 7 | `host_quality_score` | 3.5% | Composite of superhost status, response rate, and review score |
| 8 | `avg_sentiment` | 2.0% | VADER mean across all listing reviews — measures guest satisfaction |
| 9 | `has_view` | 1.2% | Binary amenity flag — water/mountain views add a measurable premium |
| 10 | `consecutive_blocked_rate` | 1.0% | Calendar feature — high block rate signals demand-driven pricing |

---

## Key Decisions & Lessons Learned

- **Capped price at $500/night.** The raw price distribution reaches $50,000 (hotel buyouts, corporate rentals) — extreme outliers that are structurally different from typical Airbnb listings. Capping at $500 covers 96.3% of listings and prevents the model from chasing unrepresentative signal. Hosts pricing above $500 are out of scope.

- **Log-transformed the target.** Raw price has skewness of 9.0 — linear models are forced to over-fit the high-price tail. Log transformation reduces skewness to 0.3, letting the model treat price as approximately Gaussian. All predictions are exponentiated at inference time.

- **Removed leakage features.** `estimated_revenue_l365d` and `estimated_occupancy_l365d` are derived from actual booking history — they contain the target. Including them produced artificially high R² (0.82+) that collapsed to chance on new hosts with no booking history. These were removed entirely.

- **Confidence from review activity, not price range.** Early versions flagged HIGH/LOW confidence based on predicted price bands. This conflated model uncertainty with listing price tier. Review activity (`days_since_last_review`, `number_of_reviews`) is a better proxy: recently-reviewed listings are actively managed, price-competitive, and closer to the training distribution.

- **Optuna showed the model is at an expressiveness ceiling.** After 30 trials, hyperparameter tuning improved R² by only 0.016 over the default XGBoost config. The plateau indicates the bottleneck is feature information, not model capacity. The NLP sentiment features (avg_sentiment, review_velocity, sentiment_trend) confirmed this — adding 19 new information signals improved R² more than tuning the existing 131 features.

---

## Project Structure

```
seattle-airbnb-price-predictor/
├── app/
│   └── streamlit_app.py          # 5-page Streamlit portfolio app
├── data/
│   ├── cleaned.csv               # 6,221 listings after cleaning
│   ├── features.csv              # 5,927 listings × 100 engineered features
│   └── neighbourhoods.geojson    # Seattle neighbourhood polygons (90 areas)
├── models/
│   └── production_model.pkl      # Trained XGBoost pipeline + metadata
├── src/
│   ├── data/                     # Data loading utilities
│   ├── features/
│   │   ├── engineer.py           # Core feature engineering pipeline
│   │   ├── calendar_features.py  # Availability & blocked-rate features
│   │   ├── review_features.py    # VADER sentiment & review velocity
│   │   └── run_features.py       # Feature pipeline entry point
│   └── models/
│       ├── baseline.py           # Linear regression baseline + prepare_features()
│       ├── train.py              # XGBoost training loop
│       ├── train_xgb_log.py      # Log-target XGBoost variant
│       ├── train_xgb_nlp.py      # NLP-augmented XGBoost variant
│       ├── train_xgb_full.py     # Full production model training
│       ├── tuning.py             # Optuna hyperparameter search
│       ├── evaluate_nlp.py       # NLP model evaluation
│       ├── error_analysis.py     # Residual & segment analysis
│       └── predict.py            # Inference utilities
├── tests/
│   ├── conftest.py               # Session-scoped pytest fixtures
│   ├── test_data_quality.py      # Quality gate + 8 data tests
│   ├── test_features.py          # 5 feature pipeline tests
│   └── test_model.py             # 7 model artifact tests
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI (test + lint)
├── Dockerfile                    # python:3.9-slim container
├── docker-compose.yml            # Compose with data/models bind mounts
├── requirements.txt              # Runtime dependencies
├── setup.py                      # Editable install for src/ package
└── README.md
```

---

## Live Demo

[https://seattle-airbnb-price-predictor-ydpo9stzscb6rckukpysxh.streamlit.app/](https://seattle-airbnb-price-predictor-ydpo9stzscb6rckukpysxh.streamlit.app/)

The app has five pages:

| Page | What it shows |
|---|---|
| **Overview** | Project summary, KPI cards, key EDA findings |
| **Explore the Data** | Price distributions, neighbourhood choropleth, correlation heatmap |
| **Model Results** | Model comparison table, residual plot, error-by-segment analysis |
| **Try It Yourself** | Interactive price prediction form with SHAP waterfall |
| **How I Built This** | Architecture diagram, tech stack, methodology notes |

---

## Author

**Zahra Ahmadi** — [github.com/ZahraBrightness](https://github.com/ZahraBrightness)

Data: [Inside Airbnb](http://insideairbnb.com/seattle) (Seattle, September 2025)
