# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.9-slim

WORKDIR /app

# System deps needed by scipy, matplotlib, xgboost (OpenMP), and shap
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached independently
# of source-code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK VADER lexicon at build time so the container is self-contained
RUN python -m nltk.downloader vader_lexicon

# Copy source in dependency order:
#   setup.py before src/ so editable install works
COPY setup.py .
COPY src/ src/
COPY app/ app/
COPY data/ data/
COPY models/ models/

# Install the local package (makes `from models.baseline import ...` work
# without the sys.path hack, though the app already handles this itself)
RUN pip install --no-cache-dir -e .

# Streamlit listens on 8501 by default
EXPOSE 8501

# Disable Streamlit's browser-open and telemetry for headless/container use
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
