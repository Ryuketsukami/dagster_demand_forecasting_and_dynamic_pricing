# Airline Stock Movement Forecasting Pipeline

A production-grade ML pipeline built with **Dagster** that predicts next-day percentage returns for four US airline stocks — DAL, UAL, AAL, LUV — by fusing market data, weather signals, and currency rates into a LightGBM regression model.

---

## What It Does

Each trading day the pipeline:

1. **Ingests** raw data from three external APIs into GCS (Bronze)
2. **Validates** and cleans each dataset into BigQuery (Silver)
3. **Engineers** 60+ features per stock per day into BigQuery (Gold)
4. **Serves** predictions via a FastAPI endpoint backed by the champion model
5. **Monitors** for feature drift daily and auto-triggers retraining when significant drift is detected

Every week, a full model retrain runs with Optuna hyperparameter optimisation. The new model only replaces the champion if its test RMSE strictly improves.

---

## Architecture Overview

```
External APIs          Bronze (GCS)           Silver (BigQuery)       Gold (BigQuery)
─────────────          ────────────           ─────────────────       ───────────────
Yahoo Finance    →  airline_market/           silver_airline_market
Open-Meteo       →  weather/             →   silver_weather         →  gold_features
Frankfurter FX   →  currency/               silver_currency

Gold (BigQuery)        Training (GCS)         Serving                 Monitoring (GCS)
───────────────        ──────────────         ───────────             ────────────────
gold_features    →  train/val/test splits  →  champion/model.pkl  →  drift_report.json
                 →  model.pkl (latest)     →  FastAPI /predict    →  drift_retrain_sensor
                 →  eval_report.json       →  serving_config.json →  retrain_job trigger
                 →  champion/ (promoted)
```

### Medallion layers

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Bronze** | GCS Parquet | Raw API responses, one file per date. Immutable. |
| **Silver** | BigQuery (partitioned by date) | Validated, typed, schema-enforced with Pandera. |
| **Gold** | BigQuery (partitioned by date × ticker) | 60+ engineered features ready for model consumption. |
| **Training** | GCS (model artifacts) | Chronological splits, Optuna HPO, champion promotion. |
| **Serving** | GCS config + FastAPI process | Real-time predictions from the champion model. |
| **Monitoring** | GCS JSON reports | Daily drift detection; triggers weekly retrains on demand. |

---

## Data Sources

| Source | What | Auth |
|--------|------|------|
| **yfinance** | Daily OHLCV for DAL, UAL, AAL, LUV | None (Yahoo Finance, non-commercial) |
| **Open-Meteo** | Temperature, precipitation, wind, weather code at ATL / LAX / ORD / DFW / JFK | None (free, no caps) |
| **Frankfurter** | USD → EUR, GBP, BRL exchange rates | None (free, no caps) |

Weekend and holiday partitions are handled gracefully: Bronze writes empty Parquet files, Silver stores empty partitions, Gold skips feature computation and writes nothing.

---

## Feature Engineering

Gold features are computed in [`lib/feature_logic.py`](src/quickstart_etl/lib/feature_logic.py) and shared between the training pipeline and the serving app.

**Market (per ticker, ~20 features)**
- Raw OHLCV
- `daily_return`, `log_return`, `high_low_range`, `overnight_gap`
- Rolling volatility: `vol_5d`, `vol_20d`
- Rolling momentum: `return_5d`, `return_20d`
- RSI(14), MACD line/signal/histogram, Bollinger Bands (upper/lower/width)
- `vol_ratio_5d` — volume relative to 5-day average

**Weather (5 cities × 5 variables = 25 features + 25 lag-1 features)**
Each ticker gets the full hub-city weather signal: `{city}_{temp_max|temp_min|precip|wind|weather_code}` for ATL, LAX, ORD, DFW, JFK, plus a one-day lag set (`lag1_` prefix).

**Currency (3 rates × 4 derived = 12 features)**
`eur_usd`, `gbp_usd`, `brl_usd` — raw rates, daily % change, and one-day lags.

**Calendar (6 features)**
`day_of_week`, `month`, `quarter`, `week_of_year`, `is_us_holiday`, `days_to_holiday`

**Target**
`target_return = (close_{t+1} − close_t) / close_t` — computed at training time, not stored in Gold.

---

## Model

**Algorithm:** LightGBM (`lgb.LGBMRegressor`, GBDT, RMSE objective)

**Hyperparameter optimisation:** Optuna (50 trials, minimise validation RMSE)

| Parameter | Search space |
|-----------|-------------|
| `n_estimators` | 200 – 2 000 |
| `learning_rate` | 1e-3 – 0.3 (log) |
| `num_leaves` | 20 – 300 |
| `max_depth` | 3 – 12 |
| `min_child_samples` | 5 – 100 |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `reg_alpha`, `reg_lambda` | 1e-8 – 10 (log) |

**Chronological split (no shuffle)**

| Split | Date range |
|-------|-----------|
| Train | 2020-01-01 → 2024-12-31 |
| Validation | 2025-01-01 → 2025-06-30 |
| Test | 2025-07-01 → present |

**Experiment tracking:** Dagster+ asset metadata. Every `trained_model` and `model_evaluation` materialisation logs params and metrics (MAE, RMSE, R², directional accuracy) as `MetadataValue` entries, visible in the Dagster+ UI asset history. MLflow (`dagster-mlflow`) is available as a future upgrade path.

**Champion promotion:** `champion_model` compares the new model's test RMSE against `champion/metrics.json` in GCS. Promotion happens only when the new RMSE is strictly lower.

---

## Serving

The champion model is served by a standalone FastAPI application:

```
POST /predict
{ "date": "2026-03-16", "ticker": "DAL" }

→ { "ticker": "DAL", "prediction_date": "2026-03-16",
    "predicted_return": 0.0142, "model_version": "rmse=0.01234",
    "prediction_timestamp": "2026-03-15T18:00:00Z" }
```

**Prediction flow:**
1. On first request, load champion `model.pkl` and `feature_cols.json` from GCS (cached in memory for the lifetime of the process)
2. Query `gold_features` from BigQuery for the requested `(date, ticker)` — the feature store is BigQuery, not a separate vector DB
3. Select columns in training order, run `model.predict(X)`

**Start the server:**
```bash
uvicorn quickstart_etl.lib.serving_app:app --host 0.0.0.0 --port 8080
```

The `serving_endpoint` Dagster asset acts as a readiness gate — it smoke-tests the champion model and writes `serving/serving_config.json` to GCS before the server is (re-)deployed.

---

## Monitoring

`drift_report` runs daily after ingestion and compares the last 7 days of `gold_features` against the training baseline using **Evidently** `DataDriftPreset`. Results are written to `gs://{bucket}/monitoring/{date}_drift_report.json`.

When more than 30% of numeric features drift, a **Prometheus** counter (`drift_detected_total`) is incremented and the **`drift_retrain_sensor`** yields a `RunRequest` for `retrain_job`.

---

## Pipeline Schedules

| Schedule | Cron | Job |
|----------|------|-----|
| `daily_ingestion_schedule` | `0 6 * * 1-5` — Mon–Fri 06:00 UTC | Bronze → Silver → Gold |
| `daily_monitoring_schedule` | `0 7 * * 1-5` — Mon–Fri 07:00 UTC | Drift report |
| `weekly_retraining_schedule` | `0 8 * * 0` — Sundays 08:00 UTC | Full model retrain |

The ingestion schedule targets trading days only (Mon–Fri). Weekend partitions are written as empty files so the asset graph stays complete.

---

## Tech Stack

| Technology | Role |
|-----------|------|
| **Dagster / Dagster+** | Pipeline orchestration, scheduling, asset lineage, experiment tracking via asset metadata |
| **Google Cloud Storage** | Bronze raw Parquet, model artifacts (`models/latest/`, `champion/`), monitoring reports |
| **BigQuery** | Silver validated tables, Gold feature store — partitioned by date |
| **`dagster-gcp` / `dagster-gcp-pandas`** | `GCSResource`, `BigQueryResource`, `BigQueryPandasIOManager` |
| **yfinance** | Airline OHLCV data |
| **Open-Meteo** | Historical weather data (no API key) |
| **Frankfurter** | FX rates (no API key) |
| **httpx** | HTTP client for Open-Meteo and Frankfurter calls with retry/backoff |
| **Pandera** | DataFrame schema validation in Silver layer |
| **LightGBM** | Gradient boosted regression model |
| **Optuna** | Bayesian hyperparameter optimisation (50 trials per retrain) |
| **FastAPI + uvicorn** | Real-time prediction serving |
| **Evidently** | Feature and prediction drift detection |
| **Prometheus client** | Drift metric exposure (`drift_detected_total` counter) |
| **Dask** | `DaskResource` with `LocalCluster` — available for large backfills |
| **DVC** | Bronze Parquet versioning via GCS remote |
| **Ruff** | Linting and formatting |
| **pre-commit** | Git hooks — runs Ruff on commit |

---

## Project Structure

```
src/quickstart_etl/
├── definitions.py               # Dagster Definitions entry point
├── partitions.py                # DailyPartitionsDefinition (2020-01-01 → today)
├── lib/
│   ├── feature_logic.py         # Pure feature functions (RSI, MACD, Bollinger, etc.)
│   └── serving_app.py           # Standalone FastAPI inference server
└── defs/
    ├── assets/
    │   ├── ingestion.py         # Bronze: raw_airline_market_data, raw_weather_data, raw_currency_rates
    │   ├── validation.py        # Silver: silver_airline_market, silver_weather, silver_currency
    │   ├── features.py          # Gold: gold_features
    │   ├── training.py          # training_dataset, trained_model, model_evaluation, champion_model
    │   ├── serving.py           # serving_endpoint (readiness gate)
    │   └── monitoring.py        # drift_report
    ├── resources/
    │   ├── storage.py           # bigquery_resource, bigquery_io_manager, gcs_resource
    │   ├── dask_resource.py     # DaskResource (LocalCluster, configurable workers)
    │   └── mlflow_resource.py   # Placeholder — MLflow available as future upgrade
    ├── jobs/
    │   └── __init__.py          # daily_ingestion_job, retrain_job
    ├── schedules/
    │   └── __init__.py          # daily_schedule, retraining_schedule, monitoring_schedule
    └── sensors/
        └── drift_sensors.py     # drift_retrain_sensor
```

---

## GCS Layout

```
gs://{GCS_BUCKET_NAME}/
  bronze/
    airline_market/{YYYY-MM-DD}.parquet
    weather/{YYYY-MM-DD}.parquet
    currency/{YYYY-MM-DD}.parquet
  training/
    splits/train.parquet
    splits/val.parquet
    splits/test.parquet
  models/
    latest/model.pkl
    latest/feature_cols.json
    latest/eval_report.json
  champion/
    model.pkl
    feature_cols.json
    metrics.json
  serving/
    serving_config.json
  monitoring/
    {YYYY-MM-DD}_drift_report.json
```

---

## Environment Variables

Copy `.env.example` → `.env` and fill in your GCP values.

```env
GCP_PROJECT_ID=your-project-id
GCP_REGION=europe-west2
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
BIGQUERY_DATASET=demand_forecasting
BIGQUERY_LOCATION=EU
GCS_BUCKET_NAME=your-bucket-name
GCS_BRONZE_PREFIX=bronze
GCS_MODELS_PREFIX=models
GCS_CHAMPION_PREFIX=champion
GCS_MONITORING_PREFIX=monitoring
DAGSTER_CLOUD_API_TOKEN=...
DAGSTER_CLOUD_URL=https://your-org.dagster.cloud/
SERVING_HOST=0.0.0.0
SERVING_PORT=8080
```

---

## Getting Started

**Install dependencies**
```bash
uv sync
```

**Lint / format**
```bash
.venv/Scripts/pre-commit run --all-files   # Windows
.venv/bin/pre-commit run --all-files       # macOS / Linux
```

**Run Dagster locally**
```bash
dagster dev
```

**Backfill historical Bronze → Silver → Gold (2020 → today)**

In the Dagster UI, select the `daily_ingestion_job`, click *Launch backfill*, and select the full partition range. The pipeline handles empty market-closed partitions automatically.

**Run the serving app**
```bash
uvicorn quickstart_etl.lib.serving_app:app --host 0.0.0.0 --port 8080
```

---

## CI/CD

The pipeline is deployed to **Dagster+** (Dagster Cloud). A GitHub Actions workflow (`.github/workflows/`) handles:

1. **On pull request** — run Ruff linter, execute unit tests with pytest
2. **On merge to `main`** — push a new code location snapshot to Dagster Cloud via the Dagster Cloud API using `DAGSTER_CLOUD_API_TOKEN`
3. **Scheduled runs** — executed by Dagster+ agents on the cloud schedule; no self-hosted runner needed for production pipeline runs

Local git hooks (pre-commit + Ruff) enforce formatting before any commit reaches CI.

---

## Design Decisions

**Why BigQuery as a feature store?**
Gold features are already partitioned by `(date, ticker)` in BigQuery. The serving app queries BigQuery at prediction time — this avoids running a separate feature store service while still benefiting from BigQuery's partitioning, caching, and SQL access for ad-hoc analysis.

**Why Dagster+ metadata instead of MLflow?**
This project targets Dagster+ deployment. Every `trained_model` and `model_evaluation` materialisation surfaces params and metrics in the Dagster+ asset history UI without requiring a separate MLflow tracking server. `dagster-mlflow` is already a dependency — it can be wired up if a tracking server becomes available.

**Why Optuna on the full dataset weekly rather than incremental updates?**
Airline return distributions shift with macro events (fuel prices, regulatory changes, seasonal demand). A weekly full retrain with Optuna re-discovers the best hyperparameters under the current regime, which is more reliable than incremental updates to a fixed model.

**Why a champion/challenger pattern?**
The `champion_model` asset ensures a bad HPO run (e.g., overfit to a volatile training period) never silently replaces a well-performing model. Promotion only occurs when test RMSE strictly improves.
