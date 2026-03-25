# Airline Stock Movement Forecasting Pipeline

A production-grade ML pipeline built with **Dagster** that predicts next-day percentage returns for four US airline stocks — DAL, UAL, AAL, LUV — by fusing market data, weather signals, and currency rates into a gradient boosted regression model.

---

## What It Does

Each trading day the pipeline:

1. **Ingests** raw data from three external APIs into GCS (Bronze)
2. **Validates** and cleans each dataset into BigQuery (Silver)
3. **Engineers** 60+ features per stock per day into BigQuery (Gold)
4. **Serves** predictions via a FastAPI endpoint backed by the champion model
5. **Monitors** for feature drift and model degradation daily; auto-triggers retraining on significant drift

Every week, a full model retrain runs with Optuna hyperparameter optimisation. The new model only replaces the champion if its test RMSE strictly improves.

---

## Architecture Overview

```
External APIs          Bronze (GCS)           Silver (BigQuery)       Gold (BigQuery)
─────────────          ────────────           ─────────────────       ───────────────
Yahoo Finance    →  airline_market/           silver_airline_market
Open-Meteo       →  weather/             →   silver_weather         →  gold_features
Frankfurter FX   →  currency/               silver_currency

Gold (BigQuery)        Training (GCS)         Serving                 Monitoring (GCS/BQ)
───────────────        ──────────────         ───────────             ───────────────────
gold_features    →  train/val/test splits  →  champion/model.pkl  →  drift_report.json
                 →  models/latest/         →  FastAPI /predict    →  drift_retrain_sensor
                 →  eval_report.json       →  serving_logs (BQ)   →  retrain_job trigger
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
| **Monitoring** | GCS JSON reports + BQ serving_logs | Data drift + concept drift; triggers retrains on demand. |

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
`target_return = (close_{t+1} − close_t) / close_t` — computed at training time by `training_dataset`, not stored in Gold. The Gold layer stores `target_return = None`.

---

## Model

**Algorithm:** `HistGradientBoostingRegressor` (scikit-learn, squared error objective)

This is **not** LightGBM. `HistGradientBoostingRegressor` is sklearn's native gradient boosted trees implementation. It requires no `libgomp` system library and runs without modification on Dagster+ Serverless.

**Hyperparameter optimisation:** Optuna (50 trials, minimise validation RMSE)

| Parameter | Search space | Notes |
|-----------|-------------|-------|
| `max_iter` | 200 – 2 000 | Number of boosting rounds |
| `learning_rate` | 1e-3 – 0.3 (log) | Shrinkage rate |
| `max_leaf_nodes` | 20 – 300 | Tree complexity |
| `max_depth` | 3 – 12 | Maximum tree depth |
| `min_samples_leaf` | 5 – 100 | Leaf regularisation |
| `l2_regularization` | 1e-8 – 10 (log) | L2 penalty on leaf values |
| `max_features` | 0.5 – 1.0 | Feature subsampling per split |

**Chronological split (no shuffle)**

| Split | Date range |
|-------|-----------|
| Train | 2020-01-01 → 2024-12-31 |
| Validation | 2025-01-01 → 2025-06-30 |
| Test | 2025-07-01 → present |

**Experiment tracking:** Dagster+ asset metadata. Every `trained_model` and `model_evaluation` materialisation logs params and metrics (MAE, RMSE, R², directional accuracy) as `MetadataValue` entries, visible as time-series charts in the Dagster+ UI asset history.

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

POST /reload
→ { "status": "reloaded", "timestamp": "2026-03-16T07:00:00Z" }
   Evicts cached model + feature cols — call after champion_model promotes.

GET /health
→ { "status": "ok" }
```

**Prediction flow:**
1. On first request, load champion `model.pkl` and `feature_cols.json` from GCS (cached via `lru_cache` for the lifetime of the process)
2. Query `gold_features` from BigQuery for the requested `(date, ticker)` using **parameterized BigQuery queries** (not f-strings)
3. Select columns in training order, run `model.predict(X)`
4. Fire-and-forget async write to `serving_logs` BigQuery table (feeds concept drift monitoring)

**Start the server:**
```bash
uvicorn quickstart_etl.lib.serving_app:app --host 0.0.0.0 --port 8080
```

The `serving_endpoint` Dagster asset acts as a readiness gate — it smoke-tests the champion model and writes `serving/serving_config.json` to GCS before the server is (re-)deployed. After a new champion is promoted, call `POST /reload` or restart the server to pick up the new artifact.

---

## Monitoring

The monitoring layer runs two complementary drift checks daily:

### Data Drift (feature distribution shift)

`drift_report` compares the last 7 days of `gold_features` against the training baseline (`training/splits/train.parquet`) using **Evidently** `DataDriftPreset`. Results are written to `gs://{bucket}/monitoring/{date}_drift_report.json`.

When more than 30% of numeric features drift, a **Prometheus** counter (`drift_detected_total`) is incremented and the **`drift_retrain_sensor`** yields a `RunRequest` for `retrain_job`.

### Concept Drift (model degradation)

`drift_report` opportunistically loads the last 7 days from the `serving_logs` BigQuery table, joins predictions against actual next-day returns from `silver_airline_market`, and runs Evidently `RegressionPreset`. The resulting `concept_mae` and `concept_rmse` are surfaced as Dagster metadata on every drift report materialisation.

This detects cases where features look stable but the model has stopped predicting accurately — a scenario that data drift alone misses.

### Sensor behaviour

`drift_retrain_sensor` uses the `@asset_sensor` pattern — it fires directly on each `drift_report` materialisation, reads metadata, and conditionally yields a `RunRequest`. No GCS polling, no cursor management.

---

## Data Quality

### Silver layer (Pandera validation)
Each Silver asset runs a Pandera `DataFrameSchema` before writing to BigQuery. Invalid rows (type mismatches, unexpected nulls, out-of-range values) raise `SchemaError` and fail the partition cleanly rather than propagating bad data downstream.

### Asset checks (planned)
The roadmap includes `@asset_check` definitions that surface validation results as named, queryable checks in the Dagster+ UI — separate from the asset materialisation itself. Planned checks include:
- `silver_market_partition_completeness` — row count in {0, 4} per partition
- `silver_weather_null_rate` — weather columns must be < 20% null
- `training_target_range` — target_return 99.9th percentile < 50%
- `training_feature_count` — expected ≥60 feature columns

Until these are implemented, failures surface as asset materialisation errors rather than check failures.

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
| **BigQuery** | Silver validated tables, Gold feature store, `serving_logs` predictions table — all partitioned |
| **`dagster-gcp` / `dagster-gcp-pandas`** | `GCSResource`, `BigQueryResource`, `BigQueryPandasIOManager` |
| **yfinance** | Airline OHLCV data |
| **Open-Meteo** | Historical weather data (no API key) |
| **Frankfurter** | FX rates (no API key) |
| **httpx** | HTTP client for Open-Meteo and Frankfurter calls with retry/backoff |
| **Pandera** | DataFrame schema validation in Silver layer |
| **scikit-learn** | `HistGradientBoostingRegressor` — gradient boosted regression, no system library deps |
| **Optuna** | Bayesian hyperparameter optimisation (50 trials per retrain) |
| **FastAPI + uvicorn** | Real-time prediction serving with `/predict`, `/reload`, `/health` endpoints |
| **Evidently** | `DataDriftPreset` (feature drift) + `RegressionPreset` (concept drift / model degradation) |
| **Prometheus client** | `drift_detected_total` counter (module-level singleton) |
| **Dask** | `DaskResource` with `LocalCluster` context manager — available for large backfills |
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
    │   └── monitoring.py        # drift_report (data drift + concept drift)
    ├── resources/
    │   ├── storage.py           # bigquery_resource, bigquery_io_manager, gcs_resource
    │   └── dask_resource.py     # DaskResource (LocalCluster, context manager)
    ├── jobs/
    │   └── __init__.py          # daily_ingestion_job, retrain_job, monitoring_job
    ├── schedules/
    │   └── __init__.py          # daily_schedule, retraining_schedule, monitoring_schedule
    └── sensors/
        └── drift_sensors.py     # drift_retrain_sensor (@asset_sensor on drift_report)
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

**Why HistGradientBoostingRegressor instead of LightGBM?**
`HistGradientBoostingRegressor` is sklearn's native GBDT implementation. It does not depend on `libgomp` (OpenMP), which is unavailable on Dagster+ Serverless containers. LightGBM requires a compiled C extension linked against libgomp and fails to import in that environment. HGBR achieves comparable accuracy with zero system-library dependencies, enabling serverless deployment without a custom container layer.

**Why BigQuery as a feature store?**
Gold features are already partitioned by `(date, ticker)` in BigQuery. The serving app queries BigQuery at prediction time — this avoids running a separate feature store service while still benefiting from BigQuery's partitioning, caching, and SQL access for ad-hoc analysis.

**Why Dagster+ metadata instead of MLflow?**
This project targets Dagster+ deployment. Every `trained_model` and `model_evaluation` materialisation surfaces params and metrics in the Dagster+ asset history UI without requiring a separate MLflow tracking server.

**Why Optuna on the full dataset weekly rather than incremental updates?**
Airline return distributions shift with macro events (fuel prices, regulatory changes, seasonal demand). A weekly full retrain with Optuna re-discovers the best hyperparameters under the current regime, which is more reliable than incremental updates to a fixed model.

**Why a champion/challenger pattern?**
The `champion_model` asset ensures a bad HPO run (e.g., overfit to a volatile training period) never silently replaces a well-performing model. Promotion only occurs when test RMSE strictly improves.

**Why two types of drift detection?**
Data drift (feature distribution shift) and concept drift (model degradation against actuals) are independent failure modes. Feature distributions can shift without degrading model performance (the relationship is stable, just the input range changed). Conversely, the model can degrade while features look normal (the relationship itself has changed). Both signals are needed for a robust monitoring layer.

**Why fire-and-forget serving logs?**
The `/predict` endpoint uses `ThreadPoolExecutor` to write prediction logs to BigQuery asynchronously. This ensures that a BigQuery outage or slow write never adds latency to the prediction response. The trade-off is that a very small number of predictions may be lost if the server crashes between prediction and log write — acceptable for this use case.

---

## Known Issues & Roadmap

| ID | Severity | Description | Status |
|----|----------|-------------|--------|
| S1 | Critical | `/reload` endpoint missing — model cache not invalidated after champion promotion | To implement |
| S2 | Critical | SQL injection risk — f-string BQ queries in serving_app.py | To fix |
| S3 | High | BQ client created per-request in serving_app.py | To fix |
| N1 | High | No `@asset_check` definitions — Silver/training validation not in Dagster UI | To implement |
| N2 | High | Training data validation gap — no feature count / target range assertions | To implement |
| D1 | Medium | Unused heavy deps in pyproject.toml (feast, tsfresh, great-expectations, etc.) | To clean |
| A4 | Medium | Concept drift does not independently trigger retraining | To extend sensor |
| R1 | Low | `gold_features` is a monolithic asset — no partial execution on domain outage | Roadmap |
