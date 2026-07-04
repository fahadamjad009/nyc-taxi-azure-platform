<div align="center">

# 🚕 NYC Taxi & Rideshare Analytics Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://nyc-taxi-azure-platform.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Azure](https://img.shields.io/badge/Azure-Data%20Factory-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)
[![Databricks](https://img.shields.io/badge/Databricks-Delta%20Lake-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com)
[![DuckDB](https://img.shields.io/badge/DuckDB-Analytics-FFC300?style=for-the-badge&logo=duckdb&logoColor=black)](https://duckdb.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**End-to-end Azure data platform ingesting 1.95 billion NYC TLC taxi & rideshare trips (2019–2025) through Azure Data Factory → Databricks Delta Lake → DuckDB → XGBoost + SHAP → interactive Streamlit dashboard.**

</div>

---

## 📊 Live Dashboard

https://nyc-taxi-azure-platform.streamlit.app/
> Click the badge above to open the deployed app. No login required.

Five analytical tabs powered by pre-aggregated Databricks outputs and a pre-trained XGBoost fare model:

| Tab | What It Shows |
|---|---|
| **📈 Market Overview** | Monthly volume trends, stacked market share, treemap, revenue, COVID-19 impact, cumulative growth index |
| **⏰ Demand Patterns** | Hour × day heatmap, hourly vehicle profiles, day-of-week bar chart, year × month seasonality heatmap |
| **🗺️ Borough Intelligence** | Borough treemap, avg fare by borough, volume over time |
| **💵 Fare Economics** | Histogram + box, box plot by borough, distance vs fare scatter + OLS, median trend, YoY fare change |
| **🤖 ML & SHAP** | XGBoost R²=0.93 fare predictor, SHAP feature importance, beeswarm, dependence plot, predicted vs actual |

---

## 🏗️ Architecture

```
NYC TLC Open Data (AWS S3 Public)
           │
           ▼
Azure Data Factory
Metadata-driven pipeline
336 Parquet files ingested
4 vehicle classes × 2019–2025
           │
           ▼
Azure Data Lake Storage Gen2
raw-tlc-data/ container
           │
           ▼
Azure Databricks
Delta Lake curation
Schema normalisation
Type casting + partitioning
1.95 Billion rows → Delta format
           │
           ▼
Databricks Aggregation Notebooks
5 summary tables (monthly trips,
hourly patterns, annual KPIs,
fare distribution, borough volumes)
+ 552K-row ML training sample
           │
           ├──► data/exports/*.parquet (committed to GitHub)
           └──► models/fare_model.pkl  (pre-trained XGBoost)
                         │
                         ▼
              Streamlit Community Cloud
              (permanent public deployment)
```

---

## 📁 Project Structure

```
nyc-taxi-azure-platform/
│
├── streamlit_app.py          # Dashboard — reads from data/exports/ + models/
├── train_model.py            # One-time script to pre-train XGBoost + save pkl
├── requirements.txt
│
├── data/
│   └── exports/              # Pre-aggregated Databricks outputs (committed)
│       ├── monthly_trips.parquet       # 335 rows — monthly trips by vehicle type
│       ├── hourly_patterns.parquet     # 3,528 rows — hourly demand by day/vehicle
│       ├── annual_kpis.parquet         # 28 rows — annual totals per vehicle type
│       ├── fare_distribution.parquet   # 21 rows — fare percentiles per year/vehicle
│       ├── borough_volumes.parquet     # 49 rows — borough pickup counts and avg fare
│       └── ml_sample.parquet          # 552K rows — individual Yellow Taxi trips for ML
│
├── models/
│   └── fare_model.pkl        # Pre-trained XGBoost + SHAP explainer bundle (3.1 MB)
│
└── .env                      # (gitignored — Azure storage key)
```

---

## 🛠️ Tech Stack

| Layer | Tool | Detail |
|---|---|---|
| Ingestion | Azure Data Factory | Metadata-driven pipeline, 4 vehicle types, 2019–2025 |
| Storage | Azure Data Lake Gen2 | `raw-tlc-data/` and `curated-tlc-data/` containers |
| Compute | Azure Databricks | Spark 4.0, Delta Lake, Standard_D4s_v3 single node |
| Transformation | PySpark + Delta | Schema normalisation, type casting, partitioned Delta tables |
| Aggregation | PySpark GroupBy | 5 dashboard-ready summary Parquets + 552K ML sample |
| Analytics | DuckDB | In-process OLAP queries over Parquet files at dashboard runtime |
| ML | XGBoost + SHAP | Fare prediction R²=0.93, SHAP explainability, pre-trained pkl |
| Dashboard | Streamlit + Plotly | Dark theme, 5 tabs, 20+ interactive Plotly charts |
| Deployment | Streamlit Community Cloud | Permanent public URL, no Snowflake/Azure dependency at runtime |
| Language | Python 3.11 | |

---

## 📦 Data Coverage

| Dataset | Rows (Full Delta) | Exported Rows | Description |
|---|---|---|---|
| Yellow Taxi | ~308M | 552K (ML sample) | Metered trips, fare + distance data |
| Green Taxi | ~12M | Aggregated | Outer borough metered trips |
| FHV (Black Car) | ~146M | Aggregated | For-hire vehicle dispatched trips |
| HVFHV (Uber/Lyft) | ~1.48B | Aggregated | High-volume rideshare (Uber, Lyft, Via) |
| **Total** | **~1.95B** | **~554K + 5 summaries** | |

> **Why aggregated exports?** The 1.95B row Delta Lake lives permanently in Azure. Databricks aggregated it into 5 dashboard-ready summary files (total ~1.4 MB) — the same pattern used in production BI platforms (Power BI Direct Query materialised views, Tableau extracts, dbt marts). The full raw data is re-queryable from Azure at any time using the Databricks notebooks in this repo.

---

## 🤖 ML Model — XGBoost Fare Predictor

Trained on 552,771 individual Yellow Taxi trips sampled from the 1.95B row Delta Lake.

| Metric | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Features | Trip distance, hour, day of week, month, year, pickup borough, dropoff borough, passenger count |
| Train size | ~441K trips |
| Test size | ~111K trips |
| R² Score | **0.9316** |
| RMSE | **$3.70** |

**SHAP Findings:**
- **Trip distance** dominates fare prediction (mean |SHAP| = $8.89) — each mile ≈ +$2.50
- **Year** is the second driver ($2.48) — capturing post-2021 fare inflation and surcharge increases
- **Hour of day** ($0.75) — surge pricing during late night and rush hours
- **Passenger count** near zero — NYC taxis charge per trip, not per person

---

## 🚀 Run Locally

```bash
git clone https://github.com/fahadamjad009/nyc-taxi-azure-platform.git
cd nyc-taxi-azure-platform
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app reads from `data/exports/` Parquet files and `models/fare_model.pkl` — no Azure account or API keys required.

To re-train the model from scratch:
```bash
python train_model.py
```

---

## 🔑 Data Source

[NYC Taxi & Limousine Commission (TLC) Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — public dataset, no license restrictions. Hosted on AWS S3 and ingested via Azure Data Factory.

---

## 👤 Author

**Fahad Amjad**
[GitHub](https://github.com/fahadamjad009) · [Portfolio](https://fahadamjad009.github.io)

---

<div align="center">
<i>Part of a data engineering portfolio demonstrating end-to-end pipelines across Azure, Snowflake, dbt, Python, and BI tooling.</i>
</div>
