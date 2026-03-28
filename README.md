# Fraud MLOps Platform

A production-grade, real-time fraud detection system built with Kubeflow Pipelines, Apache Kafka, Feast feature store, and deployed on GCP GKE. Demonstrates end-to-end MLOps: from live data ingestion to automated model training, A/B testing, and self-healing deployment.

## Architecture
```
Kafka (live transactions)
    → Feature Consumer (real-time feature engineering)
    → Redis (feature store / low-latency serving)
    → Kubeflow Pipeline (parallel training of 3 models)
        → MLflow (experiment tracking)
        → Auto-selection (best AUC wins)
    → KServe (A/B traffic splitting: 80/15/5)
    → Prometheus + Grafana (drift detection + monitoring)
    → GitHub Actions (auto-retrain on drift or code push)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Data streaming | Apache Kafka |
| Feature store | Feast + Redis |
| ML orchestration | Kubeflow Pipelines |
| Experiment tracking | MLflow |
| Model serving | FastAPI + KServe |
| Infrastructure | Terraform + GKE |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |

## Models

Three fraud detection models trained in parallel, evaluated by AUC, winner auto-promoted:

- Logistic Regression v1 — baseline
- Random Forest v2 — challenger
- XGBoost v3 — champion (AUC: 0.976)

## Quick Start
```bash
# Clone and start local stack
git clone https://github.com/Sakshi3027/fraud-mlops-platform
cd fraud-mlops-platform
docker-compose up -d

# Start MLflow
mlflow server --host 0.0.0.0 --port 5001 \
  --backend-store-uri sqlite:///mlflow_data/mlflow.db \
  --default-artifact-root $(pwd)/mlflow_data/artifacts

# Train all models
python models/logistic_regression.py
python models/random_forest.py
python models/xgboost_model.py

# Start prediction API
python -m uvicorn serving.app:app --port 8000 --reload

# Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id":"test-001","user_id":"user_1234",
       "amount":2500,"hour_of_day":3,"transactions_last_hour":12,
       "is_international":1,"unique_countries_count":5,
       "amount_zscore":3.2,"avg_amount_last_50":45.0,"std_amount_last_50":12.0}'
```

## Local Services

| Service | URL |
|---|---|
| Prediction API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

## Deploy to GCP
```bash
cd terraform
terraform init
terraform plan -var="project_id=YOUR_PROJECT_ID"
terraform apply -var="project_id=YOUR_PROJECT_ID"
```