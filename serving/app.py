import time
import json
import pickle
import numpy as np
import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
from serving.ab_router import (
    get_model_assignment, log_prediction,
    get_model_stats, TRAFFIC_CONFIG
)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with A/B model routing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total predictions made",
    ["model_role", "prediction"]
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Prediction latency",
    ["model_role"]
)
FRAUD_DETECTED = Counter(
    "fraud_detected_total",
    "Total fraud cases detected",
    ["model_role"]
)

FEATURES = [
    "amount", "amount_zscore", "avg_amount_last_50",
    "std_amount_last_50", "unique_countries_count",
    "transactions_last_hour", "hour_of_day", "is_international"
]

models = {}


class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    hour_of_day: int
    transactions_last_hour: int
    is_international: int
    unique_countries_count: int
    amount_zscore: float = 0.0
    avg_amount_last_50: float = 0.0
    std_amount_last_50: float = 0.0


class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    model_role: str
    model_name: str
    latency_ms: float


def get_mock_prediction(features: list, model_role: str) -> tuple:
    amount = features[0]
    hour = features[6]
    velocity = features[5]
    is_intl = features[7]

    base_score = 0.01
    if amount > 500:
        base_score += 0.4
    if hour < 6:
        base_score += 0.3
    if velocity > 7:
        base_score += 0.2
    if is_intl:
        base_score += 0.1

    noise = {"champion": 0.0, "challenger": 0.02, "shadow": 0.05}
    prob = min(base_score + noise.get(model_role, 0), 0.99)
    return int(prob > 0.5), prob


@app.on_event("startup")
async def startup():
    print("Fraud Detection API starting up...")
    print("Models: using rule-based scoring (swap with MLflow models in production)")


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest):
    start_time = time.time()

    model_role = get_model_assignment(transaction.user_id)
    model_info = TRAFFIC_CONFIG[model_role]

    features = [
        transaction.amount,
        transaction.amount_zscore,
        transaction.avg_amount_last_50,
        transaction.std_amount_last_50,
        transaction.unique_countries_count,
        transaction.transactions_last_hour,
        transaction.hour_of_day,
        transaction.is_international
    ]

    prediction, probability = get_mock_prediction(features, model_role)

    latency_ms = (time.time() - start_time) * 1000

    PREDICTIONS_TOTAL.labels(
        model_role=model_role,
        prediction=str(prediction)
    ).inc()
    PREDICTION_LATENCY.labels(model_role=model_role).observe(latency_ms / 1000)
    if prediction == 1:
        FRAUD_DETECTED.labels(model_role=model_role).inc()

    log_prediction(
        transaction_id=transaction.transaction_id,
        user_id=transaction.user_id,
        model_role=model_role,
        model_name=model_info["model_name"],
        prediction=prediction,
        probability=probability,
        latency_ms=latency_ms
    )

    return PredictionResponse(
        transaction_id=transaction.transaction_id,
        is_fraud=bool(prediction),
        fraud_probability=round(probability, 4),
        model_role=model_role,
        model_name=model_info["model_name"],
        latency_ms=round(latency_ms, 2)
    )


@app.get("/models/stats")
async def model_stats():
    return get_model_stats()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": list(TRAFFIC_CONFIG.keys()),
        "redis": "connected"
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return generate_latest()


@app.get("/recent-predictions")
async def recent_predictions(limit: int = 20):
    raw = redis_client.lrange("prediction_log", 0, limit - 1)
    return [json.loads(r) for r in raw]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)