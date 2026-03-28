import random
import redis
import json
from datetime import datetime

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

TRAFFIC_CONFIG = {
    "champion": {
        "model_name": "fraud-xgboost",
        "weight": 0.80,
        "version": "v3"
    },
    "challenger": {
        "model_name": "fraud-random-forest",
        "weight": 0.15,
        "version": "v2"
    },
    "shadow": {
        "model_name": "fraud-logistic-regression",
        "weight": 0.05,
        "version": "v1"
    }
}


def get_model_assignment(user_id: str) -> str:
    roll = random.random()
    cumulative = 0.0
    for role, config in TRAFFIC_CONFIG.items():
        cumulative += config["weight"]
        if roll <= cumulative:
            return role
    return "champion"


def log_prediction(
    transaction_id: str,
    user_id: str,
    model_role: str,
    model_name: str,
    prediction: int,
    probability: float,
    latency_ms: float
):
    record = {
        "transaction_id": transaction_id,
        "user_id": user_id,
        "model_role": model_role,
        "model_name": model_name,
        "prediction": prediction,
        "probability": round(probability, 4),
        "latency_ms": round(latency_ms, 2),
        "timestamp": datetime.utcnow().isoformat()
    }
    redis_client.lpush("prediction_log", json.dumps(record))
    redis_client.ltrim("prediction_log", 0, 49999)

    key = f"model_stats:{model_role}"
    redis_client.hincrby(key, "total_predictions", 1)
    if prediction == 1:
        redis_client.hincrby(key, "fraud_predictions", 1)


def get_model_stats() -> dict:
    stats = {}
    for role in TRAFFIC_CONFIG:
        key = f"model_stats:{role}"
        raw = redis_client.hgetall(key)
        total = int(raw.get("total_predictions", 0))
        fraud = int(raw.get("fraud_predictions", 0))
        stats[role] = {
            "model_name": TRAFFIC_CONFIG[role]["model_name"],
            "version": TRAFFIC_CONFIG[role]["version"],
            "weight": TRAFFIC_CONFIG[role]["weight"],
            "total_predictions": total,
            "fraud_predictions": fraud,
            "fraud_rate": round(fraud / total, 4) if total > 0 else 0.0
        }
    return stats


def update_traffic_config(champion_weight: float, challenger_weight: float):
    shadow_weight = round(1.0 - champion_weight - challenger_weight, 2)
    TRAFFIC_CONFIG["champion"]["weight"] = champion_weight
    TRAFFIC_CONFIG["challenger"]["weight"] = challenger_weight
    TRAFFIC_CONFIG["shadow"]["weight"] = shadow_weight
    print(f"Traffic updated: champion={champion_weight} "
          f"challenger={challenger_weight} shadow={shadow_weight}")