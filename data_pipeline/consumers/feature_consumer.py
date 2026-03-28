import json
import redis
import numpy as np
from kafka import KafkaConsumer
from datetime import datetime
from collections import defaultdict

KAFKA_TOPIC = "transactions"
KAFKA_BROKER = "localhost:9092"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True
)

user_stats = defaultdict(lambda: {
    "amounts": [],
    "countries": set(),
    "hourly_counts": defaultdict(int)
})

def compute_features(txn: dict) -> dict:
    user_id = txn["user_id"]
    stats = user_stats[user_id]

    stats["amounts"].append(txn["amount"])
    stats["countries"].add(txn["country"])
    hour = txn["hour_of_day"]
    stats["hourly_counts"][hour] += 1

    amounts = stats["amounts"][-50:]
    avg_amount = float(np.mean(amounts)) if amounts else 0.0
    std_amount = float(np.std(amounts)) if len(amounts) > 1 else 0.0
    amount_zscore = (
        (txn["amount"] - avg_amount) / std_amount
        if std_amount > 0 else 0.0
    )

    return {
        "transaction_id": txn["transaction_id"],
        "user_id": user_id,
        "amount": txn["amount"],
        "amount_zscore": round(amount_zscore, 4),
        "avg_amount_last_50": round(avg_amount, 2),
        "std_amount_last_50": round(std_amount, 2),
        "unique_countries_count": len(stats["countries"]),
        "transactions_last_hour": txn["transactions_last_hour"],
        "hour_of_day": txn["hour_of_day"],
        "is_international": int(txn["is_international"]),
        "is_fraud": int(txn["is_fraud"]),
        "timestamp": txn["timestamp"]
    }

def store_features(features: dict):
    key = f"features:{features['user_id']}"
    redis_client.hset(key, mapping={
        k: str(v) for k, v in features.items()
    })
    redis_client.expire(key, 3600)

    redis_client.lpush("feature_log", json.dumps(features))
    redis_client.ltrim("feature_log", 0, 9999)

def run_consumer():
    print(f"Starting feature consumer ← topic: {KAFKA_TOPIC}")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="feature-engineering-group",
        auto_offset_reset="latest"
    )

    count = 0
    fraud_count = 0

    for message in consumer:
        txn = message.value
        features = compute_features(txn)
        store_features(features)

        count += 1
        if features["is_fraud"]:
            fraud_count += 1

        if count % 100 == 0:
            print(f"Processed: {count} | "
                  f"Fraud detected: {fraud_count} | "
                  f"Fraud rate: {fraud_count/count*100:.1f}%")

if __name__ == "__main__":
    run_consumer()