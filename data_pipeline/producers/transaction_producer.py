import json
import random
import time
import uuid
from datetime import datetime
from kafka import KafkaProducer

KAFKA_TOPIC = "transactions"
KAFKA_BROKER = "localhost:9092"

MERCHANTS = ["Amazon", "Walmart", "Target", "BestBuy", "Starbucks",
             "McDonald's", "Shell", "CVS", "Walgreens", "Apple"]
COUNTRIES = ["US", "US", "US", "US", "UK", "CA", "MX", "NG", "RU", "CN"]
CARD_TYPES = ["visa", "mastercard", "amex", "discover"]

def generate_transaction(fraud_rate: float = 0.02) -> dict:
    is_fraud = random.random() < fraud_rate

    if is_fraud:
        amount = round(random.uniform(500, 5000), 2)
        country = random.choice(["NG", "RU", "CN", "MX"])
        hour = random.randint(1, 5)
        merchant = random.choice(MERCHANTS)
        velocity = random.randint(8, 20)
    else:
        amount = round(random.uniform(5, 300), 2)
        country = "US"
        hour = random.randint(8, 22)
        merchant = random.choice(MERCHANTS)
        velocity = random.randint(1, 4)

    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": f"user_{random.randint(1000, 9999)}",
        "amount": amount,
        "merchant": merchant,
        "country": country,
        "card_type": random.choice(CARD_TYPES),
        "hour_of_day": hour,
        "transactions_last_hour": velocity,
        "is_international": country != "US",
        "is_fraud": is_fraud,
        "timestamp": datetime.utcnow().isoformat()
    }

def create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8")
    )

def run_producer(transactions_per_second: int = 10):
    print(f"Starting transaction producer → topic: {KAFKA_TOPIC}")
    producer = create_producer()
    count = 0

    try:
        while True:
            for _ in range(transactions_per_second):
                txn = generate_transaction()
                producer.send(
                    KAFKA_TOPIC,
                    key=txn["user_id"],
                    value=txn
                )
                count += 1

            producer.flush()
            fraud_count = 0
            print(f"Sent {count} transactions total | "
                  f"Rate: {transactions_per_second}/sec | "
                  f"Time: {datetime.utcnow().strftime('%H:%M:%S')}")
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\nProducer stopped. Total sent: {count}")
        producer.close()

if __name__ == "__main__":
    run_producer(transactions_per_second=10)