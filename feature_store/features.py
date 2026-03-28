from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64, String
from feast import FeatureStore
import pandas as pd
import numpy as np
import os

USER_ENTITY = Entity(
    name="user_id",
    description="Unique user identifier for transaction features"
)

TRANSACTION_SOURCE = FileSource(
    name="transaction_features_source",
    path=os.path.join(
        os.path.dirname(__file__),
        "data/transaction_features.parquet"
    ),
    timestamp_field="event_timestamp"
)

TRANSACTION_FEATURE_VIEW = FeatureView(
    name="transaction_features",
    entities=[USER_ENTITY],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="amount",                  dtype=Float64),
        Field(name="amount_zscore",           dtype=Float64),
        Field(name="avg_amount_last_50",      dtype=Float64),
        Field(name="std_amount_last_50",      dtype=Float64),
        Field(name="unique_countries_count",  dtype=Int64),
        Field(name="transactions_last_hour",  dtype=Int64),
        Field(name="hour_of_day",             dtype=Int64),
        Field(name="is_international",        dtype=Int64),
    ],
    source=TRANSACTION_SOURCE,
    tags={"team": "fraud-detection", "version": "v1"}
)


def generate_feature_data(n_samples: int = 10000) -> pd.DataFrame:
    np.random.seed(42)
    n_fraud = int(n_samples * 0.02)
    n_legit = n_samples - n_fraud

    legit = pd.DataFrame({
        "user_id": [f"user_{i:04d}" for i in range(n_legit)],
        "amount": np.random.uniform(5, 300, n_legit),
        "hour_of_day": np.random.randint(8, 23, n_legit),
        "transactions_last_hour": np.random.randint(1, 5, n_legit),
        "is_international": np.random.choice([0, 1], n_legit, p=[0.9, 0.1]),
        "unique_countries_count": np.random.randint(1, 3, n_legit),
        "is_fraud": 0
    })

    fraud = pd.DataFrame({
        "user_id": [f"user_fraud_{i:04d}" for i in range(n_fraud)],
        "amount": np.random.uniform(500, 5000, n_fraud),
        "hour_of_day": np.random.randint(1, 6, n_fraud),
        "transactions_last_hour": np.random.randint(8, 20, n_fraud),
        "is_international": np.ones(n_fraud, dtype=int),
        "unique_countries_count": np.random.randint(3, 8, n_fraud),
        "is_fraud": 1
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    df["amount_zscore"] = (
        (df["amount"] - df["amount"].mean()) / df["amount"].std()
    )
    df["avg_amount_last_50"] = (
        df["amount"].rolling(50, min_periods=1).mean()
    )
    df["std_amount_last_50"] = (
        df["amount"].rolling(50, min_periods=1).std().fillna(0)
    )
    df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    return df


def materialize_features():
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "data"),
        exist_ok=True
    )
    data_path = os.path.join(
        os.path.dirname(__file__),
        "data/transaction_features.parquet"
    )

    print("Generating feature data...")
    df = generate_feature_data(10000)
    df.to_parquet(data_path, index=False)
    print(f"Saved {len(df)} rows to {data_path}")

    store = FeatureStore(
        repo_path=os.path.dirname(__file__)
    )

    print("Applying feature store...")
    store.apply([USER_ENTITY, TRANSACTION_FEATURE_VIEW])

    print("Materializing features to online store...")
    from datetime import datetime, timezone
    store.materialize_incremental(
        end_date=datetime.now(tz=timezone.utc)
    )
    print("Feature store ready!")
    return store


def get_online_features(user_ids: list) -> pd.DataFrame:
    store = FeatureStore(
        repo_path=os.path.dirname(__file__)
    )
    feature_vector = store.get_online_features(
        features=[
            "transaction_features:amount",
            "transaction_features:amount_zscore",
            "transaction_features:avg_amount_last_50",
            "transaction_features:std_amount_last_50",
            "transaction_features:unique_countries_count",
            "transaction_features:transactions_last_hour",
            "transaction_features:hour_of_day",
            "transaction_features:is_international",
        ],
        entity_rows=[{"user_id": uid} for uid in user_ids]
    ).to_df()
    return feature_vector


if __name__ == "__main__":
    store = materialize_features()
    print("\nTesting online feature retrieval...")
    features = get_online_features(["user_0001", "user_0002", "user_fraud_0001"])
    print(features.to_string())