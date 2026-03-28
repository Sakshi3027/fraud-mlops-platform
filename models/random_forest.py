import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import json

FEATURES = [
    "amount", "amount_zscore", "avg_amount_last_50",
    "std_amount_last_50", "unique_countries_count",
    "transactions_last_hour", "hour_of_day", "is_international"
]
TARGET = "is_fraud"
MODEL_NAME = "fraud-random-forest"


def generate_training_data(n_samples: int = 10000) -> pd.DataFrame:
    np.random.seed(123)
    n_fraud = int(n_samples * 0.02)
    n_legit = n_samples - n_fraud

    legit = pd.DataFrame({
        "amount": np.random.uniform(5, 300, n_legit),
        "hour_of_day": np.random.randint(8, 23, n_legit),
        "transactions_last_hour": np.random.randint(1, 5, n_legit),
        "is_international": np.random.choice([0, 1], n_legit, p=[0.9, 0.1]),
        "unique_countries_count": np.random.randint(1, 3, n_legit),
        "is_fraud": 0
    })

    fraud = pd.DataFrame({
        "amount": np.random.uniform(500, 5000, n_fraud),
        "hour_of_day": np.random.randint(1, 6, n_fraud),
        "transactions_last_hour": np.random.randint(8, 20, n_fraud),
        "is_international": np.ones(n_fraud, dtype=int),
        "unique_countries_count": np.random.randint(3, 8, n_fraud),
        "is_fraud": 1
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()
    df["avg_amount_last_50"] = df["amount"].rolling(50, min_periods=1).mean()
    df["std_amount_last_50"] = df["amount"].rolling(50, min_periods=1).std().fillna(0)

    return df


def train(mlflow_tracking_uri: str = None, artifact_location: str = None) -> dict:
    mlflow_tracking_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("fraud-detection")

    df = generate_training_data(10000)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    with mlflow.start_run(run_name="random-forest-v2") as run:
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "class_weight": "balanced",
            "random_state": 42
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "auc": round(roc_auc_score(y_test, y_prob), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4)
        }
        mlflow.log_metrics(metrics)

        importances = dict(zip(FEATURES, model.feature_importances_.tolist()))

        cm = confusion_matrix(y_test, y_pred).tolist()


        print(f"Random Forest      | AUC: {metrics['auc']} | "
              f"F1: {metrics['f1']} | Run ID: {run.info.run_id}")

        return {"run_id": run.info.run_id, "metrics": metrics, "model_name": MODEL_NAME}


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))