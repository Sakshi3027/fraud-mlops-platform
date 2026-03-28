from kfp import dsl
from kfp.dsl import Output, Metrics, Model


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "mlflow==3.10.1",
        "scikit-learn==1.5.1",
        "xgboost==2.1.1",
        "pandas==2.1.4",
        "numpy==1.26.4"
    ]
)
def train_model(
    model_type: str,
    n_samples: int,
    mlflow_tracking_uri: str,
    metrics: Output[Metrics],
    model_artifact: Output[Model]
):
    import numpy as np
    import pandas as pd
    import mlflow
    import json

    FEATURES = [
        "amount", "amount_zscore", "avg_amount_last_50",
        "std_amount_last_50", "unique_countries_count",
        "transactions_last_hour", "hour_of_day", "is_international"
    ]

    def make_data(n, seed):
        np.random.seed(seed)
        n_fraud = int(n * 0.02)
        n_legit = n - n_fraud
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
        df = pd.concat([legit, fraud]).sample(frac=1, random_state=42)
        df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()
        df["avg_amount_last_50"] = df["amount"].rolling(50, min_periods=1).mean()
        df["std_amount_last_50"] = df["amount"].rolling(50, min_periods=1).std().fillna(0)
        return df

    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    df = make_data(n_samples, seed=42)
    split = int(len(df) * 0.8)
    X_train = df.iloc[:split][FEATURES]
    y_train = df.iloc[:split]["is_fraud"]
    X_test = df.iloc[split:][FEATURES]
    y_test = df.iloc[split:]["is_fraud"]

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("fraud-detection-kubeflow")

    with mlflow.start_run(run_name=f"{model_type}-kubeflow") as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_samples", n_samples)

        if model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))
            ])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            mlflow.sklearn.log_model(model, "model")

        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            mlflow.sklearn.log_model(model, "model")

        elif model_type == "xgboost":
            import xgboost as xgb
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight, random_state=42,
                eval_metric="auc"
            )
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            mlflow.xgboost.log_model(model, "model")

        auc = round(roc_auc_score(y_test, y_prob), 4)
        f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
        precision = round(precision_score(y_test, y_pred, zero_division=0), 4)
        recall = round(recall_score(y_test, y_pred, zero_division=0), 4)

        mlflow.log_metrics({"auc": auc, "f1": f1, "precision": precision, "recall": recall})

        metrics.log_metric("auc", auc)
        metrics.log_metric("f1", f1)
        metrics.log_metric("precision", precision)
        metrics.log_metric("recall", recall)
        metrics.log_metric("run_id", 0)

        model_artifact.uri = run.info.run_id
        model_artifact.metadata["run_id"] = run.info.run_id
        model_artifact.metadata["model_type"] = model_type
        model_artifact.metadata["auc"] = str(auc)
        model_artifact.metadata["mlflow_uri"] = mlflow_tracking_uri

        print(f"{model_type} | AUC: {auc} | F1: {f1} | run_id: {run.info.run_id}")