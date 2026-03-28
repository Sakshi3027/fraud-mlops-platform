import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES = [
    "amount", "amount_zscore", "avg_amount_last_50",
    "std_amount_last_50", "unique_countries_count",
    "transactions_last_hour", "hour_of_day", "is_international"
]


def make_sample_data(n=100):
    np.random.seed(42)
    n_fraud = int(n * 0.1)
    n_legit = n - n_fraud
    legit = pd.DataFrame({
        "amount": np.random.uniform(5, 300, n_legit),
        "hour_of_day": np.random.randint(8, 23, n_legit),
        "transactions_last_hour": np.random.randint(1, 5, n_legit),
        "is_international": np.zeros(n_legit, dtype=int),
        "unique_countries_count": np.ones(n_legit, dtype=int),
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
    df = pd.concat([legit, fraud], ignore_index=True)
    df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()
    df["avg_amount_last_50"] = df["amount"].rolling(50, min_periods=1).mean()
    df["std_amount_last_50"] = df["amount"].rolling(50, min_periods=1).std().fillna(0)
    return df


class TestDataGeneration:
    def test_data_shape(self):
        df = make_sample_data(200)
        assert len(df) == 200

    def test_features_present(self):
        df = make_sample_data(100)
        for f in FEATURES:
            assert f in df.columns, f"Missing feature: {f}"

    def test_fraud_ratio(self):
        df = make_sample_data(1000)
        fraud_rate = df["is_fraud"].mean()
        assert 0.05 <= fraud_rate <= 0.20

    def test_no_nulls(self):
        df = make_sample_data(100)
        assert df[FEATURES].isnull().sum().sum() == 0

    def test_amount_range(self):
        df = make_sample_data(500)
        assert df["amount"].min() >= 0
        assert df["amount"].max() <= 6000


class TestLogisticRegression:
    def test_model_trains(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        df = make_sample_data(200)
        X = df[FEATURES]
        y = df["is_fraud"]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=100, class_weight="balanced"))
        ])
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_model_predicts_probabilities(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        df = make_sample_data(200)
        X = df[FEATURES]
        y = df["is_fraud"]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=100, class_weight="balanced"))
        ])
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_model_auc_above_threshold(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import roc_auc_score

        df = make_sample_data(500)
        X = df[FEATURES]
        y = df["is_fraud"]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
        ])
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        assert auc >= 0.7, f"AUC too low: {auc}"


class TestRandomForest:
    def test_model_trains(self):
        from sklearn.ensemble import RandomForestClassifier

        df = make_sample_data(200)
        X = df[FEATURES]
        y = df["is_fraud"]

        model = RandomForestClassifier(
            n_estimators=10, class_weight="balanced", random_state=42
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_feature_importances(self):
        from sklearn.ensemble import RandomForestClassifier

        df = make_sample_data(200)
        X = df[FEATURES]
        y = df["is_fraud"]

        model = RandomForestClassifier(
            n_estimators=10, class_weight="balanced", random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_
        assert len(importances) == len(FEATURES)
        assert abs(importances.sum() - 1.0) < 1e-6


class TestXGBoost:
    def test_model_trains(self):
        import xgboost as xgb

        df = make_sample_data(200)
        X = df[FEATURES]
        y = df["is_fraud"]

        model = xgb.XGBClassifier(
            n_estimators=10, random_state=42,
            eval_metric="auc"
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_model_auc_above_threshold(self):
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        df = make_sample_data(500)
        X = df[FEATURES]
        y = df["is_fraud"]

        model = xgb.XGBClassifier(
            n_estimators=50, random_state=42,
            eval_metric="auc"
        )
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        assert auc >= 0.7, f"AUC too low: {auc}"