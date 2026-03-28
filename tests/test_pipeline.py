import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPipelineCompilation:
    def test_pipeline_yaml_exists(self):
        assert os.path.exists("pipeline.yaml"), \
            "pipeline.yaml not found — run: python pipelines/pipeline.py"

    def test_pipeline_yaml_not_empty(self):
        assert os.path.getsize("pipeline.yaml") > 100

    def test_pipeline_yaml_valid(self):
        import yaml
        with open("pipeline.yaml") as f:
            content = yaml.safe_load(f)
        assert content is not None
        assert isinstance(content, dict)

    def test_pipeline_components_importable(self):
        from pipelines.components.train import train_model
        from pipelines.components.evaluate import evaluate_and_select
        from pipelines.components.deploy import deploy_best_model
        assert train_model is not None
        assert evaluate_and_select is not None
        assert deploy_best_model is not None

    def test_pipeline_function_importable(self):
        from pipelines.pipeline import fraud_detection_pipeline
        assert fraud_detection_pipeline is not None


class TestFeatureConsumer:
    def test_compute_features_structure(self):
        from data_pipeline.consumers.feature_consumer import compute_features

        txn = {
            "transaction_id": "test-001",
            "user_id": "user_test",
            "amount": 150.0,
            "country": "US",
            "hour_of_day": 14,
            "transactions_last_hour": 2,
            "is_international": False,
            "is_fraud": False,
            "timestamp": "2026-01-01T12:00:00"
        }
        features = compute_features(txn)

        required_keys = [
            "transaction_id", "user_id", "amount",
            "amount_zscore", "avg_amount_last_50",
            "transactions_last_hour", "hour_of_day",
            "is_international", "is_fraud"
        ]
        for key in required_keys:
            assert key in features, f"Missing key: {key}"

    def test_fraud_flag_preserved(self):
        from data_pipeline.consumers.feature_consumer import compute_features

        txn = {
            "transaction_id": "test-002",
            "user_id": "user_fraud",
            "amount": 3000.0,
            "country": "NG",
            "hour_of_day": 3,
            "transactions_last_hour": 15,
            "is_international": True,
            "is_fraud": True,
            "timestamp": "2026-01-01T03:00:00"
        }
        features = compute_features(txn)
        assert features["is_fraud"] == 1