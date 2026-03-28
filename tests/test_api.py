import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestABRouter:
    def test_model_assignment_returns_valid_role(self):
        from serving.ab_router import get_model_assignment, TRAFFIC_CONFIG
        for _ in range(50):
            role = get_model_assignment("user_test")
            assert role in TRAFFIC_CONFIG

    def test_traffic_weights_sum_to_one(self):
        from serving.ab_router import TRAFFIC_CONFIG
        total = sum(c["weight"] for c in TRAFFIC_CONFIG.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_roles_present(self):
        from serving.ab_router import TRAFFIC_CONFIG
        assert "champion" in TRAFFIC_CONFIG
        assert "challenger" in TRAFFIC_CONFIG
        assert "shadow" in TRAFFIC_CONFIG

    def test_champion_has_highest_weight(self):
        from serving.ab_router import TRAFFIC_CONFIG
        weights = {role: c["weight"] for role, c in TRAFFIC_CONFIG.items()}
        assert weights["champion"] == max(weights.values())


class TestPredictionLogic:
    def test_high_risk_transaction_flagged(self):
        from serving.app import get_mock_prediction
        features = [2500.0, 3.2, 45.0, 12.0, 5, 12, 3, 1]
        pred, prob = get_mock_prediction(features, "champion")
        assert pred == 1
        assert prob > 0.5

    def test_low_risk_transaction_cleared(self):
        from serving.app import get_mock_prediction
        features = [24.99, 0.1, 30.0, 8.0, 1, 1, 14, 0]
        pred, prob = get_mock_prediction(features, "champion")
        assert pred == 0
        assert prob < 0.5

    def test_probability_in_valid_range(self):
        from serving.app import get_mock_prediction
        test_cases = [
            [100.0, 0.5, 80.0, 20.0, 2, 3, 12, 0],
            [3000.0, 4.0, 50.0, 15.0, 6, 15, 2, 1],
            [15.0, -0.5, 25.0, 5.0, 1, 1, 18, 0],
        ]
        for features in test_cases:
            _, prob = get_mock_prediction(features, "champion")
            assert 0.0 <= prob <= 1.0