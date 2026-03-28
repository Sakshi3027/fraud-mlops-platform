#!/bin/bash
set -e

echo "================================"
echo "Auto-retrain triggered by drift"
echo "Timestamp: $(date -u)"
echo "================================"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5001}"

echo "Training Logistic Regression..."
python models/logistic_regression.py

echo "Training Random Forest..."
python models/random_forest.py

echo "Training XGBoost..."
python models/xgboost_model.py

echo "Compiling Kubeflow pipeline..."
python -c "
import sys
sys.path.insert(0, '.')
from pipelines.pipeline import compile_pipeline
compile_pipeline('pipeline.yaml')
print('Pipeline compiled')
"

echo "Running drift check post-retrain..."
python pipelines/drift_detector.py

echo "================================"
echo "Auto-retrain complete!"
echo "Timestamp: $(date -u)"
echo "================================"