from kfp import dsl
from kfp.dsl import Input, Output, Metrics, Model, Dataset


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["mlflow==3.10.1"]
)
def evaluate_and_select(
    lr_model: Input[Model],
    rf_model: Input[Model],
    xgb_model: Input[Model],
    best_model_info: Output[Dataset]
):
    import json

    candidates = [
        {
            "model_type": lr_model.metadata.get("model_type", "logistic_regression"),
            "run_id": lr_model.metadata.get("run_id", ""),
            "auc": float(lr_model.metadata.get("auc", 0)),
            "mlflow_uri": lr_model.metadata.get("mlflow_uri", "")
        },
        {
            "model_type": rf_model.metadata.get("model_type", "random_forest"),
            "run_id": rf_model.metadata.get("run_id", ""),
            "auc": float(rf_model.metadata.get("auc", 0)),
            "mlflow_uri": rf_model.metadata.get("mlflow_uri", "")
        },
        {
            "model_type": xgb_model.metadata.get("model_type", "xgboost"),
            "run_id": xgb_model.metadata.get("run_id", ""),
            "auc": float(xgb_model.metadata.get("auc", 0)),
            "mlflow_uri": xgb_model.metadata.get("mlflow_uri", "")
        }
    ]

    print("\n--- Model Comparison ---")
    for c in candidates:
        print(f"  {c['model_type']:25s} | AUC: {c['auc']:.4f}")

    best = max(candidates, key=lambda x: x["auc"])
    best["promotion_reason"] = (
        f"Highest AUC ({best['auc']:.4f}) among "
        f"{len(candidates)} candidates"
    )
    best["promoted_at"] = "pipeline_run"

    print(f"\n  Winner: {best['model_type']} | AUC: {best['auc']:.4f}")
    print(f"  Reason: {best['promotion_reason']}")

    with open(best_model_info.path, "w") as f:
        json.dump(best, f, indent=2)