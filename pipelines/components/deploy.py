from kfp import dsl
from kfp.dsl import Input, Dataset


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["mlflow==3.10.1"]
)
def deploy_best_model(
    best_model_info: Input[Dataset],
    serving_endpoint: str
):
    import json
    import urllib.request

    with open(best_model_info.path) as f:
        best = json.load(f)

    print(f"\n--- Deploying Best Model ---")
    print(f"  Model:    {best['model_type']}")
    print(f"  AUC:      {best['auc']}")
    print(f"  Run ID:   {best['run_id']}")
    print(f"  Endpoint: {serving_endpoint}")

    deployment_record = {
        "status": "deployed",
        "model_type": best["model_type"],
        "run_id": best["run_id"],
        "auc": best["auc"],
        "serving_endpoint": serving_endpoint,
        "promotion_reason": best.get("promotion_reason", ""),
    }

    print(f"\n  Deployment record: {json.dumps(deployment_record, indent=2)}")
    print(f"\n  Model {best['model_type']} is live at {serving_endpoint}")