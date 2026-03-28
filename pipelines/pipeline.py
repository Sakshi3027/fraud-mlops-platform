import kfp
from kfp import dsl, compiler
from pipelines.components.train import train_model
from pipelines.components.evaluate import evaluate_and_select
from pipelines.components.deploy import deploy_best_model

MLFLOW_TRACKING_URI = "http://host.docker.internal:5001"
SERVING_ENDPOINT = "http://localhost:8000"


@dsl.pipeline(
    name="fraud-detection-mlops-pipeline",
    description="Trains 3 fraud detection models in parallel, "
                "evaluates them, and auto-deploys the best one"
)
def fraud_detection_pipeline(
    n_samples: int = 10000,
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    serving_endpoint: str = SERVING_ENDPOINT
):
    # Train all 3 models in parallel
    lr_task = train_model(
        model_type="logistic_regression",
        n_samples=n_samples,
        mlflow_tracking_uri=mlflow_tracking_uri
    ).set_display_name("Train Logistic Regression")

    rf_task = train_model(
        model_type="random_forest",
        n_samples=n_samples,
        mlflow_tracking_uri=mlflow_tracking_uri
    ).set_display_name("Train Random Forest")

    xgb_task = train_model(
        model_type="xgboost",
        n_samples=n_samples,
        mlflow_tracking_uri=mlflow_tracking_uri
    ).set_display_name("Train XGBoost")

    # Evaluate all 3 and select best
    eval_task = evaluate_and_select(
        lr_model=lr_task.outputs["model_artifact"],
        rf_model=rf_task.outputs["model_artifact"],
        xgb_model=xgb_task.outputs["model_artifact"]
    ).set_display_name("Evaluate & Select Best Model")

    # Deploy the winner
    deploy_task = deploy_best_model(
        best_model_info=eval_task.outputs["best_model_info"],
        serving_endpoint=serving_endpoint
    ).set_display_name("Deploy Best Model")


def compile_pipeline(output_path: str = "pipeline.yaml"):
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path=output_path
    )
    print(f"Pipeline compiled → {output_path}")


if __name__ == "__main__":
    compile_pipeline()