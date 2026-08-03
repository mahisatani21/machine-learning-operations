import mlflow
from mlflow.tracking import MlflowClient
from zenml import step


MODEL_NAME = "BreastCancerModel"
PROMOTION_THRESHOLD = 0.95


@step
def register_model(run_id: str, accuracy: float) -> int:
    """Register this run's model as a new version of MODEL_NAME.

    Every call creates a new, immutable version (v1, v2, v3, ...) in the
    MLflow Model Registry — nothing is overwritten. If the new version's
    accuracy clears PROMOTION_THRESHOLD, it's tagged 'champion' and any
    previous champion is demoted to 'previous_champion'. This gives you an
    explicit, queryable model version history instead of just "whatever
    predictor.py happens to load".
    """
    mlflow.set_tracking_uri("file:/app/mlruns")
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/RandomForest"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version
    client.set_model_version_tag(MODEL_NAME, version, "accuracy", f"{accuracy:.4f}")

    if accuracy >= PROMOTION_THRESHOLD:
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if mv.version != version:
                current_tags = client.get_model_version(MODEL_NAME, mv.version).tags
                if current_tags.get("stage") == "champion":
                    client.set_model_version_tag(
                        MODEL_NAME, mv.version, "stage", "previous_champion"
                    )
        client.set_model_version_tag(MODEL_NAME, version, "stage", "champion")
        print(f"Registered {MODEL_NAME} v{version} — accuracy {accuracy:.4f} — promoted to champion")
    else:
        client.set_model_version_tag(MODEL_NAME, version, "stage", "candidate")
        print(
            f"Registered {MODEL_NAME} v{version} — accuracy {accuracy:.4f} "
            f"— below {PROMOTION_THRESHOLD} threshold, kept as candidate"
        )

    return version
