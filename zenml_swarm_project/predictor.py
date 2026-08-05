"""
Loads whichever model version is currently tagged 'champion' in the MLflow
Model Registry — not just the newest version. A newest version only becomes
champion in steps/register_model.py if its accuracy clears the promotion
threshold, so this script always predicts with the best-known model, even
if the most recent training run underperformed and stayed a 'candidate'.
"""
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "BreastCancerModel"

mlflow.set_tracking_uri("file:/app/mlruns")
client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
champions = [v for v in versions if v.tags.get("stage") == "champion"]

if not champions:
    raise SystemExit(
        f"No champion version found for '{MODEL_NAME}'. "
        "Run the training pipeline at least once first."
    )

champion = max(champions, key=lambda v: int(v.version))
print(f"Loading champion version {champion.version} "
      f"(accuracy={champion.tags.get('accuracy')})")

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{champion.version}")

sample = [[15, 20, 100, 500, 0.1, 0.2, 0.15, 0.1, 0.2, 0.06,
           0.5, 1.2, 3.5, 40, 0.006, 0.02, 0.03, 0.01, 0.02, 0.003,
           20, 25, 140, 900, 0.14, 0.3, 0.4, 0.18, 0.3, 0.08]]

prediction = model.predict(sample)
print("Prediction:", prediction)
