import os
import subprocess
from typing import Tuple, Any

from zenml import step
from typing_extensions import Annotated

import optuna
import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def _git_commit_sha() -> str:
    """Best-effort short git SHA of the current commit, for traceability.

    Falls back to 'untracked' if this isn't running inside a git repo with
    at least one commit (e.g. a fresh clone before the first `git commit`).
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "untracked"


def _dvc_data_hash(dvc_file: str = "data/breast_cancer.csv.dvc") -> str:
    """Read the md5 hash DVC recorded for the current dataset version.

    This is what ties a trained model back to the *exact* data file it was
    trained on, independent of git history — see scripts/version_data.sh.
    """
    if not os.path.exists(dvc_file):
        return "unversioned"
    try:
        with open(dvc_file) as f:
            for line in f:
                if "md5:" in line:
                    return line.split("md5:")[1].strip()
    except Exception:
        pass
    return "unknown"


@step(enable_cache=False)
def train_model(
    X_train, y_train
) -> Tuple[Annotated[Any, "model"], Annotated[str, "run_id"]]:

    mlflow.set_tracking_uri("file:/app/mlruns")
    mlflow.set_experiment("ZenML_Optuna_Swarm")

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 20)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )

        score = cross_val_score(
            clf,
            X_train,
            y_train,
            cv=5,
        ).mean()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best = study.best_params

    model = RandomForestClassifier(**best)
    model.fit(X_train, y_train)

    git_sha = _git_commit_sha()
    data_hash = _dvc_data_hash()

    with mlflow.start_run() as run:
        mlflow.log_params(best)
        mlflow.set_tag("git_commit", git_sha)
        mlflow.set_tag("data_version_md5", data_hash)
        mlflow.sklearn.log_model(model, "RandomForest")
        run_id = run.info.run_id

    return model, run_id
