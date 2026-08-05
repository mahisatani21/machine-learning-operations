import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from zenml import step


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "breast_cancer.csv")


@step
def ingest_data() -> pd.DataFrame:
    """Load the dataset and also write a versioned copy to data/.

    The CSV written here is what DVC tracks (see scripts/version_data.sh).
    ZenML separately versions this DataFrame as a pipeline artifact, so the
    data is versioned twice, in two different systems, for two different
    reasons:
      - DVC + Git: lets you check out *which exact file* fed a given
        training run, and diff/roll back raw data over time.
      - ZenML artifact store: lets you inspect the exact in-memory object
        that flowed into a specific pipeline run, without touching Git.
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    return df
