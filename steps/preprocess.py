from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step
from typing_extensions import Annotated


@step
def preprocess(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test
