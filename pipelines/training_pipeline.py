from zenml import pipeline

from steps.ingest import ingest_data
from steps.preprocess import preprocess
from steps.train import train_model
from steps.evaluate import evaluate_model
from steps.register_model import register_model


@pipeline
def training_pipeline():

    df = ingest_data()

    X_train, X_test, y_train, y_test = preprocess(df)

    model, run_id = train_model(
        X_train,
        y_train,
    )

    accuracy = evaluate_model(
        model,
        X_test,
        y_test,
    )

    register_model(run_id, accuracy)
