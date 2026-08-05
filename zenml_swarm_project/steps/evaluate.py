from zenml import step
from sklearn.metrics import accuracy_score


@step
def evaluate_model(model, X_test, y_test) -> float:

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction,
    )

    print("Accuracy =", accuracy)

    return accuracy
