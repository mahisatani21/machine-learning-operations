import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
df = pd.read_csv("data/student.csv")

# Features and target
X = df[["Pthours"]]
y = df["salary"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("Model Saved")

