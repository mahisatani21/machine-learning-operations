import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("data/student.csv")

X = df[["pthours"]]
y = df["salary"]

model = LinearRegression()

model.fit(X,y)

joblib.dump(model,"model/model.pkl")

print("Model Saved")
