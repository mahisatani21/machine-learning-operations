import joblib
import pandas as pd

model = joblib.load("model/model.pkl")

hours = pd.DataFrame({
    "pthours": [9]
})

prediction = model.predict(hours)

print("Prediction:", prediction[0])
