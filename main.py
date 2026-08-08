# Churn Prediction API

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Churn Prediction API is Running"}


@app.post("/predict")
def predict(customer: Customer):
    input_df = pd.DataFrame([customer.model_dump()])

    cat_cols = input_df.select_dtypes(include="str").columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=False)

    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_final[num_cols] = scaler.transform(input_final[num_cols])

    probability = model.predict_proba(input_final)[0][1]
    prediction = int(probability >= 0.5)

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }
