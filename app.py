# Churn Prediction Dashboard

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Customer Churn Predictor")

col1, col2 = st.columns(2)
input_df = pd.DataFrame()

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col2:
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 840.0)

if st.button("Predict Churn Risk"):
    customer = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([customer])
    cat_cols = input_df.select_dtypes(include="str").columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=False)
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_final[num_cols] = scaler.transform(input_final[num_cols])

    probability = model.predict_proba(input_final)[0][1]
    prediction = "Yes" if probability >= 0.5 else "No"

    st.subheader("Result")
    if prediction == "Yes":
        st.error(f"Likely to Churn — Probability: {probability:.1%}")
    else:
        st.success(f"Likely to Stay — Probability of Churn: {probability:.1%}")

    st.subheader("Top Risk Factors")
    importances = pd.Series(model.feature_importances_, index=model_columns)
    top_factors = importances.sort_values(ascending=False).head(5)
    st.bar_chart(top_factors)

    st.subheader("Retention Suggestion")
    if contract == "Month-to-month" and tenure < 12:
        st.info("New customer on a flexible plan offer a discounted 1-year contract to lock in loyalty.")
    elif monthly_charges > 80 and prediction == "Yes":
        st.info("High monthly charges with churn risk consider a loyalty discount or bundle offer.")
    elif online_security == "No" and tech_support == "No" and prediction == "Yes":
        st.info("No security or tech support add ons offer a free trial of these services.")
    elif prediction == "Yes":
        st.info("Customer flagged as at risk recommend proactive outreach from retention team.")
    else:
        st.info("Customer profile looks stable no urgent action needed.")