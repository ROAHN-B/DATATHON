import streamlit as st
import requests

# Set page configuration
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

st.title("Telco Customer Churn Prediction")
st.markdown("Enter customer details below to predict the likelihood of churn.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female")
    senior = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    partner = st.selectbox("Partner", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    dependents = st.selectbox("Dependents", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    phone = st.selectbox("Phone Service", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    paperless = st.selectbox("Paperless Billing", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")

with col2:
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
    security = st.selectbox("Online Security", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    backup = st.selectbox("Online Backup", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    protection = st.selectbox("Device Protection", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    support = st.selectbox("Tech Support", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    tv = st.selectbox("Streaming TV", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    movies = st.selectbox("Streaming Movies", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")

# Prepare the data dictionary
input_data = {
    "SeniorCitizen": senior,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalChargers": total,  # Matches your Pydantic model typo 'TotalChargers'
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone,
    "PaperlessBilling": paperless,
    "gender": gender,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": protection,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies
}

st.divider()

# Model Selection and Prediction
model_choice = st.radio("Choose Prediction Model", ["KNN", "Logistic Regression"], horizontal=True)

if st.button("Predict Churn Status"):
    endpoint = "predict/knn_model" if model_choice == "KNN" else "predict/Logistic_regression"
    
    try:
        response = requests.post(f"http://127.0.0.1:8000/{endpoint}", json=input_data)
        result = response.json()
        
        prediction = result["prediction"]
        
        if prediction == "YES":
            st.error(f"Prediction: **High Risk of Churn** Customer might not stay!!(Model: {result['model']})")
        else:
            st.success(f"Prediction: **Customer Likely to Stay** (Model: {result['model']})")
            
    except Exception as e:
        st.error(f"Error connecting to API. Ensure FastAPI is running! {e}")