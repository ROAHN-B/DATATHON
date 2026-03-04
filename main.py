import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app=FastAPI(tilt="Telco customer churn prediction dataset")

knn=joblib.load("models/knn_model.pkl")
log_reg=joblib.load("models/logistic_regression.pkl")
scaler=joblib.load("models/scaler.pkl")

app=FastAPI()

class CustomerData(BaseModel):
        SeniorCitizen:int
        tenure:int
        MonthlyCharges: float
        TotalChargers:float
        Partner:int
        Dependents:int
        PhoneService:int
        PaperlessBilling:int
        gender:int
        OnlineSecurity:int
        OnlineBackup:int
        DeviceProtection:int
        TechSupport:int
        StreamingTV:int
        StreamingMovies:int
@app.post("/predict/knn_model")
def predict(data:CustomerData):
        x=np.array([[
              data.SeniorCitizen,
                data.tenure,
                data.MonthlyCharges,
                data.TotalChargers,
                data.Partner,
                data.Dependents,
                data.PhoneService,
                data.PaperlessBilling,
                data.gender,
                data.OnlineSecurity,
                data.OnlineBackup,
                data.DeviceProtection,
                data.TechSupport,
                data.StreamingTV,
                data.StreamingMovies  
        ]]
        )

        x_scaled=scaler.transform(x)
        pred=knn.predict(x_scaled)[0]

        return {
                "model": "KNN CLASSIFICATION",
                "prediction": "YES" if pred==1 else "NO"
        }
@app.post("/predict/Logistic_regression")
def predict_logistic(data:CustomerData):
        x=np.array([[
                data.SeniorCitizen,
                data.tenure,
                data.MonthlyCharges,
                data.TotalChargers,
                data.Partner,
                data.Dependents,
                data.PhoneService,
                data.PaperlessBilling,
                data.gender,
                data.OnlineSecurity,
                data.OnlineBackup,
                data.DeviceProtection,
                data.TechSupport,
                data.StreamingTV,
                data.StreamingMovies
        ]])

        x_scaled=scaler.transform(x)
        pred=log_reg.predict(x_scaled)[0]

        return {
                "model": "LOGISTIC REGRESSION",
                "prediction": "YES" if pred==1 else "NO"
        }
