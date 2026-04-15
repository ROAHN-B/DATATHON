# Telco Customer Churn Prediction System

An end-to-end Machine Learning application designed to predict the likelihood of customer attrition in the telecommunications sector. This project features a full-stack architecture comprising a data science pipeline, a FastAPI backend, and a Streamlit interactive dashboard.

## 🚀 Project Overview
Customer churn is a critical KPI for telecom providers. This system allows businesses to input customer demographics and service data to receive real-time churn risk assessments using two different classification algorithms: **Logistic Regression** and **K-Nearest Neighbors (KNN)**.

## 🏗️ System Architecture
The project is structured into three main layers:
1.  **ML Pipeline (`ml.py` & `analysis.py`):** Handles data cleaning, Exploratory Data Analysis (EDA), feature scaling, and model serialization.
2.  **Backend API (`main.py`):** A RESTful API built with FastAPI that serves the trained models and processes prediction requests.
3.  **Frontend Dashboard (`app.py`):** A user-friendly Streamlit interface for data entry and visualization of results.

---

## 🛠️ Tech Stack
* **Language:** Python 3.12+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **API Framework:** FastAPI (Uvicorn)
* **Frontend:** Streamlit
* **Serialization:** Joblib

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd DATATHON-master

### 2. Create environment
python -m venv venv
venv\Scripts\activate

### 3. Install dependencies
pip install pandas numpy scikit-learn fastapi uvicorn streamlit requests joblib

---

####HOW TO RUN PROJECT
step: 1 start backend
uvicorn main:app --reload

step:2 start frontend
streamlit run app.py


####The dashboard will automatically open in your default browser at http://localhost:8501.

---

###Repository Structure 
├── Machine_learning_model/
│   └── ml.py               # Training pipeline and model saving
├── models/
│   ├── knn_model.pkl       # Serialized KNN model
│   ├── logistic_regression.pkl 
│   └── scaler.pkl          # Saved StandardScaler
├── main.py                 # FastAPI backend entry point
├── app.py                  # Streamlit frontend entry point
├── analysis.py             # Data visualization and EDA
└── Telco_customer.csv      # Source Dataset

