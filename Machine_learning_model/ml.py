from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib

df = pd.read_csv("Telco_customer.csv")

binary_columns = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "Churn",
    "gender",
]

le = LabelEncoder()
for col in binary_columns:
    df[col] = le.fit_transform(df[col])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

x = df.drop(
    columns=[
        "Churn",
        "customerID",
        "MultipleLines",
        "InternetService",
        "Contract",
        "PaymentMethod",
    ]
)

y = df["Churn"]

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(x_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)

log_pred = log_reg.predict(x_test)
knn_pred = knn_model.predict(x_test)

print(f"Logistic regression report: {classification_report(y_test, log_pred)}")
print(f"Confusion matrix of logistic regression: {confusion_matrix(y_test, log_pred)}")

print(
    "################################################################################"
)
print(f"Knn classification report: {classification_report(y_test, knn_pred)}")
print(f"Confusion matrix: {confusion_matrix(y_test, knn_pred)}")


joblib.dump(knn_model, "models/knn_model.pkl")
joblib.dump(log_reg, "models/logistic_regression.pkl")
joblib.dump(scaler, "models/scaler.pkl")
