import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Telco_customer.csv")
print(df.info())
print(df.describe())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)
sns.set_theme(style="whitegrid")

# Pie chart
plt.figure(figsize=(6, 6))
df["Churn"].value_counts().plot.pie(
    autopct="%1.1f%%", colors=["#66b3ff", "#ff9999"], startangle=90
)
plt.title("overall churn percentage")
plt.show()

# correlation heatmap
df["churn_numeric"] = df["Churn"].map({"Yes": 1, "No": 0})
numeric_df = df.select_dtypes(include=["float64", "int64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.show()

#Bar graph
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Contract', hue='Churn', palette='viridis')
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.show()

#KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, palette='magma')
plt.title('Distribution of Monthly Charges by Churn Status')
plt.show()