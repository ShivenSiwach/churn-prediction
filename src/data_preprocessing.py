import pandas as pd

df = pd.read_csv("C:/Users/Dell/churn-prediction/data/telco_churn_cleaned.csv")
print(df.shape)

#binary yes/no column - 0/1
binary_cols = [  "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

for col in binary_cols:
   df[col] = (df[col] == "Yes").astype(int)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print(df[["Partner","Dependents","Churn"]].head())   
print("Binary encoding is done")

# Multi-category columns -> one-hot encoding
multi_cols = ["InternetService","Contract","PaymentMethod","gender"]
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
print(df.shape)
print(df.columns.tolist())
print("one-hot encoding done")

# Feature engineering
import numpy as np

df["avg_monthly_spend"] = np.where(
   df["tenure"] > 0,
   df["TotalCharges"]/df["tenure"],
   df["MonthlyCharges"] #fallback for tenure=0 customers
)

print(df["avg_monthly_spend"].describe())
print("feature engineering done")

from sklearn.model_selection import train_test_split
x = df.drop("Churn", axis=1)
y = df["Churn"]

x_train ,x_test, y_train, y_test = train_test_split(
   x, y,
   test_size=0.2,
   random_state=42,
   stratify=y
)

print(f"Train;{x_train.shape[0]} rows | Test: {x_test.shape[0]} rows")
print(f"Churn rate - train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")
print("Train/test split done")

from sklearn.preprocessing import StandardScaler
import pickle
import os
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]

scaler = StandardScaler()
x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols]) # only transform, not fit!

os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
   pickle.dump(scaler,f)

print(x_train[num_cols].describe().round(2))
print("scaling done")
print("Scaler save -> model/scaler.pkl")

# Save train and test splits
x_train["Churn"] = y_train
x_test["Churn"] = y_test

x_train.to_csv("data/train.csv", index=False)
x_test.to_csv("data/test.csv", index=False)

print("Train and test saved -> data/train.csv, data/test.csv")