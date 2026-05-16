from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# create FastAPI app
app = FastAPI()

# load model and scaler 
with open ("models/xgb_model.pkl", "rb") as f:
   model = pickle.load(f)

with open ("models/scaler.pkl", "rb") as f:
   scaler = pickle.load(f)

print("model and scaler loaded")     

class CustomerData(BaseModel):
    # Numeric (4)
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    avg_monthly_spend: float
    # Binary (12)
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    PaperlessBilling: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    # One-hot encoded (8)
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Credit_card: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int
    gender_Male: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API! Go to /docs to test the model."}
def predict(data: CustomerData):
   # convert to dataframe
   df = pd.DataFrame([data.dict()]) 

   # 1. FIX: Map Pydantic safe names back to XGBoost expected names
   rename_map = {
       "InternetService_Fiber_optic": "InternetService_Fiber optic",
       "Contract_One_year": "Contract_One year",
       "Contract_Two_year": "Contract_Two year",
       "PaymentMethod_Credit_card": "PaymentMethod_Credit card (automatic)",
       "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
       "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check"
   }
   df = df.rename(columns=rename_map)

   # 2. Force the correct column order
   expected_cols = model.get_booster().feature_names
   df = df[expected_cols]

   # 3. scale numeric columns
   num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]
   df[num_cols] = scaler.transform(df[num_cols])

  # 4. predict
   prediction = model.predict(df)[0]
   
# Convert to float first, THEN round
   probability = round(float(model.predict_proba(df)[0][1]), 4)
   
   # human readable message
   if probability > 0.7:
      message = "high risk of churn"
   elif probability > 0.4:
      message = "medium risk of churn"
   else:
      message = "low risk of churn"

   return {
      "churn_prediction": int(prediction),
      "churn_probability": probability, # Now safely a native python float
      "message": message
   }