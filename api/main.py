from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
import logging

# Set up logging for production visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction Service",
    description="Production API for evaluating customer churn risk metrics.",
    version="1.0.0"
)

# Load model and scaler with basic startup verification
try:
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("ML Artifacts successfully loaded into memory.")
except Exception as e:
    logger.critical(f"Failed to load ML artifacts on startup: {str(e)}")
    raise RuntimeError(f"Startup failure: {str(e)}")

class CustomerData(BaseModel):
    # Strict validation: Non-negative values for numerical metrics
    tenure: float = Field(..., ge=0, description="Months the customer has stayed with the company")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    avg_monthly_spend: float = Field(..., ge=0)
    
    # Strict validation: Enforce exact binary choices (0 or 1)
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: int = Field(..., ge=0, le=1)
    Dependents: int = Field(..., ge=0, le=1)
    PhoneService: int = Field(..., ge=0, le=1)
    PaperlessBilling: int = Field(..., ge=0, le=1)
    MultipleLines: int = Field(..., ge=0, le=1)
    OnlineSecurity: int = Field(..., ge=0, le=1)
    OnlineBackup: int = Field(..., ge=0, le=1)
    DeviceProtection: int = Field(..., ge=0, le=1)
    TechSupport: int = Field(..., ge=0, le=1)
    StreamingTV: int = Field(..., ge=0, le=1)
    StreamingMovies: int = Field(..., ge=0, le=1)
    
    # One-hot encoded structural checks
    InternetService_Fiber_optic: int = Field(..., ge=0, le=1)
    InternetService_No: int = Field(..., ge=0, le=1)
    Contract_One_year: int = Field(..., ge=0, le=1)
    Contract_Two_year: int = Field(..., ge=0, le=1)
    PaymentMethod_Credit_card: int = Field(..., ge=0, le=1)
    PaymentMethod_Electronic_check: int = Field(..., ge=0, le=1)
    PaymentMethod_Mailed_check: int = Field(..., ge=0, le=1)
    gender_Male: int = Field(..., ge=0, le=1)

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Go to /docs to evaluate payloads."}

@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(data: CustomerData):
    try:
        # Convert incoming safe Pydantic payload directly to a dict
        payload_dict = data.dict()
        df = pd.DataFrame([payload_dict])

        # 1. Rename columns to match XGBoost expectations
        rename_map = {
            "InternetService_Fiber_optic": "InternetService_Fiber optic",
            "Contract_One_year": "Contract_One year",
            "Contract_Two_year": "Contract_Two year",
            "PaymentMethod_Credit_card": "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check"
        }
        df = df.rename(columns=rename_map)

        # 2. Enforce exact column order required by the booster
        expected_cols = model.get_booster().feature_names
        df = df[expected_cols]

        # 3. Apply feature scaling to continuous variables
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]
        df[num_cols] = scaler.transform(df[num_cols])

        # 4. Optimized Inference: Compute probability first, derive prediction class to cut latency in half
        prob_array = model.predict_proba(df)[0]
        probability = round(float(prob_array[1]), 4)
        prediction = 1 if probability >= 0.5 else 0

        # Operational categorization
        if probability > 0.7:
            message = "high risk of churn"
        elif probability > 0.4:
            message = "medium risk of churn"
        else:
            message = "low risk of churn"

        return {
            "churn_prediction": prediction,
            "churn_probability": probability,
            "risk_category": message
        }

    except Exception as e:
        logger.error(f"Inference pipeline execution error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the prediction pipeline."
        )
