# Churn Prediction — End-to-End ML Project

A production-ready machine learning project that predicts customer churn using XGBoost, served via a FastAPI REST API, tracked with MLflow, and containerized with Docker.

## Problem Statement
A telecom company wants to identify customers likely to cancel their subscription (churn) so they can intervene early and retain them.

## Dataset
* **Source:** Kaggle Telco Customer Churn
* **Size:** 7043 rows × 21 columns
* **Target:** Churn (Yes/No) — 26.5% churn rate

## Tech Stack
| Tool | Purpose |
| :--- | :--- |
| **XGBoost** | Model training and inference |
| **scikit-learn** | Data preprocessing and evaluation metrics |
| **pandas / numpy** | Data manipulation |
| **FastAPI & Pydantic** | REST API routing and strict payload validation |
| **MLflow** | Experiment tracking and artifact registry |
| **Docker** | Secure, non-root containerization |
| **Python 3.11** | Core language |

## Project Structure
churn-prediction/
├── data/               # Raw and processed data
├── src/                # Core scripts
│   ├── data_preprocessing.py
│   └── train_model.py
├── api/                # FastAPI app
│   └── main.py
├── models/             # Saved model + scaler
├── notebooks/          # EDA notebook
├── Dockerfile
├── requirements.txt
└── README.md

Model Performance

Metric             Score
AUC-ROC            0.8252
F1 Score (churn)   0.58
Precision          0.53
Recall             0.64
Accuracy           0.75
#Note: The model handles class imbalance natively using scale_pos_weight = 3.54.

ML Pipeline:>

Phase 1: Data cleaning and Exploratory Data Analysis (EDA)

Phase 2: Feature engineering and stratified train/test splitting

Phase 3: XGBoost model training

Phase 4: MLflow experiment tracking

Phase 5: FastAPI deployment with Pydantic error handling

Phase 6: Docker containerization with non-root security compliance

How to Run

Option 1 — Local Environment
# Install dependencies
pip install -r requirements.txt

# Run preprocessing and train model
python src/data_preprocessing.py
python src/train_model.py

# Start API
uvicorn api.main:app --reload

Option 2 — Docker (Recommended)

Bash
# Build the image
docker build -t churn-prediction .

# Run the container securely
docker run -p 8000:8000 churn-prediction

API Usage
Send a POST request to /predict. The API uses strict Pydantic data validation to reject improper data types and prevent backend crashes.

Payload:

JSON
{
  "tenure": 24,
  "MonthlyCharges": 65.50,
  "TotalCharges": 1572.00,
  "avg_monthly_spend": 65.50,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "PhoneService": 1,
  "PaperlessBilling": 1,
  "MultipleLines": 0,
  "OnlineSecurity": 0,
  "OnlineBackup": 1,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 1,
  "StreamingMovies": 1,
  "InternetService_Fiber_optic": 1,
  "InternetService_No": 0,
  "Contract_One_year": 0,
  "Contract_Two_year": 0,
  "PaymentMethod_Credit_card": 0,
  "PaymentMethod_Electronic_check": 1,
  "PaymentMethod_Mailed_check": 0,
  "gender_Male": 1
}
Response:

JSON
{
  "churn_prediction": 1,
  "churn_probability": 0.6058,
  "risk_category": "medium risk of churn"
}

Key Learnings
.Production API Hardening: Implemented strict Pydantic Field validation to reject bad payloads and optimized the endpoint to compute predictions natively from probabilities, cutting inference latency.

.Secure Deployment: Containerized the full pipeline with Docker, implementing a non-root user environment to align with enterprise security standards.

.Efficient Imbalance Handling: Handled class imbalance with scale_pos_weight instead of oversampling, ensuring cleaner and more efficient training.

.Leakage Prevention: Prevented data leakage by strictly fitting the scaler on the training set only.

.Observability: Tracked all experiments and artifact registries automatically with MLflow.

:-Author: Built as a production-focused end-to-end ML engineering portfolio project.
