# Churn Prediction — End-to-End ML Project

A production-ready machine learning project that predicts customer churn 
using XGBoost, served via a FastAPI REST API, tracked with MLflow, 
and containerized with Docker.

## Problem Statement
A telecom company wants to identify customers likely to cancel their 
subscription (churn) so they can intervene early and retain them.

## Dataset
- Source: Kaggle Telco Customer Churn
- Size: 7043 rows × 21 columns
- Target: Churn (Yes/No) — 26.5% churn rate

## Tech Stack
| Tool | Purpose |
|------|---------|
| XGBoost | Model training |
| scikit-learn | Preprocessing + metrics |
| pandas / numpy | Data manipulation |
| FastAPI | REST API |
| MLflow | Experiment tracking |
| Docker | Containerization |
| Python 3.11 | Core language |

## Project Structure
```
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
```

## Model Performance
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.8252 |
| F1 Score (churn) | 0.58 |
| Precision | 0.53 |
| Recall | 0.64 |
| Accuracy | 0.75 |

> Model handles class imbalance using `scale_pos_weight=3.54`

## ML Pipeline
1. **Phase 1** — Data cleaning + EDA
2. **Phase 2** — Feature engineering + train/test split
3. **Phase 3** — XGBoost model training
4. **Phase 4** — MLflow experiment tracking
5. **Phase 5** — FastAPI deployment
6. **Phase 6** — Docker containerization

## How to Run

### Option 1 — Local
```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/data_preprocessing.py

# Train model
python src/train_model.py

# Start API
uvicorn api.main:app --reload
```

### Option 2 — Docker
```bash
docker build -t churn-prediction .
docker run -p 8000:8000 churn-prediction
```

## API Usage
Send a POST request to `/predict`:

```json
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
```

### Response
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.6058,
  "message": "medium risk of churn"
}
```

## Key Learnings

- Handled class imbalance with `scale_pos_weight` instead of 
  oversampling — cleaner and more efficient
- Used stratified train/test split to maintain churn ratio
- Prevented data leakage by fitting scaler on train only
- Tracked all experiments automatically with MLflow
- Containerized the full pipeline with Docker for 
  reproducible deployments
- Built production-ready REST API with FastAPI + Pydantic 
  validation

## Author
Built as an end-to-end ML engineer portfolio project.