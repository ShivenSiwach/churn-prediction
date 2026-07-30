# 🔄 Customer Churn Prediction Pipeline

An end-to-end machine learning project designed to predict customer churn using an XGBoost classifier. This repository demonstrates a complete MLOps workflow, from data preprocessing and experiment tracking to deploying a production-ready API via Docker.

## 🚀 Project Highlights

* **Model Training & Tracking:** Automated training pipeline utilizing XGBoost, with hyperparameter, metric, and artifact tracking managed by **MLflow**.
* **High-Performance API:** Real-time prediction serving built with **FastAPI**.
* **Production-Ready Container:** Containerized application using **Docker** (Python 3.12-slim), configured with a non-root user for strict security compliance.
* **Robust Dependency Management:** Strictly pinned dependencies ensuring environment parity between development and production.

## 🛠️ Tech Stack

* **Language:** Python 3.12
* **Machine Learning:** Scikit-Learn, XGBoost, Pandas, NumPy
* **Experiment Tracking:** MLflow
* **API Framework:** FastAPI, Uvicorn
* **Deployment:** Docker

## 📂 Repository Structure

```text
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI application and endpoints
├── data/                       # Directory for datasets (ignored in version control)
├── models/
│   └── xgb_model.pkl           # Serialized XGBoost model artifact
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── outputs/                    # Visualizations (confusion matrix, feature importance)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning and feature engineering pipeline
│   └── train_model.py          # Model training, evaluation, and MLflow logging
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git tracking exclusions
├── Dockerfile                  # Production container configuration
├── requirements.txt            # Pinned environment dependencies
└── README.md                   # Project documentation
```

## ⚙️ Local Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/ShivenSiwach/churn-prediction.git
cd churn-prediction
```

**2. Create a virtual environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install --no-cache-dir -r requirements.txt
```

**4. Prepare the Data**
* Ensure your raw dataset is placed inside the `data/` directory (e.g., `data/telco_churn_cleaned.csv`).
* Run the preprocessing script to generate the training data:
```bash
python src/data_preprocessing.py
```

## 🧠 Model Training & Tracking

To train the XGBoost model and log the experiment metrics (AUC-ROC, F1, Precision, Recall) via MLflow:

```bash
python src/train_model.py
```
*Note: The script automatically registers the model, logs evaluation graphics to the `outputs/` folder, and saves the final `.pkl` file to the `models/` directory.*

## 🌐 Running the API

### Option A: Running Locally
Start the FastAPI server using Uvicorn:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive Swagger documentation at: `http://localhost:8000/docs`

### Option B: Running via Docker (Production Simulation)
Build the highly-optimized, secure Docker image:
```bash
docker build -t churn-prediction-api .
```

Run the container:
```bash
docker run -p 8000:8000 churn-prediction-api
```
Access the API locally at: `http://localhost:8000`

## 📡 API Usage

Once the server is running, you can generate a churn prediction by sending a `POST` request with a customer profile to the `/predict` endpoint.

**Endpoint:** `POST http://localhost:8000/predict`

**Example cURL Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "gender": "Female",
           "SeniorCitizen": 0,
           "Partner": "Yes",
           "Dependents": "No",
           "tenure": 12,
           "PhoneService": "Yes",
           "MultipleLines": "No",
           "InternetService": "Fiber optic",
           "OnlineSecurity": "No",
           "OnlineBackup": "No",
           "DeviceProtection": "No",
           "TechSupport": "No",
           "StreamingTV": "Yes",
           "StreamingMovies": "Yes",
           "Contract": "Month-to-month",
           "PaperlessBilling": "Yes",
           "PaymentMethod": "Electronic check",
           "MonthlyCharges": 94.20,
           "TotalCharges": 1130.40
         }'
```

**Expected JSON Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.82,
  "status": "success"
}
```
*(Note: Ensure the exact JSON keys in your request match the Pydantic schema defined in your `api/main.py` file).*

## 👤 Author
**Shiven**
