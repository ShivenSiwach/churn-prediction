import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import pickle
import os

#  Load data 
df_train = pd.read_csv("data/train.csv")
df_test  = pd.read_csv("data/test.csv")

X_train = df_train.drop("Churn", axis=1)
y_train = df_train["Churn"]
X_test  = df_test.drop("Churn", axis=1)
y_test  = df_test["Churn"]

#  Hyperparams (single source of truth) 
params = {
    "n_estimators":     100,
    "max_depth":        4,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",   
    "random_state":     42,
}

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale    = round(negative / positive, 2)
params["scale_pos_weight"] = scale

#  MLflow ─
mlflow.set_experiment("churn-prediction")

with mlflow.start_run(run_name="xgboost-baseline"):

    # log all hyperparams
    mlflow.log_params(params)

    # Train
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    print("Model training done")

    # Evaluate
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc       = round(roc_auc_score(y_test, y_pred_proba), 4)
    f1        = round(f1_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall    = round(recall_score(y_test, y_pred), 4)

    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {auc} | F1: {f1} | Precision: {precision} | Recall: {recall}")

    # log all metrics
    mlflow.log_metrics({
        "auc_roc":   auc,
        "f1_score":  f1,
        "precision": precision,
        "recall":    recall,
    })

    # Artifacts — Confusion Matrix
    os.makedirs("outputs", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("outputs/confusion_matrix.png")

    # Artifacts — Feature Importance
    xgb.plot_importance(model, max_num_features=15, importance_type="gain")
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()
    mlflow.log_artifact("outputs/feature_importance.png")

    # log model (XGBoost flavor) + register in Model Registry
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="churn-xgboost",  # creates entry in Model Registry
    )
    print("Logged to MLflow + registered model")

    # Save pickle 
    os.makedirs("models", exist_ok=True)
    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact("models/xgb_model.pkl")   # pickle also tracked
    print("Model saved → models/xgb_model.pkl")