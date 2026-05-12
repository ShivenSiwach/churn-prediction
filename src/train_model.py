import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os

# load train and test data
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

#seperate features and target
x_train = df_train.drop("Churn", axis=1)
y_train = df_train["Churn"]

x_test = df_test.drop("Churn",axis=1)
y_test = df_test["Churn"]

# set experiment
mlflow.set_experiment("churn-prediction")

with mlflow.start_run():

# calculate scale_pos_weight
  negative = (y_train == 0).sum()
  positive = (y_train == 1).sum()
  scale = negative / positive

# train XGBoost model
  model = xgb.XGBClassifier(
   n_estimators=100,
   scale_pos_weight=scale,
   random_state=42,
   eval_metrics="logloss"
)
  model.fit(x_train, y_train)
  print("model training done")

# evaluate the model
  y_pred = model.predict(x_test)
  y_pred_proba = model.predict_proba(x_test)[:, 1]
  auc = round(roc_auc_score(y_test, y_pred_proba), 4)
  print(classification_report(y_test, y_pred))
  print("AUC-ROC:", auc)
 
# log to mlflow
  mlflow.log_param("n_estimator", 100)
  mlflow.log_param("scale_pos_weight", round(scale, 2))
  mlflow.log_metric("auc_roc", auc)
  mlflow.xgboost.log_model(model,"model")
  print("logged to mlflow")

# save the model
os.makedirs("models", exist_ok=True)
with open("models/xgb_model.pkl", "wb") as f:
   pickle.dump(model, f)

print("model saved -> models/xgb-model.pkl")


