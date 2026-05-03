import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("data.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

mlflow.set_experiment("modelserve")

with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("model", "RandomForest")
    mlflow.sklearn.log_model(model, "model")

joblib.dump(model, "model.pkl")