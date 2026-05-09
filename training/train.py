"""
Model training script for ModelServe.
Trains sklearn model and registers it in MLflow.
"""

import os
import sys
import json
import logging
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "modelserve-model")
EXPERIMENT_NAME = "model_training"
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Set artifact root for MLflow (avoid /app/mlruns permissions issues)
os.environ["MLFLOW_ARTIFACT_ROOT"] = "/tmp/mlruns"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

def setup_kaggle_credentials():
    if KAGGLE_USERNAME and KAGGLE_KEY:
        kaggle_dir = "/tmp/kaggle"
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
            json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
        os.environ["KAGGLE_CONFIG_DIR"] = kaggle_dir
        logger.info("Kaggle credentials configured")
        return True
    return False

def download_kaggle_dataset(dataset: str, path: str) -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    os.makedirs(path, exist_ok=True)
    api.dataset_download_files(dataset, path=path, unzip=True)
    logger.info(f"Downloaded Kaggle dataset: {dataset}")
    for f in os.listdir(path):
        if f.endswith('.csv'):
            return os.path.join(path, f)
    raise FileNotFoundError(f"No CSV in {path}")

def prepare_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, list]:
    df = pd.read_csv(csv_path)
    target_col = None
    for col in ['Class', 'target', 'fraud', 'label']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    return X, y, numeric_cols

def create_synthetic_data(n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.Series, list]:
    np.random.seed(42)
    n_fraud = int(n_samples * 0.002)
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    normal = pd.DataFrame({col: np.random.rand(n_samples - n_fraud) for col in feature_cols})
    normal['Class'] = 0
    
    fraud = pd.DataFrame({
        col: np.random.randn(n_fraud) * 2 + 1 if col.startswith('V') else np.random.exponential(500, n_fraud)
        for col in feature_cols
    })
    fraud['Class'] = 1
    
    df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y, feature_cols

def train_model(X_train, X_test, y_train, y_test, model_type: str):
    from sklearn.ensemble import RandomForestClassifier
    
    params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42, 'n_jobs': -1}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
    }
    return model, params, metrics

def log_to_mlflow(model, X_train, metrics, model_type, params, register: bool):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        from mlflow.models import infer_signature
        signature = infer_signature(X_train.head(), model.predict(X_train.head()))
        
        # Log model WITHOUT registration (older API compatible)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        logger.info(f"MLflow run ID: {run_id}")
    return run_id

def register_model_version(stage: str = "Production") -> Optional[str]:
    client = MlflowClient()
    try:
        try:
            client.create_registered_model(MODEL_NAME)
            logger.info(f"Created registered model: {MODEL_NAME}")
        except mlflow.exceptions.MlflowException:
            pass  # Already exists
        
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment:
            runs = client.search_runs(experiment.experiment_id, max_results=1, order_by=["start_time DESC"])
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                result = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=run_id)
                version = result.version
                logger.info(f"Registered model version: {version}")
                if stage:
                    client.transition_model_version_stage(MODEL_NAME, version, stage)
                return str(version)
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        import traceback
        traceback.print_exc()
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dataset", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--model-type", type=str, default="random_forest")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--stage", type=str, default="Production")
    args = parser.parse_args()
    
    if args.kaggle_dataset:
        setup_kaggle_credentials()
        csv_path = download_kaggle_dataset(args.kaggle_dataset, "/tmp/kaggle_data")
        X, y, _ = prepare_data(csv_path)
    elif args.data_path:
        X, y, _ = prepare_data(args.data_path)
    else:
        logger.info("Using synthetic data")
        X, y, _ = create_synthetic_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training: {X_train.shape}, Test: {X_test.shape}")
    
    model, params, metrics = train_model(X_train, X_test, y_train, y_test, args.model_type)
    logger.info(f"Metrics: {metrics}")
    
    run_id = log_to_mlflow(model, X_train, metrics, args.model_type, params, args.register)
    
    if args.register:
        version = register_model_version(args.stage)
        if version:
            logger.info(f"Model registered as version {version}")
    
    print(f"\nTraining complete! Run ID: {run_id}")