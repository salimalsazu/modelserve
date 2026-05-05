"""
Model training script for ModelServe.
Trains sklearn/XGBoost model and registers it in MLflow.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import joblib
import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "modelserve")
MODEL_NAME = os.getenv("MODEL_NAME", "modelserve-model")
DATA_PATH = os.getenv("DATA_PATH", "training/data.csv")
MODEL_TYPE = os.getenv("MODEL_TYPE", "random_forest")


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load training data from CSV file."""
    logger.info(f"Loading data from {path}")
    
    if not os.path.exists(path):
        # Generate synthetic data for demonstration
        logger.warning(f"Data file not found: {path}. Generating synthetic data.")
        return generate_synthetic_data()
    
    df = pd.read_csv(path)
    
    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic data for demonstration purposes."""
    np.random.seed(42)
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Create target with some signal
    y = (
        X["feature_0"] * 0.5 +
        X["feature_1"] * 0.3 +
        X["feature_2"] * 0.2 +
        np.random.randn(n_samples) * 0.5
    > 0
    ).astype(int)
    
    # Save to file
    os.makedirs("training", exist_ok=True)
    df = pd.concat([X, y.rename("target")], axis=1)
    df.to_csv("training/data.csv", index=False)
    
    logger.info(f"Generated synthetic data: {n_samples} samples")
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    hyperparams: Optional[dict] = None,
) -> object:
    """Train a model based on the specified type."""
    logger.info(f"Training {model_type} model")
    
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100) if hyperparams else 100,
            max_depth=hyperparams.get("max_depth", 10) if hyperparams else 10,
            min_samples_split=hyperparams.get("min_samples_split", 5) if hyperparams else 5,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=hyperparams.get("n_estimators", 100) if hyperparams else 100,
            max_depth=hyperparams.get("max_depth", 5) if hyperparams else 5,
            learning_rate=hyperparams.get("learning_rate", 0.1) if hyperparams else 0.1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    logger.info(f"Model training complete: {type(model).__name__}")
    
    return model


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = None
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }
    
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            pass
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_test, y_test, cv=5)
    metrics["cv_mean"] = cv_scores.mean()
    metrics["cv_std"] = cv_scores.std()
    
    logger.info(f"Metrics: {metrics}")
    
    return metrics


def log_to_mlflow(
    model: object,
    metrics: dict,
    X_train: pd.DataFrame,
    model_type: str,
    tags: Optional[dict] = None,
) -> str:
    """Log model and metrics to MLflow."""
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Start run
    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        
        # Log parameters
        params = {
            "model_type": model_type,
            "n_features": X_train.shape[1],
            "n_samples": X_train.shape[0],
        }
        
        if hasattr(model, "n_estimators"):
            params["n_estimators"] = model.n_estimators
        if hasattr(model, "max_depth"):
            params["max_depth"] = model.max_depth
        
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME,
        )
        
        # Log feature names
        mlflow.log_param("feature_names", ",".join(X_train.columns.tolist()))
        
        # Add custom tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        # Log training timestamp
        mlflow.set_tag("training_timestamp", datetime.utcnow().isoformat())
        
        logger.info(f"Logged to MLflow: run_id={run_id}")
        
        return run_id


def register_model_version(stage: str = "Production") -> Optional[str]:
    """Register the latest model version in MLflow registry."""
    try:
        client = MlflowClient()
        
        # Get latest model versions
        versions = client.get_latest_versions(MODEL_NAME)
        
        if not versions:
            logger.warning("No model versions found in registry")
            return None
        
        latest_version = versions[0].version
        
        # Transition to target stage
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version,
            stage=stage,
        )
        
        logger.info(f"Registered model version {latest_version} as {stage}")
        
        # Add description
        client.update_model_version(
            name=MODEL_NAME,
            version=latest_version,
            description=f"Production model registered at {datetime.utcnow().isoformat()}",
        )
        
        return str(latest_version)
    
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        return None


def archive_previous_production() -> None:
    """Archive the current production model before deploying new one."""
    try:
        client = MlflowClient()
        
        # Find current production version
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        
        for v in versions:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=v.version,
                stage="Archived",
            )
            logger.info(f"Archived previous production model: version {v.version}")
    
    except Exception as e:
        logger.warning(f"Failed to archive previous production model: {e}")


@click.command()
@click.option("--model-type", default="random_forest", help="Model type (random_forest, gradient_boosting)")
@click.option("--n-estimators", default=100, help="Number of estimators")
@click.option("--max-depth", default=10, help="Maximum tree depth")
@click.option("--register", is_flag=True, help="Register model in MLflow registry")
@click.option("--stage", default="Production", help="Target stage for registration")
def main(model_type: str, n_estimators: int, max_depth: int, register: bool, stage: str):
    """Train model and optionally register in MLflow."""
    logger.info("Starting model training pipeline")
    logger.info(f"Configuration: model_type={model_type}, n_estimators={n_estimators}, max_depth={max_depth}")
    
    # Load data
    X, y = load_data(DATA_PATH)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Prepare hyperparameters
    hyperparams = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }
    
    # Train model
    model = train_model(X_train, y_train, model_type, hyperparams)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Log to MLflow
    run_id = log_to_mlflow(
        model,
        metrics,
        X_train,
        model_type,
        tags={"environment": os.getenv("ENV", "development")},
    )
    
    # Register if requested
    if register:
        archive_previous_production()
        version = register_model_version(stage)
        
        if version:
            logger.info(f"Model registered as version {version} in stage {stage}")
        else:
            logger.warning("Model not registered - no versions available")
    
    # Save model locally
    os.makedirs("training", exist_ok=True)
    joblib.dump(model, "training/model.pkl")
    logger.info("Model saved to training/model.pkl")
    
    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Model Type: {model_type}")
    print(f"Run ID: {run_id}")
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()