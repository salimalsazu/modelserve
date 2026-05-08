"""Model training script for ModelServe.
Trains sklearn/XGBoost model and registers it in MLflow.
Supports Kaggle datasets, synthetic data, and local CSV files.
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


def download_kaggle_dataset(dataset: str, output_path: str = "training/data") -> str:
    """Download dataset from Kaggle.
    
    Args:
        dataset: Kaggle dataset slug (e.g., 'username/dataset-name')
        output_path: Local path to save downloaded files
        
    Returns:
        Path to downloaded CSV file
    """
    try:
        import kaggle
    except ImportError:
        logger.error("Kaggle library not installed. Install with: pip install kaggle")
        raise ImportError("Kaggle library not installed. Run: pip install kaggle")
    
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Downloading Kaggle dataset: {dataset}")
    kaggle.api.dataset_download_files(
        dataset,
        path=output_path,
        unzip=True,
    )
    
    # Find the CSV file
    csv_files = [f for f in os.listdir(output_path) if f.endswith('.csv')]
    if csv_files:
        csv_path = os.path.join(output_path, csv_files[0])
        logger.info(f"Downloaded to: {csv_path}")
        return csv_path
    
    raise FileNotFoundError(f"No CSV file found in downloaded dataset: {output_path}")


def load_data(path: str, kaggle_dataset: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load training data from CSV, Kaggle, or generate synthetic.
    
    Args:
        path: Local CSV file path
        kaggle_dataset: Kaggle dataset slug to download (e.g., 'user/dataset')
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # If Kaggle dataset specified, download it
    if kaggle_dataset:
        csv_path = download_kaggle_dataset(kaggle_dataset, "training/kaggle_data")
        
        # Load and prepare Kaggle dataset
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded Kaggle dataset: {df.shape}")
        
        # Try to auto-detect target column
        if 'target' in df.columns:
            X = df.drop("target", axis=1)
            y = df["target"]
        elif 'label' in df.columns:
            X = df.drop("label", axis=1)
            y = df["label"]
        elif 'y' in df.columns:
            X = df.drop("y", axis=1)
            y = df["y"]
        elif 'class' in df.columns:
            X = df.drop("class", axis=1)
            y = df["class"]
        else:
            # Use last column as target
            target_col = df.columns[-1]
            X = df.drop(target_col, axis=1)
            y = df[target_col]
        
        # Convert target to binary if needed
        if y.dtype == 'object' or y.nunique() > 2:
            # Encode target
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y))
        
        return X, y
    
    # Load from local path if exists
    if os.path.exists(path):
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        
        if 'target' in df.columns:
            X = df.drop("target", axis=1)
            y = df["target"]
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        return X, y
    
    logger.warning(f"Data file not found: {path}. Generating synthetic data.")
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
) -> RandomForestClassifier:
    """Train a sklearn model."""
    logger.info(f"Training {model_type} model")
    
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth or 10,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth or 5,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    logger.info(f"Model training complete: {model.__class__.__name__}")
    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
    }
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_test, y_test, cv=5)
    metrics["cv_mean"] = cv_scores.mean()
    metrics["cv_std"] = cv_scores.std()
    
    logger.info(f"Metrics: {metrics}")
    return metrics


def log_to_mlflow(
    model,
    metrics: dict,
    X_train: pd.DataFrame,
    input_example,
) -> str:
    """Log model, metrics, and artifacts to MLflow."""
    import warnings
    # Suppress deprecation warning for artifact_path
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        # Log parameters
        mlflow.log_params({
            "model_type": MODEL_TYPE,
            "n_features": X_train.shape[1],
            "n_samples": X_train.shape[0],
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model using artifact_path (deprecated but compatible)
        artifact_path = "model"
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            registered_model_name=MODEL_NAME,
        )
        
        # Log feature importance if available
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            importance_path = "feature_importance.csv"
            importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
        
        logger.info(f"Model logged to MLflow with registered name: {MODEL_NAME}")
    
    return run_id


def register_model_version(stage: str = "Production") -> Optional[str]:
    """Register a model version and transition to stage."""
    client = MlflowClient()
    
    try:
        # Get all versions of the model
        versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production"])
        if versions:
            latest_version = versions[0].version
            logger.info(f"Latest model version: {latest_version}")
            
            # Transition to stage
            client.transition_model_version_stage(
                MODEL_NAME,
                latest_version,
                stage,
            )
            logger.info(f"Model version {latest_version} transitioned to {stage}")
            return str(latest_version)
        else:
            logger.warning(f"No model versions found for {MODEL_NAME}")
    except Exception as e:
        logger.error(f"Error registering model version: {e}")
    
    return None


@click.command()
@click.option("--model-type", default="random_forest", help="Model type (random_forest, gradient_boosting)")
@click.option("--n-estimators", default=100, help="Number of estimators")
@click.option("--max-depth", default=None, help="Max depth of trees")
@click.option("--data-path", default="training/data.csv", help="Path to training data")
@click.option("--kaggle-dataset", default=None, help="Kaggle dataset slug (e.g., 'username/dataset-name')")
@click.option("--register", is_flag=True, help="Register model in MLflow")
@click.option("--stage", default="Production", help="Model stage (Production, Staging)")
def main(
    model_type: str,
    n_estimators: int,
    max_depth: Optional[int],
    data_path: str,
    kaggle_dataset: Optional[str],
    register: bool,
    stage: str,
):
    """Main training pipeline."""
    logger.info("Starting model training pipeline")
    logger.info(f"Configuration: model_type={model_type}, n_estimators={n_estimators}, max_depth={max_depth}")
    
    # Load data (from Kaggle, local file, or synthetic)
    X, y = load_data(data_path, kaggle_dataset)
    
    if kaggle_dataset:
        logger.info(f"Loaded Kaggle dataset: {len(X)} samples, {X.shape[1]} features")
    else:
        logger.info(f"Data loaded: {len(X)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train, model_type, n_estimators, max_depth)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Log to MLflow if requested
    if register:
        run_id = log_to_mlflow(
            model,
            metrics,
            X_train,
            input_example=X_train.head(),
        )
        logger.info(f"Run ID: {run_id}")
        
        # Register version
        version = register_model_version(stage)
        if version:
            logger.info(f"Model registered as version {version} in {stage}")
    
    # Save model locally
    model_path = f"models/{model_type}_model.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()