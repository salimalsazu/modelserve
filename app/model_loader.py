"""
Model loader for MLflow registry.
Supports loading models from MLflow model registry with fallback to local models.
"""

import os
import logging
from typing import Optional, Dict, Any
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import joblib

logger = logging.getLogger(__name__)

# Default configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "modelserve-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "training/model.pkl")


def get_mlflow_client() -> MlflowClient:
    """Create and return MLflow client."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def get_model_uri(stage: Optional[str] = None) -> str:
    """Get model URI from MLflow registry."""
    stage = stage or MODEL_STAGE
    return f"models:/{MODEL_NAME}/{stage}"


def load_model(stage: Optional[str] = None) -> Any:
    """
    Load model from MLflow registry with predict_proba support.
    
    Args:
        stage: Model stage (Production, Staging, None for latest)
    
    Returns:
        Loaded model object with predict and predict_proba support
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = get_model_uri(stage)
    
    logger.info(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Wrap the pyfunc model to add predict_proba support
    class PyFuncWrapper:
        def __init__(self, pyfunc_model):
            self._model = pyfunc_model
            self._sklearn_model = None
            # Try to get underlying sklearn model
            try:
                impl = pyfunc_model._model_impl
                if hasattr(impl, 'model'):
                    self._sklearn_model = impl.model
            except Exception:
                pass
        
        def predict(self, X):
            return self._model.predict(X)
        
        def predict_proba(self, X):
            if self._sklearn_model and hasattr(self._sklearn_model, 'predict_proba'):
                return self._sklearn_model.predict_proba(X)
            # Fallback: predict and convert to probabilities
            pred = self._model.predict(X)
            # Handle DataFrame input
            if hasattr(X, 'values'):
                n = len(X.values) if len(X.values.shape) > 1 else len(X)
            else:
                n = len(X) if hasattr(X, '__len__') else 1
            proba = np.zeros((n, 2))
            for i, p in enumerate(pred.flatten() if hasattr(pred, 'flatten') else [pred]):
                proba[i, 1] = float(p)
                proba[i, 0] = 1.0 - float(p)
            return proba
        
        def __getattr__(self, name):
            return getattr(self._model, name)
    
    logger.info("Model loaded successfully from MLflow registry")
    return PyFuncWrapper(model)


def load_local_model(path: Optional[str] = None) -> Any:
    """
    Load model from local file system.
    
    Args:
        path: Path to model file
    
    Returns:
        Loaded model or None
    """
    model_path = path or LOCAL_MODEL_PATH
    
    if not os.path.exists(model_path):
        logger.error(f"Local model not found: {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from local path: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        return None


def get_model_info() -> Dict[str, Any]:
    """
    Get model metadata from MLflow registry.

    Returns:
        Dictionary with model version, stage, and other metadata
    """
    import concurrent.futures

    def _fetch():
        client = get_mlflow_client()
        model_uri = get_model_uri()
        parts = model_uri.replace("models:/", "").split("/")
        model_name = parts[0]
        stage = parts[1] if len(parts) > 1 else None
        versions = client.get_latest_versions(model_name, stages=[stage] if stage else None)
        if versions:
            latest = versions[0]
            return {
                "name": latest.name,
                "version": str(latest.version),
                "stage": latest.current_stage,
                "status": latest.status,
                "run_id": latest.run_id,
                "created_at": str(latest.creation_timestamp),
            }
        return {
            "name": MODEL_NAME,
            "version": "unknown",
            "stage": MODEL_STAGE,
        }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            return future.result(timeout=5)
    except Exception as e:
        logger.warning(f"Failed to get model info: {e}")
        return {
            "name": MODEL_NAME,
            "version": "unknown",
            "stage": MODEL_STAGE,
            "error": str(e),
        }


def list_model_versions() -> list:
    """List all versions of the registered model."""
    try:
        client = get_mlflow_client()
        versions = client.get_latest_versions(MODEL_NAME)
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "status": v.status,
            }
            for v in versions
        ]
    except Exception as e:
        logger.error(f"Failed to list model versions: {e}")
        return []


def transition_model_stage(version: int, target_stage: str) -> bool:
    """
    Transition model version to a different stage.
    
    Args:
        version: Model version number
        target_stage: Target stage (Production, Staging, Archived)
    
    Returns:
        True if successful
    """
    try:
        client = get_mlflow_client()
        client.transition_model_version_stage(MODEL_NAME, version, target_stage)
        logger.info(f"Model {version} transitioned to {target_stage}")
        return True
    except Exception as e:
        logger.error(f"Failed to transition model: {e}")
        return False


if __name__ == "__main__":
    # Test model loading
    print("Testing model loader...")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Model: {MODEL_NAME}")
    print(f"Stage: {MODEL_STAGE}")
    
    model = load_model()
    if model:
        print(f"Model loaded: {type(model)}")
        print(f"Model info: {get_model_info()}")
    else:
        print("Failed to load model")