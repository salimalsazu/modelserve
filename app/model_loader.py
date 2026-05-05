"""
Model loader for MLflow registry.
Supports loading models from MLflow model registry with fallback to local models.
"""

import os
import logging
from typing import Optional, Dict, Any
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
    Load model from MLflow registry.
    
    Args:
        stage: Model stage (Production, Staging, None for latest)
    
    Returns:
        Loaded model object
    
    Raises:
        Exception: If model loading fails
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = get_model_uri(stage)
        
        logger.info(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        logger.info("Model loaded successfully from MLflow registry")
        return model
    
    except Exception as e:
        logger.warning(f"Failed to load from MLflow registry: {e}")
        
        # Try loading the latest version
        try:
            model_uri = f"models:/{MODEL_NAME}/latest"
            logger.info(f"Trying latest version: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Loaded latest model version")
            return model
        except Exception:
            pass
        
        # Fallback to local model
        return load_local_model()


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
    try:
        client = get_mlflow_client()
        model_uri = get_model_uri()
        
        # Parse model name and stage from URI
        parts = model_uri.replace("models:/", "").split("/")
        model_name = parts[0]
        stage = parts[1] if len(parts) > 1 else None
        
        # Get latest model version
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