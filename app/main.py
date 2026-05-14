"""
ModelServe FastAPI Application v2
Production-grade inference API with MLflow, Feast, and Prometheus monitoring.
"""

from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os

from app.model_loader import load_model, get_model_info, load_local_model
from app.feature_client import FeatureStoreClient
from app.metrics import (
    prediction_requests,
    prediction_errors,
    prediction_latency,
    model_version_gauge,
    feature_store_latency,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
model = None
feature_client = None
mlflow_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown."""
    global model, feature_client, mlflow_client
    
    logger.info("Starting ModelServe...")
    
    # Initialize MLflow client
    mlflow_client = MlflowClient()
    
    # Load model from MLflow registry with local fallback
    try:
        model = load_model()
        logger.info("Model loaded successfully from MLflow")
    except Exception as e:
        logger.warning(f"MLflow model loading failed: {e}. Trying local model...")
        local_model_path = os.getenv("LOCAL_MODEL_PATH", "training/model.pkl")
        model = load_local_model(local_model_path)
        if model:
            logger.info("Model loaded from local path successfully")
        else:
            logger.error("Both MLflow and local model loading failed")

    if model is None:
        logger.error("No model available - predictions will fail")
    
    # Initialize feature store client
    try:
        feature_client = FeatureStoreClient()
        logger.info("Feature store client initialized")
    except Exception as e:
        logger.warning(f"Feature store initialization failed: {e}")
        feature_client = None
    
    # Update model version metric
    try:
        model_info = get_model_info()
        model_version_gauge.labels(
            stage=model_info.get("stage", "Unknown"),
            version=model_info.get("version", "Unknown")
        ).set(1)
    except Exception:
        pass
    
    yield
    
    logger.info("Shutting down ModelServe...")


app = FastAPI(
    title="ModelServe",
    description="Production ML Inference API with Feature Store and Monitoring",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    entity_id: int = Field(..., description="Unique entity identifier")
    features: List[float] = Field(..., description="Feature vector for prediction (V1-V28 + Amount)")
    feature_names: Optional[List[str]] = Field(None, description="Optional feature names")


class PredictResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    model_version: str
    model_stage: str
    timestamp: str
    latency_ms: float


class ExplainResponse(PredictResponse):
    feature_importance: dict
    shap_values: Optional[List[float]] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_stage: str
    feature_store_connected: bool
    mlflow_tracking_uri: str
    timestamp: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint for load balancers and orchestration.
    Returns system status and component health.
    """
    model_info = get_model_info()
    
    # Check feature store connectivity
    feature_store_ok = False
    if feature_client:
        try:
            feature_client.health_check()
            feature_store_ok = True
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_version=model_info.get("version", "unknown"),
        model_stage=model_info.get("stage", "unknown"),
        feature_store_connected=feature_store_ok,
        mlflow_tracking_uri=mlflow.get_tracking_uri(),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    """
    Make predictions using the registered MLflow model.
    Supports batch predictions for multiple entities.
    """
    model_info = get_model_info()
    endpoint = "predict"
    stage = model_info.get("stage", "unknown")
    
    prediction_requests.labels(endpoint=endpoint, model_stage=stage).inc()
    
    if model is None:
        prediction_errors.labels(endpoint=endpoint, error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with prediction_latency.labels(endpoint=endpoint, model_stage=stage).time():
            # Create DataFrame with correct feature names for MLflow model
            feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
            features_df = pd.DataFrame([data.features], columns=feature_columns)
            
            # Get prediction
            prediction = model.predict(features_df)
            
            # Get probability if available
            probability = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(features_df)
                    probability = float(proba[0][1])  # Class 1 probability
                except Exception:
                    pass
            
            return PredictResponse(
                prediction=int(prediction[0]),
                probability=probability,
                model_version=model_info.get("version", "unknown"),
                model_stage=stage,
                timestamp=datetime.utcnow().isoformat() + "Z",
                latency_ms=0,  # Latency tracked by histogram
            )
    
    except Exception as e:
        prediction_errors.labels(endpoint=endpoint, error_type="prediction_failed").inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{entity_id}", response_model=ExplainResponse)
def predict_with_explain(
    entity_id: int,
    explain: bool = Query(False, description="Include SHAP explanations"),
    feature_names: Optional[List[str]] = None,
):
    """
    Get features from Feast and make predictions with optional explanations.
    """
    model_info = get_model_info()
    endpoint = "predict_explain"
    stage = model_info.get("stage", "unknown")
    
    prediction_requests.labels(endpoint=endpoint, model_stage=stage).inc()
    
    if model is None:
        prediction_errors.labels(endpoint=endpoint, error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with prediction_latency.labels(endpoint=endpoint, model_stage=stage).time():
            # Fetch features from feature store
            with feature_store_latency.labels(entity_type="user").time():
                if feature_client is None:
                    raise HTTPException(status_code=503, detail="Feature store not available")
                
                features = feature_client.get_online_features(entity_id)
            
            # Convert to numpy array
            feature_array = np.array(list(features.values())).reshape(1, -1)
            
            # Get prediction
            prediction = model.predict(feature_array)
            
            # Get probability
            probability = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(feature_array)
                    probability = float(proba[0][1])
                except Exception:
                    pass
            
            # Generate feature importance (model-dependent)
            feature_importance = {}
            if hasattr(model, "feature_importances_"):
                names = feature_names or [f"feature_{i}" for i in range(len(features))]
                importances = model.feature_importances_
                feature_importance = dict(zip(names, importances.tolist()))
            elif hasattr(model, "coef_"):
                names = feature_names or [f"feature_{i}" for i in range(len(features))]
                coefs = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
                feature_importance = dict(zip(names, coefs.tolist()))
            
            # SHAP values if requested
            shap_values = None
            if explain and feature_importance:
                # Simple SHAP approximation using feature importance
                shap_values = [
                    feature_importance.get(f"feature_{i}", 0.0) 
                    for i in range(len(features))
                ]
            
            return ExplainResponse(
                prediction=int(prediction[0]),
                probability=probability,
                model_version=model_info.get("version", "unknown"),
                model_stage=stage,
                timestamp=datetime.utcnow().isoformat() + "Z",
                latency_ms=0,
                feature_importance=feature_importance,
                shap_values=shap_values,
            )
    
    except HTTPException:
        raise
    except Exception as e:
        prediction_errors.labels(endpoint=endpoint, error_type="prediction_failed").inc()
        logger.error(f"Prediction with explain error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """
    Prometheus metrics endpoint.
    Exposes latency percentiles, request rates, and model version.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
def model_info():
    """Get current model metadata from MLflow registry."""
    return get_model_info()


@app.get("/")
def root():
    """Root endpoint with API documentation link."""
    return {
        "service": "ModelServe",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)