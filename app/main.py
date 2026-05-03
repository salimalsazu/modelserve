from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from model_loader import load_model

app = FastAPI()

model = load_model()

class RequestData(BaseModel):
    entity_id: int
    features: list

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": "Production"
    }

@app.post("/predict")
def predict(data: RequestData):
    try:
        prediction = model.predict([data.features])
        return {
            "prediction": int(prediction[0]),
            "model_version": "Production",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))