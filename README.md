# ModelServe

Production-grade MLOps platform for model serving with FastAPI, MLflow, Feast, and comprehensive monitoring.

## Features

- **FastAPI Inference API** - High-performance async prediction endpoints
- **MLflow Model Registry** - Centralized model versioning and stage management
- **Feast Feature Store** - Consistent features for training and serving
- **Redis Online Store** - Sub-millisecond feature retrieval
- **Prometheus + Grafana** - Full-stack observability with latency percentiles
- **GitHub Actions CI/CD** - Automated testing, building, and deployment
- **Pulumi IaC** - Reproducible AWS infrastructure

## Quick Start

```bash
# Clone and start all services
git clone https://github.com/your-org/modelserve.git
cd modelserve
docker-compose up -d

# Train a model
python training/train.py --model-type random_forest --register

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"entity_id": 1, "features": [0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7, 0.3]}'

# View metrics
curl http://localhost:8000/metrics

# Access dashboards
# - API:      http://localhost:8000
# - MLflow:   http://localhost:5000
# - Grafana:  http://localhost:3000 (admin/admin)
# - Swagger:  http://localhost:8000/docs
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client Requests                            │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Inference API                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  /health     │  │  /predict    │  │  /metrics    │              │
│  │  /model/info │  │  /predict/:id│  │  /docs       │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└────────┬─────────────────┬──────────────────────────────┬──────────┘
         │                 │                              │
         │                 ▼                              ▼
         │  ┌──────────────────────────┐    ┌─────────────────────────┐
         │  │      MLflow Registry   │    │     Prometheus + Grafana │
         │  │   (Model Staging)      │    │   (Metrics Collection)   │
         │  └──────────────────────────┘    └─────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Feature Store (Feast)                          │
│  ┌─────────────────────┐              ┌──────────────────────────┐ │
│  │   Parquet Offline   │              │     Redis Online         │ │
│  │   (Training Data)   │              │  
