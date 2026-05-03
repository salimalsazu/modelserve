# ModelServe

## Run locally

docker compose up --build

## Endpoints

- GET /health
- POST /predict

## Example

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @training/sample_request.json
