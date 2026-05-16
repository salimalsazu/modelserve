# ModelServe

Production-grade ML inference platform for real-time fraud detection. Built with FastAPI, MLflow, and deployed on AWS via Pulumi + GitHub Actions CI/CD.

---

## Architecture

```
Developer pushes to main
         ↓
GitHub Actions (OIDC — no stored AWS credentials)
         ↓
 ┌───────────────┐     ┌──────────────────┐
 │  Run Tests    │────▶│  Build & Push    │
 │  (pytest)     │     │  API + MLflow    │
 └───────────────┘     │  images → ECR    │
                       └────────┬─────────┘
                                ↓
                       ┌──────────────────┐
                       │  Deploy to EC2   │
                       │  SSH (stored key)│
                       │  docker compose  │
                       │  up -d           │
                       └────────┬─────────┘
                                ↓
                       Health check → live
```

**Stack:**

| Layer | Tool |
|---|---|
| API | FastAPI + Uvicorn |
| Model Registry | MLflow 2.14 |
| Monitoring | Prometheus + Grafana |
| Infrastructure | Pulumi (Python) on AWS |
| CI/CD | GitHub Actions + OIDC |
| Container Registry | AWS ECR |
| Compute | AWS EC2 t3.medium (Amazon Linux 2023) |

---

## Live Endpoints

| Service | URL |
|---|---|
| API | http://13.228.174.98:8000 |
| API Docs (Swagger) | http://13.228.174.98:8000/docs |
| MLflow UI | http://13.228.174.98:5000 |
| Prometheus | http://13.228.174.98:9090 |
| Grafana | http://13.228.174.98:3000 (admin / admin) |

---

## Quick Start (Local)

```bash
git clone https://github.com/salimalsazu/modelserve.git
cd modelserve
pip install -r requirements.txt

# Run all services locally
docker compose up -d
```

Local URLs match the live endpoints above but on `localhost`.

---

## Infrastructure Setup (First Time Only)

Follow these steps once when setting up a new environment from scratch.

### Prerequisites

| Tool | Install |
|---|---|
| AWS CLI | https://aws.amazon.com/cli/ |
| Pulumi CLI | https://www.pulumi.com/docs/install/ |
| Python 3.10+ | https://www.python.org/ |
| Docker | https://docs.docker.com/get-docker/ |

### Step 1 — Configure AWS credentials

```bash
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region:        ap-southeast-1
# Default output:        json
```

### Step 2 — Generate the deploy SSH key pair

This key is used by GitHub Actions to SSH into EC2 for deployments.

**Windows (PowerShell):**
```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\deploy_key" -N '""'
```

**Mac/Linux:**
```bash
ssh-keygen -t ed25519 -f ~/.ssh/deploy_key -N ""
```

Keep both files — you will need them in the next two steps.

### Step 3 — Bootstrap the infrastructure

```powershell
# Windows
cd infrastructure
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Set the public key in Pulumi config
$pubkey = Get-Content "$env:USERPROFILE\deploy_key.pub"
pulumi config set ssh_public_key $pubkey

# Preview and apply
pulumi up --yes
```

```bash
# Mac/Linux
cd infrastructure
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pulumi config set ssh_public_key "$(cat ~/.ssh/deploy_key.pub)"
pulumi up --yes
```

Note the outputs — you will need `ec2_public_ip` and `github_actions_role_arn`.

### AWS resources created

| Resource | Details |
|---|---|
| VPC | 10.0.0.0/16 with public subnet in ap-southeast-1a |
| EC2 | t3.medium, Amazon Linux 2023, 30 GB gp3 |
| Elastic IP | Static public IP (survives instance replacement) |
| Security Group | Ports 22, 80, 8000, 5000, 9090, 3000 open |
| ECR | `modelserve-api-prod` + `modelserve-mlflow-prod` |
| S3 | Private bucket for MLflow artifacts + deploy staging |
| IAM Instance Role | ECR read + S3 full access + SSM core |
| IAM GitHub Role | Assumed via OIDC — no static AWS keys needed |

### Step 4 — Add GitHub repository secret

Go to **Settings → Secrets and variables → Actions → New repository secret**:

| Name | Value |
|---|---|
| `EC2_SSH_KEY` | Full contents of your `deploy_key` private key file |

**Windows:** `Get-Content "$env:USERPROFILE\deploy_key"`  
**Mac/Linux:** `cat ~/.ssh/deploy_key`

### Step 5 — Add GitHub repository variable

Go to **Settings → Secrets and variables → Actions → Variables → New repository variable**:

| Name | Value |
|---|---|
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID |

### Step 6 — Push to trigger first deploy

```bash
git commit --allow-empty -m "chore: trigger initial deploy"
git push origin main
```

The pipeline will run automatically. Watch progress at:
`https://github.com/salimalsazu/modelserve/actions`

---

## CI/CD Pipeline

File: [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)

### Triggers

- Every push to `main`
- Manual via GitHub Actions UI (`workflow_dispatch`)

### Jobs

```
test ──▶ build-push ──▶ deploy
```

**test**
- Installs Python dependencies
- Runs `pytest app/tests/` with short tracebacks

**build-push**
- Assumes AWS role via OIDC (no stored credentials)
- Builds API image from `Dockerfile`
- Builds MLflow image from `Dockerfile.mlflow`
- Pushes both to ECR tagged with commit SHA + `latest`
- Uses GitHub Actions layer cache for fast rebuilds

**deploy**
- Looks up EC2 instance by tag `Name=modelserve-prod`
- Writes `EC2_SSH_KEY` secret to a temp file
- Waits up to 5 minutes for SSH to become ready (handles fresh instances)
- Copies `docker-compose.yml`, `monitoring/`, and `fraud_model_v2.pkl` to EC2 via SCP
- SSHs into EC2 and runs:
  - `docker login` to ECR
  - `docker pull` new images
  - `docker compose up -d --remove-orphans`
- Health checks `GET /health` (10 retries × 15 s)
- Prints live service URLs to the job summary

### How OIDC works (no AWS secrets in GitHub)

```
GitHub runner requests OIDC token
         ↓
aws-actions/configure-aws-credentials exchanges it with AWS STS
         ↓
Temporary credentials issued (1 hour TTL)
         ↓
Runner can push to ECR and describe EC2 instances
```

`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are never stored anywhere.

### Re-deploying without a code change

```bash
git commit --allow-empty -m "chore: redeploy"
git push origin main
```

---

## Reprovisioning the EC2 Instance

If you need to replace the EC2 instance (e.g. after rotating the SSH key):

```powershell
cd infrastructure

# Generate new key pair
ssh-keygen -t ed25519 -f "$env:USERPROFILE\deploy_key" -N '""'

# Update Pulumi config with new public key
$pubkey = Get-Content "$env:USERPROFILE\deploy_key.pub"
pulumi config set ssh_public_key $pubkey

# Replace the instance (user_data_replace_on_change=True handles this)
pulumi up --yes
```

Then update the `EC2_SSH_KEY` GitHub secret with the new private key.  
The Elastic IP stays the same — no DNS changes needed.

---

## Model Training

### Train locally (synthetic data)

```bash
python training/train.py
```

### Train with a CSV file

```bash
python training/train.py --data-path /path/to/creditcard.csv
```

### Train and register to MLflow

```bash
# Point at the production MLflow server
export MLFLOW_TRACKING_URI=http://13.228.174.98:5000

python training/train.py --register --stage Production
```

### Train with Kaggle dataset

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
python training/train.py --kaggle-dataset mlg-ulb/creditcardfraud --register
```

### Register model on the EC2 instance directly

```bash
ssh -i ~/deploy_key ec2-user@13.228.174.98
docker exec modelserve-api python training/train.py --register --stage Production
```

**Metrics tracked:**

| Metric | Description |
|---|---|
| `accuracy` | Overall classification accuracy |
| `precision` | Precision on fraud class |
| `recall` | Recall on fraud class |
| `f1` | F1 score |
| `roc_auc` | Area under ROC curve |

---

## API Reference

Base URL: `http://13.228.174.98:8000`

### POST `/predict`

Predict fraud from a raw feature vector (V1–V28 + Amount = 29 features).

**Request:**
```json
{
  "entity_id": 1,
  "features": [-1.0, 2.0, 1.5, -0.5, 1.2, 0.8, -1.0, -0.3,
               -1.5, -0.9, -0.2, -1.1, -0.7, -0.8, 0.4, -1.3,
               -0.9, 0.6, 1.0, 0.3, 0.5, 0.8, -1.2, 0.9,
                1.1, -0.4, 0.7, 0.2, 27.0]
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.04,
  "model_version": "1",
  "model_stage": "Production",
  "timestamp": "2026-05-16T10:00:00Z",
  "latency_ms": 8.2
}
```

`prediction: 0` = legitimate, `prediction: 1` = fraud.

### GET `/predict/{entity_id}`

Fetch features from the feature store and predict with optional SHAP explanations.

```bash
curl "http://13.228.174.98:8000/predict/123?explain=true"
```

### GET `/health`

```bash
curl http://13.228.174.98:8000/health
```

```json
{
  "status": "healthy",
  "model_version": "1",
  "model_stage": "Production",
  "feature_store_connected": false,
  "mlflow_tracking_uri": "http://mlflow:5000",
  "timestamp": "2026-05-16T10:00:00Z"
}
```

### GET `/model/info`

Returns current model metadata from MLflow registry.

```bash
curl http://13.228.174.98:8000/model/info
```

### GET `/metrics`

Prometheus metrics in text format.

```bash
curl http://13.228.174.98:8000/metrics
```

### GET `/docs`

Interactive Swagger UI — open in a browser:

```
http://13.228.174.98:8000/docs
```

---

## Model Loading Strategy

The API tries three sources in order at startup:

```
1. MLflow registry  →  models:/modelserve-model/Production
        ↓ fails (registry empty or unreachable)
2. Local pickle     →  /app/models/model.pkl
        ↓ fails (file missing)
3. Degraded mode    →  GET /health returns "degraded"
                        POST /predict returns HTTP 503
```

---

## Monitoring

### Prometheus metrics

| Metric | Description |
|---|---|
| `prediction_requests_total` | Request count by endpoint and model stage |
| `prediction_errors_total` | Error count by error type |
| `prediction_latency_seconds` | Latency histogram (p50/p95/p99) |
| `model_version_info` | Current model version and stage |
| `feature_store_latency_seconds` | Feast online lookup latency |

### Dashboards

| Service | URL | Credentials |
|---|---|---|
| Grafana | http://13.228.174.98:3000 | admin / admin |
| Prometheus | http://13.228.174.98:9090 | — |
| MLflow | http://13.228.174.98:5000 | — |

---

## Destroying Infrastructure

```powershell
cd infrastructure
.\.venv\Scripts\Activate.ps1
pulumi destroy --yes
```

This terminates the EC2 instance, deletes ECR images, removes the S3 bucket, and tears down all networking. The operation is irreversible.

---

## Project Structure

```
modelserve/
├── app/
│   ├── main.py              # FastAPI app, all endpoints, startup cache
│   ├── model_loader.py      # MLflow + local model loading (5 s timeout)
│   ├── feature_client.py    # Feast online feature retrieval
│   ├── metrics.py           # Prometheus metric definitions
│   └── tests/
│       └── test_predict.py  # Pytest suite (26 tests)
├── training/
│   └── train.py             # Model training + MLflow registration
├── infrastructure/
│   ├── __main__.py          # Pulumi — all AWS resources
│   ├── Pulumi.yaml          # Project name + runtime
│   ├── Pulumi.prod.yaml     # Stack config (ssh_public_key stored here)
│   └── requirements.txt     # Pulumi Python providers
├── monitoring/
│   ├── prometheus/          # prometheus.yml scrape config + alert rules
│   └── grafana/             # Dashboard JSON + datasource provisioning
├── .github/workflows/
│   └── deploy.yml           # Full CI/CD: test → build → deploy
├── Dockerfile               # API image (python:3.10-slim)
├── Dockerfile.mlflow        # MLflow tracking server image
├── docker-compose.yml       # All services for local and production
├── fraud_model_v2.pkl       # Pre-trained fallback model
└── requirements.txt         # Python runtime dependencies
```

---

## Troubleshooting

**Deploy fails with "SSH not ready"**  
The EC2 instance may still be booting. The workflow retries for 5 minutes automatically. If it consistently times out, check that `EC2_SSH_KEY` secret matches the public key set in `pulumi config`.

**Deploy fails with "Permission denied (publickey)"**  
The SSH key in GitHub Secrets doesn't match the one embedded in the instance. Regenerate the key pair, update `pulumi config set ssh_public_key`, run `pulumi up --yes`, and update the `EC2_SSH_KEY` secret.

**API returns 503**  
No model is loaded. Either register a model in MLflow or ensure `fraud_model_v2.pkl` was copied to the instance (the deploy step does this automatically).

**`pulumi up` shows "no project file found"**  
You must run Pulumi from the `infrastructure/` directory, not the repo root.
