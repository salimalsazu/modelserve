# ModelServe — Architecture Documentation

> **Exam Capstone | MLOps S2 | 2026**  
> Production-grade ML inference platform for real-time fraud detection.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagrams](#2-architecture-diagrams)
3. [Architecture Decision Records (ADRs)](#3-architecture-decision-records-adrs)
4. [CI/CD Pipeline Documentation](#4-cicd-pipeline-documentation)
5. [Runbook](#5-runbook)
6. [Known Limitations](#6-known-limitations)

---

## 1. System Overview

### Purpose

ModelServe is a production-grade MLOps platform that serves a real-time fraud detection model. It ingests raw transaction features, looks up pre-computed features from a Feast feature store, runs inference through an MLflow-registered Random Forest model, and returns fraud predictions with latency metrics.

### Key Design Goals

| Goal | Decision |
|------|----------|
| Zero stored AWS credentials | GitHub Actions OIDC + temporary STS tokens |
| Reproducible infrastructure | Pulumi (Python IaC) for all AWS resources |
| Model lifecycle management | MLflow Model Registry with stage promotion |
| Training-serving consistency | Feast feature store (same definitions for train + serve) |
| Observability | Prometheus metrics + Grafana dashboards |
| Graceful degradation | Model fallback chain: MLflow → local pickle → 503 |

### Component Summary

| Layer | Component | Technology |
|-------|-----------|------------|
| Inference API | REST endpoints | FastAPI + Uvicorn |
| Model Registry | Versioned model storage | MLflow 2.14 + PostgreSQL |
| Feature Store | Online feature retrieval | Feast + Redis |
| Monitoring | Metrics + dashboards | Prometheus + Grafana |
| Infrastructure | AWS resource provisioning | Pulumi (Python) |
| CI/CD | Build, test, deploy | GitHub Actions + OIDC |
| Container Registry | Docker image storage | AWS ECR |
| Compute | Application hosting | AWS EC2 t3.medium |
| Artifact Store | MLflow model artifacts | AWS S3 |

### Data Flow (Inference)

```
Client
  │
  │  POST /predict { entity_id, features[] }
  ▼
FastAPI (port 8000)
  │
  ├─► Feast Redis Online Store ──► feature lookup by entity_id
  │         (sub-millisecond)
  │
  ├─► MLflow Model (in-process, pre-loaded)
  │         RandomForest.predict(features)
  │
  ├─► Prometheus Counter/Histogram (fire-and-forget)
  │
  └─► JSON Response
        { prediction, probability, model_version, latency_ms }
```

---

## 2. Architecture Diagrams

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS ap-southeast-1                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    VPC (10.0.0.0/16)                     │   │
│  │                                                          │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │              Public Subnet (10.0.1.0/24)           │  │   │
│  │  │                                                    │  │   │
│  │  │  ┌─────────────────────────────────────────────┐  │  │   │
│  │  │  │          EC2 t3.medium (Docker host)         │  │  │   │
│  │  │  │  Elastic IP: 52.221.26.39                   │  │  │   │
│  │  │  │                                             │  │  │   │
│  │  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │  │   │
│  │  │  │  │ FastAPI  │  │  MLflow  │  │PostgreSQL│  │  │  │   │
│  │  │  │  │ :8000    │  │  :5000   │  │  :5432   │  │  │  │   │
│  │  │  │  └──────────┘  └──────────┘  └──────────┘  │  │  │   │
│  │  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │  │   │
│  │  │  │  │  Redis   │  │Prometheus│  │ Grafana  │  │  │  │   │
│  │  │  │  │ :6379    │  │  :9090   │  │  :3000   │  │  │  │   │
│  │  │  │  └──────────┘  └──────────┘  └──────────┘  │  │  │   │
│  │  │  └─────────────────────────────────────────────┘  │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   ECR Repos     │    │    S3 Bucket     │                    │
│  │  modelserve-api │    │  mlflow-artifacts│                    │
│  │  modelserve-mlf │    │  (private)       │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 CI/CD Pipeline

```
Developer
  │  git push → main
  ▼
GitHub Actions (ubuntu-latest runner)
  │
  ├─[Job 1: test]────────────────────────────────────────────────┐
  │   pip install deps                                           │
  │   pytest app/tests/ -v --tb=short                           │
  └──────────────────────────────────────────────────────────────┘
  │  (must pass before build)
  ▼
  ├─[Job 2: build-push]──────────────────────────────────────────┐
  │   OIDC → STS AssumeRoleWithWebIdentity (1 hour TTL)         │
  │   aws-actions/amazon-ecr-login                              │
  │   docker buildx build → push API image (sha + latest)       │
  │   docker buildx build → push MLflow image (sha + latest)    │
  └──────────────────────────────────────────────────────────────┘
  │  (needs build-push)
  ▼
  └─[Job 3: deploy]──────────────────────────────────────────────┐
      OIDC → STS (re-assumed)                                   │
      aws ec2 describe-instances --filter tag:Name=modelserve-prod│
      ssh-keygen -t ed25519 (ephemeral, in-memory)              │
      aws ec2-instance-connect send-ssh-public-key (60s TTL)    │
      scp docker-compose.yml + monitoring/ + model.pkl → EC2    │
      ssh → docker pull + docker compose up -d --remove-orphans │
      curl /health (10 retries × 15s)                           │
      Write GitHub Step Summary with service URLs               │
    ────────────────────────────────────────────────────────────┘
```

### 2.3 OIDC Authentication Flow (no stored credentials)

```
GitHub Runner
  │
  │  1. Request OIDC JWT from GitHub token endpoint
  ▼
GitHub OIDC Provider (token.actions.githubusercontent.com)
  │
  │  2. Return signed JWT:
  │     { sub: "repo:salimalsazu/modelserve:ref:refs/heads/main",
  │       aud: "sts.amazonaws.com" }
  ▼
AWS STS (AssumeRoleWithWebIdentity)
  │  3. Validate JWT signature against thumbprint
  │  4. Check Condition: StringLike sub matches repo pattern
  │
  │  5. Return temporary credentials (1 hour TTL)
  │     { AccessKeyId, SecretAccessKey, SessionToken }
  ▼
GitHub Runner (uses temp creds for ECR push + EC2 connect)
```

### 2.4 Model Loading Fallback Chain

```
API Startup
  │
  ├─► Try: MLflow Registry
  │         mlflow://models:/modelserve-model/Production
  │         ↓ success → serve from registry
  │         ↓ fail    ──────────────────────────────────┐
  │                                                     │
  ├─► Try: Local Pickle                                 │
  │         /app/models/model.pkl               ◄───────┘
  │         ↓ success → serve from pickle
  │         ↓ fail    ──────────────────────────────────┐
  │                                                     │
  └─► Degraded Mode                             ◄───────┘
        /health returns { "status": "degraded" }
        /predict returns HTTP 503
```

### 2.5 Feature Store Data Flow

```
Training time:
  train.py → pandas DataFrame → Feast offline store (Parquet)
                                          │
                             feast materialize-incremental
                                          │
                                          ▼
Serving time:                    Redis online store
  POST /predict → feature_client.get_online_features(entity_id)
                        │
                        └─► HGETALL redis key → feature dict
                                      │
                                  fallback: mock features from entity_id hash
```

---

## 3. Architecture Decision Records (ADRs)

### ADR-001: FastAPI as the Inference API Framework

**Status:** Accepted  
**Date:** 2026-05-01  
**Deciders:** Project team

#### Context

We needed a Python-native HTTP framework to serve ML predictions with sub-100ms p95 latency, automatic OpenAPI documentation, and Prometheus middleware support.

#### Decision

Use **FastAPI** with Uvicorn ASGI server.

#### Rationale

1. **ASGI throughput**: Uvicorn handles concurrent requests without GIL blocking on I/O
2. **Pydantic validation**: Automatic request/response schema with error messages at no extra code cost
3. **OpenAPI**: `/docs` endpoint generated automatically — no manual documentation
4. **Type hints**: Same Python type system used in ML code; no cognitive context switch
5. **Prometheus middleware**: `prometheus-fastapi-instrumentator` integrates in 3 lines

#### Consequences

- **Positive**: ~2x throughput vs synchronous Flask under concurrent load; automatic /docs
- **Negative**: Async patterns require care — CPU-bound prediction must run in thread pool to avoid blocking event loop

#### Alternatives Rejected

| Framework | Reason Rejected |
|-----------|----------------|
| Flask | Synchronous only; no built-in validation |
| Starlette | More boilerplate; FastAPI is built on it anyway |
| Django REST | Heavyweight ORM features irrelevant to inference |

---

### ADR-002: MLflow Model Registry for Model Lifecycle Management

**Status:** Accepted  
**Date:** 2026-05-01

#### Context

Multiple model versions needed to be tracked, compared, and promoted to production without redeploying the API. We needed artifact storage and a promotion workflow.

#### Decision

Use **MLflow Model Registry** backed by PostgreSQL (metadata) and S3 (artifacts).

#### Rationale

1. **Stage promotion**: `Staging → Production` transition without code change
2. **Artifact lineage**: Every model version links to its training run, parameters, and metrics
3. **API-first**: `mlflow.pyfunc.load_model("models:/name/Production")` — one line to load latest Production model
4. **Self-hosted**: Full control; no per-prediction cost

#### Consequences

- **Positive**: Model rollback in seconds; full experiment history
- **Negative**: Adds PostgreSQL dependency; MLflow server is a single point of failure (mitigated by local pickle fallback)

#### Alternatives Rejected

| Option | Reason Rejected |
|--------|----------------|
| Weights & Biases | Paid SaaS; data leaves environment |
| SageMaker Model Registry | AWS lock-in; higher operational complexity |
| Plain S3 versioning | No promotion workflow; no experiment tracking |

---

### ADR-003: Pulumi (Python) for Infrastructure as Code

**Status:** Accepted  
**Date:** 2026-05-01

#### Context

All AWS resources — VPC, EC2, ECR, S3, IAM roles, OIDC provider — needed to be provisioned reproducibly and version-controlled alongside application code.

#### Decision

Use **Pulumi with the Python AWS provider** instead of Terraform or CloudFormation.

#### Rationale

1. **Same language**: Python IaC means ML engineers can read and modify infrastructure
2. **Real loops and functions**: ECR lifecycle policy applied to both repos with one Python loop — no HCL `count` tricks
3. **`pulumi preview`**: Diff before apply, identical to Terraform plan but in a familiar tool
4. **State management**: Pulumi Cloud (free tier) stores state; no S3 backend to manage
5. **Data sources**: `aws.iam.get_open_id_connect_provider()` to reference existing OIDC provider without owning it

#### Consequences

- **Positive**: Testable infrastructure code; familiar language
- **Negative**: Smaller community than Terraform; state file is in Pulumi Cloud (third-party dependency)

#### Alternatives Rejected

| Tool | Reason Rejected |
|------|----------------|
| Terraform | HCL language; no native Python |
| AWS CDK | TypeScript-first; Python CDK is second-class |
| CloudFormation | YAML/JSON only; no loops; very verbose |

---

### ADR-004: GitHub Actions OIDC (No Stored AWS Credentials)

**Status:** Accepted  
**Date:** 2026-05-01

#### Context

CI/CD pipelines typically store long-lived `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` as GitHub Secrets. These credentials never expire and create a persistent blast radius if leaked.

#### Decision

Use **GitHub Actions OIDC** to exchange short-lived GitHub JWTs for temporary AWS STS credentials.

#### Rationale

1. **Zero stored secrets**: No AWS credentials in GitHub at all — not even encrypted
2. **1-hour TTL**: Temporary STS credentials expire; leaked tokens are quickly worthless
3. **Repo-scoped**: IAM trust policy uses `StringLike` on `sub` claim — only this repo's pushes can assume the role
4. **Audit trail**: Every STS AssumeRoleWithWebIdentity call is logged in CloudTrail
5. **Industry standard**: AWS, Google, and Azure all support OIDC federation natively

#### Implementation Details

```yaml
# IAM Trust Policy (condition restricts to this exact repo)
"Condition": {
  "StringLike": {
    "token.actions.githubusercontent.com:sub": "repo:salimalsazu/modelserve:*"
  }
}
```

```yaml
# GitHub Actions workflow
- uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: arn:aws:iam::679835924810:role/modelserve-github-actions-prod
    aws-region: ap-southeast-1
```

#### Consequences

- **Positive**: Eliminates credential rotation burden; reduces blast radius
- **Negative**: Slightly more complex initial setup (OIDC provider must exist in AWS account)

#### Alternatives Rejected

| Approach | Reason Rejected |
|----------|----------------|
| Long-lived IAM user keys | Never expire; full blast radius if leaked |
| Vault/external secrets | Additional infrastructure to maintain |
| Manual deploy only | Not automated; human error prone |

---

### ADR-005: Docker Compose for Container Orchestration (Single-Node)

**Status:** Accepted  
**Date:** 2026-05-01

#### Context

The platform runs six services (API, MLflow, PostgreSQL, Redis, Prometheus, Grafana) that must be networked together with persistent volumes. We needed an orchestration layer appropriate for a single EC2 instance.

#### Decision

Use **Docker Compose** (v2 plugin) on a single EC2 t3.medium instance.

#### Rationale

1. **Simplicity**: One `docker-compose.yml` defines all services, networks, and volumes
2. **Shared network**: `modelserve` bridge network lets containers address each other by service name (e.g., `http://mlflow:5000`)
3. **Volume persistence**: Named volumes (`postgres_data`, `redis_data`, `mlflow_data`) survive container restarts
4. **Env var images**: `${API_IMAGE:-default}` pattern lets CI inject specific image tags without modifying the compose file
5. **Appropriate scale**: For a capstone / development environment, Kubernetes overhead is not justified

#### Deployment Pattern

```bash
# CI pushes env vars; EC2 pulls and restarts
export API_IMAGE=679835924810.dkr.ecr.ap-southeast-1.amazonaws.com/modelserve-api-prod:abc123
export MLFLOW_IMAGE=679835924810.dkr.ecr.ap-southeast-1.amazonaws.com/modelserve-mlflow-prod:abc123
docker compose up -d --remove-orphans
```

#### Consequences

- **Positive**: Zero Kubernetes complexity; fast iteration; one command to start everything
- **Negative**: No auto-scaling; single point of failure; Kubernetes migration would require rewrite

#### Future Migration Path

When traffic justifies it: ECS Fargate (minimal change) or EKS (full migration). The Docker images are portable — only orchestration changes.

#### Alternatives Rejected

| Option | Reason Rejected |
|--------|----------------|
| Kubernetes / EKS | Significant operational complexity for single-node dev |
| ECS Fargate | AWS lock-in for orchestration layer |
| Nomad | Small community; less tooling |

---

## 4. CI/CD Pipeline Documentation

### Pipeline Overview

File: [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml)

**Triggers:**
- Push to `main` branch (automatic)
- `workflow_dispatch` (manual trigger via GitHub UI)

**Jobs:** `test` → `build-push` → `deploy`

---

### Job 1: Test

**Runs on:** `ubuntu-latest`  
**Blocks:** build-push does not start until all tests pass

```
Steps:
1. actions/checkout@v4
2. Set up Python 3.10
3. pip install -r requirements.txt
4. pytest app/tests/ -v --tb=short
```

**What's tested:**

| Test Class | Scope |
|------------|-------|
| `TestHealthEndpoint` | `/health` returns 200, correct fields |
| `TestPredictEndpoint` | `/predict` validation, response structure, error codes |
| `TestMetricsEndpoint` | `/metrics` Prometheus format |
| `TestModelInfoEndpoint` | `/model/info` structure |
| `TestErrorHandling` | 503 when no model, 422 on bad input |
| `TestPerformance` | Latency < 1s, concurrent requests |

---

### Job 2: Build & Push

**Runs on:** `ubuntu-latest`  
**Needs:** `test`

```
Steps:
1. Checkout code
2. Assume AWS role via OIDC (no stored credentials)
3. Login to ECR
4. Set image tag = github.sha (40-char commit SHA)
5. docker/setup-buildx-action (layer caching enabled)
6. Build + push API image  → ECR with :sha and :latest tags
7. Build + push MLflow image → ECR with :sha and :latest tags
```

**Image tags:** `679835924810.dkr.ecr.ap-southeast-1.amazonaws.com/modelserve-api-prod:<sha>`

**Outputs:**
- `image_tag` — 40-char SHA passed to deploy job
- `ecr_registry` — registry hostname passed to deploy job

---

### Job 3: Deploy

**Runs on:** `ubuntu-latest`  
**Needs:** `build-push`  
**Environment:** `production` (can add required reviewers in GitHub settings)

```
Steps:
1. Checkout code
2. Assume AWS role via OIDC
3. aws ec2 describe-instances → get instance_id + public_ip by tag Name=modelserve-prod
4. ssh-keygen -t ed25519 (generate ephemeral key pair, in-memory)
5. aws ec2-instance-connect send-ssh-public-key (key has 60s TTL)
6. scp docker-compose.yml → ~/modelserve/docker-compose.yml
7. scp monitoring/ → ~/modelserve/monitoring/
8. scp fraud_model_v2.pkl → ~/modelserve/models/model.pkl (fallback model)
9. SSH: aws ecr get-login-password | docker login
10. SSH: docker pull $API_IMAGE && docker pull $MLFLOW_IMAGE
11. SSH: docker compose up -d --remove-orphans
12. Health check: curl /health (10 retries × 15s = 150s total budget)
13. Write GitHub Step Summary with all service URLs
```

**SSH Security:** No PEM file stored anywhere. Ephemeral ed25519 key sent via EC2 Instance Connect API and expires in 60 seconds. The `StrictHostKeyChecking=no` flag is acceptable because the EC2 IP is fetched dynamically from AWS API (trusted source) immediately before connection.

---

### Environment Variables

| Variable | Source | Purpose |
|----------|--------|---------|
| `AWS_REGION` | Hardcoded (`ap-southeast-1`) | AWS region |
| `AWS_ACCOUNT_ID` | GitHub Actions Variable | 12-digit account ID for ARN construction |
| `API_IMAGE` | build-push job output | ECR image URL with commit SHA |
| `MLFLOW_IMAGE` | build-push job output | MLflow ECR image URL with commit SHA |
| `EC2_IP` | aws describe-instances | Dynamic EC2 public IP |

**No secrets stored.** `AWS_ACCOUNT_ID` is a non-sensitive repository variable (not a secret). AWS credentials come from OIDC temporary tokens.

---

### Rollback Procedure

```bash
# Option 1: Revert commit and push (triggers new deploy)
git revert HEAD
git push origin main

# Option 2: Manual re-deploy of previous SHA
# Trigger workflow_dispatch from GitHub UI
# Or SSH to EC2 and pull previous tag manually:
ssh ec2-user@52.221.26.39
docker pull 679835924810.dkr.ecr.ap-southeast-1.amazonaws.com/modelserve-api-prod:<PREV_SHA>
export API_IMAGE=...:<PREV_SHA>
docker compose up -d
```

---

## 5. Runbook

> For the full operations runbook, see [`RUNBOOK.md`](./RUNBOOK.md).

### Quick Reference — Common Operations

#### Start Everything (EC2)

```bash
ssh ec2-user@52.221.26.39
cd ~/modelserve
docker compose up -d
docker compose ps    # verify all containers are Up
```

#### Check API Health

```bash
curl http://52.221.26.39:8000/health
# Expected: { "status": "healthy", "model_version": "...", ... }
```

#### Make a Test Prediction

```bash
curl -X POST http://52.221.26.39:8000/predict \
  -H "Content-Type: application/json" \
  -d @training/sample_request.json
# Expected: { "prediction": 0, "probability": 0.39, ... }
```

#### View Live Logs

```bash
docker compose logs -f api           # API logs
docker compose logs -f mlflow        # MLflow logs
docker compose logs --tail=50 api    # Last 50 lines
```

#### Register a Model

```bash
# On EC2 inside the API container:
docker exec modelserve-api python training/train.py --register --stage Production
```

#### View Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://52.221.26.39:3000 | admin / admin |
| Prometheus | http://52.221.26.39:9090 | — |
| MLflow | http://52.221.26.39:5000 | — |
| API Docs | http://52.221.26.39:8000/docs | — |

#### Destroy Infrastructure

```powershell
# Windows PowerShell (from repo root)
cd infrastructure
$env:PULUMI_CONFIG_PASSPHRASE = ""
.\.venv\Scripts\Activate.ps1
pulumi destroy --yes
```

#### Incident Response — API Down

```bash
# 1. Check container status
docker compose ps

# 2. Check logs for errors
docker compose logs --tail=100 api

# 3. Restart API container only
docker compose restart api

# 4. If still down, restart everything
docker compose down && docker compose up -d

# 5. If images are corrupted, re-pull
docker compose pull && docker compose up -d
```

#### Incident Response — Model Not Loading

```bash
# 1. Check MLflow server
curl http://52.221.26.39:5000/api/2.0/mlflow/registered-models/list

# 2. Verify local fallback model exists
docker exec modelserve-api ls -la /app/models/model.pkl

# 3. Force retrain and register
docker exec modelserve-api python training/train.py --register --stage Production
```

---

## 6. Known Limitations

> For the full limitations document, see [`LIMITATIONS.md`](./LIMITATIONS.md).

### Critical Limitations (Current Implementation)

| # | Limitation | Impact | Mitigation |
|---|-----------|--------|-----------|
| 1 | **Single EC2 node** | No high availability; instance failure = outage | Elastic IP remaps quickly; ECS/EKS migration path exists |
| 2 | **No API authentication** | Any IP can call `/predict` | Network-level: security group could restrict source IPs |
| 3 | **MLflow on SQLite** (compose default) | Not production-grade for concurrent writes | Switch `--backend-store-uri` to PostgreSQL URI |
| 4 | **Model cold start ~500ms** | First request after restart is slow | Health check warmup via `start_period: 10s` + scheduled pings |
| 5 | **No drift detection** | Model accuracy degrades silently | Evidently integration planned |
| 6 | **Redis data loss on restart** | Feature store loses materialized features | AOF persistence enabled (`appendonly yes`) |
| 7 | **No HTTPS** | Traffic to EC2 is plaintext | ACM certificate + ALB or Nginx termination required for production |
| 8 | **Grafana default credentials** | admin/admin is publicly accessible | Change immediately on production; use AWS Secrets Manager |
| 9 | **t3.medium memory limit** | ~3.9 GB RAM constrains model size | Instance type upgrade or model quantization |
| 10 | **No canary / A/B testing** | All traffic goes to one model version | Nginx weighted upstream or AWS ALB weighted target groups |

### Security Posture

| Control | Status | Notes |
|---------|--------|-------|
| AWS credentials in CI | ✅ None stored | OIDC only |
| SSH keys in CI | ✅ None stored | Ephemeral via EC2 Instance Connect |
| Container user | ✅ Non-root (`appuser`) | Dockerfile creates dedicated user |
| ECR image scanning | ✅ Enabled | `scan_on_push = True` in Pulumi |
| S3 bucket | ✅ Fully private | Public access blocked in Pulumi |
| Security group | ⚠️ Wide open | All ports open to 0.0.0.0/0; restrict to known IPs for production |
| API auth | ❌ None | No token/key required |
| HTTPS | ❌ None | HTTP only |

### Performance Baseline

Measured on t3.medium with pre-loaded RandomForest (fraud_model_v2.pkl):

| Metric | Value |
|--------|-------|
| p50 latency (`/predict`) | ~12 ms |
| p95 latency (`/predict`) | ~25 ms |
| p99 latency (`/predict`) | ~45 ms |
| Cold start (model load) | ~500 ms |
| Throughput (single core) | ~200 RPS |
| Model size in memory | ~85 MB |
