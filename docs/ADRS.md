# Architecture Decision Records (ADRs)

## ADR-001: FastAPI for Inference API

**Status:** Accepted  
**Date:** 2024-01-15

### Context
We need a high-performance inference API that can handle real-time predictions with low latency while integrating with MLflow, Feast, and monitoring systems.

### Decision
We chose **FastAPI** as the inference API framework.

### Reasons
1. **Performance**: FastAPI is one of the fastest Python web frameworks, comparable to Node.js and Go
2. **Type Safety**: Pydantic integration provides automatic request validation and documentation
3. **Async Support**: Native async/await for I/O-bound operations (Redis, MLflow)
4. **OpenAPI**: Automatic API documentation with Swagger UI
5. **Ecosystem**: Strong integration with Starlette middleware (Prometheus, CORS, etc.)

### Consequences
- **Positive**: Rapid development, automatic docs, type validation
- **Negative**: Steeper learning curve for async patterns vs Flask

### Alternatives Considered
| Framework | Pros | Cons |
|-----------|------|------|
| Flask | Simple, familiar | No async, manual validation |
| Starlette | Low-level | More boilerplate |
| Django | Full-featured | Overkill for API-only |

---

## ADR-002: MLflow for Model Registry

**Status:** Accepted  
**Date:** 2024-01-15

### Context
We need centralized model management with versioning, metadata tracking, and deployment support.

### Decision
We use **MLflow Model Registry** as our model management system.

### Reasons
1. **Industry Standard**: Widely adopted in MLOps community
2. **Model Lineage**: Track which data/pipeline created each model
3. **Staging Workflow**: Production promotion with stage transitions
4. **Plugin Architecture**: Easy integration with custom serving frameworks
5. **Metadata Storage**: Metrics, params, and tags with each version

### Consequences
- **Positive**: Unified model management, easy deployment, experiment tracking
- **Negative**: Additional PostgreSQL dependency, potential scaling issues at very high volume

### Alternatives Considered
| Solution | Pros | Cons |
|----------|------|------|
| Weights & Biases | Great UI | Cost, closed-source |
| MLflow (self-hosted) | Control, free | Operational overhead |
| Custom DB | Full control | Reinventing wheel |

---

## ADR-003: Feast for Feature Store

**Status:** Accepted  
**Date:** 2024-01-15

### Context
We need consistent feature engineering between training and serving to prevent training-serving skew.

### Decision
We implement **Feast** as our feature store with Redis online store.

### Reasons
1. **Consistency**: Same feature definitions for training and serving
2. **Abstraction**: Unified API regardless of storage backend
3. **Online/Offline**: Both real-time and batch feature access
4. **Popular**: Active community, enterprise adoption
5. **Integration**: Native MLflow and Kubernetes support

### Consequences
- **Positive**: Eliminated training-serving skew, feature reuse
- **Negative**: Added complexity, Feast learning curve, Redis dependency

### Alternatives Considered
| Solution | Pros | Cons |
|----------|------|------|
| Tecton | Managed, enterprise | Cost, vendor lock-in |
| Hopsworks | Full platform | Heavy, complex |
| Custom Redis | Simple | No offline store, no lineage |

### Implementation Notes
- Use Parquet files for offline store (training data)
- Redis for online store (serving with <1ms latency)
- Mock fallback when Feast unavailable

---

## ADR-004: Prometheus + Grafana for Monitoring

**Status:** Accepted  
**Date:** 2024-01-15

### Context
We need comprehensive observability including latency percentiles, request rates, and model drift detection.

### Decision
We deploy **Prometheus** for metrics collection and **Grafana** for visualization.

### Reasons
1. **Metrics Standard**: Prometheus format is industry standard
2. **Histogram Buckets**: Native support for p50, p95, p99 calculations
3. **Alerting**: Prometheus alerting rules + Alertmanager
4. **Dashboards**: Pre-built Grafana dashboards for common metrics
5. **Cost**: Open-source, self-hosted, no per-metric pricing

### Consequences
- **Positive**: Rich metrics, alerting, visualization
- **Negative**: Operational complexity, cardinality concerns

### Metrics Tracked
```python
prediction_duration_seconds  # Histogram with p50, p95, p99
prediction_requests_total    # Counter by endpoint
prediction_errors_total      # Counter by error type
model_version_info           # Gauge with labels
```

### Alternatives Considered
| Solution | Pros | Cons |
|----------|------|------|
| DataDog | Managed, APM | Cost, vendor lock-in |
| CloudWatch | AWS native | Limited percentiles |
| OpenTelemetry | Vendor-neutral | Additional complexity |

---

## ADR-005: Pulumi for Infrastructure as Code

**Status:** Accepted  
**Date:** 2024-01-15

### Context
We need reproducible infrastructure provisioning for AWS resources (EC2, S3, ECR) with proper state management.

### Decision
We use **Pulumi** (Python) for infrastructure as code.

### Reasons
1. **Real Language**: Python for infrastructure, same language as ML code
2. **Abstractions**: Reusable components, loops, functions
3. **State Management**: Built-in state backend, drift detection
4. **Preview**: `pulumi preview` before apply
5. **Testing**: Unit tests with pytest

### Consequences
- **Positive**: Familiar language, testable, preview before apply
- **Negative**: New tool for team, Python-only SDK

### Alternatives Considered
| Tool | Pros | Cons |
|------|------|------|
| Terraform | Mature, widely used | HCL language |
| AWS CDK | AWS native | TypeScript/Python hybrid |
| CloudFormation | AWS native | YAML/JSON, verbose |

### Resources Provisioned
- EC2 instance with Docker
- S3 bucket (MLflow artifacts)
- ECR repository with lifecycle policy
- Security groups
- IAM roles

---

## ADR-006: GitHub Actions for CI/CD

**Status:** Accepted  
**Date:** 2024-01-15

### Context
We need automated testing, Docker builds, and deployments triggered by code changes.

### Decision
We use **GitHub Actions** for the CI/CD pipeline.

### Reasons
1. **Native GitHub Integration**: PR checks, deployments on merge
2. **Marketplace**: Pre-built actions (AWS, Docker, Pulumi)
3. **Cost**: Free for open source, generous free tier
4. **Secrets**: Built-in secrets management
5. **YAML**: Workflows as code in repo

### Consequences
- **Positive**: Native integration, extensive actions marketplace
- **Negative**: GitHub-only (migration would require rewrite)

### Pipeline Stages
1. **Test**: pytest + linting + coverage upload
2. **Build**: Docker buildx + push to ECR
3. **Deploy Dev**: SSH + docker pull/restart
4. **Deploy Prod**: Rolling deployment with health check
5. **Infrastructure**: Pulumi preview/apply

---

## ADR-007: Docker Multi-Stage Build for Production Images

**Status:** Accepted  
**Date:** 2024-01-15

### Context
We need small, secure Docker images for production deployment.

### Decision
We use **multi-stage Docker builds** with non-root users.

### Reasons
1. **Image Size**: Separate build and runtime stages reduce size
2. **Security**: Non-root user, no package managers in final image
3. **Reproducibility**: Build args for timestamps, git refs
4. **Layer Caching**: Optimized layer ordering for cache efficiency

### Consequences
- **Positive**: Images under 800MB, secure by default
- **Negative**: More complex Dockerfile

### Build Optimization
```dockerfile
# Builder stage
FROM python:3.10-slim as builder
RUN pip install --user -r requirements.txt

# Runtime stage  
FROM python:3.10-slim
COPY --from=builder /root/.local /home/appuser/.local
USER appuser
```

### Size Target
- Development image: ~2GB (includes dev tools)
- Production image: <800MB (runtime only)

---

## ADR-008: PostgreSQL for MLflow Backend

**Status:** Accepted  
**Date:** 2024-01-15

### Context
MLflow requires a backend store for metadata. We need ACID compliance and concurrent access support.

### Decision
We use **PostgreSQL** as the MLflow backend store.

### Reasons
1. **ACID Compliance**: Safe concurrent writes
2. **Scalability**: Supports many concurrent connections
3. **Popular**: Well-understood, managed options available
4. **MLflow Native**: First-class PostgreSQL support
5. **Replication**: PostgreSQL replication for high availability

### Consequences
- **Positive**: Reliable metadata storage, concurrent access
- **Negative**: Additional database to operate

### Alternatives Considered
| Backend | Pros | Cons |
|---------|------|------|
| SQLite | Simple | No concurrent access |
| MySQL | Common | Less MLflow optimization |
| PostgreSQL | ACID, replication | Operational overhead |

---

## ADR-009: Redis for Online Feature Store

**Status:** Accepted  
**Date:** 2024-01-15

### Context
Feature serving requires sub-millisecond latency for real-time predictions.

### Decision
We use **Redis** as the online feature store.

### Reasons
1. **Latency**: Sub-millisecond reads, in-memory storage
2. **Data Structures**: Native support for hashes, sorted sets
3. **Persistence**: AOF/RDB persistence options
4. **Clustering**: Horizontal scaling for high availability
5. **Feast Native**: Built-in Redis integration

### Consequences
- **Positive**: Ultra-low latency, simple operations
- **Negative**: In-memory limits, single point of failure without clustering

### Fallback Strategy
If Redis is unavailable, feature client returns mock features generated deterministically from entity_id.

---

## ADR-010: Blue-Green Deployment for Production

**Status:** Proposed  
**Date:** 2024-01-15

### Context
Production deployments need zero-downtime with instant rollback capability.

### Decision
We implement **blue-green deployment** for production.

### Reasons
1. **Zero Downtime**: New version receives traffic immediately
2. **Instant Rollback**: Previous version remains ready
3. **Testing**: Verify new version before traffic switch
4. **CI/CD Integration**: Simple SSH-based implementation

### Implementation
```bash
# Deploy new version
ssh ec2 "docker pull $NEW_IMAGE && docker run -d --name modelserve-new"

# Health check
sleep 30 && ssh ec2 "curl -f http://localhost:8000/health"

# Switch traffic (remove old, rename new)
ssh ec2 "docker stop modelserve-old; docker rm modelserve-old; docker rename modelserve-new modelserve"
```

### Consequences
- **Positive**: Zero downtime, instant rollback
- **Negative**: Double container resources during deploy

---

## Future ADRs to Consider

1. **ADR-011**: Kubernetes vs Docker Compose for orchestration
2. **ADR-012**: gRPC inference endpoint for lower latency
3. **ADR-013**: Model A/B testing framework
4. **ADR-014**: Feature drift detection with Evidently
5. **ADR-015**: Automated retraining triggers based on performance degradation