# Known Limitations & Future Work

## Current Limitations

### 1. Model Loading
- **Issue**: First prediction has ~500ms cold start latency
- **Impact**: High-latency spike for initial requests after restart
- **Workaround**: Keep-alive requests, scheduled health checks
- **Future**: Implement model pre-loading, warmup endpoint

### 2. Feature Store Consistency
- **Issue**: Redis and Parquet stores can drift during materialization
- **Impact**: Training-serving skew if materialization fails
- **Workaround**: Manual verification, monitoring alerts
- **Future**: Transactional materialization, CDC pipeline

### 3. Horizontal Scaling
- **Issue**: Each FastAPI instance loads model independently
- **Impact**: Memory overhead, potential version mismatch
- **Workaround**: Shared model cache (e.g., Redis)
- **Future**: Model caching layer, distributed serving

### 4. Authentication
- **Issue**: No built-in authentication for API endpoints
- **Impact**: Anyone can make predictions
- **Workaround**: API Gateway with API keys, network isolation
- **Future**: OAuth2, JWT tokens, rate limiting

### 5. Model Versioning
- **Issue**: Only one "Production" stage at a time
- **Impact**: No A/B testing or canary deployments
- **Workaround**: Manual traffic splitting
- **Future**: Built-in traffic management, feature flags

### 6. Feature Engineering
- **Issue**: Limited to pre-computed features
- **Impact**: No real-time feature computation
- **Workaround**: Pre-compute all features during training
- **Future**: Streaming feature computation with Kafka

### 7. Monitoring Gaps
- **Issue**: No model performance monitoring
- **Impact**: No drift detection, accuracy tracking
- **Workaround**: Manual analysis
- **Future**: Evidently integration, ground truth collection

### 8. Data Storage
- **Issue**: Parquet files as offline store are not scalable
- **Impact**: Can't handle large training datasets
- **Workaround**: Manual partition management
- **Future**: Snowflake/BigQuery integration

### 9. Deployment Flexibility
- **Issue**: Docker Compose for orchestration
- **Impact**: No auto-scaling, limited to single node
- **Workaround**: Manual scaling, careful capacity planning
- **Future**: Kubernetes/EKS migration

### 10. CI/CD Testing
- **Issue**: Integration tests require full stack
- **Impact**: Slow feedback loop
- **Workaround**: Docker Compose for local testing
- **Future**: Test containers, mocks for dependencies

---

## Technology Debt

### High Priority
| Item | Description | Estimated Effort |
|------|-------------|-----------------|
| K8s Migration | Move to Kubernetes | 2 weeks |
| Auth Implementation | Add OAuth2/JWT | 1 week |
| Drift Detection | Evidently integration | 1 week |

### Medium Priority
| Item | Description | Estimated Effort |
|------|-------------|-----------------|
| gRPC API | Lower latency endpoint | 3 days |
| Model A/B Testing | Traffic splitting | 1 week |
| Streaming Features | Kafka integration | 2 weeks |

### Low Priority
| Item | Description | Estimated Effort |
|------|-------------|-----------------|
| Documentation | API docs, examples | 2 days |
| Benchmarking | Performance tests | 3 days |
| Multi-region | Disaster recovery | 1 week |

---

## Risks

### Operational Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Redis data loss | Low | High | Regular backups, AOF persistence |
| Model registry corruption | Low | High | PostgreSQL backups |
| S3 outage | Very Low | Medium | Local model fallback |
| EC2 instance failure | Medium | High | Auto-scaling, health checks |

### Security Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API key exposure | Low | Critical | Rotate immediately, audit logs |
| Container breakout | Very Low | Critical | Non-root user, AppArmor |
| Data exfiltration | Very Low | Critical | Network policies, encryption |

### Performance Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Memory exhaustion | Medium | High | Container limits, monitoring |
| Latency spikes | Medium | Medium | Caching, connection pooling |
| Database overload | Low | High | Query optimization, indexing |

---

## Out of Scope

The following are explicitly out of scope for the current implementation:

1. **Real-time ML Training**: Continuous training pipeline
2. **AutoML**: Automated hyperparameter tuning
3. **Data Versioning**: DVC-style dataset versioning
4. **Feature Store UI**: Admin interface for feature management
5. **Multi-tenancy**: Isolation between different clients
6. **GraphQL API**: Alternative API protocol
7. **Mobile SDK**: Native mobile integration
8. **Real-time Streaming**: WebSocket connections
9. **Federated Learning**: Privacy-preserving training
10. **Explainability**: SHAP integration for all models

---

## Assumptions

1. **Traffic Patterns**: Assumes <1000 RPS with burst capacity to 2000
2. **Model Size**: Assumes models <500MB (RAM constraint for t3.medium)
3. **Feature Count**: Assumes <100 features per entity
4. **User Base**: Assumes <100 concurrent users
5. **Data Volume**: Assumes <10M training samples
6. **Latency Budget**: Assumes 100ms p95 latency target is acceptable
7. **Availability**: 99.5% uptime (4.5 hours downtime/month)
8. **Budget**: AWS costs <$500/month for development environment

---

## Changelog

### v1.0.0 (Current)
- Initial production deployment
- FastAPI inference API
- MLflow model registry integration
- Feast feature store with Redis
- Prometheus + Grafana monitoring
- GitHub Actions CI/CD
- Pulumi infrastructure (EC2, S3, ECR)

### v0.9.0
- Beta release with limited features
- Basic monitoring setup
- Docker Compose deployment

### v0.1.0
- Proof of concept
- Flask-based API
- Local MLflow tracking
- Manual deployments