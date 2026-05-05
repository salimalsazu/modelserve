# ModelServe Operations Runbook

## Overview

This runbook covers operational procedures for the ModelServe MLOps platform.

## Table of Contents
1. [Deployment Procedures](#deployment-procedures)
2. [Monitoring & Alerts](#monitoring--alerts)
3. [Troubleshooting](#troubleshooting)
4. [Backup & Recovery](#backup--recovery)
5. [Scaling](#scaling)
6. [Security](#security)

---

## Deployment Procedures

### Local Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Run tests
pytest app/tests/ -v

# Train model locally
python training/train.py --model-type random_forest --register
```

### Deploy to Development

```bash
# Build and push to dev ECR
docker build -t modelserve-dev:$GIT_SHA .
docker tag modelserve-dev:$GIT_SHA $ECR_REGISTRY/modelserve-dev:$GIT_SHA
docker push $ECR_REGISTRY/modelserve-dev:$GIT_SHA

# Deploy via SSH
ssh dev-server "docker pull $ECR_REGISTRY/modelserve-dev:$GIT_SHA"
ssh dev-server "docker-compose -f docker-compose.yml up -d"
```

### Deploy to Production

1. **Create Release PR** with version bump
2. **Merge to main** triggers CI/CD
3. **Manual Approval** required for production
4. **Health Check** runs automatically

```bash
# Verify deployment
curl -f https://api.modelserve.example.com/health

# Check model version
curl https://api.modelserve.example.com/model/info
```

---

## Monitoring & Alerts

### Accessing Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | N/A |
| MLflow | http://localhost:5000 | N/A |

### Key Metrics

#### Latency (p95 < 100ms target)
```promql
histogram_quantile(0.95, rate(prediction_duration_seconds_bucket[5m]))
```

#### Request Rate (>100 rps target)
```promql
rate(prediction_requests_total[5m])
```

#### Error Rate (<1% target)
```promql
rate(prediction_errors_total[5m]) / rate(prediction_requests_total[5m])
```

### Alert Thresholds

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| HighLatency | p95 > 500ms for 5min | Warning | Check model loading |
| CriticalLatency | p95 > 2s for 2min | Critical | Rollback deployment |
| HighErrorRate | error rate > 5% | Critical | Disable new features |
| ModelDown | model not loaded | Critical | Emergency rollback |
| RedisDown | Redis unreachable | Warning | Check network |

### Grafana Dashboard Panels

1. **API Overview**
   - Request rate over time
   - Error rate trend
   - Active connections

2. **Latency**
   - p50, p95, p99 percentiles
   - Latency histogram
   - Slow requests table

3. **Model**
   - Current model version
   - Prediction distribution
   - Feature store latency

4. **System**
   - CPU/Memory usage
   - Container status
   - Disk usage

---

## Troubleshooting

### Service Not Responding

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs api --tail=100

# Restart service
docker-compose restart api

# Check health
curl http://localhost:8000/health
```

### Model Loading Failed

```bash
# Check MLflow connectivity
curl http://mlflow:5000/api/2.0/preview/mlflow/registered-models

# Check S3 bucket
aws s3 ls s3://modelserve-artifacts/

# Manually load model
python training/train.py --model-type random_forest
```

### High Memory Usage

```bash
# Check memory per container
docker stats

# Restart with memory limit
docker-compose up -d --memory=2g api

# Profile Python memory
import tracemalloc
tracemalloc.start()
# ... run predictions ...
print(tracemalloc.get_traced_memory())
```

### Slow Predictions

1. **Check Model Size**
   ```bash
   docker exec modelserve-api ls -la training/model.pkl
   ```

2. **Profile Latency**
   ```bash
   # Add timing to requests
   time curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"entity_id": 1, "features": [1,2,3,4,5]}'
   ```

3. **Check Feature Store**
   ```bash
   redis-cli ping
   docker exec modelserve-redis redis-cli info memory
   ```

### Database Connection Issues

```bash
# Test PostgreSQL
docker exec modelserve-postgres psql -U mlflow -d mlflow -c "SELECT 1"

# Check MLflow logs
docker-compose logs mlflow | grep -i error

# Restart PostgreSQL
docker-compose restart postgres
```

---

## Backup & Recovery

### MLflow Metadata Backup

```bash
# Backup PostgreSQL
docker exec modelserve-postgres pg_dump -U mlflow mlflow > mlflow_backup.sql

# Schedule daily backups
crontab -e
0 2 * * * docker exec modelserve-postgres pg_dump -U mlflow mlflow > /backups/mlflow_$(date +%Y%m%d).sql
```

### MLflow Artifacts Backup

```bash
# Sync S3 bucket
aws s3 sync s3://modelserve-artifacts/mlflow/ /backups/mlflow-artifacts/

# Versioning enabled (point-in-time recovery)
aws s3api list-object-versions --bucket modelserve-artifacts
```

### Redis Data Backup

```bash
# BGSAVE
docker exec modelserve-redis redis-cli BGSAVE

# Copy RDB file
docker cp modelserve-redis:/data/dump.rdb /backups/redis_dump.rdb
```

### Full System Backup

```bash
#!/bin/bash
# backup_modelserve.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/backups/$DATE

mkdir -p $BACKUP_DIR

# PostgreSQL
docker exec modelserve-postgres pg_dump -U mlflow mlflow > $BACKUP_DIR/mlflow.sql

# S3 artifacts
aws s3 sync s3://modelserve-artifacts/ $BACKUP_DIR/s3/ --exact-timestamps

# Redis
docker exec modelserve-redis redis-cli SAVE
docker cp modelserve-redis:/data/dump.rdb $BACKUP_DIR/redis.rdb

# Docker volumes
docker run --rm -v modelserve_postgres_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/postgres_vol.tar.gz -C /data .

echo "Backup complete: $BACKUP_DIR"
```

### Recovery Procedure

```bash
# 1. Stop services
docker-compose down

# 2. Restore PostgreSQL
docker exec -i modelserve-postgres psql -U mlflow mlflow < mlflow_backup.sql

# 3. Restore Redis
docker cp redis_dump.rdb modelserve-redis:/data/dump.rdb
docker exec modelserve-redis redis-cli BGSAVE

# 4. Restore S3 (if needed)
aws s3 sync /backups/s3/ s3://modelserve-artifacts/

# 5. Start services
docker-compose up -d
```

---

## Scaling

### Horizontal Scaling (FastAPI)

```bash
# Scale API instances
docker-compose up -d --scale api=3

# Update load balancer configuration
# Add instances to target group
```

### Vertical Scaling (EC2)

```bash
# In Pulumi infrastructure/__main__.py:
# Change instance_type from "t3.medium" to "t3.large"

# Apply changes
cd infrastructure
pulumi up --stack prod
```

### Feature Store Scaling

```bash
# Enable Redis clustering
# Update feature_store.yaml:
# online_store:
#   type: redis_cluster
#   hosts: ["redis-1", "redis-2", "redis-3"]
```

### Database Scaling

```bash
# Read replica for MLflow
# Add to docker-compose.yml:
mlflow-replica:
  image: postgres:15-alpine
  command: --replica
  depends_on: [postgres]
```

---

## Security

### Rotate AWS Credentials

```bash
# Update in AWS Console
# IAM > Users > modelserve-deploy > Security credentials

# Update GitHub Secrets
# Settings > Secrets > Actions > Update AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

### Update Docker Images

```bash
# Scan image for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image modelserve:latest

# Push new version
docker build --push -t $ECR_REGISTRY/modelserve:secure-latest .
```

### Enable HTTPS (Production)

```bash
# Request certificate
aws acm request-certificate \
  --domain-name api.modelserve.example.com \
  --validation-method DNS

# Update security group to allow 443
# Configure ALB with HTTPS listener
```

### Audit Logs

```bash
# Enable CloudTrail
aws cloudtrail create-trail \
  --name modelserve-audit \
  --s3-bucket-name modelserve-audit-logs

# Query recent API calls
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=Username,AttributeValue=deploy
```

---

## Emergency Procedures

### Complete Service Outage

1. **Check Status Page**: https://status.modelserve.example.com
2. **Notify Team**: Page on-call via PagerDuty
3. **Assess Damage**: Check logs, metrics, error rates
4. **Rollback**: Deploy previous version
5. **Communicate**: Update status page every 15min
6. **Postmortem**: Review within 24 hours

### Rollback Procedure

```bash
# Get previous image tag
git log --oneline -5
PREV_SHA=$(git rev-parse HEAD~1)

# Deploy previous version
docker pull $ECR_REGISTRY/modelserve:$PREV_SHA
docker stop modelserve && docker rm modelserve
docker run -d --name modelserve \
  -p 8000:8000 \
  $ECR_REGISTRY/modelserve:$PREV_SHA

# Verify
curl -f http://localhost:8000/health
```

### Database Corruption

1. **Stop all services**
2. **Identify corruption point**: Check logs for first error
3. **Restore from backup**: Use most recent clean backup
4. **Replay transactions**: If using WAL, replay from point of corruption
5. **Verify integrity**: Run pg_checksums
6. **Resume service**: Start services, verify health

### Security Incident

1. **Isolate**: Block affected IPs immediately
2. **Assess**: Determine scope of breach
3. **Preserve**: Capture logs, memory dumps
4. **Remediate**: Patch vulnerability, rotate credentials
5. **Report**: Document for compliance
6. **Review**: Update security measures

---

## Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | oncall@modelserve.example.com | PagerDuty |
| DevOps Lead | devops@modelserve.example.com | Slack #incidents |
| Security Team | security@modelserve.example.com | emergency@ |
| ML Platform | ml-platform@modelserve.example.com | Slack #ml-platform |