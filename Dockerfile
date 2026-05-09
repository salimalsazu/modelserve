# Multi-stage Dockerfile for ModelServe
# Stage 1: Builder - Install dependencies         
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies     
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production image
FROM python:3.10-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r appgroup && \
    useradd -r -g appgroup appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code (to /app directly, not nested)
COPY --chown=appuser:appgroup app/ ./app/        
COPY --chown=appuser:appgroup training/ ./training/
COPY --chown=appuser:appgroup feast_repo/ ./feast_repo/

# Create __init__.py files to make app a package
RUN touch /app/app/__init__.py /app/training/__init__.py

# Set Python path
ENV PATH=/home/appuser/.local/bin:$PATH    
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV MLFLOW_ARTIFACT_ROOT=/tmp/mlruns
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Create mlruns directory for MLflow artifact storage (symlink to tmp)
RUN mkdir -p /tmp/mlruns && rm -rf /app/mlruns && ln -s /tmp/mlruns /app/mlruns && chown -R appuser:appgroup /tmp/mlruns /app/mlruns

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]