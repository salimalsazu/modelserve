"""
Prometheus metrics for ModelServe.
Tracks prediction latency, request rates, error rates, and model version.
"""

from prometheus_client import Counter, Histogram, Gauge, Enum

# Prediction metrics
prediction_requests = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint", "model_stage"]
)

prediction_errors = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
    ["endpoint", "error_type"]
)

prediction_latency = Histogram(
    "prediction_duration_seconds",
    "Prediction latency in seconds",
    ["endpoint", "model_stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# Feature store metrics
feature_store_requests = Counter(
    "feature_store_requests_total",
    "Total number of feature store requests",
    ["entity_type"]
)

feature_store_latency = Histogram(
    "feature_store_duration_seconds",
    "Feature store lookup latency in seconds",
    ["entity_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

feature_store_errors = Counter(
    "feature_store_errors_total",
    "Total number of feature store errors"
)

# Model metrics
model_version_gauge = Gauge(
    "model_version_info",
    "Current model version information",
    ["stage", "version"]
)

model_load_latency = Histogram(
    "model_load_duration_seconds",
    "Model loading duration in seconds"
)

# System metrics
active_requests = Gauge(
    "active_requests",
    "Number of currently active requests",
    ["endpoint"]
)

request_size_bytes = Histogram(
    "request_size_bytes",
    "Request payload size in bytes",
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000)
)

response_size_bytes = Histogram(
    "response_size_bytes",
    "Response payload size in bytes",
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000)
)

# Health check metrics
health_check_latency = Histogram(
    "health_check_duration_seconds",
    "Health check latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1)
)

component_health = Gauge(
    "component_health",
    "Health status of system components (1=healthy, 0=unhealthy)",
    ["component"]
)