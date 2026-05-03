from prometheus_client import Counter, Histogram

prediction_requests = Counter(
    "prediction_requests_total", "Total prediction requests"
)

prediction_errors = Counter(
    "prediction_errors_total", "Total prediction errors"
)

prediction_latency = Histogram(
    "prediction_duration_seconds", "Prediction latency"
)