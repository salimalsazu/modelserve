"""
Feast Feature Definitions for ModelServe.

This module defines the feature views and feature services for the ML pipeline.
Features are stored in parquet files and served online via Redis.
"""

from feast import Entity, Feature, FeatureView, FileSource, FeatureService
from feast.types import Float64, Int64
from datetime import timedelta
import os

# Define entity
entity = Entity(
    name="entity_id",
    join_keys=["entity_id"],
    description="Unique identifier for prediction entities"
)

# Define file source for offline features
entity_features_source = FileSource(
    name="entity_features_source",
    path=f"{os.path.dirname(os.path.abspath(__file__))}/data/entity_features.parquet",
    timestamp_field="event_timestamp",
)

# Define feature view for entity features
entity_feature_view = FeatureView(
    name="entity_features",
    entities=["entity_id"],
    ttl=timedelta(days=30),
    schema=[
        Feature(name="feature_0", dtype=Float64),
        Feature(name="feature_1", dtype=Float64),
        Feature(name="feature_2", dtype=Float64),
        Feature(name="feature_3", dtype=Float64),
        Feature(name="feature_4", dtype=Float64),
        Feature(name="feature_5", dtype=Float64),
        Feature(name="feature_6", dtype=Float64),
        Feature(name="feature_7", dtype=Float64),
        Feature(name="feature_8", dtype=Float64),
        Feature(name="feature_9", dtype=Float64),
    ],
    source=entity_features_source,
)

# Define aggregated feature view
aggregated_features_source = FileSource(
    name="aggregated_features_source",
    path=f"{os.path.dirname(os.path.abspath(__file__))}/data/aggregated_features.parquet",
    timestamp_field="event_timestamp",
)

aggregated_feature_view = FeatureView(
    name="aggregated_features",
    entities=["entity_id"],
    ttl=timedelta(days=7),
    schema=[
        Feature(name="count_last_7d", dtype=Int64),
        Feature(name="sum_last_7d", dtype=Float64),
        Feature(name="avg_last_7d", dtype=Float64),
        Feature(name="max_last_7d", dtype=Float64),
        Feature(name="min_last_7d", dtype=Float64),
    ],
    source=aggregated_features_source,
)

# Define feature service for model inference
model_features = FeatureService(
    name="model_features",
    features=[
        entity_feature_view[[f"feature_{i}" for i in range(10)]],
        aggregated_feature_view[["count_last_7d", "avg_last_7d"]],
    ],
)

# Define feature service for training
training_features = FeatureService(
    name="training_features",
    features=[
        entity_feature_view,
        aggregated_feature_view,
    ],
)