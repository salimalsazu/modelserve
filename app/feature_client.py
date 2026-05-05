"""
Feature store client for Feast.
Provides online feature retrieval from Redis and offline feature loading.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)

# Configuration
FEAST_CONFIG_PATH = os.getenv("FEAST_CONFIG_PATH", "feast_repo/feature_store.yaml")
FEAST_ENTITY_ID = os.getenv("FEAST_ENTITY_ID", "entity_id")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


class FeatureStoreClient:
    """
    Client for interacting with Feast feature store.
    Supports both online (Redis) and offline (parquet) feature retrieval.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature store client.
        
        Args:
            config_path: Path to feature_store.yaml
        """
        self.config_path = config_path or FEAST_CONFIG_PATH
        self._client = None
        self._initialized = False
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Feast client."""
        try:
            from feast import FeatureStore
            
            self._client = FeatureStore(repo_path=self.config_path.replace("/feature_store.yaml", "") if self.config_path.endswith("feature_store.yaml") else self.config_path)
            self._initialized = True
            logger.info(f"Feature store initialized with config: {self.config_path}")
            
        except ImportError as e:
            logger.warning(f"Feast not installed: {e}")
            self._initialized = False
        except Exception as e:
            logger.warning(f"Failed to initialize Feast: {e}")
            self._initialized = False
    
    def health_check(self) -> bool:
        """Check feature store connectivity."""
        if not self._initialized:
            return False
        
        try:
            # Test Redis connectivity
            import redis
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2)
            r.ping()
            return True
        except Exception as e:
            logger.error(f"Feature store health check failed: {e}")
            return False
    
    def get_online_features(self, entity_id: int) -> Dict[str, Any]:
        """
        Get online features for an entity from Redis.
        
        Args:
            entity_id: Unique entity identifier
        
        Returns:
            Dictionary of feature values
        """
        if not self._initialized:
            return self._get_mock_features(entity_id)
        
        try:
            # Use Feast to get online features
            from feast import FeatureStore
            
            fs = FeatureStore(repo_path=self.config_path.replace("/feature_store.yaml", "") if self.config_path.endswith("feature_store.yaml") else self.config_path)
            
            feature_service = fs.get_feature_service("model_features")
            
            entity_df = pd.DataFrame({"entity_id": [entity_id]})
            
            features = fs.get_online_features(
                entity_df=entity_df,
                features=feature_service,
            )
            
            return dict(features.to_dict())
        
        except Exception as e:
            logger.error(f"Failed to get online features: {e}")
            return self._get_mock_features(entity_id)
    
    def get_historical_features(self, entity_ids: List[int]) -> pd.DataFrame:
        """
        Get historical features for batch processing.
        
        Args:
            entity_ids: List of entity identifiers
        
        Returns:
            DataFrame with entity_id and feature columns
        """
        if not self._initialized:
            return self._get_mock_batch_features(entity_ids)
        
        try:
            from feast import FeatureStore
            
            fs = FeatureStore(repo_path=self.config_path.replace("/feature_store.yaml", "") if self.config_path.endswith("feature_store.yaml") else self.config_path)
            
            entity_df = pd.DataFrame({"entity_id": entity_ids})
            
            features = fs.get_historical_features(
                entity_df=entity_df,
                features=["driver_hourly_stats:total_trips", "driver_hourly_stats:avg_distance"],
            )
            
            return features.to_df()
        
        except Exception as e:
            logger.error(f"Failed to get historical features: {e}")
            return self._get_mock_batch_features(entity_ids)
    
    def _get_mock_features(self, entity_id: int) -> Dict[str, float]:
        """
        Generate mock features for testing when Feast is unavailable.
        
        Args:
            entity_id: Entity identifier
        
        Returns:
            Dictionary of mock feature values
        """
        import hashlib
        
        # Generate deterministic mock features based on entity_id
        seed = int(hashlib.md5(str(entity_id).encode()).hexdigest()[:8], 16)
        
        return {
            "feature_0": float((seed % 100) / 10),
            "feature_1": float(((seed * 7) % 100) / 10),
            "feature_2": float(((seed * 13) % 100) / 10),
            "feature_3": float(((seed * 17) % 100) / 10),
        }
    
    def _get_mock_batch_features(self, entity_ids: List[int]) -> pd.DataFrame:
        """Generate mock batch features."""
        return pd.DataFrame([
            {**{"entity_id": eid}, **self._get_mock_features(eid)}
            for eid in entity_ids
        ])


class RedisClient:
    """
    Direct Redis client for simple feature retrieval.
    Used when Feast is not available.
    """
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or REDIS_HOST
        self.port = port or REDIS_PORT
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            import redis
            self._client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        return self._client
    
    def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
            return None
    
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with optional TTL."""
        try:
            self.client.set(key, value, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Redis set failed: {e}")
            return False
    
    def mget(self, keys: List[str]) -> List[Optional[str]]:
        """Get multiple values from Redis."""
        try:
            return self.client.mget(keys)
        except Exception as e:
            logger.error(f"Redis mget failed: {e}")
            return [None] * len(keys)
    
    def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self.client.ping()
        except Exception:
            return False


if __name__ == "__main__":
    # Test feature store client
    print("Testing feature store client...")
    
    client = FeatureStoreClient()
    print(f"Initialized: {client._initialized}")
    
    features = client.get_online_features(1)
    print(f"Features for entity 1: {features}")
    
    batch = client.get_historical_features([1, 2, 3])
    print(f"Batch features: {len(batch)} rows")