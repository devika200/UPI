"""ML Model loader and manager"""
import os
import logging
import joblib
import sys
import numpy as np

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML models for fraud detection"""
    
    def __init__(self):
        self.hmm_model = None
        self.crf_model = None
        self.scaler = None
        self.label_encoder = None
        self.dataset = None
    
    def init_app(self, app):
        """Initialize models with Flask app configuration"""
        # Import AutoRegressiveHMM if available
        try:
            from model import AutoRegressiveHMM
            sys.modules["__main__"].AutoRegressiveHMM = AutoRegressiveHMM
        except ImportError:
            logger.warning("model.py not found, will use basic prediction")
        
        # Load models
        self._load_hmm_model(app.config['HMM_MODEL_PATHS'])
        self._load_crf_model(app.config['CRF_MODEL_PATHS'])
        self._load_scaler(app.config['SCALER_PATHS'])
        self._load_encoder(app.config['ENCODER_PATHS'])
        self._load_dataset(app.config['DATASET_PATHS'])
    
    def _load_hmm_model(self, paths):
        """Load HMM model from multiple possible paths"""
        for path in paths:
            if os.path.exists(path):
                try:
                    self.hmm_model = joblib.load(path)
                    logger.info(f"[OK] AR-HMM model loaded from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load HMM model from {path}: {e}")
        
        logger.error("[ERR] No HMM model loaded. Predictions will use rule-based approach.")
    
    def _load_crf_model(self, paths):
        """Load CRF model from multiple possible paths"""
        for path in paths:
            if os.path.exists(path):
                try:
                    self.crf_model = joblib.load(path)
                    logger.info(f"[OK] CRF model loaded from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load CRF model from {path}: {e}")
        
        logger.warning("[WARN] CRF model not found, using HMM only")
    
    def _load_scaler(self, paths):
        """Load feature scaler from multiple possible paths"""
        for path in paths:
            if os.path.exists(path):
                try:
                    self.scaler = joblib.load(path)
                    logger.info(f"[OK] Scaler loaded from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load scaler from {path}: {e}")
        
        logger.warning("[WARN] Scaler not found - predictions may be less accurate")
    
    def _load_encoder(self, paths):
        """Load label encoder from multiple possible paths"""
        for path in paths:
            if os.path.exists(path):
                try:
                    self.label_encoder = joblib.load(path)
                    logger.info(f"[OK] Label encoder loaded from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load encoder from {path}: {e}")
        
        logger.warning("[WARN] Label encoder not found - predictions may be less accurate")
    
    def _load_dataset(self, paths):
        """Load fallback dataset for feature calculation"""
        import pandas as pd
        
        for path in paths:
            if os.path.exists(path):
                try:
                    self.dataset = pd.read_csv(path, parse_dates=["Timestamp"])
                    logger.info(f"Fallback dataset loaded from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load dataset from {path}: {e}")
        
        logger.warning("No fallback dataset loaded. Will use MongoDB history only.")
    
    def is_ready(self):
        """Check if at least one model is loaded"""
        return self.hmm_model is not None
    
    def transform_with_lagging(self, features):
        """Scale features, handling both 10-feature and 30-feature inputs
        
        For 10-feature input (2D): Apply scaler directly
        For 30-feature input (2D): Apply scaler to each 10-feature segment independently
        
        Returns:
            Scaled features with same shape as input
        """
        try:
            if self.scaler is None:
                logger.warning("[WARN] [ModelManager] Scaler not loaded, returning unscaled features")
                return features
            
            # Handle 2D input only
            if len(features.shape) == 2:
                # Detect input dimensionality
                if features.shape[1] == 10:
                    # 10-feature input: apply scaler directly
                    return self.scaler.transform(features)
                elif features.shape[1] == 30:
                    # 30-feature input: apply scaler to each 10-feature segment independently
                    scaled_features = np.zeros_like(features, dtype=float)
                    
                    # Apply scaler to each segment [0:10], [10:20], [20:30]
                    for i in range(3):
                        start_idx = i * 10
                        end_idx = (i + 1) * 10
                        segment = features[:, start_idx:end_idx]
                        scaled_segment = self.scaler.transform(segment)
                        scaled_features[:, start_idx:end_idx] = scaled_segment
                    
                    # Ensure output is finite
                    if not np.all(np.isfinite(scaled_features)):
                        logger.error("[ERROR] [ModelManager] Scaled features contain NaN or infinity, returning input unchanged")
                        return features
                    
                    return scaled_features
                else:
                    logger.error(f"[ERROR] [ModelManager] Unexpected feature dimension: {features.shape[1]}, expected 10 or 30")
                    return features
            else:
                logger.error(f"[ERROR] [ModelManager] Unexpected feature shape: {features.shape}, expected 2D")
                return features
        except Exception as e:
            logger.error(f"[ERROR] [ModelManager] Scaling failed: {e}, returning input unchanged")
            return features


# Global model manager instance
model_manager = ModelManager()
