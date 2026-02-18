"""Fraud detection service"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from app.utils import _extract_features_from_transaction

logger = logging.getLogger(__name__)


class FraudDetectionService:
    """Service for fraud detection logic"""
    
    def __init__(self, model_manager, db, feature_columns):
        self.model_manager = model_manager
        self.db = db
        self.feature_columns = feature_columns
        self.n_lags = 3
    
    
    
    
    def _convert_to_crf_format(self, feature_array, feature_names=None):
        """Convert feature array to CRF dict format
        
        Input: np.ndarray of shape (1, 10)
        Output: [[{'feature_0': 0.5, 'feature_1': 0.3, ...}]]
        
        Returns:
            List of sequences, each sequence is list of dicts
        """
        try:
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(feature_array.shape[1])]
            
            # Convert numpy array to list of dicts
            crf_input = []
            for features in feature_array:
                feature_dict = {feature_names[i]: float(features[i]) for i in range(len(features))}
                crf_input.append([feature_dict])
            
            return crf_input
        except Exception as e:
            logger.error(f"[ERROR] [FraudDetectionService] CRF format conversion failed: {e}")
            return None
    
    def _get_user_transaction_history(self, username, transactions_collection, limit=2):
        """Get user's recent transactions from MongoDB for lagging
        
        Returns:
            List of transaction dicts, oldest first
        """
        if transactions_collection is None:
            return []
        
        try:
            txs = list(transactions_collection.find(
                {"username": username},
                {"Transaction Amount (INR)": 1, "Transaction Time": 1, "Receiver UPI ID": 1}
            ).sort("Transaction Time", -1).limit(limit))
            return list(reversed(txs))  # Oldest first
        except Exception as e:
            logger.error(f"[ERROR] [FraudDetectionService] [username={username}] Could not get transaction history: {e}")
            return []
    
    def predict(self, feature_vector, username, transactions_collection=None):
        """Make fraud prediction using ensemble of models with proper lagging

        Process:
        1. Get last 2 transactions from MongoDB
        2. Extract features from each historical transaction
        3. Create 3-row feature matrix: [t-2_features, t-1_features, t_features]
        4. Pass to HMM which creates lagging internally
        5. Scale current 10 features
        6. CRF prediction: 10 features (dict format) → label → probability
        7. Ensemble: average probabilities
        8. Classify: score >= 0.5 = fraud, 0.3-0.5 = suspicious, < 0.3 = normal

        Returns:
            (fraud_detected: bool, fraud_score: float, label: int, confidence: str)
        """
        if self.model_manager.hmm_model is None:
            logger.warning("[WARN] [FraudDetectionService] HMM model not loaded")
            return False, 0.5, 1, "Medium"

        try:
            # Log input feature vector
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Current Features (10): {feature_vector}")
            
            # Get last 2 transactions from MongoDB
            history = self._get_user_transaction_history(username, transactions_collection, limit=2)
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Transaction History Count: {len(history)}")
            
            # Build feature matrix with historical + current transactions
            feature_matrix = []
            
            # Add historical transactions
            for i, hist_tx in enumerate(history):
                hist_features = _extract_features_from_transaction(hist_tx, transactions_collection, self.model_manager.dataset)
                logger.info(f"[INFO] [FraudDetectionService] [username={username}] Historical Tx {i} Features: {hist_features}")
                feature_matrix.append(hist_features)
            
            # Add current transaction
            feature_matrix.append(feature_vector)
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Feature Matrix Shape: {np.array(feature_matrix).shape}")
            
            # Convert to numpy array
            feature_matrix = np.array(feature_matrix)
            
            # Scale the feature matrix
            try:
                scaled_matrix = self.model_manager.scaler.transform(feature_matrix)
                logger.info(f"[INFO] [FraudDetectionService] [username={username}] Scaled Matrix Shape: {scaled_matrix.shape}")
                logger.info(f"[INFO] [FraudDetectionService] [username={username}] Scaled Matrix:\n{scaled_matrix}")
            except Exception as e:
                logger.error(f"[ERROR] [FraudDetectionService] [username={username}] Scaling features failed: {e}")
                scaled_matrix = feature_matrix
            
            # HMM prediction - pass multiple rows so it can create lagging internally
            # The HMM's create_lag_features() will shift rows to create lagged features
            try:
                hmm_predictions = self.model_manager.hmm_model.predict(scaled_matrix)
                logger.info(f"[INFO] [FraudDetectionService] [username={username}] HMM Predictions: {hmm_predictions}")
                # Get the prediction for the last row (current transaction)
                hmm_state = int(hmm_predictions[-1])
                logger.info(f"[INFO] [FraudDetectionService] [username={username}] HMM State (last row): {hmm_state}")
            except Exception as e:
                logger.error(f"[ERROR] [FraudDetectionService] [username={username}] HMM prediction failed: {e}")
                return False, 0.5, 1, "Medium"

            # Map HMM hidden state to risk probability
            # State 0: low risk (0.0), State 1: medium risk (0.5), State 2: high risk (1.0)
            hmm_prob = float(hmm_state) / 2.0
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] HMM Probability: {hmm_prob:.2f}")

            # Scale current features for CRF
            try:
                scaled_current = self.model_manager.scaler.transform([feature_vector])
                logger.info(f"[INFO] [FraudDetectionService] [username={username}] Scaled Current Features: {scaled_current[0]}")
            except Exception as e:
                logger.error(f"[ERROR] [FraudDetectionService] [username={username}] Scaling current features failed: {e}")
                scaled_current = np.array([feature_vector])

            # CRF prediction (returns class labels: '0', '1', or '2')
            crf_prob = hmm_prob  # Default to HMM probability
            crf_label = None
            if self.model_manager.crf_model:
                try:
                    # Convert scaled features to CRF dict format
                    crf_input = self._convert_to_crf_format(scaled_current)
                    logger.info(f"[INFO] [FraudDetectionService] [username={username}] CRF Input Format: {crf_input}")
                    if crf_input is not None:
                        crf_predictions = self.model_manager.crf_model.predict(crf_input)
                        logger.info(f"[INFO] [FraudDetectionService] [username={username}] CRF Predictions: {crf_predictions}")
                        crf_label = str(crf_predictions[0][0])
                        logger.info(f"[INFO] [FraudDetectionService] [username={username}] CRF Label (string): {crf_label}")
                        # Map CRF label to risk probability
                        crf_prob = float(crf_label) / 2.0
                        logger.info(f"[INFO] [FraudDetectionService] [username={username}] CRF Probability: {crf_prob:.2f}")
                    else:
                        logger.warning(f"[WARN] [FraudDetectionService] [username={username}] CRF format conversion returned None")
                except Exception as e:
                    logger.error(f"[ERROR] [FraudDetectionService] [username={username}] CRF prediction failed: {e}")
                    crf_prob = hmm_prob

            # Ensemble: average the probabilities
            ensemble_fraud_score = (hmm_prob + crf_prob) / 2.0
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Ensemble Score: {ensemble_fraud_score:.2f}")

            # Classify based on probability thresholds
            # Adjusted thresholds to reduce false positives
            if ensemble_fraud_score >= 0.9:  # Fraud (very high confidence)
                final_label = 2
                confidence = "High"
            elif ensemble_fraud_score >= 0.7:  # Suspicious (high confidence)
                final_label = 1
                confidence = "Medium"
            else:  # Normal
                final_label = 0
                confidence = "Low"

            fraud_detected = (final_label >= 1)

            logger.info(f"[INFO] [FraudDetectionService] [username={username}] ===== FINAL PREDICTION =====")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] HMM State: {hmm_state} (prob: {hmm_prob:.2f})")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] CRF Label: {crf_label} (prob: {crf_prob:.2f})")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Ensemble Score: {ensemble_fraud_score:.2f}")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Final Label: {final_label} ({self.decode_label(final_label)})")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Confidence: {confidence}")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Fraud Detected: {fraud_detected}")

            return fraud_detected, ensemble_fraud_score, final_label, confidence

        except Exception as e:
            logger.error(f"[ERROR] [FraudDetectionService] [username={username}] Model prediction failed: {e}")
            import traceback
            logger.error(f"[ERROR] [FraudDetectionService] [username={username}] Traceback: {traceback.format_exc()}")
            return False, 0.5, 1, "Medium"

    
    def decode_label(self, label):
        """Decode numeric label to text"""
        fraud_types = {0: "Normal", 1: "Suspicious", 2: "Fraud"}
        return fraud_types.get(int(label), "Unknown")
