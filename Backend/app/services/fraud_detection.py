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
    
    
    
    
    # Notebook's exact feature column names (must match CRF training attributes)
    FEATURE_NAMES = [
        'Transaction Amount (INR)', 'Transaction_Amount_Diff',
        'Transaction_Frequency_Score', 'Time_Anomaly_Score',
        'Recipient_Total_Transactions', 'Recipient_Avg_Transaction_Amount',
        'Risk_Score', 'hour', 'day_of_week', 'Location_Cluster'
    ]

    def _convert_to_crf_format(self, feature_array, feature_names=None):
        """Convert feature array to CRF dict format using notebook's exact attribute names
        
        Input: np.ndarray of shape (1, 10)
        Output: [[{'Transaction Amount (INR)': 0.5, ...}]]
        """
        try:
            if feature_names is None:
                feature_names = self.FEATURE_NAMES
            
            crf_input = []
            for features in feature_array:
                feature_dict = {feature_names[i]: float(features[i]) for i in range(len(features))}
                crf_input.append([feature_dict])
            
            return crf_input
        except Exception as e:
            logger.error(f"[ERROR] [FraudDetectionService] CRF format conversion failed: {e}")
            return None
    
    def _get_user_transaction_history(self, username, transactions_collection, limit=3):
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
            
            # HMM prediction: only use if we have enough history (n_lags rows)
            # Otherwise, HMM with zero-padding is unreliable
            n_lags = self.n_lags
            hmm_state = None
            
            if len(scaled_matrix) >= n_lags:
                try:
                    # Create one lagged observation from the last (n_lags+1) rows
                    # Matches notebook: np.hstack([X[i-j] for j in range(n_lags)])
                    lagged_row = np.hstack([scaled_matrix[-1 - j] for j in range(n_lags)])
                    X_hmm = lagged_row.reshape(1, -1)  # shape (1, 30)
                    logger.info(f"[INFO] [FraudDetectionService] [username={username}] HMM input shape: {X_hmm.shape}")
                    hmm_state = int(self.model_manager.hmm_model.predict(X_hmm)[0])
                    logger.info(f"[INFO] [FraudDetectionService] [username={username}] HMM State: {hmm_state}")
                except Exception as e:
                    logger.error(f"[ERROR] [FraudDetectionService] [username={username}] HMM prediction failed: {e}")
                    hmm_state = None
            else:
                logger.info(f"[INFO] [FraudDetectionService] [username={username}] Insufficient history ({len(scaled_matrix)} rows < {n_lags + 1}), skipping HMM")

            # CRF prediction on current (scaled) features using notebook's named attributes
            scaled_current = scaled_matrix[-1].reshape(1, -1)  # last row = current tx
            crf_label = None
            if self.model_manager.crf_model:
                try:
                    crf_input = self._convert_to_crf_format(scaled_current)
                    if crf_input is not None:
                        crf_label = int(self.model_manager.crf_model.predict(crf_input)[0][0])
                        logger.info(f"[INFO] [FraudDetectionService] [username={username}] CRF Label: {crf_label}")
                except Exception as e:
                    logger.error(f"[ERROR] [FraudDetectionService] [username={username}] CRF prediction failed: {e}")

            # Ensemble: average HMM and CRF probabilities
            # Both models contribute equally to fraud detection
            hmm_prob = float(hmm_state) / 2.0 if hmm_state is not None else 0.5
            crf_prob = float(crf_label) / 2.0 if crf_label is not None else 0.5
            fraud_score = (hmm_prob + crf_prob) / 2.0
            
            # Map score to label: 0=Normal, 1=Suspicious, 2=Fraud
            if fraud_score >= 0.67:
                final_label = 2
                confidence = "High"
            elif fraud_score >= 0.33:
                final_label = 1
                confidence = "Medium"
            else:
                final_label = 0
                confidence = "Low"

            fraud_detected = (final_label >= 1)

            logger.info(f"[INFO] [FraudDetectionService] [username={username}] ===== FINAL PREDICTION =====")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] HMM State: {hmm_state} (prob: {hmm_prob:.2f})")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] CRF Label: {crf_label} (prob: {crf_prob:.2f})")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Ensemble Score: {fraud_score:.2f}")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Final Label: {final_label} ({self.decode_label(final_label)})")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Confidence: {confidence}")
            logger.info(f"[INFO] [FraudDetectionService] [username={username}] Fraud Detected: {fraud_detected}")

            return fraud_detected, fraud_score, final_label, confidence

        except Exception as e:
            logger.error(f"[ERROR] [FraudDetectionService] [username={username}] Model prediction failed: {e}")
            import traceback
            logger.error(f"[ERROR] [FraudDetectionService] [username={username}] Traceback: {traceback.format_exc()}")
            return False, 0.5, 1, "Medium"

    
    def decode_label(self, label):
        """Decode numeric label to text"""
        fraud_types = {0: "Normal", 1: "Suspicious", 2: "Fraud"}
        return fraud_types.get(int(label), "Unknown")
