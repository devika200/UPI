"""Risk analysis service"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """Service for analyzing transaction risk factors"""
    
    def __init__(self, db):
        self.db = db
    
    def analyze_risk_factors(self, username, transaction_amount, feature_vector):
        """Analyze and generate risk factors for a transaction"""
        risk_factors = []
        
        if self.db.transactions_collection is None:
            return risk_factors
        
        try:
            user_txs = list(self.db.transactions_collection.find(
                {"username": username},
                {"Transaction Amount (INR)": 1, "Transaction Time": 1, "timestamp": 1}
            ))
            
            if not user_txs:
                return risk_factors
            
            amounts = [tx.get("Transaction Amount (INR)", 0) for tx in user_txs]
            user_avg_amount = np.mean(amounts)
            user_max_amount = np.max(amounts)
            user_std_amount = np.std(amounts) if len(amounts) > 1 else 0
            
            # Get last transaction amount
            user_txs_sorted = sorted(user_txs, key=lambda t: self._get_tx_time(t), reverse=True)
            last_amount = user_txs_sorted[0].get("Transaction Amount (INR)", 0) if user_txs_sorted else transaction_amount
            amount_diff_vs_last = abs(float(transaction_amount) - float(last_amount))
            
            # Check if amount is unusual for this user
            if user_std_amount > 0:
                z_score = (transaction_amount - user_avg_amount) / user_std_amount
                if z_score > 3:
                    risk_factors.append(f"Amount is {z_score:.1f}x higher than your typical pattern (avg: Rs.{user_avg_amount:,.0f})")
                elif z_score > 2:
                    risk_factors.append(f"Amount is significantly higher than your usual (avg: Rs.{user_avg_amount:,.0f})")
            
            # Check if it's the highest transaction ever
            if transaction_amount > user_max_amount:
                risk_factors.append(f"This is your highest transaction ever (previous max: Rs.{user_max_amount:,.0f})")
            
            # Large change vs last transaction
            if amount_diff_vs_last > user_avg_amount:
                if transaction_amount > last_amount:
                    risk_factors.append(f"Large jump from your last transaction (Rs.{amount_diff_vs_last:,.0f} increase)")
                else:
                    risk_factors.append(f"Large drop from your last transaction (Rs.{amount_diff_vs_last:,.0f} decrease)")
            
        except Exception as e:
            logger.warning(f"Could not calculate user statistics: {e}")
        
        # Frequency analysis
        if feature_vector[2] > 10:
            risk_factors.append(f"Unusually high transaction frequency for you ({feature_vector[2]:.1f} transactions/month)")
        elif feature_vector[2] > 5:
            risk_factors.append(f"Higher than normal transaction frequency ({feature_vector[2]:.1f} transactions/month)")
        
        # Time anomaly
        if feature_vector[3] > 0.8:
            risk_factors.append("Transaction at highly unusual time compared to your pattern")
        elif feature_vector[3] > 0.5:
            risk_factors.append("Transaction at unusual time for you")
        
        # New recipient check
        if feature_vector[4] == 0:
            risk_factors.append("First transaction to this recipient")
        
        return risk_factors
    
    @staticmethod
    def _get_tx_time(tx):
        """Safely get transaction time"""
        ts = tx.get("timestamp") or tx.get("Transaction Time")
        if ts is None:
            return datetime.min
        try:
            return pd.to_datetime(ts, errors="coerce") or datetime.min
        except Exception:
            return datetime.min
    
    @staticmethod
    def generate_recommendation(fraud_score):
        """Generate recommendation based on fraud score"""
        if fraud_score >= 0.8:
            return "HIGH RISK: This transaction shows multiple suspicious patterns. We strongly recommend canceling this transaction and verifying with the recipient through alternate means."
        elif fraud_score >= 0.6:
            return "SUSPICIOUS: This transaction shows concerning patterns. Please verify the recipient details and transaction purpose before proceeding."
        elif fraud_score >= 0.4:
            return "MODERATE RISK: Some unusual patterns detected. Double-check the transaction details before confirming."
        else:
            return "Transaction appears normal based on your historical patterns. Safe to proceed."
