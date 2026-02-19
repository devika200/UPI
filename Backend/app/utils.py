"""Utility functions"""
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def convert_types(obj):
    """Convert numpy/datetime types to JSON-serializable types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _extract_features_from_transaction(transaction, transactions_collection=None, dataset=None):
    """Extract 10 features from a historical transaction using identical logic to calculate_features()
    
    Returns:
        List of 10 floats with actual values (not zeros)
    """
    try:
        amount = float(transaction.get("Transaction Amount (INR)", 0))
        transaction_time = pd.to_datetime(transaction.get("Transaction Time", datetime.now()))
        # If timezone-aware, convert to naive (remove timezone info)
        if transaction_time.tzinfo is not None:
            transaction_time = transaction_time.replace(tzinfo=None)
        username = transaction.get("username", "")
        recipient_id = transaction.get("Receiver UPI ID", "")
        
        # Get user and recipient data for feature calculation
        user_df = pd.DataFrame()
        recipient_df = pd.DataFrame()
        
        if transactions_collection is not None:
            try:
                user_data = list(transactions_collection.find({"username": username}))
                recipient_data = list(transactions_collection.find({"Receiver UPI ID": recipient_id}))
                
                if user_data:
                    user_df = pd.DataFrame(user_data)
                    user_df["Transaction Time"] = pd.to_datetime(user_df["Transaction Time"], errors='coerce')
                
                if recipient_data:
                    recipient_df = pd.DataFrame(recipient_data)
                    recipient_df["Transaction Time"] = pd.to_datetime(recipient_df["Transaction Time"], errors='coerce')
            except Exception as db_error:
                logger.error(f"[ERROR] [utils] [username={username}] Database query failed: {db_error}")
        
        # Fallback to dataset if available
        if user_df.empty and dataset is not None and not dataset.empty:
            try:
                user_df = dataset[dataset.get('Transaction ID', pd.Series()) == username]
            except Exception as e:
                logger.error(f"[ERROR] [utils] [username={username}] Dataset query failed: {e}")
        
        if recipient_df.empty and dataset is not None and not dataset.empty:
            try:
                recipient_df = dataset[dataset.get('Recipient ID', pd.Series()) == recipient_id]
            except Exception as e:
                logger.error(f"[ERROR] [utils] [username={username}] Dataset query failed: {e}")
        
        # Calculate features
        if not user_df.empty:
            # Ensure transaction_time is timezone-naive for comparison
            if hasattr(transaction_time, 'tz_localize') and transaction_time.tzinfo is not None:
                transaction_time = transaction_time.tz_localize(None)
            
            # Convert all Transaction Time columns to naive datetimes
            if "Transaction Time" in user_df.columns:
                user_df["Transaction Time"] = pd.to_datetime(user_df["Transaction Time"], utc=True).dt.tz_localize(None)
            
            last_amount = user_df["Transaction Amount (INR)"].iloc[-1] if "Transaction Amount (INR)" in user_df.columns else amount
            amount_diff = abs(amount - last_amount)
            
            recent_tx = user_df[user_df["Transaction Time"] > transaction_time - pd.Timedelta(days=30)]
            frequency_score = len(recent_tx) / 10.0
            
            mean_time = user_df["Transaction Time"].mean()
            # Calculate time anomaly: check if current hour is unusual, scaled by amount
            try:
                typical_hours = user_df["Transaction Time"].dt.hour.mode()
                current_hour = transaction_time.hour
                
                if len(typical_hours) > 0 and current_hour in typical_hours.values:
                    # Current hour matches typical transaction hours
                    time_anomaly_score = 0.0
                else:
                    # Unusual hour: scale by amount ratio (capped at 1.0)
                    avg_amount = user_df["Transaction Amount (INR)"].mean() if "Transaction Amount (INR)" in user_df.columns else amount
                    amount_ratio = min(amount / avg_amount, 2.0) if avg_amount > 0 else 1.0
                    time_anomaly_score = 0.5 * (amount_ratio / 2.0)  # Scale to 0.0-0.5 range
            except Exception as hour_error:
                logger.warning(f"[WARN] [utils] [username={username}] Hour-based anomaly check failed: {hour_error}, using 0.0")
                time_anomaly_score = 0.0
        else:
            amount_diff = 0.0
            frequency_score = 0.1
            time_anomaly_score = 0.0
        
        if not recipient_df.empty:
            recipient_txns = len(recipient_df)
            recipient_avg = float(recipient_df["Transaction Amount (INR)"].mean()) if "Transaction Amount (INR)" in recipient_df.columns else amount
        else:
            recipient_txns = 0
            recipient_avg = amount
        
        # Calculate derived features matching training data
        risk_score = (frequency_score + time_anomaly_score) / 2.0  # Average of frequency and time anomaly
        
        return [
            float(amount),
            float(amount_diff),
            float(frequency_score),
            float(time_anomaly_score),
            float(recipient_txns),
            float(recipient_avg),
            float(risk_score),
            float(transaction_time.hour),
            float(transaction_time.weekday()),
            0.0  # location_cluster
        ]
    except Exception as e:
        logger.error(f"[ERROR] [utils] Feature extraction failed: {e}")
        return [0.0] * 10


def calculate_features(username, recipient_id, transaction_amount, transaction_time, 
                      transactions_collection, dataset=None):
    """
    Calculate 10 features matching notebook's exact pipeline:
    1. Amount, Amount_Diff, Frequency_Score, Time_Anomaly_Score
    2. Recipient_Total_Transactions, Recipient_Avg_Amount
    3. Risk_Score, hour, day_of_week, Location_Cluster
    
    NOTE: These 10 features are DERIVED from notebook's feature engineering.
    The scaler in models/ was fit on these exact 10 features.
    """
    try:
        transaction_time = pd.to_datetime(transaction_time)
        if transaction_time.tzinfo is not None:
            transaction_time = transaction_time.replace(tzinfo=None)
    except Exception as e:
        logger.error(f"[ERROR] [utils] [username={username}] Invalid transaction time: {e}")
        transaction_time = datetime.now()

    # Get user and recipient transaction history
    user_df = pd.DataFrame()
    recipient_df = pd.DataFrame()
    
    if transactions_collection is not None:
        try:
            user_data = list(transactions_collection.find({"username": username}))
            recipient_data = list(transactions_collection.find({"Receiver UPI ID": recipient_id}))
            
            if user_data:
                user_df = pd.DataFrame(user_data)
                user_df["Transaction Time"] = pd.to_datetime(user_df["Transaction Time"], errors='coerce')
            
            if recipient_data:
                recipient_df = pd.DataFrame(recipient_data)
                recipient_df["Transaction Time"] = pd.to_datetime(recipient_df["Transaction Time"], errors='coerce')
        except Exception as db_error:
            logger.error(f"[ERROR] [utils] [username={username}] Database query failed: {db_error}")

    # Fallback to dataset
    if user_df.empty and dataset is not None and not dataset.empty:
        try:
            user_df = dataset[dataset.get('Transaction ID', pd.Series()) == username]
        except Exception as e:
            logger.error(f"[ERROR] [utils] [username={username}] Dataset query failed: {e}")
    
    if recipient_df.empty and dataset is not None and not dataset.empty:
        try:
            recipient_df = dataset[dataset.get('Recipient ID', pd.Series()) == recipient_id]
        except Exception as e:
            logger.error(f"[ERROR] [utils] [username={username}] Dataset query failed: {e}")

    # Default features if no history
    if user_df.empty or recipient_df.empty:
        return [
            float(transaction_amount), 0.0, 0.1, 0.0,
            0.0, 0.0, 0.1,
            float(transaction_time.hour), float(transaction_time.weekday()), 0.0
        ]

    try:
        # Ensure timezone-naive
        if hasattr(transaction_time, 'tz_localize') and transaction_time.tzinfo is not None:
            transaction_time = transaction_time.tz_localize(None)
        if "Transaction Time" in user_df.columns:
            user_df["Transaction Time"] = pd.to_datetime(user_df["Transaction Time"], utc=True).dt.tz_localize(None)

        # Feature 0: Transaction Amount (INR) — current amount
        amount = float(transaction_amount)

        # Feature 1: Transaction_Amount_Diff — difference from last transaction
        last_amount = user_df["Transaction Amount (INR)"].iloc[-1] if "Transaction Amount (INR)" in user_df.columns else transaction_amount
        amount_diff = abs(transaction_amount - last_amount)

        # Feature 2: Transaction_Frequency_Score — recent transactions in 30 days / 10
        recent_tx = user_df[user_df["Transaction Time"] > transaction_time - pd.Timedelta(days=30)]
        frequency_score = len(recent_tx) / 10.0

        # Feature 3: Time_Anomaly_Score — unusual hour flag scaled by amount ratio
        try:
            typical_hours = user_df["Transaction Time"].dt.hour.mode()
            current_hour = transaction_time.hour
            if len(typical_hours) > 0 and current_hour in typical_hours.values:
                time_anomaly_score = 0.0
            else:
                # Unusual hour: scale by amount ratio (capped at 1.0)
                avg_amount = user_df["Transaction Amount (INR)"].mean() if "Transaction Amount (INR)" in user_df.columns else transaction_amount
                amount_ratio = min(transaction_amount / avg_amount, 2.0) if avg_amount > 0 else 1.0
                time_anomaly_score = 0.5 * (amount_ratio / 2.0)  # Scale to 0.0-0.5 range
        except Exception:
            time_anomaly_score = 0.0

        # Feature 4: Recipient_Total_Transactions — count of transactions to this recipient
        recipient_txns = float(len(recipient_df))

        # Feature 5: Recipient_Avg_Transaction_Amount — average amount to this recipient
        recipient_avg = float(recipient_df["Transaction Amount (INR)"].mean()) if "Transaction Amount (INR)" in recipient_df.columns else transaction_amount

        # Feature 6: Risk_Score — (frequency_score + time_anomaly_score) / 2
        risk_score = (frequency_score + time_anomaly_score) / 2.0

        # Feature 7: hour — hour of day (0-23)
        hour = float(transaction_time.hour)

        # Feature 8: day_of_week — day of week (0-6)
        day_of_week = float(transaction_time.weekday())

        # Feature 9: Location_Cluster — placeholder (0.0)
        location_cluster = 0.0

        features = [amount, amount_diff, frequency_score, time_anomaly_score, 
                   recipient_txns, recipient_avg, risk_score, hour, day_of_week, location_cluster]
        
        logger.info(f"[INFO] [utils] [username={username}] Features: "
                   f"amount={amount:.2f}, diff={amount_diff:.2f}, freq={frequency_score:.2f}, "
                   f"time_anom={time_anomaly_score:.2f}, recip_txns={recipient_txns:.0f}, "
                   f"recip_avg={recipient_avg:.2f}, risk={risk_score:.2f}, hour={hour:.0f}, dow={day_of_week:.0f}")
        
        return features
    except Exception as e:
        logger.error(f"[ERROR] [utils] [username={username}] Feature calculation failed: {e}")
        return [float(transaction_amount), 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 
                float(transaction_time.hour), float(transaction_time.weekday()), 0.0]


def get_user_statistics(username, transactions_collection):
    """Get user transaction statistics"""
    user_stats = {
        'count': 1,
        'avg_amount': 0,
        'max_amount': 0,
        'frequency': 1
    }
    
    if transactions_collection is None:
        return user_stats
    
    try:
        user_txs = list(transactions_collection.find(
            {"username": username},
            {"Transaction Amount (INR)": 1, "Transaction Time": 1}
        ))
        
        if user_txs:
            amounts = [tx.get("Transaction Amount (INR)", 0) for tx in user_txs]
            user_stats['count'] = len(amounts)
            user_stats['avg_amount'] = np.mean(amounts)
            user_stats['max_amount'] = np.max(amounts)
            
            # Calculate frequency (transactions per month)
            if len(user_txs) > 1:
                times = [pd.to_datetime(tx.get("Transaction Time", datetime.now())) for tx in user_txs]
                time_span_days = (max(times) - min(times)).days
                if time_span_days > 0:
                    user_stats['frequency'] = (len(user_txs) / time_span_days) * 30
    except Exception as e:
        logger.warning(f"Could not get user stats: {e}")
    
    return user_stats
