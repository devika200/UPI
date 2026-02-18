#!/usr/bin/env python3
"""
Improved CRF Model Retraining Script

This script retrains the CRF model with better synthetic data that includes
more nuanced "Suspicious" transactions to improve classification granularity.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import sklearn_crfsuite
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from utils import calculate_features


def generate_improved_training_data(n_samples=2000):
    """
    Generate improved synthetic training data with better class distribution.
    
    This creates more nuanced transactions:
    - Normal: Low amount, business hours
    - Suspicious: Medium amount OR unusual hour (but not both)
    - Fraud: High amount AND unusual hour
    
    Args:
        n_samples: Number of training samples to generate
        
    Returns:
        DataFrame with features and fraud labels
    """
    print(f"[INFO] Generating {n_samples} improved synthetic training samples...")
    
    data = []
    
    # Generate synthetic transactions with better distribution
    for i in range(n_samples):
        username = f"user_{i % 100}@upi"
        recipient_id = f"recipient_{i % 50}@upi"
        
        # Create more nuanced fraud labels
        rand = np.random.random()
        
        if rand < 0.70:  # 70% normal transactions
            transaction_amount = np.random.uniform(50, 500)
            hour = np.random.choice(list(range(9, 18)))  # Business hours
            fraud_label = 0  # Normal
            
        elif rand < 0.85:  # 15% suspicious transactions
            # Suspicious: either medium amount OR unusual hour (but not both)
            if np.random.random() < 0.5:
                # Medium amount, business hours
                transaction_amount = np.random.uniform(1000, 3000)
                hour = np.random.choice(list(range(9, 18)))
            else:
                # Normal amount, unusual hour
                transaction_amount = np.random.uniform(100, 500)
                hour = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23])
            fraud_label = 1  # Suspicious
            
        else:  # 15% fraudulent transactions
            # Fraud: high amount AND unusual hour
            transaction_amount = np.random.uniform(5000, 50000)
            hour = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23])
            fraud_label = 2  # Fraud
        
        # Create transaction time
        base_time = datetime.now() - timedelta(days=np.random.randint(0, 365))
        transaction_time = base_time.replace(hour=hour, minute=0, second=0)
        
        # Create mock transaction dict
        transaction = {
            'username': username,
            'Receiver UPI ID': recipient_id,
            'Transaction Amount (INR)': transaction_amount,
            'Transaction Time': transaction_time,
            'Transaction ID': f"txn_{i}"
        }
        
        # Extract features using the correct formula
        features = calculate_features(
            username=username,
            recipient_id=recipient_id,
            transaction_amount=transaction_amount,
            transaction_time=transaction_time,
            transactions_collection=None,  # No MongoDB
            dataset=None
        )
        
        # Add fraud label
        data.append(features + [fraud_label])
    
    # Create DataFrame
    feature_names = [
        'amount', 'amount_diff', 'frequency', 'time_anomaly',
        'recipient_txns', 'recipient_avg', 'risk_score',
        'hour', 'weekday', 'location_cluster', 'fraud_label'
    ]
    
    df = pd.DataFrame(data, columns=feature_names)
    
    print(f"[INFO] Generated {len(df)} samples")
    print(f"[INFO] Fraud distribution:")
    print(df['fraud_label'].value_counts().sort_index())
    print(f"\n[INFO] Class distribution:")
    print(f"  - Normal (0): {(df['fraud_label'] == 0).sum()} ({(df['fraud_label'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"  - Suspicious (1): {(df['fraud_label'] == 1).sum()} ({(df['fraud_label'] == 1).sum()/len(df)*100:.1f}%)")
    print(f"  - Fraud (2): {(df['fraud_label'] == 2).sum()} ({(df['fraud_label'] == 2).sum()/len(df)*100:.1f}%)")
    
    return df


def retrain_crf_model(df, output_dir='models'):
    """
    Retrain the CRF model with improved data and better parameters.
    
    Args:
        df: DataFrame with features and fraud labels
        output_dir: Directory to save the retrained model
    """
    print("\n[INFO] Preparing data for CRF training...")
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col != 'fraud_label']
    X = df[feature_columns].values
    y = df['fraud_label'].values
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Training set size: {len(X_train)}")
    print(f"[INFO] Test set size: {len(X_test)}")
    print(f"[INFO] Training set distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = {0: 'Normal', 1: 'Suspicious', 2: 'Fraud'}.get(label, 'Unknown')
        print(f"  - {label_name} ({label}): {count} ({count/len(y_train)*100:.1f}%)")
    
    # Scale features
    print("[INFO] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to CRF format (list of sequences with dict features)
    print("[INFO] Converting to CRF format...")
    X_train_seq = []
    for data_point in X_train_scaled:
        feature_dict = {f'feature_{i}': float(value) for i, value in enumerate(data_point)}
        X_train_seq.append([feature_dict])  # Wrap in list for sequence format
    
    X_test_seq = []
    for data_point in X_test_scaled:
        feature_dict = {f'feature_{i}': float(value) for i, value in enumerate(data_point)}
        X_test_seq.append([feature_dict])
    
    # Convert labels to string format (required by CRF)
    y_train_seq = [[str(int(label))] for label in y_train]
    y_test_seq = [[str(int(label))] for label in y_test]
    
    # Train CRF with optimized parameters
    print("[INFO] Training CRF model with optimized parameters...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=300,  # More iterations for better convergence
        all_possible_transitions=True,
        c1=0.05,  # Slightly lower L1 regularization for better class separation
        c2=0.05,  # Slightly lower L2 regularization
        verbose=1
    )
    
    crf.fit(X_train_seq, y_train_seq)
    
    # Evaluate on test set
    print("\n[INFO] Evaluating on test set...")
    y_pred = crf.predict(X_test_seq)
    y_pred_flat = [int(label[0]) for label in y_pred]
    
    # Calculate accuracy
    accuracy = sum(1 for pred, true in zip(y_pred_flat, y_test) if pred == true) / len(y_test)
    print(f"[INFO] Test Accuracy: {accuracy:.4f}")
    
    # Print confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred_flat)
    print("\n[INFO] Confusion Matrix:")
    print(cm)
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred_flat, target_names=['Normal', 'Suspicious', 'Fraud']))
    
    # Save models
    print(f"\n[INFO] Saving models to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(crf, os.path.join(output_dir, 'crf_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    print("[INFO] ✅ CRF model retrained and saved successfully!")
    print(f"[INFO] Model saved to: {os.path.join(output_dir, 'crf_model.pkl')}")
    print(f"[INFO] Scaler saved to: {os.path.join(output_dir, 'scaler.pkl')}")
    
    return crf, scaler


def main():
    """Main retraining workflow"""
    print("=" * 60)
    print("Improved CRF Model Retraining Script")
    print("=" * 60)
    
    # Generate improved synthetic training data
    df = generate_improved_training_data(n_samples=2000)
    
    # Retrain CRF model
    crf, scaler = retrain_crf_model(df, output_dir='models')
    
    print("\n" + "=" * 60)
    print("✅ Improved Retraining complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Restart the Flask server: python mainapp.py")
    print("2. Test with different transaction amounts and times:")
    print("   - Normal: 100 INR at 10 AM → fraud_score < 0.3")
    print("   - Suspicious: 2000 INR at 10 AM → fraud_score 0.3-0.5")
    print("   - Suspicious: 100 INR at 3 AM → fraud_score 0.3-0.5")
    print("   - Fraud: 10000 INR at 3 AM → fraud_score >= 0.5")


if __name__ == '__main__':
    main()
