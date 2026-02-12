from flask import Flask, request, jsonify
from model import AutoRegressiveHMM  # Import your HMM model class
import joblib
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
from functools import wraps
import json
from typing import Dict, Any, List, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration"""
    MODEL_PATH = os.getenv('MODEL_PATH', 'hmm_fraud_model.pkl')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    
    # Feature columns expected by the model
    FEATURE_COLUMNS = [
        "Transaction Amount (INR)", "Transaction_Amount_Diff", "Transaction_Frequency_Score",
        "Time_Anomaly_Score", "Recipient_Total_Transactions", "Recipient_Avg_Transaction_Amount",
        "Fraud_Type", "Risk_Score", "hour", "day_of_week"
    ] + [f"Location_Hash_{i}" for i in range(10)]

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        print("Received data:", data)

        # Convert input to DataFrame
        feature_columns = [
            "Transaction Amount (INR)", "Transaction_Amount_Diff", "Transaction_Frequency_Score",
            "Time_Anomaly_Score", "Recipient_Total_Transactions", "Recipient_Avg_Transaction_Amount",
            "Fraud_Type", "Risk_Score", "hour", "day_of_week"
        ] + [f"Location_Hash_{i}" for i in range(10)]
        
        df = pd.DataFrame([data], columns=feature_columns)

        # Convert to NumPy array
        input_data = df.values

        # Make fraud prediction
        anomaly_prediction = model.detect_anomalies(input_data)
        result = bool(anomaly_prediction[0])  # Convert NumPy bool to Python bool

        return jsonify({"fraud_detected": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
