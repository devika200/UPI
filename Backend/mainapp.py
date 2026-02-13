from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from pymongo import MongoClient
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import sys

# Import model class if available
try:
    from model import AutoRegressiveHMM
except ImportError:
    AutoRegressiveHMM = None
    logging.warning("model.py not found, will use basic prediction")

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

app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)

# JWT secret
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "1234567890")  # Use env variable in production
jwt = JWTManager(app)

# MongoDB setup (optional - will work without it)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
    client.server_info()  # Test connection
    db = client["upi_fraud_detection"]
    users_collection = db["users"]
    transactions_collection = db["transactions"]
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.warning(f"MongoDB connection failed: {e}. Running without database.")
    client = None
    db = None
    users_collection = None
    transactions_collection = None

# Load models
hmm_model = None
model_paths = [
    "hmm_fraud_model.pkl",
    "../project/Backend/models/arlg_hmm_model (2).pkl",
    "models/arlg_hmm_model (2).pkl"
]

for model_path in model_paths:
    if os.path.exists(model_path):
        try:
            hmm_model = joblib.load(model_path)
            logger.info(f"HMM model loaded successfully from {model_path}")
            break
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")

if hmm_model is None:
    logger.warning("No HMM model loaded. Predictions will use rule-based approach.")

# Fallback dataset for feature calculation
dataset = pd.DataFrame()
dataset_paths = [
    "balanced_dataset.csv",
    "../project/Backend/balanced_dataset.csv"
]

for dataset_path in dataset_paths:
    if os.path.exists(dataset_path):
        try:
            dataset = pd.read_csv(dataset_path, parse_dates=["Timestamp"])
            logger.info(f"Fallback dataset loaded successfully from {dataset_path}")
            break
        except Exception as e:
            logger.warning(f"Failed to load dataset from {dataset_path}: {e}")

if dataset.empty:
    logger.warning("No fallback dataset loaded. Will use MongoDB history only.")

feature_columns = [
    'Transaction Amount (INR)', 'Transaction_Amount_Diff',
    'Transaction_Frequency_Score', 'Time_Anomaly_Score',
    'Recipient_Total_Transactions', 'Recipient_Avg_Transaction_Amount',
    'Fraud_Type', 'Risk_Score', 'hour', 'day_of_week'
] + [f'Location_Hash_{i}' for i in range(10)]

# Type conversion for JSON/MongoDB safety
def convert_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def calculate_features(username, recipient_id, transaction_amount, transaction_time):
    """Calculate transaction features from historical data"""
    try:
        transaction_time = pd.to_datetime(transaction_time)
    except Exception:
        transaction_time = datetime.now()

    # Try to get data from MongoDB first
    user_df = pd.DataFrame()
    recipient_df = pd.DataFrame()
    
    if transactions_collection is not None:
        user_data = list(transactions_collection.find({"username": username}))
        recipient_data = list(transactions_collection.find({"Receiver UPI ID": recipient_id}))
        
        if user_data:
            user_df = pd.DataFrame(user_data)
            user_df["Transaction Time"] = pd.to_datetime(user_df["Transaction Time"], errors='coerce')
        
        if recipient_data:
            recipient_df = pd.DataFrame(recipient_data)
            recipient_df["Transaction Time"] = pd.to_datetime(recipient_df["Transaction Time"], errors='coerce')

    # Fallback to CSV dataset if MongoDB has no data
    if user_df.empty and not dataset.empty:
        user_df = dataset[dataset.get('Transaction ID', pd.Series()) == username]
    
    if recipient_df.empty and not dataset.empty:
        recipient_df = dataset[dataset.get('Recipient ID', pd.Series()) == recipient_id]

    # If still no data, use defaults
    if user_df.empty or recipient_df.empty:
        logger.info(f"No historical data found for user: {username}. Using default values.")
        return [
            transaction_amount, 0, 1, 0,
            0, 0, 0, 0.5,
            transaction_time.hour, transaction_time.weekday()
        ] + [0] * 10  # 10 location hash values

    # Calculate features from historical data
    last_amount = user_df["Transaction Amount (INR)"].iloc[-1] if "Transaction Amount (INR)" in user_df.columns else transaction_amount
    transaction_amount_diff = abs(transaction_amount - last_amount)

    recent_tx = user_df[user_df["Transaction Time"] > transaction_time - pd.Timedelta(days=30)]
    transaction_frequency_score = len(recent_tx) / 10

    mean_time = user_df["Transaction Time"].mean()
    time_anomaly_score = abs((transaction_time - mean_time).total_seconds()) / 86400

    recipient_total_transactions = len(recipient_df)
    recipient_avg_transaction_amount = recipient_df["Transaction Amount (INR)"].mean() if "Transaction Amount (INR)" in recipient_df.columns else transaction_amount
    
    fraud_type = 0  # Default: no fraud
    risk_score = (transaction_frequency_score + time_anomaly_score) / 2

    # Location hash (simplified - using random for now)
    location_hashes = [np.random.randint(0, 2) for _ in range(10)]

    return [
        transaction_amount, transaction_amount_diff, transaction_frequency_score,
        time_anomaly_score, recipient_total_transactions, recipient_avg_transaction_amount,
        fraud_type, risk_score, transaction_time.hour, transaction_time.weekday()
    ] + location_hashes

@app.route('/api/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        username = data.get("username")
        email = data.get("email")
        password = data.get("password")

        if not username or not password or not email:
            return jsonify({"error": "Username, email and password are required"}), 400

        if users_collection is None:
            return jsonify({"error": "Database not available"}), 503

        if users_collection.find_one({"email": email}):
            return jsonify({"error": "User already exists"}), 400

        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed,
            "created_at": datetime.now()
        })
        
        logger.info(f"User registered: {email}")
        return jsonify({"message": "User registered successfully"}), 201
    
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        if users_collection is None:
            return jsonify({"error": "Database not available"}), 503

        user = users_collection.find_one({"email": email})
        if not user or not bcrypt.check_password_hash(user["password"], password):
            return jsonify({"error": "Invalid email or password"}), 401

        token = create_access_token(identity=email)
        logger.info(f"User logged in: {email}")
        return jsonify({
            "access_token": token,
            "message": "Login successful",
            "username": user.get("username", email)
        }), 200
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/check_fraud', methods=['POST'])
@jwt_required()
def check_fraud():
    """
    Main fraud detection endpoint - uses logged-in user's history
    Requires JWT authentication
    """
    try:
        if hmm_model is None:
            return jsonify({"error": "Model not loaded"}), 503

        # Get current user's email from JWT token
        current_user_email = get_jwt_identity()
        logger.info(f"Fraud check for user: {current_user_email}")

        data = request.get_json()

        # Extract frontend data
        transaction_amount = float(data.get("amount", 0))
        sender_upi = data.get("sender", current_user_email)  # Use logged-in user as sender
        receiver_upi = data.get("receiver", "")
        timestamp_str = data.get("timestamp", "")
        description = data.get("description", "")
        category = data.get("category", "")

        if not transaction_amount or not receiver_upi:
            return jsonify({"error": "Amount and receiver are required"}), 400

        # Parse timestamp
        try:
            transaction_time = pd.to_datetime(timestamp_str)
        except:
            transaction_time = datetime.now()

        # Calculate features using THIS USER's history
        feature_vector = calculate_features(
            current_user_email, receiver_upi, transaction_amount, transaction_time
        )

        # Prepare data for model
        df = pd.DataFrame([feature_vector], columns=feature_columns)
        input_data = df.values

        # Make prediction
        if hmm_model is not None:
            try:
                anomaly_prediction = hmm_model.detect_anomalies(input_data)
                fraud_detected = bool(anomaly_prediction[0])
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}. Using rule-based approach.")
                fraud_detected = False
        else:
            # Rule-based fraud detection if no model
            fraud_detected = False
            
            # Check for suspicious patterns
            if transaction_amount > 50000:  # Very high amount
                fraud_detected = True
            elif feature_vector[1] > transaction_amount * 0.8:  # Huge difference from last transaction
                fraud_detected = True
            elif feature_vector[2] > 10:  # Extremely high frequency
                fraud_detected = True
            elif feature_vector[3] > 0.8:  # Very unusual time
                fraud_detected = True

        # Calculate fraud score (0.0 to 1.0)
        if fraud_detected:
            fraud_score = min(0.9, 0.5 + (feature_vector[7] * 0.4))  # Based on risk score
        else:
            fraud_score = max(0.1, feature_vector[7] * 0.3)  # Low risk
        
        # Determine risk factors
        risk_factors = []
        if transaction_amount > 10000:
            risk_factors.append("High transaction amount")
        if feature_vector[2] > 5:  # High frequency
            risk_factors.append("Unusual transaction frequency")
        if feature_vector[3] > 0.5:  # Time anomaly
            risk_factors.append("Unusual transaction time")
        if fraud_detected:
            risk_factors.append("Pattern matches known fraud signatures")

        # Generate recommendation
        if fraud_detected:
            recommendation = "This transaction shows suspicious patterns. We recommend additional verification before proceeding."
        else:
            recommendation = "Transaction appears normal based on historical patterns."

        # Save to database with user's email
        if transactions_collection is not None:
            try:
                transactions_collection.insert_one({
                    "username": current_user_email,  # Store user's email
                    "Sender UPI ID": sender_upi,
                    "Receiver UPI ID": receiver_upi,
                    "Transaction Amount (INR)": transaction_amount,
                    "Transaction Time": convert_types(transaction_time),
                    "Category": category,
                    "Description": description,
                    "Fraud Detected": fraud_detected,
                    "Fraud Score": fraud_score,
                    "Risk Factors": risk_factors,
                    "timestamp": datetime.now()
                })
                logger.info(f"Transaction saved for user: {current_user_email}")
            except Exception as db_error:
                logger.warning(f"Database save error: {db_error}")

        return jsonify({
            "is_fraud": fraud_detected,
            "fraud_score": fraud_score,
            "risk_factors": risk_factors if risk_factors else ["No significant risk factors detected"],
            "recommendation": recommendation
        })

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/history', methods=['GET'])
@jwt_required()
def history():
    """Get transaction history for logged-in user only"""
    try:
        if transactions_collection is None:
            return jsonify([]), 200  # Return empty array if no database
        
        # Get current user's email from JWT token
        current_user_email = get_jwt_identity()
        logger.info(f"Fetching history for user: {current_user_email}")
        
        # Get only THIS user's transactions
        transactions = list(transactions_collection.find(
            {"username": current_user_email},  # Filter by username (which is email)
            {"_id": 0}
        ).sort("timestamp", -1).limit(100))
        
        # Convert to frontend format
        formatted_transactions = []
        for tx in transactions:
            formatted_transactions.append({
                "id": str(tx.get("timestamp", "")),
                "amount": tx.get("Transaction Amount (INR)", 0),
                "sender": tx.get("Sender UPI ID", ""),
                "receiver": tx.get("Receiver UPI ID", ""),
                "timestamp": tx.get("Transaction Time", ""),
                "status": "Fraudulent" if tx.get("Fraud Detected", False) else "Legitimate",
                "risk_score": tx.get("Fraud Score", 0),
                "category": tx.get("Category", ""),
                "description": tx.get("Description", "")
            })
        
        logger.info(f"Found {len(formatted_transactions)} transactions for {current_user_email}")
        return jsonify(formatted_transactions)
    
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify([]), 200  # Return empty array on error

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": hmm_model is not None,
        "database_connected": users_collection is not None,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting UPI Fraud Detection API...")
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)


@app.route('/api/admin/stats', methods=['GET'])
def admin_stats():
    """Admin endpoint to view database statistics"""
    try:
        if users_collection is None or transactions_collection is None:
            return jsonify({"error": "Database not available"}), 503
        
        # User stats
        user_count = users_collection.count_documents({})
        
        # Transaction stats
        tx_count = transactions_collection.count_documents({})
        fraud_count = transactions_collection.count_documents({"Fraud Detected": True})
        legitimate_count = tx_count - fraud_count
        fraud_rate = (fraud_count / tx_count * 100) if tx_count > 0 else 0
        
        # Recent transactions
        recent_txs = list(transactions_collection.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(10))
        
        # Recent users
        recent_users = list(users_collection.find(
            {}, {"_id": 0, "password": 0}
        ).sort("created_at", -1).limit(10))
        
        return jsonify({
            "users": {
                "total": user_count,
                "recent": recent_users
            },
            "transactions": {
                "total": tx_count,
                "fraudulent": fraud_count,
                "legitimate": legitimate_count,
                "fraud_rate": round(fraud_rate, 2),
                "recent": recent_txs
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/users', methods=['GET'])
def admin_users():
    """Get all users (without passwords)"""
    try:
        if users_collection is None:
            return jsonify({"error": "Database not available"}), 503
        
        users = list(users_collection.find({}, {"_id": 0, "password": 0}))
        return jsonify({"users": users, "count": len(users)})
    
    except Exception as e:
        logger.error(f"Admin users error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/transactions', methods=['GET'])
def admin_transactions():
    """Get all transactions with optional filters"""
    try:
        if transactions_collection is None:
            return jsonify({"error": "Database not available"}), 503
        
        # Get query parameters
        limit = int(request.args.get('limit', 100))
        fraud_only = request.args.get('fraud_only', 'false').lower() == 'true'
        
        # Build query
        query = {}
        if fraud_only:
            query["Fraud Detected"] = True
        
        # Get transactions
        txs = list(transactions_collection.find(
            query, {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        
        return jsonify({
            "transactions": txs,
            "count": len(txs),
            "total": transactions_collection.count_documents(query)
        })
    
    except Exception as e:
        logger.error(f"Admin transactions error: {e}")
        return jsonify({"error": str(e)}), 500
