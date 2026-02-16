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
import sklearn_crfsuite
from sklearn.preprocessing import StandardScaler

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
CORS(app, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "https://upi-secure.netlify.app"
])
bcrypt = Bcrypt(app)

# JWT secret
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "1234567890")  # Use env variable in production
jwt = JWTManager(app)

# MongoDB setup (optional - will work without it)
# Escape username/password in URI per RFC 3986 (fixes "must be escaped" error on Render)
def _safe_mongodb_uri(uri):
    if not uri or "@" not in uri or "://" not in uri:
        return uri
    from urllib.parse import urlparse, quote_plus, urlunparse
    parsed = urlparse(uri)
    if parsed.username is not None or parsed.password is not None:
        netloc = parsed.hostname or ""
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        user = quote_plus(parsed.username) if parsed.username is not None else ""
        passwd = quote_plus(parsed.password) if parsed.password is not None else ""
        if user or passwd:
            netloc = f"{user}:{passwd}@{netloc}" if passwd else f"{user}@{netloc}"
        parsed = parsed._replace(netloc=netloc)
        return urlunparse(parsed)
    return uri

MONGODB_URI = _safe_mongodb_uri(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
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

# ============= MODEL LOADING =============
hmm_model = None
crf_model = None
scaler = None
label_encoder = None

# Load AR-HMM Model (pickle expects __main__.AutoRegressiveHMM if saved from a script)
if AutoRegressiveHMM is not None:
    import sys
    sys.modules["__main__"].AutoRegressiveHMM = AutoRegressiveHMM

model_paths = [
    "models/arlg_hmm_model.pkl",
    "hmm_fraud_model.pkl",
    "../project/Backend/models/arlg_hmm_model.pkl"
]

for model_path in model_paths:
    if os.path.exists(model_path):
        try:
            hmm_model = joblib.load(model_path)
            logger.info(f"[OK] AR-HMM model loaded successfully from {model_path}")
            break
        except Exception as e:
            logger.warning(f"Failed to load HMM model from {model_path}: {e}")

if hmm_model is None:
    logger.error("[ERR] No HMM model loaded. Predictions will use rule-based approach.")

# Load CRF Model
crf_paths = [
    "models/crf_model.pkl",
    "../project/Backend/models/crf_model.pkl"
]

for crf_path in crf_paths:
    if os.path.exists(crf_path):
        try:
            crf_model = joblib.load(crf_path)
            logger.info(f"[OK] CRF model loaded successfully from {crf_path}")
            break
        except Exception as e:
            logger.warning(f"Failed to load CRF model from {crf_path}: {e}")

if crf_model is None:
    logger.warning("[WARN] CRF model not found, using HMM only")

# Load Scaler and Encoder (CRITICAL for accuracy)
scaler_paths = [
    "models/scaler.pkl",
    "../project/Backend/models/scaler.pkl"
]

encoder_paths = [
    "models/label_encoder.pkl",
    "../project/Backend/models/label_encoder.pkl"
]

for scaler_path in scaler_paths:
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            logger.info(f"[OK] Scaler loaded successfully from {scaler_path}")
            break
        except Exception as e:
            logger.warning(f"Failed to load scaler from {scaler_path}: {e}")

for encoder_path in encoder_paths:
    if os.path.exists(encoder_path):
        try:
            label_encoder = joblib.load(encoder_path)
            logger.info(f"[OK] Label encoder loaded successfully from {encoder_path}")
            break
        except Exception as e:
            logger.warning(f"Failed to load encoder from {encoder_path}: {e}")

if scaler is None or label_encoder is None:
    logger.warning("[WARN] Scaler/encoder not found - predictions may be less accurate")

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

        # Make prediction with HMM model
        fraud_detected = False
        model_confidence = 0.0
        
        if hmm_model is not None:
            try:
                # Get user's recent transactions for context (lag features)
                if transactions_collection is not None:
                    recent_txs = list(transactions_collection.find(
                        {"username": current_user_email}
                    ).sort("timestamp", -1).limit(5))
                    
                    # Build sequence of feature vectors
                    sequence_data = []
                    for tx in reversed(recent_txs[-3:]):  # Last 3 transactions
                        try:
                            past_features = calculate_features(
                                current_user_email,
                                tx.get("Receiver UPI ID", receiver_upi),
                                float(tx.get("Transaction Amount (INR)", 0)),
                                pd.to_datetime(tx.get("Transaction Time", datetime.now()))
                            )
                            sequence_data.append(past_features)
                        except:
                            pass
                    
                    # Add current transaction
                    sequence_data.append(feature_vector)
                    
                    # Need at least 2 transactions for the model
                    if len(sequence_data) >= 2:
                        sequence_array = np.array(sequence_data)
                        anomaly_prediction = hmm_model.detect_anomalies(sequence_array)
                        fraud_detected = bool(anomaly_prediction[-1])  # Last one is current transaction
                        model_confidence = 0.8 if fraud_detected else 0.2
                        logger.info(f"HMM Model prediction: {fraud_detected} (with {len(sequence_data)} transactions in sequence)")
                    else:
                        # Not enough history, use single transaction with padding
                        logger.info("Not enough transaction history for HMM, using single transaction")
                        # Pad with current transaction repeated
                        padded_data = np.array([feature_vector] * 3)
                        anomaly_prediction = hmm_model.detect_anomalies(padded_data)
                        fraud_detected = bool(anomaly_prediction[-1])
                        model_confidence = 0.5  # Lower confidence with padding
                else:
                    # No database, use single transaction with padding
                    padded_data = np.array([feature_vector] * 3)
                    anomaly_prediction = hmm_model.detect_anomalies(padded_data)
                    fraud_detected = bool(anomaly_prediction[-1])
                    model_confidence = 0.5
                    
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                logger.error(f"Feature vector shape: {np.array([feature_vector]).shape}")
                logger.error(f"Feature vector: {feature_vector}")
                fraud_detected = False
                model_confidence = 0.0
        
        # ============= ENSEMBLE PREDICTION (HMM + CRF) =============
        # Get HMM label (0=Normal, 1=Suspicious, 2=Fraud)
        if fraud_detected:
            hmm_label = 2  # Fraud
        else:
            hmm_label = 0  # Normal
        
        # Prepare features for scaling and CRF
        # Get user statistics for feature preparation
        user_stats = {
            'count': 1,
            'avg_amount': transaction_amount,
            'max_amount': transaction_amount,
            'frequency': 1
        }
        
        if transactions_collection is not None:
            try:
                user_txs = list(transactions_collection.find(
                    {"username": current_user_email},
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
        
        # Extract time features
        hour = transaction_time.hour
        day_of_week = transaction_time.dayofweek
        
        # Calculate time anomaly (simplified)
        time_anomaly = feature_vector[3] if len(feature_vector) > 3 else 0.0
        
        # Prepare raw features for scaling (8 features to match training)
        raw_features = np.array([[
            float(transaction_amount),
            float(user_stats['avg_amount']),
            float(user_stats['frequency']),
            float(time_anomaly),
            float(user_stats['count']),
            float(hour),
            float(day_of_week),
            0  # location_cluster placeholder
        ]])
        
        # Scale features if scaler is available
        if scaler:
            try:
                scaled_features = scaler.transform(raw_features)
                logger.info("[OK] Features scaled successfully")
            except Exception as e:
                logger.warning(f"[WARN] Scaling failed: {e}, using raw features")
                scaled_features = raw_features
        else:
            scaled_features = raw_features
            logger.warning("[WARN] No scaler available, using raw features")
        
        # CRF Prediction (if available)
        confidence = "Medium (HMM only)"
        final_label = hmm_label
        
        if crf_model:
            try:
                # Prepare features for CRF (expects dict format)
                feature_names = [
                    'amount', 'avg_amount', 'frequency', 'time_anomaly',
                    'past_transactions', 'hour', 'day_of_week', 'location_cluster'
                ]
                feature_dict = {
                    name: float(scaled_features[0][i]) 
                    for i, name in enumerate(feature_names)
                }
                
                # CRF prediction (expects list of sequences)
                crf_pred = crf_model.predict([[feature_dict]])[0][0]
                crf_label = int(crf_pred)
                
                # Ensemble Decision
                if hmm_label == crf_label:
                    final_label = hmm_label
                    confidence = "High"
                    logger.info(f"[OK] Models agree: {final_label}")
                else:
                    # When they disagree, use CRF (better at feature detection)
                    final_label = crf_label
                    confidence = "Medium"
                    logger.info(f"[WARN] Models disagree - HMM: {hmm_label}, CRF: {crf_label}, Using: {final_label}")
                    
            except Exception as e:
                logger.error(f"[ERR] CRF prediction failed: {e}")
                final_label = hmm_label
                confidence = "Low"
        else:
            logger.warning("[WARN] CRF not available, using HMM only")
        
        # Decode label to text
        if label_encoder:
            try:
                prediction_text = label_encoder.inverse_transform([final_label])[0]
                logger.info(f"[OK] Prediction decoded: {prediction_text}")
            except Exception as e:
                logger.error(f"[ERR] Label decoding failed: {e}")
                # Fallback mapping
                fraud_types = {0: "Normal", 1: "Suspicious", 2: "Fraud"}
                prediction_text = fraud_types.get(final_label, "Unknown")
        else:
            # Fallback mapping if encoder not available
            fraud_types = {0: "Normal", 1: "Suspicious", 2: "Fraud"}
            prediction_text = fraud_types.get(final_label, "Unknown")
        
        # Update fraud_detected based on ensemble result
        fraud_detected = (final_label >= 1)  # 1=Suspicious, 2=Fraud
        model_confidence = 0.9 if confidence == "High" else (0.7 if confidence == "Medium" else 0.5)
        
        logger.info(f"Ensemble prediction: {prediction_text} (label: {final_label}, confidence: {confidence})")
        
        logger.info(f"Model fraud decision: {fraud_detected} (confidence: {model_confidence})")

        # Calculate fraud score based on model confidence
        if fraud_detected:
            fraud_score = max(0.6, model_confidence)
        else:
            fraud_score = min(0.3, 1.0 - model_confidence)
        
        # Determine risk factors based on MODEL's analysis and USER's patterns
        risk_factors = []
        
        if fraud_detected:
            risk_factors.append("🤖 ML Model detected anomalous transaction pattern")
        
        # Dynamic thresholds based on user's history
        if transactions_collection is not None:
            try:
                # Get user's transaction statistics
                user_txs = list(transactions_collection.find(
                    {"username": current_user_email},
                    {"Transaction Amount (INR)": 1, "Transaction Time": 1, "timestamp": 1}
                ))
                
                if user_txs:
                    amounts = [tx.get("Transaction Amount (INR)", 0) for tx in user_txs]
                    user_avg_amount = np.mean(amounts)
                    user_max_amount = np.max(amounts)
                    user_std_amount = np.std(amounts) if len(amounts) > 1 else 0
                    
                    # Chronologically last transaction (for "last amount" message)
                    def _tx_time(t):
                        ts = t.get("timestamp") or t.get("Transaction Time")
                        if ts is None:
                            return datetime.min
                        try:
                            return pd.to_datetime(ts, errors="coerce") or datetime.min
                        except Exception:
                            return datetime.min
                    try:
                        user_txs_sorted = sorted(user_txs, key=lambda t: _tx_time(t), reverse=True)
                    except Exception:
                        user_txs_sorted = user_txs
                    last_amount = user_txs_sorted[0].get("Transaction Amount (INR)", 0) if user_txs_sorted else transaction_amount
                    amount_diff_vs_last = abs(float(transaction_amount) - float(last_amount))
                    
                    # Check if current amount is unusual FOR THIS USER
                    if user_std_amount > 0:
                        z_score = (transaction_amount - user_avg_amount) / user_std_amount
                        if z_score > 3:  # More than 3 standard deviations
                            risk_factors.append(f"📊 Amount is {z_score:.1f}x higher than your typical pattern (avg: ₹{user_avg_amount:,.0f})")
                        elif z_score > 2:
                            risk_factors.append(f"📊 Amount is significantly higher than your usual (avg: ₹{user_avg_amount:,.0f})")
                    
                    # Check if it's the highest transaction ever
                    if transaction_amount > user_max_amount:
                        risk_factors.append(f"💰 This is your highest transaction ever (previous max: ₹{user_max_amount:,.0f})")
                    
                    # Large change vs last transaction (show correct direction: increase vs decrease)
                    if amount_diff_vs_last > user_avg_amount:
                        if transaction_amount > last_amount:
                            risk_factors.append(f"⚡ Large jump from your last transaction (₹{amount_diff_vs_last:,.0f} increase)")
                        else:
                            risk_factors.append(f"⚡ Large drop from your last transaction (₹{amount_diff_vs_last:,.0f} decrease)")
            except Exception as e:
                logger.warning(f"Could not calculate user statistics: {e}")
        
        # Frequency analysis (already personalized)
        if feature_vector[2] > 10:
            risk_factors.append(f"⚡ Unusually high transaction frequency for you ({feature_vector[2]:.1f} transactions/month)")
        elif feature_vector[2] > 5:
            risk_factors.append(f"⚡ Higher than normal transaction frequency ({feature_vector[2]:.1f} transactions/month)")
        
        # Time anomaly (already personalized to user's habits)
        if feature_vector[3] > 0.8:
            risk_factors.append("🕐 Transaction at highly unusual time compared to your pattern")
        elif feature_vector[3] > 0.5:
            risk_factors.append("🕐 Transaction at unusual time for you")
        
        # New recipient check
        if feature_vector[4] == 0:
            risk_factors.append("👤 First transaction to this recipient")
        
        # If model detected fraud but no specific factors, add generic message
        if fraud_detected and len(risk_factors) == 1:  # Only has the ML detection message
            risk_factors.append("🔍 Transaction pattern deviates from your normal behavior")
        
        if not risk_factors:
            risk_factors.append("✅ Transaction matches your normal patterns")

        # Generate recommendation based on fraud score
        if fraud_score >= 0.8:
            recommendation = "⚠️ HIGH RISK: This transaction shows multiple suspicious patterns. We strongly recommend canceling this transaction and verifying with the recipient through alternate means."
        elif fraud_score >= 0.6:
            recommendation = "⚠️ SUSPICIOUS: This transaction shows concerning patterns. Please verify the recipient details and transaction purpose before proceeding."
        elif fraud_score >= 0.4:
            recommendation = "⚠️ MODERATE RISK: Some unusual patterns detected. Double-check the transaction details before confirming."
        else:
            recommendation = "✅ Transaction appears normal based on your historical patterns. Safe to proceed."

        # Map label to status for storage (0=Normal, 1=Suspicious, 2=Fraud)
        prediction_status = "Normal" if final_label == 0 else ("Suspicious" if final_label == 1 else "Fraud")

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
                    "Prediction Status": prediction_status,  # Normal | Suspicious | Fraud
                    "Fraud Score": fraud_score,
                    "Risk Factors": risk_factors,
                    "timestamp": datetime.now()
                })
                logger.info(f"Transaction saved for user: {current_user_email}")
            except Exception as db_error:
                logger.warning(f"Database save error: {db_error}")

        # Ensure JSON-serializable types (numpy int64/float64 are not serializable)
        def _native(val):
            if isinstance(val, (np.integer, np.int64)):
                return int(val)
            if isinstance(val, (np.floating, np.float64)):
                return float(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

        return jsonify({
            "is_fraud": bool(fraud_detected),
            "fraud_score": float(_native(fraud_score)),
            "risk_factors": risk_factors if risk_factors else ["No significant risk factors detected"],
            "recommendation": recommendation,
            "confidence": confidence,
            "prediction": prediction_status,  # Normal | Suspicious | Fraud (so frontend shows 3 tiers)
            "model_details": {
                "hmm_available": hmm_model is not None,
                "crf_available": crf_model is not None,
                "scaler_available": scaler is not None
            }
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
        
        # Convert to frontend format (Normal | Suspicious | Fraudulent)
        status_map = {"Normal": "Normal", "Suspicious": "Suspicious", "Fraud": "Fraudulent"}
        formatted_transactions = []
        for tx in transactions:
            stored_status = tx.get("Prediction Status")
            if stored_status in status_map:
                status = status_map[stored_status]
            else:
                # Backward compatibility: old records only have Fraud Detected
                status = "Fraudulent" if tx.get("Fraud Detected", False) else "Legitimate"
            formatted_transactions.append({
                "id": str(tx.get("timestamp", "")),
                "amount": tx.get("Transaction Amount (INR)", 0),
                "sender": tx.get("Sender UPI ID", ""),
                "receiver": tx.get("Receiver UPI ID", ""),
                "timestamp": tx.get("Transaction Time", ""),
                "status": status,
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
