"""Fraud detection routes"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import pandas as pd
from datetime import datetime
import logging

from app.database import db
from app.models import model_manager
from app.utils import calculate_features, convert_types
from app.services import FraudDetectionService, RiskAnalysisService

logger = logging.getLogger(__name__)
fraud_bp = Blueprint('fraud', __name__, url_prefix='/api')


@fraud_bp.route('/check_fraud', methods=['POST'])
@jwt_required()
def check_fraud():
    """Main fraud detection endpoint"""
    try:
        if not model_manager.is_ready():
            logger.error("[ERROR] [fraud.py] Model not loaded")
            return jsonify({"error": "Model not loaded"}), 503

        # Get current user
        current_user_email = get_jwt_identity()

        data = request.get_json()

        # Extract transaction data
        transaction_amount = float(data.get("amount", 0))
        sender_upi = data.get("sender", current_user_email)
        receiver_upi = data.get("receiver", "")
        timestamp_str = data.get("timestamp", "")
        description = data.get("description", "")
        category = data.get("category", "")

        if not transaction_amount or not receiver_upi:
            logger.warning(f"[WARN] [fraud.py] [username={current_user_email}] Missing required fields")
            return jsonify({"error": "Amount and receiver are required"}), 400

        # Parse timestamp
        try:
            transaction_time = pd.to_datetime(timestamp_str)
        except:
            transaction_time = datetime.now()

        # Calculate features
        try:
            feature_vector = calculate_features(
                current_user_email, 
                receiver_upi, 
                transaction_amount, 
                transaction_time,
                db.transactions_collection,
                model_manager.dataset
            )
        except Exception as e:
            logger.error(f"[ERROR] [fraud.py] [username={current_user_email}] Feature calculation failed: {e}")
            return jsonify({"error": "Feature calculation failed"}), 500

        # Initialize services
        fraud_service = FraudDetectionService(
            model_manager, 
            db, 
            fraud_bp.config.get('FEATURE_COLUMNS', [])
        )
        risk_service = RiskAnalysisService(db)

        # Make prediction (pass transactions collection for lagging)
        try:
            fraud_detected, fraud_score, final_label, confidence = fraud_service.predict(
                feature_vector, 
                current_user_email,
                db.transactions_collection
            )
        except Exception as e:
            logger.error(f"[ERROR] [fraud.py] [username={current_user_email}] Prediction failed: {e}")
            return jsonify({"error": "Prediction failed"}), 500

        # Decode prediction
        prediction_text = fraud_service.decode_label(final_label)

        # Analyze risk factors
        risk_factors = risk_service.analyze_risk_factors(
            current_user_email, 
            transaction_amount, 
            feature_vector
        )

        if fraud_detected:
            risk_factors.insert(0, "ML Model detected anomalous transaction pattern")

        if fraud_detected and len(risk_factors) == 1:
            risk_factors.append("Transaction pattern deviates from your normal behavior")

        if not risk_factors:
            risk_factors.append("Transaction matches your normal patterns")

        # Generate recommendation
        recommendation = risk_service.generate_recommendation(fraud_score)

        # Map label to status
        prediction_status = "Normal" if final_label == 0 else ("Suspicious" if final_label == 1 else "Fraud")

        # Save to database
        if db.transactions_collection is not None:
            try:
                db.transactions_collection.insert_one({
                    "username": current_user_email,
                    "Sender UPI ID": sender_upi,
                    "Receiver UPI ID": receiver_upi,
                    "Transaction Amount (INR)": transaction_amount,
                    "Transaction Time": convert_types(transaction_time),
                    "Category": category,
                    "Description": description,
                    "Fraud Detected": fraud_detected,
                    "Prediction Status": prediction_status,
                    "Fraud Score": fraud_score,
                    "Risk Factors": risk_factors,
                    "timestamp": datetime.now()
                })
            except Exception as db_error:
                logger.error(f"[ERROR] [fraud.py] [username={current_user_email}] Database save error: {db_error}")

        logger.info(f"[INFO] [fraud.py] [username={current_user_email}] Prediction: {prediction_status}, Score: {fraud_score:.2f}")

        return jsonify({
            "is_fraud": bool(fraud_detected),
            "fraud_score": float(fraud_score),
            "risk_factors": risk_factors,
            "recommendation": recommendation,
            "confidence": confidence,
            "prediction": prediction_status,
            "model_details": {
                "hmm_available": model_manager.hmm_model is not None,
                "crf_available": model_manager.crf_model is not None,
                "scaler_available": model_manager.scaler is not None
            }
        })

    except ValueError as ve:
        logger.error(f"[ERROR] [fraud.py] Validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"[ERROR] [fraud.py] Prediction error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@fraud_bp.route('/history', methods=['GET'])
@jwt_required()
def history():
    """Get transaction history for logged-in user"""
    try:
        if db.transactions_collection is None:
            return jsonify([]), 200

        current_user_email = get_jwt_identity()

        transactions = list(db.transactions_collection.find(
            {"username": current_user_email},
            {"_id": 0}
        ).sort("timestamp", -1).limit(100))

        # Format for frontend
        status_map = {"Normal": "Normal", "Suspicious": "Suspicious", "Fraud": "Fraudulent"}
        formatted_transactions = []
        
        for tx in transactions:
            stored_status = tx.get("Prediction Status")
            if stored_status in status_map:
                status = status_map[stored_status]
            else:
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

        return jsonify(formatted_transactions)

    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify([]), 200


@fraud_bp.route('/fraud/predict', methods=['POST'])
def predict_fraud():
    """Public fraud prediction endpoint (for testing)"""
    try:
        if not model_manager.is_ready():
            logger.error("[ERROR] [fraud.py] Model not loaded")
            return jsonify({"error": "Model not loaded"}), 503

        data = request.get_json()

        # Extract transaction data
        username = data.get("username", "test_user")
        recipient_id = data.get("recipient_id", "test_recipient")
        transaction_amount = float(data.get("transaction_amount", 0))
        transaction_time_str = data.get("transaction_time", "")

        if not transaction_amount or not recipient_id:
            logger.warning(f"[WARN] [fraud.py] Missing required fields")
            return jsonify({"error": "transaction_amount and recipient_id are required"}), 400

        # Parse timestamp with IST timezone
        try:
            import pytz
            # Parse the timestamp
            transaction_time = pd.to_datetime(transaction_time_str)
            # If no timezone info, assume IST
            if transaction_time.tzinfo is None:
                ist = pytz.timezone('Asia/Kolkata')
                transaction_time = ist.localize(transaction_time)
            # Convert to naive datetime (remove timezone) for feature calculation
            transaction_time = transaction_time.replace(tzinfo=None)
        except:
            transaction_time = datetime.now()

        # Calculate features
        try:
            feature_vector = calculate_features(
                username, 
                recipient_id, 
                transaction_amount, 
                transaction_time,
                db.transactions_collection,
                model_manager.dataset
            )
        except Exception as e:
            logger.error(f"[ERROR] [fraud.py] [username={username}] Feature calculation failed: {e}")
            return jsonify({"error": "Feature calculation failed"}), 500

        # Initialize fraud service
        fraud_service = FraudDetectionService(
            model_manager, 
            db, 
            fraud_bp.config.get('FEATURE_COLUMNS', [])
        )

        # Make prediction (pass transactions collection for lagging)
        try:
            fraud_detected, fraud_score, final_label, confidence = fraud_service.predict(
                feature_vector, 
                username,
                db.transactions_collection
            )
        except Exception as e:
            logger.error(f"[ERROR] [fraud.py] [username={username}] Prediction failed: {e}")
            import traceback
            logger.error(f"[ERROR] [fraud.py] Traceback: {traceback.format_exc()}")
            return jsonify({"error": "Prediction failed"}), 500

        # Decode prediction
        prediction_text = fraud_service.decode_label(final_label)

        logger.info(f"[INFO] [fraud.py] [username={username}] Prediction: {prediction_text}, Score: {fraud_score:.2f}")

        return jsonify({
            "fraud_detected": bool(fraud_detected),
            "fraud_score": float(fraud_score),
            "label": int(final_label),
            "prediction": prediction_text,
            "confidence": confidence,
            "model_details": {
                "hmm_available": model_manager.hmm_model is not None,
                "crf_available": model_manager.crf_model is not None,
                "scaler_available": model_manager.scaler is not None
            }
        })

    except ValueError as ve:
        logger.error(f"[ERROR] [fraud.py] Validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"[ERROR] [fraud.py] Prediction error: {e}")
        import traceback
        logger.error(f"[ERROR] [fraud.py] Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# Store config reference for services
fraud_bp.config = {}
