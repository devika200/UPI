"""Admin routes"""
from flask import Blueprint, request, jsonify
from datetime import datetime
import logging

from app.database import db

logger = logging.getLogger(__name__)
admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')


@admin_bp.route('/stats', methods=['GET'])
def admin_stats():
    """Admin endpoint to view database statistics"""
    try:
        if db.users_collection is None or db.transactions_collection is None:
            return jsonify({"error": "Database not available"}), 503

        # User stats
        user_count = db.users_collection.count_documents({})

        # Transaction stats
        tx_count = db.transactions_collection.count_documents({})
        fraud_count = db.transactions_collection.count_documents({"Fraud Detected": True})
        legitimate_count = tx_count - fraud_count
        fraud_rate = (fraud_count / tx_count * 100) if tx_count > 0 else 0

        # Recent transactions
        recent_txs = list(db.transactions_collection.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(10))

        # Recent users
        recent_users = list(db.users_collection.find(
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


@admin_bp.route('/users', methods=['GET'])
def admin_users():
    """Get all users (without passwords)"""
    try:
        if db.users_collection is None:
            return jsonify({"error": "Database not available"}), 503

        users = list(db.users_collection.find({}, {"_id": 0, "password": 0}))
        return jsonify({"users": users, "count": len(users)})

    except Exception as e:
        logger.error(f"Admin users error: {e}")
        return jsonify({"error": str(e)}), 500


@admin_bp.route('/transactions', methods=['GET'])
def admin_transactions():
    """Get all transactions with optional filters"""
    try:
        if db.transactions_collection is None:
            return jsonify({"error": "Database not available"}), 503

        # Get query parameters
        limit = int(request.args.get('limit', 100))
        fraud_only = request.args.get('fraud_only', 'false').lower() == 'true'

        # Build query
        query = {}
        if fraud_only:
            query["Fraud Detected"] = True

        # Get transactions
        txs = list(db.transactions_collection.find(
            query, {"_id": 0}
        ).sort("timestamp", -1).limit(limit))

        return jsonify({
            "transactions": txs,
            "count": len(txs),
            "total": db.transactions_collection.count_documents(query)
        })

    except Exception as e:
        logger.error(f"Admin transactions error: {e}")
        return jsonify({"error": str(e)}), 500
