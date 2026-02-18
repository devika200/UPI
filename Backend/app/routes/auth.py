"""Authentication routes"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from datetime import datetime
import logging

from app.database import db

logger = logging.getLogger(__name__)
auth_bp = Blueprint('auth', __name__, url_prefix='/api')


@auth_bp.route('/register', methods=['POST'])
def register():
    """User registration endpoint"""
    from app import bcrypt
    
    try:
        data = request.get_json()
        username = data.get("username")
        email = data.get("email")
        password = data.get("password")

        if not username or not password or not email:
            return jsonify({"error": "Username, email and password are required"}), 400

        if db.users_collection is None:
            return jsonify({"error": "Database not available"}), 503

        if db.users_collection.find_one({"email": email}):
            return jsonify({"error": "User already exists"}), 400

        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        db.users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed,
            "created_at": datetime.now()
        })
        
        return jsonify({"message": "User registered successfully"}), 201
    
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    from app import bcrypt
    
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        if db.users_collection is None:
            return jsonify({"error": "Database not available"}), 503

        user = db.users_collection.find_one({"email": email})
        if not user or not bcrypt.check_password_hash(user["password"], password):
            return jsonify({"error": "Invalid email or password"}), 401

        token = create_access_token(identity=email)
        return jsonify({
            "access_token": token,
            "message": "Login successful",
            "username": user.get("username", email)
        }), 200
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": str(e)}), 500
