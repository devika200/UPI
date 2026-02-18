"""Flask application factory"""
import logging

# Suppress pymongo DEBUG logs IMMEDIATELY before anything else
logging.getLogger('pymongo').setLevel(logging.ERROR)
logging.getLogger('pymongo.topology').setLevel(logging.ERROR)
logging.getLogger('pymongo.connection').setLevel(logging.ERROR)
logging.getLogger('pymongo.serverSelection').setLevel(logging.ERROR)
logging.getLogger('pymongo.command').setLevel(logging.ERROR)
logging.getLogger('pymongo.server').setLevel(logging.ERROR)
logging.getLogger('pymongo.pool').setLevel(logging.ERROR)

from flask import Flask, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from datetime import datetime

from app.config import get_config
from app.database import db
from app.models import model_manager

# Initialize extensions
bcrypt = Bcrypt()
jwt = JWTManager()


def create_app(config_name=None):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    if config_name:
        from app.config import config
        app.config.from_object(config[config_name])
    else:
        app.config.from_object(get_config())
    
    # Setup logging
    setup_logging(app)
    
    # Initialize extensions
    CORS(app, origins=app.config['CORS_ORIGINS'])
    bcrypt.init_app(app)
    jwt.init_app(app)
    
    # Initialize database
    db.init_app(app)
    
    # Initialize models
    model_manager.init_app(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "model_loaded": model_manager.is_ready(),
            "database_connected": db.is_connected(),
            "timestamp": datetime.now().isoformat()
        })
    
    return app


def setup_logging(app):
    """Configure application logging"""
    # Suppress pymongo DEBUG/INFO/WARNING logs completely
    logging.getLogger('pymongo').setLevel(logging.ERROR)
    logging.getLogger('pymongo.topology').setLevel(logging.ERROR)
    logging.getLogger('pymongo.connection').setLevel(logging.ERROR)
    logging.getLogger('pymongo.serverSelection').setLevel(logging.ERROR)
    logging.getLogger('pymongo.command').setLevel(logging.ERROR)
    logging.getLogger('pymongo.server').setLevel(logging.ERROR)
    logging.getLogger('pymongo.pool').setLevel(logging.ERROR)
    
    # Suppress werkzeug (Flask) request logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Configure application logging
    logging.basicConfig(
        level=getattr(logging, app.config['LOG_LEVEL']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(app.config['LOG_FILE']),
            logging.StreamHandler()
        ]
    )


def register_blueprints(app):
    """Register Flask blueprints"""
    from app.routes import auth_bp, fraud_bp, admin_bp
    
    # Pass config to fraud blueprint for services
    fraud_bp.config = app.config
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(fraud_bp)
    app.register_blueprint(admin_bp)


def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Resource not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        app.logger.error(f"Unhandled exception: {error}")
        return jsonify({"error": "An unexpected error occurred"}), 500
