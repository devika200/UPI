"""Routes package"""
from .auth import auth_bp
from .fraud import fraud_bp
from .admin import admin_bp

__all__ = ['auth_bp', 'fraud_bp', 'admin_bp']
