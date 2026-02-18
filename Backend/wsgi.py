"""WSGI entry point for production"""
import os
import logging
from app import create_app

logger = logging.getLogger(__name__)

# Create application instance
app = create_app()

if __name__ == '__main__':
    logger.info("Starting UPI Fraud Detection API...")
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
