"""
Backward compatibility entry point
Imports from the new modular structure
"""
from wsgi import app

if __name__ == '__main__':
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("Starting UPI Fraud Detection API...")
    
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
