import os
from typing import List

class Config:
    """Base configuration class"""
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'hmm_fraud_model.pkl')
    
    # Server settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    
    # API settings
    API_VERSION = '1.0.0'
    MAX_BATCH_SIZE = 100
    
    # Feature columns expected by the model
    FEATURE_COLUMNS: List[str] = [
        "Transaction Amount (INR)", "Transaction_Amount_Diff", "Transaction_Frequency_Score",
        "Time_Anomaly_Score", "Recipient_Total_Transactions", "Recipient_Avg_Transaction_Amount",
        "Fraud_Type", "Risk_Score", "hour", "day_of_week"
    ] + [f"Location_Hash_{i}" for i in range(10)]
    
    # Validation rules
    VALIDATION_RULES = {
        "hour": {"min": 0, "max": 23},
        "day_of_week": {"min": 0, "max": 6},
        "Risk_Score": {"min": 0, "max": 1},
        "Transaction_Frequency_Score": {"min": 0, "max": 1},
        "Time_Anomaly_Score": {"min": 0, "max": 1}
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    # Add production-specific settings here

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}





