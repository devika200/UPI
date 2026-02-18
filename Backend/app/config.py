"""Application configuration"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # JWT
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', '1234567890')
    JWT_ACCESS_TOKEN_EXPIRES = 86400  # 24 hours
    
    # MongoDB
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DB_NAME = 'upi_fraud_detection'
    MONGODB_TIMEOUT = 2000
    
    # CORS
    CORS_ORIGINS = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://upi-secure.netlify.app"
    ]
    
    # Model paths
    MODEL_DIR = 'models'
    HMM_MODEL_PATHS = [
        'models/arlg_hmm_model.pkl',
        'hmm_fraud_model.pkl',
        '../project/Backend/models/arlg_hmm_model.pkl'
    ]
    CRF_MODEL_PATHS = [
        'models/crf_model.pkl',
        '../project/Backend/models/crf_model.pkl'
    ]
    SCALER_PATHS = [
        'models/scaler.pkl',
        '../project/Backend/models/scaler.pkl'
    ]
    ENCODER_PATHS = [
        'models/label_encoder.pkl',
        '../project/Backend/models/label_encoder.pkl'
    ]
    DATASET_PATHS = [
        'balanced_dataset.csv',
        '../project/Backend/balanced_dataset.csv'
    ]
    
    # Feature configuration
    FEATURE_COLUMNS = [
        'Transaction Amount (INR)', 'Transaction_Amount_Diff',
        'Transaction_Frequency_Score', 'Time_Anomaly_Score',
        'Recipient_Total_Transactions', 'Recipient_Avg_Transaction_Amount',
        'Fraud_Type', 'Risk_Score', 'hour', 'day_of_week'
    ] + [f'Location_Hash_{i}' for i in range(10)]
    
    # Logging
    LOG_FILE = 'api.log'
    LOG_LEVEL = 'INFO'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    MONGODB_URI = 'mongodb://localhost:27017/'
    MONGODB_DB_NAME = 'upi_fraud_detection_test'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
