import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from functools import wraps
from flask import request, jsonify
import traceback

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = 'INFO', log_file: str = 'api.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def log_request_info():
    """Log request information for debugging"""
    logger.info(f"Request: {request.method} {request.path}")
    logger.info(f"Headers: {dict(request.headers)}")
    if request.is_json:
        logger.info(f"Body: {json.dumps(request.get_json(), indent=2)}")

def create_response(data: Any = None, status: str = "success", 
                   message: str = "", error: str = "", 
                   status_code: int = 200) -> Tuple[Dict[str, Any], int]:
    """Create standardized API response"""
    response = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "message": message
    }
    
    if data is not None:
        response["data"] = data
    
    if error:
        response["error"] = error
    
    return jsonify(response), status_code

def validate_transaction_data(data: Dict[str, Any], 
                            feature_columns: List[str],
                            validation_rules: Dict[str, Dict[str, int]]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate transaction data against expected schema
    
    Args:
        data: Input transaction data
        feature_columns: Expected feature columns
        validation_rules: Validation rules for specific fields
    
    Returns:
        tuple: (is_valid, error_message, cleaned_data)
    """
    try:
        # Check if data is a dictionary
        if not isinstance(data, dict):
            return False, "Input data must be a JSON object", {}
        
        cleaned_data = {}
        
        # Validate all required fields
        for field in feature_columns:
            if field not in data:
                return False, f"Missing required field: {field}", {}
            
            value = data[field]
            if not isinstance(value, (int, float)):
                return False, f"Field '{field}' must be a number", {}
            
            cleaned_data[field] = float(value)
        
        # Apply validation rules
        for field, rules in validation_rules.items():
            if field in cleaned_data:
                value = cleaned_data[field]
                if value < rules["min"] or value > rules["max"]:
                    return False, f"Field '{field}' must be between {rules['min']} and {rules['max']}", {}
        
        return True, "", cleaned_data
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False, f"Validation error: {str(e)}", {}

def handle_exceptions(f):
    """Decorator to handle exceptions and return proper error responses"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return create_response(
                status="error",
                error=error_msg,
                status_code=500
            )
    return decorated_function

def rate_limit(max_requests: int = 100, window: int = 60):
    """Simple rate limiting decorator (in-memory)"""
    request_counts = {}
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = datetime.now().timestamp()
            
            # Clean old entries
            request_counts[client_ip] = [
                req_time for req_time in request_counts.get(client_ip, [])
                if current_time - req_time < window
            ]
            
            # Check rate limit
            if len(request_counts[client_ip]) >= max_requests:
                return create_response(
                    status="error",
                    error="Rate limit exceeded",
                    status_code=429
                )
            
            # Add current request
            request_counts[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def generate_transaction_id(transaction_data: Dict[str, Any]) -> str:
    """Generate a unique transaction ID"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    data_hash = hash(str(transaction_data)) % 10000
    return f"txn_{timestamp}_{data_hash:04d}"

def format_prediction_response(fraud_detected: bool, 
                             transaction_data: Dict[str, Any],
                             confidence: str = None) -> Dict[str, Any]:
    """Format prediction response"""
    if confidence is None:
        confidence = "high" if fraud_detected else "low"
    
    return {
        "fraud_detected": fraud_detected,
        "confidence": confidence,
        "transaction_id": generate_transaction_id(transaction_data),
        "timestamp": datetime.now().isoformat()
    }

def validate_batch_size(batch_data: List[Dict[str, Any]], 
                       max_size: int) -> Tuple[bool, str]:
    """Validate batch size"""
    if not isinstance(batch_data, list):
        return False, "Input must be an array of transaction objects"
    
    if len(batch_data) == 0:
        return False, "Empty transaction array"
    
    if len(batch_data) > max_size:
        return False, f"Batch size too large. Maximum {max_size} transactions allowed"
    
    return True, ""

def calculate_batch_statistics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for batch predictions"""
    total_count = len(predictions)
    fraudulent_count = sum(1 for p in predictions if p["fraud_detected"])
    fraud_rate = (fraudulent_count / total_count) * 100 if total_count > 0 else 0
    
    return {
        "total_transactions": total_count,
        "fraudulent_count": fraudulent_count,
        "fraud_rate_percentage": round(fraud_rate, 2),
        "legitimate_count": total_count - fraudulent_count
    }





