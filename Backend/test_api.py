import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, load_model
from config import Config

class TestFraudDetectionAPI(unittest.TestCase):
    """Test cases for the Fraud Detection API"""
    
    def setUp(self):
        """Set up test environment"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        
        # Sample valid transaction data
        self.valid_transaction = {
            "Transaction Amount (INR)": 5000,
            "Transaction_Amount_Diff": 200,
            "Transaction_Frequency_Score": 0.8,
            "Time_Anomaly_Score": 0.5,
            "Recipient_Total_Transactions": 15,
            "Recipient_Avg_Transaction_Amount": 4500,
            "Fraud_Type": 0,
            "Risk_Score": 0.9,
            "hour": 14,
            "day_of_week": 3,
            "Location_Hash_0": 1,
            "Location_Hash_1": 0,
            "Location_Hash_2": 1,
            "Location_Hash_3": 0,
            "Location_Hash_4": 1,
            "Location_Hash_5": 0,
            "Location_Hash_6": 1,
            "Location_Hash_7": 0,
            "Location_Hash_8": 1,
            "Location_Hash_9": 0
        }
        
        self.valid_batch = [self.valid_transaction.copy() for _ in range(3)]

    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
        self.assertIn('version', data)

    def test_model_info_without_model(self):
        """Test model info endpoint when model is not loaded"""
        response = self.client.get('/model/info')
        self.assertEqual(response.status_code, 503)

    @patch('app.model')
    @patch('app.model_loaded', True)
    def test_model_info_with_model(self, mock_model):
        """Test model info endpoint when model is loaded"""
        response = self.client.get('/model/info')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('model_type', data)
        self.assertIn('features', data)
        self.assertIn('feature_count', data)

    def test_predict_invalid_content_type(self):
        """Test predict endpoint with invalid content type"""
        response = self.client.post('/predict', data='invalid data')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)

    def test_predict_missing_fields(self):
        """Test predict endpoint with missing fields"""
        incomplete_data = {"Transaction Amount (INR)": 5000}
        response = self.client.post('/predict', 
                                  json=incomplete_data,
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)

    def test_predict_invalid_field_types(self):
        """Test predict endpoint with invalid field types"""
        invalid_data = self.valid_transaction.copy()
        invalid_data["Transaction Amount (INR)"] = "not_a_number"
        
        response = self.client.post('/predict', 
                                  json=invalid_data,
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)

    def test_predict_invalid_ranges(self):
        """Test predict endpoint with invalid field ranges"""
        invalid_data = self.valid_transaction.copy()
        invalid_data["hour"] = 25  # Invalid hour
        
        response = self.client.post('/predict', 
                                  json=invalid_data,
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)

    @patch('app.model')
    @patch('app.model_loaded', True)
    def test_predict_success(self, mock_model):
        """Test successful prediction"""
        # Mock the model's detect_anomalies method
        mock_model.detect_anomalies.return_value = [True]
        
        response = self.client.post('/predict', 
                                  json=self.valid_transaction,
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('fraud_detected', data)
        self.assertIn('confidence', data)
        self.assertIn('transaction_id', data)
        self.assertTrue(data['fraud_detected'])

    def test_batch_predict_invalid_input(self):
        """Test batch predict with invalid input"""
        response = self.client.post('/predict/batch', 
                                  json="not_an_array",
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)

    def test_batch_predict_empty_array(self):
        """Test batch predict with empty array"""
        response = self.client.post('/predict/batch', 
                                  json=[],
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)

    def test_batch_predict_too_large(self):
        """Test batch predict with too many transactions"""
        large_batch = [self.valid_transaction.copy() for _ in range(101)]
        response = self.client.post('/predict/batch', 
                                  json=large_batch,
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)

    @patch('app.model')
    @patch('app.model_loaded', True)
    def test_batch_predict_success(self, mock_model):
        """Test successful batch prediction"""
        # Mock the model's detect_anomalies method
        mock_model.detect_anomalies.return_value = [True, False, True]
        
        response = self.client.post('/predict/batch', 
                                  json=self.valid_batch,
                                  content_type='application/json')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('predictions', data)
        self.assertIn('total_transactions', data)
        self.assertIn('fraudulent_count', data)
        self.assertEqual(data['total_transactions'], 3)
        self.assertEqual(data['fraudulent_count'], 2)

    def test_404_error(self):
        """Test 404 error handling"""
        response = self.client.get('/nonexistent_endpoint')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 404)
        self.assertIn('error', data)

    def test_405_error(self):
        """Test 405 error handling"""
        response = self.client.get('/predict')  # GET not allowed on /predict
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 405)
        self.assertIn('error', data)

class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_validate_transaction_data_valid(self):
        """Test validation with valid data"""
        from utils import validate_transaction_data
        
        valid_data = {
            "Transaction Amount (INR)": 5000,
            "Transaction_Amount_Diff": 200,
            "Transaction_Frequency_Score": 0.8,
            "Time_Anomaly_Score": 0.5,
            "Recipient_Total_Transactions": 15,
            "Recipient_Avg_Transaction_Amount": 4500,
            "Fraud_Type": 0,
            "Risk_Score": 0.9,
            "hour": 14,
            "day_of_week": 3,
            "Location_Hash_0": 1,
            "Location_Hash_1": 0,
            "Location_Hash_2": 1,
            "Location_Hash_3": 0,
            "Location_Hash_4": 1,
            "Location_Hash_5": 0,
            "Location_Hash_6": 1,
            "Location_Hash_7": 0,
            "Location_Hash_8": 1,
            "Location_Hash_9": 0
        }
        
        is_valid, error_msg, cleaned_data = validate_transaction_data(
            valid_data, Config.FEATURE_COLUMNS, Config.VALIDATION_RULES
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
        self.assertIsInstance(cleaned_data, dict)

    def test_validate_transaction_data_invalid(self):
        """Test validation with invalid data"""
        from utils import validate_transaction_data
        
        invalid_data = {"Transaction Amount (INR)": "not_a_number"}
        
        is_valid, error_msg, cleaned_data = validate_transaction_data(
            invalid_data, Config.FEATURE_COLUMNS, Config.VALIDATION_RULES
        )
        
        self.assertFalse(is_valid)
        self.assertIn("must be a number", error_msg)

if __name__ == '__main__':
    unittest.main()





