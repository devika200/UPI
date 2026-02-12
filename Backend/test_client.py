#!/usr/bin/env python3
"""
Test client for the UPI Secure Fraud Detection API
"""

import requests
import json
import time
from typing import Dict, Any

class FraudDetectionClient:
    """Client for interacting with the Fraud Detection API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'FraudDetectionClient/1.0'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "error"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "error"}
    
    def predict_single(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud for a single transaction"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=transaction_data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "error"}
    
    def predict_batch(self, transactions: list) -> Dict[str, Any]:
        """Predict fraud for multiple transactions"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=transactions
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "error"}

def create_sample_transaction() -> Dict[str, Any]:
    """Create a sample transaction for testing"""
    return {
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

def create_sample_batch(size: int = 3) -> list:
    """Create a batch of sample transactions"""
    base_transaction = create_sample_transaction()
    batch = []
    
    for i in range(size):
        transaction = base_transaction.copy()
        # Vary some values to make them different
        transaction["Transaction Amount (INR)"] += i * 100
        transaction["Transaction_Amount_Diff"] += i * 50
        transaction["hour"] = (transaction["hour"] + i) % 24
        batch.append(transaction)
    
    return batch

def main():
    """Main test function"""
    print("🚀 UPI Secure Fraud Detection API Test Client")
    print("=" * 50)
    
    # Initialize client
    client = FraudDetectionClient()
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    health_result = client.health_check()
    print(f"Health Status: {json.dumps(health_result, indent=2)}")
    
    if health_result.get("status") != "healthy":
        print("❌ API is not healthy. Please check if the server is running.")
        return
    
    # Test 2: Model Info
    print("\n2. Testing Model Info...")
    model_info = client.get_model_info()
    print(f"Model Info: {json.dumps(model_info, indent=2)}")
    
    # Test 3: Single Prediction
    print("\n3. Testing Single Transaction Prediction...")
    sample_transaction = create_sample_transaction()
    prediction_result = client.predict_single(sample_transaction)
    print(f"Prediction Result: {json.dumps(prediction_result, indent=2)}")
    
    # Test 4: Batch Prediction
    print("\n4. Testing Batch Transaction Prediction...")
    sample_batch = create_sample_batch(3)
    batch_result = client.predict_batch(sample_batch)
    print(f"Batch Result: {json.dumps(batch_result, indent=2)}")
    
    # Test 5: Error Handling
    print("\n5. Testing Error Handling...")
    
    # Test invalid transaction (missing fields)
    invalid_transaction = {"Transaction Amount (INR)": 5000}
    error_result = client.predict_single(invalid_transaction)
    print(f"Error Result: {json.dumps(error_result, indent=2)}")
    
    # Test 6: Performance Test
    print("\n6. Testing Performance (10 requests)...")
    start_time = time.time()
    
    for i in range(10):
        result = client.predict_single(sample_transaction)
        if result.get("status") == "success":
            print(f"Request {i+1}: ✅ Success")
        else:
            print(f"Request {i+1}: ❌ Failed")
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 10
    print(f"Average response time: {avg_time:.3f} seconds")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()





