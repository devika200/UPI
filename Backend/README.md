# UPI Secure Fraud Detection API

A production-ready Flask API for UPI transaction fraud detection using Hidden Markov Models (HMM).

## Features

- **Real-time Fraud Detection**: Detect fraudulent UPI transactions using AutoRegressive HMM
- **Batch Processing**: Process multiple transactions efficiently
- **Comprehensive Validation**: Input validation with detailed error messages
- **Production Ready**: Logging, error handling, CORS, and configuration management
- **Health Monitoring**: Health check and model information endpoints
- **Rate Limiting**: Built-in rate limiting for API protection
- **Comprehensive Testing**: Unit tests for all endpoints and utilities

## API Endpoints

### Health Check
```
GET /health
```
Returns API health status and version information.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00.123456",
  "version": "1.0.0"
}
```

### Model Information
```
GET /model/info
```
Returns model information and feature details.

**Response:**
```json
{
  "model_type": "AutoRegressiveHMM",
  "features": ["Transaction Amount (INR)", "Transaction_Amount_Diff", ...],
  "feature_count": 20,
  "model_path": "hmm_fraud_model.pkl",
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

### Single Transaction Prediction
```
POST /predict
```
Predict fraud for a single transaction.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "fraud_detected": true,
  "confidence": "high",
  "status": "success",
  "timestamp": "2024-01-15T10:30:00.123456",
  "transaction_id": "txn_20240115_103000_1234"
}
```

### Batch Transaction Prediction
```
POST /predict/batch
```
Predict fraud for multiple transactions (max 100).

**Request Body:**
```json
[
  {
    "Transaction Amount (INR)": 5000,
    "Transaction_Amount_Diff": 200,
    ...
  },
  {
    "Transaction Amount (INR)": 5100,
    "Transaction_Amount_Diff": 150,
    ...
  }
]
```

**Response:**
```json
{
  "predictions": [
    {
      "transaction_index": 0,
      "fraud_detected": true,
      "confidence": "high",
      "transaction_id": "txn_20240115_103000_0_1234"
    },
    {
      "transaction_index": 1,
      "fraud_detected": false,
      "confidence": "low",
      "transaction_id": "txn_20240115_103000_1_5678"
    }
  ],
  "total_transactions": 2,
  "fraudulent_count": 1,
  "status": "success",
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Backend
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Ensure model file exists:**
Make sure `hmm_fraud_model.pkl` is in the Backend directory.

## Configuration

The API can be configured using environment variables:

- `MODEL_PATH`: Path to the model file (default: `hmm_fraud_model.pkl`)
- `DEBUG`: Enable debug mode (default: `False`)
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `5000`)

Example:
```bash
export MODEL_PATH=/path/to/model.pkl
export DEBUG=True
export PORT=8080
```

## Running the API

### Development Mode
```bash
python app.py
```

### Production Mode
```bash
export DEBUG=False
python app.py
```

The API will be available at `http://localhost:5000`

## Testing

Run the test suite:
```bash
python test_api.py
```

Or run specific test classes:
```bash
python -m unittest test_api.TestFraudDetectionAPI
python -m unittest test_api.TestUtils
```

## Usage Examples

### Using curl

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @payload.json
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Using Python

```python
import requests
import json

# API base URL
base_url = "http://localhost:5000"

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# Single prediction
transaction_data = {
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

response = requests.post(
    f"{base_url}/predict",
    json=transaction_data,
    headers={"Content-Type": "application/json"}
)
result = response.json()
print(f"Fraud detected: {result['fraud_detected']}")
```

## Error Handling

The API returns standardized error responses:

```json
{
  "error": "Error description",
  "status": "error",
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (validation errors)
- `404`: Not Found
- `405`: Method Not Allowed
- `413`: Request Entity Too Large
- `429`: Rate Limit Exceeded
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## Logging

The API logs all requests and errors to:
- Console output
- `api.log` file

Log levels can be configured via the `LOG_LEVEL` environment variable.

## Security Features

- Input validation and sanitization
- Rate limiting (100 requests per minute per IP)
- CORS configuration
- Request size limits (16MB)
- Error message sanitization

## Performance

- Threaded server for concurrent requests
- Efficient batch processing
- Optimized model loading
- Memory-efficient data handling

## Monitoring

Monitor the API using:
- Health check endpoint
- Log files
- Model information endpoint
- Response timestamps

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]





