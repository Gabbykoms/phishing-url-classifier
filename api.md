# Phishing URL Classifier API Documentation

A high-performance REST API for real-time phishing URL detection using Deep Neural Networks. Built with FastAPI and PyTorch.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [API Endpoints](#api-endpoints)
5. [Request/Response Formats](#requestresponse-formats)
6. [Usage Examples](#usage-examples)
7. [Error Handling](#error-handling)
8. [Model Information](#model-information)
9. [Development](#development)
10. [Deployment](#deployment)

---

## Overview

The Phishing URL Classifier API exposes a trained neural network model that classifies URLs as either **legitimate** or **phishing** based on 50 extracted numerical features.

**Key Features:**
- Real-time predictions with high accuracy (99.99% test accuracy)
- Type-safe input validation with Pydantic
- Automatic interactive documentation (Swagger UI)
- CORS enabled for cross-origin requests
- GPU support (automatically uses CUDA if available)
- Production-ready error handling

---

## Architecture

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI | Building REST API |
| ASGI Server | Uvicorn | Running the application |
| ML Model | PyTorch | Neural network inference |
| Data Validation | Pydantic | Request/response validation |
| API Documentation | Swagger UI/ReDoc | Interactive documentation |

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT APPLICATIONS                      │
│          (Web App, Mobile App, Security Tools, etc.)         │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP Requests
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Endpoints: /health, /model/info, /predict, /           ││
│  │ CORS Middleware (allow all origins)                    ││
│  │ Request Validation (Pydantic Schemas)                  ││
│  └─────────────────────────────────────────────────────────┘│
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Manager                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Loads model from disk on startup                        ││
│  │ Manages model state and device placement                ││
│  │ Handles inference with preprocessing                    ││
│  └─────────────────────────────────────────────────────────┘│
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   PyTorch Neural Network                     │
│  Input (50) → Linear → BatchNorm → ReLU → Dropout          │
│  Hidden (128) → Linear → BatchNorm → ReLU → Dropout        │
│  Hidden (64) → Linear → BatchNorm → ReLU → Dropout         │
│  Hidden (32) → Linear → Output (1) → Sigmoid               │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture

The deployed model is a Multi-Layer Perceptron (MLP) with:
- **Input**: 50 numerical features extracted from URLs
- **Hidden Layers**: 128 → 64 → 32 neurons
- **Activation**: ReLU with Batch Normalization
- **Regularization**: 0.3 dropout rate
- **Output**: Sigmoid (probability 0-1)

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- FastAPI
- Uvicorn
- Pydantic

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify model files exist:**
   ```bash
   ls models/phishing_mlp_optimized.pt
   ls models/phishing_mlp_optimized_info.json
   ```

3. **Start the server:**
   ```bash
   python run_api.py
   ```

4. **Access the API:**
   - API: `http://localhost:8000`
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Configuration

The API runs with default settings:
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8000
- **Reload**: Enabled (auto-restart on code changes)
- **Log Level**: Info
- **Device**: GPU (CUDA) if available, else CPU

To modify settings, edit [run_api.py](run_api.py):
```python
uvicorn.run(
    "src.main:app",
    host="0.0.0.0",
    port=8000,  # Change port here
    reload=False,  # Disable for production
    log_level="info"  # Change to "warning" for less verbose output
)
```

---

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Purpose:** Verify API and model status

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Status Codes:**
- `200`: Healthy and operational
- `503`: Model not loaded (unhealthy)

---

### 2. Model Information

**Endpoint:** `GET /model/info`

**Purpose:** Retrieve model architecture and performance metrics

**Response:**
```json
{
  "model_name": "PhishingMLP",
  "input_size": 50,
  "hidden_sizes": [128, 64, 32],
  "dropout_rate": 0.3,
  "test_accuracy": 0.9999,
  "test_precision": 0.9998,
  "test_recall": 1.0,
  "test_f1_score": 0.9999,
  "test_roc_auc": 1.0,
  "use_batch_norm": true
}
```

**Status Codes:**
- `200`: Successfully retrieved model info
- `503`: Model not loaded

---

### 3. Make Prediction (Main Endpoint)

**Endpoint:** `POST /predict`

**Purpose:** Classify a URL as phishing or legitimate

**Request Body:**
```json
{
  "features": [
    0.5, 0.3, 0.7, 0.2, 0.9,
    0.4, 0.6, 0.1, 0.8, 0.5,
    0.3, 0.7, 0.2, 0.9, 0.4,
    0.6, 0.1, 0.8, 0.5, 0.3,
    0.7, 0.2, 0.9, 0.4, 0.6,
    0.1, 0.8, 0.5, 0.3, 0.7,
    0.2, 0.9, 0.4, 0.6, 0.1,
    0.8, 0.5, 0.3, 0.7, 0.2,
    0.9, 0.4, 0.6, 0.1, 0.8,
    0.5, 0.3, 0.7, 0.2, 0.9
  ]
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.9876
}
```

**Field Descriptions:**
- `prediction`: `0` = Legitimate URL, `1` = Phishing URL
- `probability`: Confidence score (0.0 - 1.0)

**Status Codes:**
- `200`: Prediction successful
- `400`: Invalid input (wrong number of features, wrong type)
- `503`: Model not loaded

---

### 4. Root Information

**Endpoint:** `GET /`

**Purpose:** API metadata and available endpoints

**Response:**
```json
{
  "message": "Phishing URL Classifier API",
  "version": "1.0.0",
  "endpoints": {
    "docs": "/docs",
    "redoc": "/redoc",
    "health": "/health",
    "model_info": "/model/info",
    "predict": "/predict"
  }
}
```

---

## Request/Response Formats

### Input Schema (PredictionRequest)

```python
{
  "features": List[float]  # Exactly 50 floats, no more, no less
}
```

**Validation Rules:**
- Must have exactly 50 features
- Each feature must be a float
- Invalid requests return 400 Bad Request

**Example of Invalid Request:**
```json
{
  "features": [0.5, 0.3]  # Only 2 features instead of 50
}
```
Response:
```json
{
  "detail": "ensure this value has at most 50 items"
}
```

### Output Schema (PredictionResponse)

```python
{
  "prediction": int,   # 0 or 1
  "probability": float  # 0.0 to 1.0
}
```

### Model Info Schema

```python
{
  "model_name": str,
  "input_size": int,
  "hidden_sizes": List[int],
  "dropout_rate": float,
  "test_accuracy": float,
  "test_precision": float,
  "test_recall": float,
  "test_f1_score": float,
  "test_roc_auc": float,
  "use_batch_norm": bool
}
```

---

## Usage Examples

### Python Requests

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Get model info
response = requests.get(f"{BASE_URL}/model/info")
model_info = response.json()
print(f"Model Accuracy: {model_info['test_accuracy']}")

# Make prediction
features = [0.5] * 50  # 50 features

response = requests.post(
    f"{BASE_URL}/predict",
    json={"features": features}
)

prediction = response.json()
print(f"Prediction: {'Phishing' if prediction['prediction'] == 1 else 'Legitimate'}")
print(f"Confidence: {prediction['probability']:.4f}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6, 0.1, 0.8, 0.5, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6, 0.1, 0.8, 0.5, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6, 0.1, 0.8, 0.5, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6, 0.1, 0.8, 0.5, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6, 0.1, 0.8, 0.5, 0.3, 0.7, 0.2, 0.9]
  }'
```

### JavaScript/Fetch

```javascript
// Make prediction
const features = Array(50).fill(0.5);

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ features })
})
  .then(response => response.json())
  .then(data => {
    console.log(`Prediction: ${data.prediction === 1 ? 'Phishing' : 'Legitimate'}`);
    console.log(`Confidence: ${(data.probability * 100).toFixed(2)}%`);
  });
```

### Interactive Testing

The easiest way to test is through the Swagger UI:

1. Start the API: `python run_api.py`
2. Open: `http://localhost:8000/docs`
3. Click on the `/predict` endpoint
4. Click "Try it out"
5. Paste your features JSON
6. Click "Execute"

---

## Error Handling

The API returns standard HTTP status codes and detailed error messages:

### Common Error Scenarios

| Status Code | Scenario | Example Response |
|-------------|----------|------------------|
| `400` | Invalid input (wrong feature count) | `{"detail": "ensure this value has exactly 50 items"}` |
| `400` | Feature is not a number | `{"detail": "value is not a valid float"}` |
| `500` | Prediction error | `{"detail": "Prediction error: [error details]"}` |
| `503` | Model not loaded | `{"detail": "Model not loaded"}` |

### Error Response Format

All errors return JSON:
```json
{
  "detail": "Human-readable error message"
}
```

### Handling Errors in Code

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"features": [0.5] * 50}
    )
    response.raise_for_status()  # Raise exception for bad status codes
    result = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e.response.status_code}")
    print(f"Message: {e.response.json()['detail']}")
except requests.exceptions.RequestException as e:
    print(f"Connection Error: {e}")
```

---

## Model Information

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 99.99% |
| **Test Precision** | 99.98% |
| **Test Recall** | 100% |
| **Test F1-Score** | 99.99% |
| **Test ROC-AUC** | 1.0 |

### Model Files

- **Model Weights**: `models/phishing_mlp_optimized.pt`
- **Model Info**: `models/phishing_mlp_optimized_info.json`

### Feature Extraction

The 50 features are extracted from URLs using domain-based analysis. They capture characteristics such as:
- URL length statistics
- Domain structure patterns
- Character distributions
- Special character presence
- Lexical features

**Note**: Feature extraction is handled externally. The API expects pre-extracted features.

### Decision Threshold

The model uses a **0.5 probability threshold**:
- Probability > 0.5 → Phishing (prediction = 1)
- Probability ≤ 0.5 → Legitimate (prediction = 0)

To adjust the threshold, modify [src/model_loader.py](src/model_loader.py):
```python
prediction = 1 if probability > 0.5 else 0  # Change 0.5 to desired threshold
```

---

## Development

### Project Structure

```
phishing-url-classifier/
├── run_api.py              # Entry point - starts the server
├── requirements.txt        # Python dependencies
├── src/
│   ├── main.py            # FastAPI application & endpoints
│   ├── schemas.py         # Pydantic request/response schemas
│   ├── model_loader.py    # ModelManager class for inference
│   ├── model.py           # PhishingMLP neural network definition
│   └── __init__.py        # Package init
├── models/
│   ├── phishing_mlp_optimized.pt     # Trained model weights
│   └── phishing_mlp_optimized_info.json  # Model metadata
└── notebooks/             # Jupyter notebooks for research/development
```

### Adding New Endpoints

1. Define request schema in [src/schemas.py](src/schemas.py)
2. Add endpoint function in [src/main.py](src/main.py)
3. Use `@app.get()` or `@app.post()` decorator
4. FastAPI automatically validates and documents it

Example:
```python
from fastapi import FastAPI
from .schemas import MyRequest, MyResponse

@app.post("/new-endpoint", response_model=MyResponse)
async def new_endpoint(request: MyRequest):
    """Endpoint description"""
    return MyResponse(result="value")
```

### Running Tests

```bash
# Run API
python run_api.py

# In another terminal, test with curl or Python requests
python -c "
import requests
response = requests.get('http://localhost:8000/health')
print(response.json())
"
```

### Code Organization Best Practices

1. **Schemas** (`schemas.py`): All request/response validation
2. **Models** (`model.py`): Neural network architecture definitions
3. **Model Loading** (`model_loader.py`): Inference logic
4. **API Routes** (`main.py`): HTTP endpoints and FastAPI setup
5. **Entry Point** (`run_api.py`): Server startup configuration

---

## Deployment

### Production Checklist

Before deploying to production:

- [ ] Disable `reload=True` in [run_api.py](run_api.py)
- [ ] Restrict CORS origins to specific domains (not `["*"]`)
- [ ] Use a production ASGI server (Gunicorn, Hypercorn)
- [ ] Set up logging and monitoring
- [ ] Use environment variables for sensitive config
- [ ] Enable HTTPS/SSL
- [ ] Set up rate limiting and authentication
- [ ] Add request logging and error tracking

### Production Server (Gunicorn)

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.main:app --bind 0.0.0.0:8000
```

### Docker Deployment

Example Dockerfile:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t phishing-classifier-api .
docker run -p 8000:8000 phishing-classifier-api
```

### Environment Variables

Create `.env` file:
```
MODEL_PATH=models/phishing_mlp_optimized.pt
MODEL_INFO_PATH=models/phishing_mlp_optimized_info.json
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
ALLOWED_ORIGINS=https://example.com,https://app.example.com
```

Load in [src/main.py](src/main.py):
```python
from dotenv import load_dotenv
import os

load_dotenv()
model_path = os.getenv("MODEL_PATH")
```

---

## Support & Troubleshooting

### Issue: Model not loading on startup

**Error**: `Model files not found`

**Solution**: Verify model files exist:
```bash
ls -la models/phishing_mlp_optimized.pt
ls -la models/phishing_mlp_optimized_info.json
```

---

### Issue: Port 8000 already in use

**Error**: `Address already in use`

**Solution**: Use a different port or kill the existing process:
```bash
# Kill existing process
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Or use a different port
python run_api.py --port 8001
```

---

### Issue: Slow predictions

**Cause**: Running on CPU instead of GPU

**Solution**: Ensure GPU is available:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

---

## Contributing

When modifying the API:

1. Test changes locally with `python run_api.py`
2. Verify endpoints in Swagger UI: `http://localhost:8000/docs`
3. Use curl/Python to test actual requests
4. Update documentation if endpoints change
5. Commit changes with clear messages

---

## License

See [LICENSE](LICENSE) file for details.

---

**Last Updated**: January 24, 2026
