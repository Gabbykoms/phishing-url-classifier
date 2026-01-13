# Real-Time Phishing URL Classifier: An ML and Economics Perspective

## Project Overview

Phishing remains the most prevalent entry point for cyberattacks, costing the global economy billions annually. This project approaches phishing detection as both a technical and economic challenge—optimizing for the trade-off between **False Positives** (blocking legitimate users) and **False Negatives** (allowing malicious breaches).

This repository contains a **production-ready** end-to-end Machine Learning pipeline that classifies URLs in real-time using deep learning. The project bridges Computer Science and Economics principles, connecting deep learning theory with practical, deployable software.

---

## Project Status

✅ **Completed:**
- Days 1-4: Data exploration, baseline models, neural architecture, and hyperparameter optimization
- FastAPI REST API with full model integration
- Pre-trained optimized neural network (saved and ready for inference)
- Comprehensive Jupyter notebooks with analysis and training logs
- Economic cost-function evaluation framework

🚀 **Current:** API is fully functional and ready for deployment

---

## Technical Architecture

### The Model

A **Multi-Layer Perceptron (MLP)** trained with PyTorch:

```
Input (50 features)
    ↓
Hidden Layer 1: 128 neurons + ReLU + BatchNorm + Dropout(0.3)
    ↓
Hidden Layer 2: 64 neurons + ReLU + BatchNorm + Dropout(0.3)
    ↓
Hidden Layer 3: 32 neurons + ReLU + BatchNorm + Dropout(0.3)
    ↓
Output Layer: 1 neuron + Sigmoid (Binary Classification)
```

**Key Components:**
- **Activation:** ReLU (prevents vanishing gradients)
- **Loss Function:** Binary Cross-Entropy with Logits
- **Optimizer:** Adam with adaptive learning rates
- **Regularization:** Batch Normalization + Dropout (0.3)
- **Training:** Optimized via hyperparameter tuning with recorded results in `models/hyperparameter_tuning_results.csv`

### The Economic Angle: Cost of Misclassification

Rather than optimizing purely for accuracy, this model balances two distinct costs:

| Error Type | Cost | Impact |
|-----------|------|--------|
| **False Positive (FP)** | User friction | Legitimate customer blocked → lost revenue |
| **False Negative (FN)** | Security breach | Phishing link clicked → data exfiltration, ransomware |

The model threshold is tuned to **minimize aggregate economic loss** rather than raw accuracy.

---

## Project Structure

```
phishing-url-classifier/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── model.py                # PyTorch model definition
│   ├── model_loader.py         # Model loading and inference
│   ├── schemas.py              # Pydantic request/response models
│   └── __init__.py
├── models/
│   ├── phishing_mlp_optimized.pt          # Pre-trained model (optimized)
│   ├── phishing_mlp_optimized_info.json   # Model metadata
│   ├── hyperparameter_tuning_results.csv  # Tuning log
│   └── model_info.json                    # Additional info
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis
│   ├── 02_baseline_models.ipynb           # Logistic Regression & Random Forest
│   ├── 03_neural_architecture.ipynb       # MLP Design & Training
│   └── 04_optimization.ipynb              # Hyperparameter Tuning
├── lesson/
│   ├── data.csv / data.json               # Dataset
│   ├── panda_basics.py                    # Data manipulation examples
│   ├── panda_cleaning.py                  # Data cleaning pipeline
│   └── *.md                               # Learning notes from each day
├── requirements.txt
├── run_api.py                  # Entry point for FastAPI
└── README.md                   # This file
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Clone & Install

```bash
git clone https://github.com/Gabbykoms/phishing-url-classifier.git
cd phishing-url-classifier
```

Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the API

### Start the Server

```bash
python run_api.py
```

Or directly with uvicorn:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

- **Interactive Docs:** http://localhost:8000/docs (Swagger UI)
- **ReDoc:** http://localhost:8000/redoc

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://legitimate-example.com",
    "features": [0.5, 0.2, 0.8, ...]  // 50 numerical features
  }'
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Classify a URL; returns phishing probability |
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Get model metadata and performance metrics |

---

## Development Workflow

### Jupyter Notebooks

All analysis and training is documented in the `notebooks/` directory:

1. **01_eda.ipynb** - Feature correlation, class imbalance analysis
2. **02_baseline_models.ipynb** - Logistic Regression & Random Forest benchmarks
3. **03_neural_architecture.ipynb** - MLP design, training, and validation
4. **04_optimization.ipynb** - Hyperparameter tuning with results logging

Run notebooks with Jupyter:
```bash
jupyter notebook notebooks/
```

---

## Model Performance

Detailed results logged in `models/hyperparameter_tuning_results.csv`. The optimized model balances:
- High recall on phishing URLs (minimize false negatives)
- Reasonable false positive rate (minimize legitimate URL blocks)

---

## Technologies Used

| Category | Tools |
|----------|-------|
| **Data** | pandas, NumPy, Matplotlib, Seaborn |
| **ML/DL** | scikit-learn, PyTorch |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Dataset** | UCI ML Phishing URLs Dataset |

---

## About

**Author:** Gabriel Koomson

**Interests:** Machine Learning, Cybersecurity, International Finance

This project demonstrates practical applications of deep learning in production environments with economic considerations.
