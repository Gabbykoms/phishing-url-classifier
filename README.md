# Real-Time Phishing URL Classifier: An ML and Economics Perspective

## Project Overview

Phishing remains the most prevalent entry point for cyberattacks, costing the global economy billions annually. This project approaches phishing detection as both a technical and economic challenge—optimizing for the trade-off between **False Positives** (blocking legitimate users) and **False Negatives** (allowing malicious breaches).

This repository contains an end-to-end Machine Learning pipeline that classifies URLs in real-time. This project serves as a practical application of Computer Science and Economics principles, bridging the gap between deep learning theory and production-grade software.

---

## 7-Day Roadmap (Winter 2026 Sprinting)

**Day 1: Data and EDA**  
Feature correlation and Class Imbalance • *Pandas, Matplotlib*

**Day 2: Statistical Baseline**  
Logistic Regression and Random Forest • *Scikit-Learn*

**Day 3: Neural Architecture**  
Multi-Layer Perceptron (MLP) Design • *PyTorch, ReLU*

**Day 4: Optimization**  
Hyperparameter Tuning and Model Saving • *PyTorch, NumPy*

**Day 5: API Engineering**  
Creating a high-performance REST API • *FastAPI, Pydantic*

**Day 6: Cloud Deployment**  
Containerization and Model Hosting • *Docker, HuggingFace*

**Day 7: Documentation**  
Technical Writing and Economic Impact Analysis • *Markdown*

---

## Technical Deep Dive

### The Model Architecture

The core classifier is a Deep Neural Network built in PyTorch.

- **Activation Function:** ReLU (Rectified Linear Unit) to prevent vanishing gradients
- **Loss Function:** Binary Cross-Entropy (BCE) with Logits
- **Optimization:** Adam Optimizer for adaptive learning rates

### The Economic Angle: The Cost of Misclassification

In this project, the model is evaluated through the lens of a Cybersecurity Cost Function:

- **Type I Error (False Positive):** High friction. A legitimate customer is blocked, leading to lost revenue and user frustration.
- **Type II Error (False Negative):** High risk. A phishing link is clicked, leading to data exfiltration, ransomware, or financial loss.

**Goal:** Tune the model threshold to minimize the aggregate Economic Loss rather than just maximizing raw Accuracy.

---

## How to Run Locally

### Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/phishing-url-classifier-ml.git
cd phishing-url-classifier
```

### Set up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn app.main:app --reload
```

---

## About the Author
Gabriel Koomson

**Interests:** Machine Learning, Cybersecurity (MobyPhish Research), International Finance.