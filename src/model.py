"""
PyTorch Model Definition for Phishing URL Classifier
"""

import torch
import torch.nn as nn


class PhishingMLP(nn.Module):
    """
    Multi-Layer Perceptron for Phishing URL Classification
    
    Architecture:
    - Input: 50 features
    - Hidden layers: 128 -> 64 -> 32 (with ReLU activation)
    - Batch Normalization after each hidden layer
    - Dropout (0.3) for regularization
    - Output: 1 (sigmoid for binary classification)
    """
    
    def __init__(self, input_size=50, hidden_sizes=None, dropout_rate=0.3):
        super(PhishingMLP, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        layers = []
        prev_size = input_size
        
        # Hidden layers with batch normalization
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))
