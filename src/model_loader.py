"""
Model loading and initialization utilities
"""

import torch
import json
from pathlib import Path
from .model import PhishingMLP


class ModelManager:
    """Manages model loading, inference, and metadata"""
    
    def __init__(self, model_path: str, info_path: str, device: str = "cpu"):
        """
        Initialize the model manager
        
        Args:
            model_path: Path to the saved PyTorch model
            info_path: Path to the model info JSON file
            device: Device to load model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model = None
        self.model_info = None
        self.model_path = model_path
        self.info_path = info_path
        
    def load_model(self):
        """Load model and metadata from disk"""
        try:
            # Load model info
            with open(self.info_path, 'r') as f:
                self.model_info = json.load(f)
            
            # Initialize model with stored hyperparameters
            self.model = PhishingMLP(
                input_size=self.model_info.get('input_size', 50),
                hidden_sizes=self.model_info.get('hidden_sizes', [128, 64, 32]),
                dropout_rate=self.model_info.get('dropout_rate', 0.3)
            ).to(self.device)
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()  # Set to evaluation mode
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, features: list) -> dict:
        """
        Make prediction on input features
        
        Args:
            features: List of 50 numerical features
            
        Returns:
            Dictionary with prediction and probability
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert to tensor
            x = torch.FloatTensor([features]).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                probability = self.model(x).item()
            
            # Convert probability to binary prediction (threshold at 0.5)
            prediction = 1 if probability > 0.5 else 0
            
            return {
                "prediction": prediction,
                "probability": round(probability, 6)
            }
        except Exception as e:
            raise ValueError(f"Error during prediction: {e}")
    
    def get_model_info(self) -> dict:
        """Get model metadata"""
        if self.model_info is None:
            raise RuntimeError("Model info not loaded. Call load_model() first.")
        
        return {
            "model_name": "PhishingMLP",
            **self.model_info
        }
