"""
Phishing URL Classifier API package
"""

from .model import PhishingMLP
from .model_loader import ModelManager
from .main import app

__all__ = ["PhishingMLP", "ModelManager", "app"]
