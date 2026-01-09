"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request schema for URL prediction"""
    features: List[float] = Field(
        ..., 
        description="List of 50 numerical features extracted from the URL",
        min_items=50,
        max_items=50
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5] * 50  # Example with 50 features
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: int = Field(..., description="0 = Legitimate, 1 = Phishing")
    probability: float = Field(..., description="Confidence score (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.9876
            }
        }


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""
    model_name: str
    input_size: int
    hidden_sizes: List[int]
    dropout_rate: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1_score: float
    test_roc_auc: float
    use_batch_norm: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "PhishingMLP",
                "input_size": 50,
                "hidden_sizes": [128, 64, 32],
                "dropout_rate": 0.3,
                "test_accuracy": 0.9999,
                "test_precision": 0.9998,
                "test_recall": 1.0,
                "test_f1_score": 0.9999,
                "test_roc_auc": 1.0,
                "use_batch_norm": True
            }
        }


class HealthCheckResponse(BaseModel):
    """Response schema for health check"""
    status: str = Field(..., description="Health status of the API")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
