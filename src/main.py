"""
FastAPI Application for Real-Time Phishing URL Classification
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from pathlib import Path

from .schemas import PredictionRequest, PredictionResponse, ModelInfoResponse, HealthCheckResponse
from .model_loader import ModelManager

# Initialize FastAPI app
app = FastAPI(
    title="Phishing URL Classifier API",
    description="A high-performance REST API for real-time phishing URL detection using Deep Neural Networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager (global state)
model_manager = None


def get_model_paths():
    """Get paths to model files"""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "phishing_mlp_optimized.pt"
    info_path = project_root / "models" / "phishing_mlp_optimized_info.json"
    return str(model_path), str(info_path)


@app.on_event("startup")
async def startup_event():
    """Initialize model on app startup"""
    global model_manager
    
    try:
        model_path, info_path = get_model_paths()
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(info_path):
            raise FileNotFoundError(f"Model files not found. Expected: {model_path} and {info_path}")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        model_manager = ModelManager(model_path, info_path, device=device)
        success = model_manager.load_model()
        
        if not success:
            raise RuntimeError("Failed to load model")
        
        print(f"✓ Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"✗ Failed to load model during startup: {e}")
        raise


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = model_manager is not None and model_manager.model is not None
    
    return HealthCheckResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information and performance metrics"""
    if model_manager is None or model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        info = model_manager.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model info: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a phishing classification prediction
    
    Returns:
    - prediction: 0 = Legitimate URL, 1 = Phishing URL
    - probability: Confidence score (0-1)
    """
    if model_manager is None or model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Validate input features
        if len(request.features) != 50:
            raise ValueError(f"Expected 50 features, got {len(request.features)}")
        
        # Make prediction
        result = model_manager.predict(request.features)
        
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return JSONResponse({
        "message": "Phishing URL Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict"
        }
    })


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )
