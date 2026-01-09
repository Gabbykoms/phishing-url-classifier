"""
Script to run the FastAPI application
Usage: python run_api.py
"""

import uvicorn
import sys


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
