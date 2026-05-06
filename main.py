"""
FastAPI Server Launcher for PDF Processing Pipeline
Run with: python main.py
Or use: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn

if __name__ == "__main__":
    # Start FastAPI server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )
