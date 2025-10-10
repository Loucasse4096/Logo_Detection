#!/usr/bin/env python3
"""
Start the Logo Detection API server
"""

import uvicorn

if __name__ == "__main__":
    print("Starting Logo Detection API")
    print("📍 http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    print("🧪 Test: python test.py ../Video_1.mp4 neurons_logo.jpg")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
