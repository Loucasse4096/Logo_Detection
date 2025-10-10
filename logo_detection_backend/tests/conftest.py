"""
Test configuration
Basic fixtures for tests
"""

import pytest
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """Test client for FastAPI"""
    return TestClient(app)

@pytest.fixture
def temp_video():
    """Create temporary video file"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        # Write minimal video data (just for testing)
        f.write(b'fake video content')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def temp_logo():
    """Create temporary logo file"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Write minimal image data (just for testing)
        f.write(b'fake image content')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)
