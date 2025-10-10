"""
Detector tests
Test detector functionality
"""

import pytest
import tempfile
import os
import cv2
import numpy as np
from app.detectors import (
    create_detector, get_available_detectors, 
    SIFTDetector, ORBDetector
)

def test_get_available_detectors():
    """Test getting available detectors"""
    detectors = get_available_detectors()
    assert isinstance(detectors, list)
    assert len(detectors) > 0
    assert "orb" in detectors  # ORB should always be available

def test_create_detector_orb():
    """Test creating ORB detector"""
    # Create a real image file for testing
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
    
    try:
        cv2.imwrite(temp_path, test_image)
        detector = create_detector("orb", temp_path)
        assert isinstance(detector, ORBDetector)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_create_detector_invalid():
    """Test creating detector with invalid type"""
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
    
    try:
        cv2.imwrite(temp_path, test_image)
        with pytest.raises(ValueError):
            create_detector("invalid_detector", temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_orb_detector_creation():
    """Test ORB detector creation"""
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
    
    try:
        cv2.imwrite(temp_path, test_image)
        detector = ORBDetector(temp_path)
        assert detector.logo_path == temp_path
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_detector_with_nonexistent_file():
    """Test detector with non-existent file"""
    with pytest.raises(ValueError):  # The detector raises ValueError, not FileNotFoundError
        ORBDetector("nonexistent_file.jpg")

def test_detector_detect_in_frame_no_matches():
    """Test detector with frame that has no matches"""
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
    
    try:
        cv2.imwrite(temp_path, test_image)
        detector = ORBDetector(temp_path)
        # Create a simple test frame
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # This should not crash, just return empty results
        detections = detector.detect_in_frame(test_frame)
        assert isinstance(detections, list)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
