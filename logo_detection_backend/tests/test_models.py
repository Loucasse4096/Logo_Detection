"""
Model validation tests
Test Pydantic models
"""

import pytest
from app.models import (
    DetectionRequest, DetectionResponse, JobStatusResponse,
    BoundingBox, Detection, DetectionResult, JobStatus
)

def test_job_status_enum():
    """Test JobStatus enum values"""
    assert JobStatus.PENDING == "pending"
    assert JobStatus.PROCESSING == "processing"
    assert JobStatus.COMPLETED == "completed"
    assert JobStatus.FAILED == "failed"

def test_bounding_box_model():
    """Test BoundingBox model"""
    bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.95)
    assert bbox.x == 10
    assert bbox.y == 20
    assert bbox.width == 100
    assert bbox.height == 50
    assert bbox.confidence == 0.95

def test_detection_model():
    """Test Detection model"""
    bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.95)
    detection = Detection(frame_number=1, timestamp=1.5, bounding_boxes=[bbox])
    
    assert detection.frame_number == 1
    assert detection.timestamp == 1.5
    assert len(detection.bounding_boxes) == 1
    assert detection.bounding_boxes[0] == bbox

def test_detection_result_model():
    """Test DetectionResult model"""
    bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.95)
    detection = Detection(frame_number=1, timestamp=1.5, bounding_boxes=[bbox])
    
    result = DetectionResult(
        job_id="test-job",
        total_frames=100,
        frames_with_logo=5,
        detections=[detection],
        processing_time=2.5
    )
    
    assert result.job_id == "test-job"
    assert result.total_frames == 100
    assert result.frames_with_logo == 5
    assert len(result.detections) == 1
    assert result.processing_time == 2.5

def test_detection_response_model():
    """Test DetectionResponse model"""
    response = DetectionResponse(
        job_id="test-job",
        status=JobStatus.PENDING,
        message="Job created"
    )
    
    assert response.job_id == "test-job"
    assert response.status == JobStatus.PENDING
    assert response.message == "Job created"

def test_job_status_response_model():
    """Test JobStatusResponse model"""
    response = JobStatusResponse(
        job_id="test-job",
        status=JobStatus.PROCESSING,
        progress=50,
        message="Processing...",
        result_url="/download/test-job",
        error=None
    )
    
    assert response.job_id == "test-job"
    assert response.status == JobStatus.PROCESSING
    assert response.progress == 50
    assert response.message == "Processing..."
    assert response.result_url == "/download/test-job"
    assert response.error is None
