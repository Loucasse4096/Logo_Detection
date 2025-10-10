"""
Pydantic models for the API
"""

from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class JobStatus(str, Enum):
    """Possible job statuses"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DetectionRequest(BaseModel):
    """Detection request"""
    job_id: str
    video_filename: str
    logo_filename: str


class DetectionResponse(BaseModel):
    """Response after upload"""
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """Job status"""
    job_id: str
    status: JobStatus
    progress: int  # 0-100
    message: str
    result_url: Optional[str] = None
    error: Optional[str] = None


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int
    confidence: float


class Detection(BaseModel):
    """Detection in a frame"""
    frame_number: int
    timestamp: float
    bounding_boxes: List[BoundingBox]


class DetectionResult(BaseModel):
    """Complete detection results"""
    job_id: str
    total_frames: int
    frames_with_logo: int
    detections: List[Detection]
    processing_time: float


# End of models
