"""
Logo Detection API
Simple FastAPI backend for logo detection in videos
"""

import uuid
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import aiofiles

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    DetectionResponse, JobStatusResponse, DetectionResult, JobStatus
)
from .detectors import create_detector, get_available_detectors, benchmark_detectors

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Logo Detection API - NEURONS",
    description="API for logo detection in videos",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
JOBS_FILE = Path("jobs_history.json")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Job persistence functions

def load_jobs_from_file() -> Dict[str, Dict[str, Any]]:
    """Load jobs from file"""
    try:
        if JOBS_FILE.exists():
            with open(JOBS_FILE, 'r', encoding='utf-8') as f:
                jobs_data = json.load(f)
                # Convert list to dict with job_id as key
                jobs_dict = {}
                for job in jobs_data:
                    if 'job_id' in job:
                        jobs_dict[job['job_id']] = job
                return jobs_dict
        return {}
    except Exception as e:
        logger.error(f"Error loading jobs: {e}")
        return {}

def save_jobs_to_file(jobs: Dict[str, Dict[str, Any]]) -> bool:
    """Save jobs to file"""
    try:
        # Convert dict to list for JSON
        jobs_list = []
        for job_id, job_data in jobs.items():
            # Create serializable job copy
            serializable_job = {
                "job_id": job_id,
                "status": str(job_data.get("status", "unknown")).replace("JobStatus.", "").lower(),
                "created_at": job_data.get("created_at", ""),
                "detector": job_data.get("detector", "unknown"),
                "video_file": job_data.get("video_file", ""),
                "logo_file": job_data.get("logo_file", ""),
                "progress": job_data.get("progress", 0),
                "message": job_data.get("message", ""),
                "error": job_data.get("error", ""),
                "total_frames": job_data.get("total_frames", 0),
                "frames_with_logo": job_data.get("frames_with_logo", 0),
                "processing_time": job_data.get("processing_time", 0)
            }
            
            # Add result details if available
            if "detection_result" in job_data and job_data["detection_result"]:
                result = job_data["detection_result"]
                if hasattr(result, 'dict'):
                    serializable_job["detection_result"] = result.dict()
                else:
                    serializable_job["detection_result"] = result
            
            jobs_list.append(serializable_job)
        
        with open(JOBS_FILE, 'w', encoding='utf-8') as f:
            json.dump(jobs_list, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving jobs: {e}")
        return False

def add_or_update_job(job_id: str, job_data: Dict[str, Any]) -> bool:
    """Add or update job in history"""
    try:
        jobs = load_jobs_from_file()
        jobs[job_id] = job_data
        return save_jobs_to_file(jobs)
    except Exception as e:
        logger.error(f"Error updating job: {e}")
        return False

def get_all_jobs() -> List[Dict[str, Any]]:
    """Get all jobs from history"""
    try:
        jobs = load_jobs_from_file()
        return list(jobs.values())
    except Exception as e:
        logger.error(f"Error retrieving jobs: {e}")
        return []

def clear_jobs_history() -> bool:
    """Clear jobs history"""
    try:
        if JOBS_FILE.exists():
            os.remove(JOBS_FILE)
        return True
    except Exception as e:
        logger.error(f"Error deleting history: {e}")
        return False

# Load existing jobs on startup
jobs: Dict[str, Dict[str, Any]] = load_jobs_from_file()
logger.info(f"Loaded {len(jobs)} jobs from persistent history")


# Detection endpoints

@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    logo: UploadFile = File(...),
    detector: str = "hybrid"  # Default hybrid (best for textured backgrounds)
):
    """
    Detect logo in video (async)
    
    - Returns immediately with job_id
    - Processing happens in background
    - detector parameter: sift, orb, color_based, edge_based, hybrid, template_matching
    """
    # Validate detector
    available_detectors = get_available_detectors()
    if detector not in available_detectors:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid detector: {detector}. Available: {available_detectors}"
        )
    
    # Generate unique ID
    job_id = str(uuid.uuid4())
    
    logger.info(f"New job: {job_id}")
    
    # Save files
    video_path = UPLOAD_DIR / f"{job_id}_video.mp4"
    logo_path = UPLOAD_DIR / f"{job_id}_logo.jpg"
    
    async with aiofiles.open(video_path, 'wb') as f:
        await f.write(await video.read())
    
    async with aiofiles.open(logo_path, 'wb') as f:
        await f.write(await logo.read())
    
    # Initialize job
    jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "progress": 0,
        "message": "Waiting",
        "video_path": str(video_path),
        "logo_path": str(logo_path),
        "video_file": video.filename or "unknown",
        "logo_file": logo.filename or "unknown",
        "detector": detector,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result_path": None,
        "error": None
    }
    
    # Start background processing
    background_tasks.add_task(process_video_async, job_id)
    
    return DetectionResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job created, processing started"
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        result_url=f"/download/{job_id}" if job["result_path"] else None,
        error=job["error"]
    )


@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download video with bounding boxes"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if not job["result_path"] or not Path(job["result_path"]).exists():
        raise HTTPException(status_code=404, detail="Result not yet available")
    
    return FileResponse(
        job["result_path"],
        media_type="video/mp4",
        filename=f"result_{job_id}.mp4"
    )


@app.get("/result/{job_id}", response_model=DetectionResult)
async def get_result_details(job_id: str):
    """Get detection details"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not yet completed")
    
    if "detection_result" not in job:
        raise HTTPException(status_code=404, detail="Results not available")
    
    return job["detection_result"]


# Models info

@app.get("/models")
async def list_models():
    """List available models for inference"""
    available_detectors = get_available_detectors()
    
    detector_info = {
        "sift": {
            "name": "SIFT Feature Matching",
            "description": "Robust to scale, rotation, illumination variations with neurons branding validation",
            "robust_to": ["rotation", "scale", "illumination", "false_positives"],
            "speed": "moderate",
            "best_for": "High accuracy with brand-compliant validation"
        },
        "orb": {
            "name": "ORB (Oriented FAST and Rotated BRIEF)",
            "description": "Fast and efficient, perfect for real-time",
            "robust_to": ["rotation", "scale"],
            "speed": "fast",
            "best_for": "Real-time applications, clean backgrounds"
        },
        "color_based": {
            "name": "Color-Based Detection",
            "description": "Segmentation by distinctive logo colors",
            "robust_to": ["textured_backgrounds", "scale", "rotation"],
            "speed": "very_fast",
            "best_for": "Textured backgrounds, colorful logos"
        },
        "edge_based": {
            "name": "Edge-Based Detection",
            "description": "Detection by contours with noise reduction",
            "robust_to": ["textured_backgrounds", "scale"],
            "speed": "fast",
            "best_for": "Textured backgrounds, geometric logos"
        },
        "hybrid": {
            "name": "Hybrid Detection",
            "description": "Combines color and edge detection for maximum robustness",
            "robust_to": ["textured_backgrounds", "scale", "rotation", "illumination"],
            "speed": "moderate",
            "best_for": "Complex backgrounds, maximum accuracy"
        },
        "template_matching": {
            "name": "Template Matching (Optimized)",
            "description": "Fast template matching with pyramid matching and optimized NMS",
            "robust_to": ["scale", "clean_backgrounds"],
            "speed": "very_fast",
            "best_for": "Clean backgrounds, exact logo matches, high performance"
        },
    }
    
    models = []
    for detector in available_detectors:
        if detector in detector_info:
            models.append({
                "id": detector,
                **detector_info[detector]
            })
    
    return {
        "models": models,
        "default": "hybrid",  # Hybrid is best for textured backgrounds
        "recommended_for_textured_backgrounds": ["sift", "hybrid", "color_based", "edge_based"],
        "recommended_for_clean_backgrounds": ["sift", "orb", "template_matching"],
        "total_available": len(models)
    }


@app.post("/benchmark")
async def benchmark_all_detectors(
    video: UploadFile = File(...),
    logo: UploadFile = File(...),
    sample_frames: int = 50
):
    """
    Benchmark all available detectors
    
    - Temporary file upload
    - Test on sample frames
    - Returns performance results
    """
    # Validate parameters
    if sample_frames < 10 or sample_frames > 200:
        raise HTTPException(
            status_code=400,
            detail="sample_frames must be between 10 and 200"
        )
    
    # Save files temporarily
    temp_video = UPLOAD_DIR / f"benchmark_{uuid.uuid4()}_video.mp4"
    temp_logo = UPLOAD_DIR / f"benchmark_{uuid.uuid4()}_logo.jpg"
    
    async with aiofiles.open(temp_video, 'wb') as f:
        await f.write(await video.read())
    
    async with aiofiles.open(temp_logo, 'wb') as f:
        await f.write(await logo.read())
    
    try:
        # Run benchmark
        results = benchmark_detectors(
            video_path=str(temp_video),
            logo_path=str(temp_logo),
            sample_frames=sample_frames
        )
        
        return {
            "benchmark_results": results,
            "sample_frames": sample_frames,
            "total_detectors": len(results)
        }
        
    finally:
        # Clean temporary files
        if temp_video.exists():
            temp_video.unlink()
        if temp_logo.exists():
            temp_logo.unlink()


# Health & info

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok", 
        "service": "logo_detection_inference",
        "active_jobs": len([j for j in jobs.values() if j["status"] == JobStatus.PROCESSING]),
        "total_jobs": len(jobs)
    }


@app.get("/")
async def root():
    """Home page"""
    return {
        "message": "Logo Detection API - Inference",
        "version": "1.0.0",
        "endpoints": {
            "detect": "POST /detect",
            "status": "GET /status/{job_id}",
            "download": "GET /download/{job_id}",
            "models": "GET /models",
            "docs": "/docs"
        }
    }

# Job history endpoints

@app.get("/jobs")
async def get_all_jobs_history():
    """Get complete job history"""
    try:
        all_jobs = get_all_jobs()
        
        # Format jobs for frontend
        formatted_jobs = []
        for job in all_jobs:
            formatted_job = {
                "job_id": job.get("job_id"),
                "status": job.get("status", "unknown"),
                "created_at": job.get("created_at"),
                "detector": job.get("detector", "unknown"),
                "video_file": job.get("video_file", "unknown"),
                "logo_file": job.get("logo_file", "unknown"),
                "total_frames": job.get("total_frames", 0),
                "frames_with_logo": job.get("frames_with_logo", 0),
                "processing_time": job.get("processing_time", 0),
                "progress": job.get("progress", 0),
                "message": job.get("message", ""),
                "error": job.get("error", "")
            }
            formatted_jobs.append(formatted_job)
        
        return {
            "jobs": formatted_jobs,
            "total": len(formatted_jobs),
            "completed": len([j for j in formatted_jobs if j["status"] == "completed"]),
            "failed": len([j for j in formatted_jobs if j["status"] == "failed"]),
            "processing": len([j for j in formatted_jobs if j["status"] == "processing"])
        }
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/jobs")
async def clear_jobs_history_endpoint():
    """Clear job history"""
    try:
        success = clear_jobs_history()
        if success:
            # Clear in-memory dict too
            jobs.clear()
            return {"message": "Job history deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Error during deletion")
    except Exception as e:
        logger.error(f"Error deleting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/jobs/{job_id}")
async def delete_job_endpoint(job_id: str):
    """Delete a specific job"""
    try:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        
        # Delete associated files
        import os
        from pathlib import Path
        
        # Delete video file
        if "video_path" in job and job["video_path"] and os.path.exists(job["video_path"]):
            try:
                os.remove(job["video_path"])
            except Exception as e:
                logger.warning(f"Could not delete video file {job['video_path']}: {e}")
        
        # Delete logo file
        if "logo_path" in job and job["logo_path"] and os.path.exists(job["logo_path"]):
            try:
                os.remove(job["logo_path"])
            except Exception as e:
                logger.warning(f"Could not delete logo file {job['logo_path']}: {e}")
        
        # Delete result video
        if "result_path" in job and job["result_path"] and os.path.exists(job["result_path"]):
            try:
                os.remove(job["result_path"])
            except Exception as e:
                logger.warning(f"Could not delete result file {job['result_path']}: {e}")
        
        # Remove from in-memory dict
        del jobs[job_id]
        
        # Update persistent file
        save_jobs_to_file(jobs)
        
        return {"message": "Job deleted successfully", "detail": f"Job {job_id} and associated files have been deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background processing

def process_video_async(job_id: str):
    """Process video in background"""
    try:
        job = jobs[job_id]
        job["status"] = JobStatus.PROCESSING
        job["message"] = "Processing..."
        
        # Initialize chosen detector
        detector = create_detector(
            detector_type=job["detector"],
            logo_path=job["logo_path"]
        )
        
        # Output path
        output_path = RESULTS_DIR / f"{job_id}_result.mp4"
        
        # Progress callback
        def progress_callback(progress, current, total):
            job["progress"] = progress
            job["message"] = f"Frame {current}/{total}"
            # Save progress to persistent history in real-time
            add_or_update_job(job_id, job)
        
        # Process video
        result = detector.process_video(
            video_path=job["video_path"],
            output_path=str(output_path),
            progress_callback=progress_callback
        )
        
        # Update job
        job["status"] = JobStatus.COMPLETED
        job["progress"] = 100
        job["message"] = "Completed"
        job["result_path"] = str(output_path)
        job["detection_result"] = DetectionResult(
            job_id=job_id,
            total_frames=result["total_frames"],
            frames_with_logo=result["frames_with_logo"],
            detections=result["detections"],
            processing_time=result["processing_time"]
        )
        
        # Save to persistent history
        add_or_update_job(job_id, job)
        
        logger.info(f"Job {job_id} completed: {result['frames_with_logo']} frames with logo")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job["status"] = JobStatus.FAILED
        job["message"] = "Error"
        job["error"] = str(e)
        
        # Save error to history
        add_or_update_job(job_id, job)