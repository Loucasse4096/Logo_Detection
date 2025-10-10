"""
API endpoint tests
Simple tests for main endpoints
"""

import pytest
from fastapi.testclient import TestClient

def test_health_endpoint(client: TestClient):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data

def test_root_endpoint(client: TestClient):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data

def test_models_endpoint(client: TestClient):
    """Test models endpoint"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "default" in data
    assert "total_available" in data
    assert len(data["models"]) > 0

def test_detect_endpoint_missing_files(client: TestClient):
    """Test detect endpoint with missing files"""
    response = client.post("/detect")
    assert response.status_code == 422  # Validation error

def test_detect_endpoint_invalid_detector(client: TestClient, temp_video, temp_logo):
    """Test detect endpoint with invalid detector"""
    with open(temp_video, 'rb') as video, open(temp_logo, 'rb') as logo:
        response = client.post(
            "/detect",
            files={"video": video, "logo": logo},
            params={"detector": "invalid_detector"}
        )
    assert response.status_code == 400

def test_detect_endpoint_valid(client: TestClient, temp_video, temp_logo):
    """Test detect endpoint with valid files"""
    with open(temp_video, 'rb') as video, open(temp_logo, 'rb') as logo:
        response = client.post(
            "/detect",
            files={"video": video, "logo": logo},
            params={"detector": "orb"}
        )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "status" in data

def test_status_endpoint_not_found(client: TestClient):
    """Test status endpoint with non-existent job"""
    response = client.get("/status/non-existent-job")
    assert response.status_code == 404

def test_download_endpoint_not_found(client: TestClient):
    """Test download endpoint with non-existent job"""
    response = client.get("/download/non-existent-job")
    assert response.status_code == 404

def test_jobs_endpoint(client: TestClient):
    """Test jobs history endpoint"""
    response = client.get("/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert "total" in data

def test_delete_job_endpoint(client: TestClient):
    """Test delete specific job endpoint"""
    # Test deleting non-existent job
    response = client.delete("/jobs/non-existent-job")
    assert response.status_code == 404
    
    # Test deleting existing job (if any exist)
    jobs_response = client.get("/jobs")
    if jobs_response.status_code == 200:
        jobs_data = jobs_response.json()
        if jobs_data.get("jobs"):
            job_id = jobs_data["jobs"][0]["job_id"]
            response = client.delete(f"/jobs/{job_id}")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "Job deleted successfully" in data["message"]
