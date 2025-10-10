# Logo Detection Backend

## Overview
FastAPI backend for logo detection in videos using multiple computer vision algorithms (SIFT, ORB, Color-based, Edge-based, Hybrid).

## Project Structure

```
logo_detection_backend/
├── app/
│   ├── main.py                  # FastAPI app with all endpoints
│   ├── models.py                # Pydantic schemas
│   └── detectors.py             # Logo detectors (SIFT, ORB, Color, Edge, Hybrid)
├── tests/                       # Unit tests (pytest)
│   ├── conftest.py              # Common fixtures
│   ├── test_models.py           # Pydantic model tests
│   ├── test_detectors.py        # Detector tests
│   ├── test_api.py              # API endpoint tests
│   └── README_TESTS.md          # Test documentation
├── results/                     # Videos with detections
├── uploads/                     # Temporary files
├── requirements.txt             # Dependencies (includes pytest)
├── pytest.ini                   # Pytest configuration
├── run.py                       # Server launcher
├── test.py                      # Functional test script
├── test_benchmark.py            # Benchmark script
└── README.md                    # Documentation
```

## Quick Start

### Option 1: Docker (Recommended)
```bash
# From project root
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs logo-detection-backend
```

### Option 2: Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start server
python run.py

# 3. Test with Hybrid (recommended for textured backgrounds)
python test.py ../Video_1.mp4 neurons_logo.jpg --detector hybrid

# 4. Benchmark all detectors
python test_benchmark.py ../Video_1.mp4 neurons_logo.jpg

# 5. Run unit tests
pytest
```

## Docker

The backend is fully dockerized and can be run with Docker Compose:

```bash
# Build and start
docker compose up --build

# Run tests in container
docker compose exec logo-detection-backend pytest

# View logs
docker compose logs -f logo-detection-backend
```

**Docker Features:**
- Multi-stage build for optimization
- Health checks
- Volume mounts for uploads/results
- Environment configuration

## API Endpoints

### Logo Detection
- **`POST /detect`** - Detect logo in video
  - Parameters: `video` (file), `logo` (file), `detector` (string)
  - Available detectors: `sift`, `orb`, `color_based`, `edge_based`, `hybrid`
  - Returns: `job_id` for async tracking

### Job Tracking
- **`GET /status/{job_id}`** - Get job status
  - Returns: status, progress, message
- **`GET /result/{job_id}`** - Get detection results
  - Returns: frame count, detections, processing time
- **`GET /download/{job_id}`** - Download video with bounding boxes

### Models and Detectors
- **`GET /models`** - List available detectors
  - Returns: detectors, descriptions, speeds

### Benchmark
- **`POST /benchmark`** - Compare all detectors
  - Parameters: `video` (file), `logo` (file), `sample_frames` (int)
  - Returns: performance results for each detector

### System
- **`GET /health`** - Health check
- **`GET /`** - General API information


## Testing

### Unit Tests (pytest)
```bash
# Run all tests
pytest

# Run with verbosity
pytest -v

# Run specific module
pytest tests/test_models.py
pytest tests/test_detectors.py
pytest tests/test_api.py

# Run with code coverage
pytest --cov=app --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

**Complete documentation**: See [tests/README_TESTS.md](tests/README_TESTS.md)

### Functional Test Scripts

#### Simple Test
```bash
# See available models
python test.py ../Video_1.mp4 neurons_logo.jpg --models

# Test with Hybrid (default, best for textured backgrounds)
python test.py ../Video_1.mp4 neurons_logo.jpg --detector hybrid

# Test with different detectors
python test.py ../Video_1.mp4 neurons_logo.jpg --detector sift
python test.py ../Video_1.mp4 neurons_logo.jpg --detector color_based
python test.py ../Video_1.mp4 neurons_logo.jpg --detector edge_based
```

#### Complete Benchmark
```bash
# Quick benchmark (30 frames)
python test_benchmark.py ../Video_1.mp4 neurons_logo.jpg

# Extended benchmark (100 frames)
python test_benchmark.py ../Video_1.mp4 neurons_logo.jpg --sample-frames 100
```

## Technical Architecture

### Unified Detectors (`detectors.py`)
- **Base class**: `BaseDetector` to standardize interface
- **Factory pattern**: `create_detector()` for dynamic instantiation
- **Integrated benchmark**: `benchmark_detectors()` for comparisons

### Async API (`main.py`)
- **Background processing**: Async jobs with FastAPI BackgroundTasks
- **Dynamic validation**: Check available detectors
- **Error handling**: Pydantic validation and exception handling

### Pydantic Models (`models.py`)
- **Standardized structures**: JobStatus, DetectionResult, BoundingBox
- **Automatic validation**: Type and format control
- **Integrated documentation**: Automatic OpenAPI schemas

## Key Points

1. **Multiple detection strategies**: 5 different algorithms (SIFT, ORB, Color-based, Edge-based, Hybrid)
2. **Hybrid detector**: Combines color + edge detection for maximum accuracy on textured backgrounds
3. **Brand guideline validation**: SIFT detector includes structural validation (Safe Zone, aspect ratio)
4. **Flexible architecture**: Detector choice via API
5. **Optimized for Neurons logo**: Color ranges based on official brand colors
6. **Maintainable code**: Unified and extensible detector architecture
7. **Complete unit tests**: Comprehensive test coverage with pytest
