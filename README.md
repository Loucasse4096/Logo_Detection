# Logo Detection System

A complete logo detection system with FastAPI backend and Streamlit frontend, supporting multiple computer vision algorithms (SIFT, ORB, BRISK, SURF, AKAZE).

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs

# Stop services
docker compose down
```

**Access the services:**
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Frontend UI**: http://localhost:8501

### Option 2: Local Development

#### Backend Setup

```bash
cd logo_detection_backend

# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

#### Frontend Setup

```bash
cd logo_detection_frontend

# Install dependencies
pip install -r requirements.txt

# Start the frontend
streamlit run app.py
```

## 🧪 Running Tests

### With Docker

```bash
# Run all tests in container
docker compose exec logo-detection-backend pytest -v

# Run specific test file
docker compose exec logo-detection-backend pytest tests/test_api.py -v

# Run with coverage
docker compose exec logo-detection-backend pytest --cov=app
```

### Local Development

```bash
cd logo_detection_backend

# Run all tests
pytest -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest --cov=app

# Run tests with detailed output
pytest -v --tb=short
```

## 📁 Project Structure

```
senior_ai_engineer_assignment/
├── logo_detection_backend/          # FastAPI backend
│   ├── app/
│   │   ├── main.py                  # FastAPI application
│   │   ├── models.py                # Pydantic models
│   │   └── detectors.py             # Logo detection algorithms
│   ├── tests/                       # Unit tests
│   │   ├── test_api.py              # API endpoint tests
│   │   ├── test_models.py           # Model validation tests
│   │   ├── test_detectors.py        # Detector tests
│   │   └── conftest.py              # Test fixtures
│   ├── uploads/                     # Uploaded files
│   ├── results/                     # Processed videos
│   ├── Dockerfile                   # Docker configuration
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # Backend documentation
├── logo_detection_frontend/         # Streamlit frontend
│   ├── app.py                       # Streamlit application
│   └── Dockerfile                   # Docker configuration
├── docker-compose.yml               # Docker Compose configuration
└── README.md                        # This file
```

## 🔧 API Endpoints

### Logo Detection
- **`POST /detect`** - Detect logo in video
  - Parameters: `video` (file), `logo` (file), `detector` (string)
  - Available detectors: `sift`, `orb`, `brisk`, `surf`, `akaze`

### Job Management
- **`GET /status/{job_id}`** - Get job status
- **`GET /download/{job_id}`** - Download processed video
- **`GET /result/{job_id}`** - Get detection results
- **`GET /jobs`** - List all jobs

### System Info
- **`GET /health`** - Health check
- **`GET /models`** - List available detectors
- **`POST /benchmark`** - Benchmark all detectors

## 🎯 Available Detectors

| Detector | Speed | Robustness |
|----------|-------|------------|
| **ORB** | Fast | Rotation, Scale |
| **SIFT** | Moderate | Rotation, Scale, Illumination |
| **BRISK** | Fast | Rotation, Scale |
| **SURF** | Moderate | Rotation, Scale, Illumination |
| **AKAZE** | Moderate | Rotation, Scale |

**Recommended:** ORB for best performance/accuracy balance.

## 🧪 Test Coverage

The project includes comprehensive unit tests:

- **API Tests** (9 tests): Endpoint validation, error handling
- **Model Tests** (6 tests): Pydantic model validation
- **Detector Tests** (6 tests): Algorithm functionality

**Total: 21 tests** - All passing ✅

### Test Categories

```bash
# API endpoint tests
pytest tests/test_api.py

# Model validation tests  
pytest tests/test_models.py

# Detector algorithm tests
pytest tests/test_detectors.py
```

## 🐳 Docker Details

### Backend Container Features
- **Base Image:** Python 3.11-slim
- **Architecture:** Multi-platform (ARM64/AMD64)
- **Dependencies:** OpenCV, NumPy, FastAPI
- **Health Checks:** Automatic monitoring
- **Volumes:** Persistent uploads/results

### Docker Commands

```bash
# Build all services
docker compose build

# Start with logs
docker compose up

# Run tests in container
docker compose exec logo-detection-backend pytest

# View container logs
docker compose logs -f

# Stop and cleanup
docker compose down
```

## 🔍 Usage Examples

### Using the API

```bash
# Test with curl
curl -X POST "http://localhost:8000/detect" \
  -F "video=@video.mp4" \
  -F "logo=@logo.jpg" \
  -F "detector=orb"

# Check job status
curl "http://localhost:8000/status/{job_id}"

# Download result
curl "http://localhost:8000/download/{job_id}" -o result.mp4
```

### Using Python

```python
import requests

# Upload files
with open('video.mp4', 'rb') as video, open('logo.jpg', 'rb') as logo:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'video': video, 'logo': logo},
        params={'detector': 'orb'}
    )

job_id = response.json()['job_id']

# Check status
status = requests.get(f'http://localhost:8000/status/{job_id}')
print(status.json())
```

## 🛠️ Development

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)
- OpenCV dependencies

### Local Development Setup

```bash
# Clone and setup
git clone <repository>
cd senior_ai_engineer_assignment

# Backend
cd logo_detection_backend
pip install -r requirements.txt
python run.py

# Run tests
pytest -v
```

### Adding New Detectors

1. Create detector class in `app/detectors.py`
2. Add to `create_detector()` function
3. Update tests in `tests/test_detectors.py`
4. Update API documentation

## 📊 Performance

The system supports benchmarking all detectors:

```bash
# Via API
curl -X POST "http://localhost:8000/benchmark" \
  -F "video=@test_video.mp4" \
  -F "logo=@test_logo.jpg" \
  -F "sample_frames=50"
```

## 🐛 Troubleshooting

### Common Issues

**Docker build fails:**
```bash
# Clean and rebuild
docker compose down
docker compose build --no-cache logo-detection-backend
```

**Tests fail:**
```bash
# Check Python version
python --version  # Should be 3.11+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**API not responding:**
```bash
# Check container status
docker compose ps

# View logs
docker compose logs logo-detection-backend
```

## 📝 License

This project is for educational purposes as part of an AI Engineer assignment.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

**Ready to detect logos? Start with `docker compose up -d` and visit http://localhost:8501 for the UI or http://localhost:8000/docs for the API!** 🚀
