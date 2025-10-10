# Logo Detection Frontend - Simplified Version

Simple and clean Streamlit interface for the logo detection system. Two essential views: create jobs and monitor history.

## Features

- **🎯 Simple Interface**: Only two main pages
- **🚀 Job Creation**: Video/logo upload with detector selection
- **📊 Real-Time Tracking**: Live job progress
- **📋 History**: List of all jobs with statuses
- **🤖 Multi-Detectors**: Choice between ORB, SIFT, BRISK, AKAZE
- **📥 Download**: Videos with integrated detections
- **✅ API Validation**: Automatic connectivity verification

## Installation

1. **Navigate to frontend folder**
```bash
cd logo_detection_frontend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start the backend API** (in another terminal)
```bash
cd ../logo_detection_backend
python run.py
```

4. **Launch the frontend**
```bash
streamlit run app.py
```

## Usage

### 🚀 "New Job" Page
1. **Choose a detector**: ORB (recommended), SIFT, BRISK, or AKAZE
2. **Upload files**: Video (MP4/AVI/MOV) + Logo (JPG/PNG)
3. **Start detection**: Real-time tracking with progress
4. **Download result**: Video with bounding boxes

### 📊 "Jobs & History" Page
1. **View statistics**: Total, completed, failed, processing
2. **Job list**: All created jobs with details
3. **Refresh statuses**: Real-time updates
4. **Download results**: Processed videos available

## Project Structure

```
logo_detection_frontend/
├── app.py                    # Main application (all-in-one)
├── requirements.txt          # Minimal dependencies
├── README.md                # Documentation
└── Dockerfile               # Container (optional)
```

## API Integration

The frontend communicates directly with the backend API via simple HTTP calls:

```python
# Health check
requests.get("http://localhost:8000/health")

# Job creation
requests.post("http://localhost:8000/detect", files=files, params={'detector': 'orb'})

# Status tracking
requests.get(f"http://localhost:8000/status/{job_id}")

# Download
requests.get(f"http://localhost:8000/download/{job_id}")
```

## Configuration

### API Endpoint
Default: `http://localhost:8000`

To change the endpoint, modify the `API_BASE_URL` variable in `app.py`.

### Available Detectors
- **ORB**: Fast, recommended (default)
- **SIFT**: Robust
- **BRISK**: Fast
- **AKAZE**: Speed/accuracy tradeoff

## Troubleshooting

### Common Issues

1. **API unavailable**
   - Check that backend is started: `python run.py`
   - Check port 8000

2. **Upload failed**
   - Supported formats: MP4/AVI/MOV for videos, JPG/PNG for logos
   - Check file sizes

3. **Slow processing**
   - ORB is the fastest
   - Check system resources

## Performance

- **ORB**: ~30 fps, 37% detection (recommended)
- **BRISK**: ~8 fps, 20% detection
- **SIFT**: ~4 fps, 7% detection
- **AKAZE**: ~8 fps, 0% detection (context-dependent)

## License

This project is part of the Neurons AI Engineer assignment.
