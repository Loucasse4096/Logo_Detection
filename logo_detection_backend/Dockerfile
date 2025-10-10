FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including ffmpeg for H.264 codec support
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && (apt-get install -y libgl1-mesa-glx || apt-get install -y libgl1) \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results temp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "run.py"]
