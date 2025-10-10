#!/bin/bash

echo "🔄 Rebuilding and restarting Docker services..."

# Stop all services
echo "⏹️  Stopping services..."
docker compose down

# Build all services
echo "🔨 Building services..."
docker compose build

# Start all services
echo "🚀 Starting services..."
docker compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check status
echo "📊 Service status:"
docker compose ps

echo "✅ Rebuild complete!"
echo "🌐 Frontend: http://localhost:8501"
echo "🔧 Backend API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
