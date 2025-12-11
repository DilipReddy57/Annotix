# ANNOTIX Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/

# Create data directories
RUN mkdir -p /tmp/annotix-data /tmp/annotix-models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/tmp/annotix-data
ENV MODELS_DIR=/tmp/annotix-models
ENV ENVIRONMENT=production

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
