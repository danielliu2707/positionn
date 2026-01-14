# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .
COPY similar_player_stats.py .
COPY similar_player_dimensions.py .
COPY models/ ./models/

# Expose port
EXPOSE 8080

# Run the application (Fly.io sets PORT env var, default to 8080)
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}
