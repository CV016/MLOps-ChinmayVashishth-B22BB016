# -----------------------------------------------------------------------------
# Dockerfile for ML/DL Ops Assignment 3 - Fine-Tuning BERT for Classification
# Base: Official Python slim image for smaller size and security updates
# -----------------------------------------------------------------------------

# Use official Python 3.11 slim image (Debian Bookworm)
# Slim variant reduces image size while keeping standard library
FROM python:3.11-slim-bookworm

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
# All subsequent commands run from /app unless overridden
WORKDIR /app

# Install system dependencies required by some Python packages
# (e.g. build tools for compiling native extensions if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first to leverage Docker layer caching
# Rebuild only this layer and below when requirements change
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir reduces image size; upgrade pip for latest resolver
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code (notebook and any local scripts) into the container
# Copy after pip install so code changes do not invalidate dependency layer
COPY . .

# Default command: start a shell so you can run the notebook or verification
# Override when running: e.g. docker run ... python verify_environment.py
CMD ["/bin/bash"]
