# ---- 1. Base Stage ----
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt pyproject.toml ./
COPY main.py server.py ./
COPY .env ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy any remaining files (e.g. README, configs)
COPY . .

# ---- 2. Runtime Stage ----
FROM python:3.12-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install runtime system packages
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment dependencies from base image
COPY --from=base /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /app /app

# Default command
CMD ["python", "server.py"]