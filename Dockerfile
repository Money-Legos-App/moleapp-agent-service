# Alkebulan Agent Service Dockerfile (Standalone — Render deploy)
# Python 3.11 with FastAPI, FAISS, LangGraph

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for FAISS and other native libs
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager for faster dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency files first for better caching
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies using UV
RUN uv pip install --system -e . --no-cache

# Copy application code
COPY app ./app
COPY data_pipeline ./data_pipeline

# Create data directories
RUN mkdir -p /app/data/faiss_index /app/data/raw

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (Render private services default to 10000)
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD wget -q -O - http://localhost:10000/health || exit 1

# Run the application (Render dockerCommand overrides this)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
