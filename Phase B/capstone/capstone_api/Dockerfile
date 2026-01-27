# Stage 1: Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY capstone_api/pyproject.toml ./
COPY capstone_api/app ./app

# Install dependencies using uv (system-wide installation, no venv)
# This reads dependencies from pyproject.toml
RUN uv pip install --system --no-cache -e .

# Stage 2: Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install only minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (only what's needed at runtime)
COPY capstone_api/app ./app

# Clean up Python cache and optimize
RUN find /usr/local -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local -type f -name "*.pyc" -delete && \
    find /usr/local -type f -name "*.pyo" -delete

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run DB init and start server
CMD ["bash", "-c", "python -m app.db.init_db && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
