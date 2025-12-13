# Use a slim Python image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency installation
RUN pip install uv

# Copy project files
COPY inference/pyproject.toml ./
COPY inference/inference ./inference

# Install dependencies from requirements.txt if it exists, otherwise from pyproject.toml
COPY inference/inference/app/requirements.txt ./
RUN uv pip install --system -r requirements.txt

# data_storage will be mounted at runtime, no need to copy

ENV PYTHONPATH=/app

EXPOSE 8001

# Run the inference worker
CMD ["python", "-m", "inference.app.main"]
