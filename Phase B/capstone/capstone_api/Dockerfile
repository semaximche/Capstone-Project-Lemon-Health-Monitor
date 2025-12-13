FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY capstone_api/pyproject.toml ./
COPY capstone_api/app ./app
COPY capstone_api/requirements.txt* ./

# Install dependencies using uv
RUN uv sync

# data_storage will be mounted at runtime, no need to copy

ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Run DB init and start server
CMD ["bash", "-c", "python -m app.db.init_db && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
