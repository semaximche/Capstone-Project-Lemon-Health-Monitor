FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create a folder for the SQLite DB
RUN mkdir -p /app/data

ENV PYTHONPATH=/app

EXPOSE 8000

# Run DB init and start server
CMD ["bash", "-c", "python -m app.db.init_db && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
