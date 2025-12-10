FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set PYTHONPATH so Python sees the package
ENV PYTHONPATH=/app


WORKDIR /app
RUN mkdir -p /app/data
# Initialize SQLite database
RUN python -m app.db.init_db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
