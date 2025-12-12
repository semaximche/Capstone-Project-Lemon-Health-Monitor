# Use a slim Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .


ENV PYTHONPATH=/app


EXPOSE 8001

# Run your app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
