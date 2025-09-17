# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY main.py .
COPY iris_model.pkl .

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]