# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY iris_model.pkl .

# Expose port (Render uses PORT env variable)
EXPOSE $PORT

# Command to run the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
